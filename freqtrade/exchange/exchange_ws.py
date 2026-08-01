import asyncio
import logging
from copy import deepcopy
from functools import partial
from threading import Event, RLock, Thread

import ccxt

from freqtrade.constants import Config, PairWithTimeframe
from freqtrade.enums.candletype import CandleType
from freqtrade.exceptions import TemporaryError
from freqtrade.exchange.common import retrier
from freqtrade.exchange.exchange import timeframe_to_seconds
from freqtrade.exchange.exchange_types import OHLCVResponse
from freqtrade.util import dt_ts, format_ms_time, format_ms_time_det


logger = logging.getLogger(__name__)


class ExchangeWS:
    def __init__(self, config: Config, ccxt_object: ccxt.Exchange) -> None:
        self.config = config
        self._ccxt_object = ccxt_object
        self._background_tasks: set[asyncio.Task] = set()
        self._state_lock = RLock()
        self._loop_ready = Event()

        self._klines_watching: set[PairWithTimeframe] = set()
        self._klines_scheduled: set[PairWithTimeframe] = set()
        self._klines_last_refresh: dict[PairWithTimeframe, float] = {}
        self._klines_last_request: dict[PairWithTimeframe, float] = {}
        self._thread = Thread(name="ccxt_ws", target=self._start_forever)
        self._thread.start()

    def _start_forever(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._loop_ready.set()
        try:
            self._loop.run_forever()
        finally:
            if not self._loop.is_closed():
                # Cancel remaining tasks and close the loop in the owning thread.
                pending = asyncio.all_tasks(self._loop)
                for task in pending:
                    task.cancel()
                if pending:
                    self._loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                self._loop.run_until_complete(self._loop.shutdown_asyncgens())
                self._loop.close()
            self._loop_ready.clear()

    def _wait_for_loop(self, timeout: float = 1.0) -> bool:
        """
        Wait for the event loop to be ready
        Returns True once the loop is ready.
        Will probably only return false during startup/shutdown.
        """
        if hasattr(self, "_loop"):
            return True
        return self._loop_ready.wait(timeout=timeout) and hasattr(self, "_loop")

    def cleanup(self) -> None:
        logger.debug("Cleanup called - stopping")
        with self._state_lock:
            self._klines_watching.clear()
            tasks = list(self._background_tasks)
        for task in tasks:
            task.cancel()
        if self._wait_for_loop(timeout=0.2) and not self._loop.is_closed():
            self.reset_connections(cleanup=True)
            self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5)
        if self._thread.is_alive():
            logger.warning("Websocket loop thread did not stop within timeout.")
        logger.debug("Stopped")

    def reset_connections(self, cleanup: bool = False) -> None:
        """
        Reset all connections - avoids "connection-reset" errors that happen after ~9 days
        """
        if self._wait_for_loop() and not self._loop.is_closed():
            logger.info(f"{'Cleaning up' if cleanup else 'Resetting'} exchange WS connections.")
            try:
                fut = asyncio.run_coroutine_threadsafe(self._cleanup_async(), loop=self._loop)
                fut.result(timeout=10)
            except TimeoutError:
                logger.warning("Timed out while resetting websocket connections.")
            except Exception:
                logger.exception("Exception while resetting websocket connections")

    async def _cleanup_async(self) -> None:
        try:
            await self._ccxt_object.close()
            # Clear the cache.
            # Not doing this will cause problems on startup with dynamic pairlists
            self._ccxt_object.ohlcvs.clear()
        except Exception:
            logger.exception("Exception in _cleanup_async")

    def _pop_history(self, paircomb: PairWithTimeframe) -> None:
        """
        Remove history for a pair/timeframe combination from ccxt cache
        """
        with self._state_lock:
            self._ccxt_object.ohlcvs.get(paircomb[0], {}).pop(paircomb[1], None)
            self._klines_last_refresh.pop(paircomb, None)

    @retrier(retries=3)
    def ohlcvs(self, pair: str, timeframe: str) -> list[list]:
        """
        Returns a copy of the klines for a pair/timeframe combination
        Note: this will only contain the data received from the websocket
            so the data will build up over time.
        """
        try:
            return deepcopy(self._ccxt_object.ohlcvs.get(pair, {}).get(timeframe, []))
        except RuntimeError as e:
            # Capture runtime errors and retry
            # TemporaryError does not cause backoff - so we're essentially retrying immediately
            raise TemporaryError(f"Error deepcopying: {e}") from e

    def get_ohlcv_with_refresh(
        self, pair: str, timeframe: str, candle_type: CandleType
    ) -> tuple[list[list], float]:
        """
        Get deepcopied klines and update the last refresh time
        """
        ohlcvs = self.ohlcvs(pair, timeframe)
        with self._state_lock:
            last_refresh = self._klines_last_refresh.get((pair, timeframe, candle_type), 0)
        return ohlcvs, last_refresh

    def cleanup_expired(self) -> None:
        """
        Remove pairs from watchlist if they've not been requested within
        the last timeframe (+ offset)
        """
        changed = False
        with self._state_lock:
            for p in list(self._klines_watching):
                _, timeframe, _ = p
                timeframe_s = timeframe_to_seconds(timeframe)
                last_refresh = self._klines_last_request.get(p, 0)
                if last_refresh > 0 and (dt_ts() - last_refresh) > ((timeframe_s + 20) * 1000):
                    logger.info(f"Removing {p} from websocket watchlist.")
                    self._klines_watching.discard(p)
                    # Pop history to avoid getting stale data
                    self._pop_history(p)
                    changed = True
        if changed:
            logger.info(f"Removal done: new watch list ({len(self._klines_watching)})")

    async def _schedule_while_true(self) -> None:
        # For the ones we should be watching
        with self._state_lock:
            pairs_to_check = list(self._klines_watching)

        for p in pairs_to_check:
            # Check if they're already scheduled
            with self._state_lock:
                if p in self._klines_scheduled:
                    continue
                self._klines_scheduled.add(p)
            pair, timeframe, candle_type = p
            task = asyncio.create_task(
                self._continuously_async_watch_ohlcv(pair, timeframe, candle_type)
            )
            with self._state_lock:
                self._background_tasks.add(task)
            task.add_done_callback(
                partial(
                    self._continuous_stopped,
                    pair=pair,
                    timeframe=timeframe,
                    candle_type=candle_type,
                )
            )

    def exchange_has(self, endpoint: str) -> bool:
        """
        Checks if exchange implements a specific API endpoint.
        Wrapper around ccxt 'has' attribute
        :param endpoint: Name of endpoint (e.g. 'fetchOHLCV', 'fetchTickers')
        :return: bool
        """
        return endpoint in self._ccxt_object.has and self._ccxt_object.has[endpoint]

    async def _unwatch_ohlcv(self, pair: str, timeframe: str, candle_type: CandleType) -> None:
        try:
            if self.exchange_has("unWatchOHLCVForSymbols"):
                await self._ccxt_object.un_watch_ohlcv_for_symbols([[pair, timeframe]])
            elif self.exchange_has("unWatchOHLCV"):
                await self._ccxt_object.un_watch_ohlcv(pair, timeframe)
            else:
                logger.debug("un_watch_ohlcv not supported for %s, %s", pair, timeframe)

        except ccxt.NotSupported as e:
            logger.debug("un_watch_ohlcv_for_symbols not supported: %s", e)
            pass
        except ccxt.NetworkError as e:
            # Network errors are common on shutdown so we can ignore them.
            # It's a network error - which most likely means that the connection is already closed.
            logger.debug("Network error during unwatch for %s, %s: %s", pair, timeframe, e)
        except Exception:
            logger.exception(f"Exception in _unwatch_ohlcv for {pair}, {timeframe},")

    def _continuous_stopped(
        self, task: asyncio.Task, pair: str, timeframe: str, candle_type: CandleType
    ) -> None:
        with self._state_lock:
            self._background_tasks.discard(task)
        result = "done"
        try:
            if task.cancelled():
                result = "cancelled"
            else:
                if (result1 := task.result()) is not None:
                    result = str(result1)
        except Exception:
            result = "error"
            logger.exception(f"Unhandled exception in watch task callback for {pair}, {timeframe}")
        finally:
            logger.info(f"{pair}, {timeframe}, {candle_type} - Task finished - {result}")
            if hasattr(self, "_loop") and not self._loop.is_closed():
                asyncio.run_coroutine_threadsafe(
                    self._unwatch_ohlcv(pair, timeframe, candle_type), loop=self._loop
                )

            with self._state_lock:
                self._klines_scheduled.discard((pair, timeframe, candle_type))
            self._pop_history((pair, timeframe, candle_type))

    async def _continuously_async_watch_ohlcv(
        self, pair: str, timeframe: str, candle_type: CandleType
    ) -> None:
        try:
            while True:
                with self._state_lock:
                    if (pair, timeframe, candle_type) not in self._klines_watching:
                        break
                start = dt_ts()
                data = await self._ccxt_object.watch_ohlcv(pair, timeframe)
                with self._state_lock:
                    self._klines_last_refresh[(pair, timeframe, candle_type)] = dt_ts()
                logger.debug(
                    f"watch done {pair}, {timeframe}, data {len(data)} "
                    f"in {(dt_ts() - start) / 1000:.3f}s"
                )
        except ccxt.ExchangeClosedByUser:
            logger.debug("Exchange connection closed by user")
        except ccxt.BaseError:
            logger.exception(f"Exception in continuously_async_watch_ohlcv for {pair}, {timeframe}")
        finally:
            with self._state_lock:
                self._klines_watching.discard((pair, timeframe, candle_type))

    def schedule_ohlcv(self, pair: str, timeframe: str, candle_type: CandleType) -> None:
        """
        Schedule a pair/timeframe combination to be watched
        """
        if not self._wait_for_loop():
            logger.warning(f"Websocket loop not ready. Could not schedule {pair}, {timeframe}.")
            return
        with self._state_lock:
            self._klines_watching.add((pair, timeframe, candle_type))
            self._klines_last_request[(pair, timeframe, candle_type)] = dt_ts()
        # asyncio.run_coroutine_threadsafe(self.schedule_schedule(), loop=self._loop)
        asyncio.run_coroutine_threadsafe(self._schedule_while_true(), loop=self._loop)
        self.cleanup_expired()

    async def get_ohlcv(
        self,
        pair: str,
        timeframe: str,
        candle_type: CandleType,
        candle_ts: int,
    ) -> OHLCVResponse:
        """
        Returns cached klines from ccxt's "watch" cache.
        :param candle_ts: timestamp of the end-time of the candle we expect.
        """
        candles, refresh_date = self.get_ohlcv_with_refresh(pair, timeframe, candle_type)
        received_ts = candles[-1][0] if candles else 0
        drop_hint = received_ts >= candle_ts
        if refresh_date and received_ts > refresh_date:
            logger.warning(
                f"{pair}, {timeframe} - Candle date > last refresh "
                f"({format_ms_time(received_ts)} > {format_ms_time_det(refresh_date)}). "
                "This usually suggests a problem with time synchronization."
            )
        logger.debug(
            f"watch result for {pair}, {timeframe} with length {len(candles)}, "
            f"r_ts={format_ms_time(received_ts)}, "
            f"lref={format_ms_time_det(refresh_date)}, "
            f"candle_ts={format_ms_time(candle_ts)}, {drop_hint=}"
        )
        return pair, timeframe, candle_type, candles, drop_hint
