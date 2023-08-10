
import asyncio
import logging
import time
from datetime import datetime
from threading import Thread
from typing import Dict, Set

import ccxt

from freqtrade.constants import Config, PairWithTimeframe
from freqtrade.enums.candletype import CandleType
from freqtrade.exchange.exchange import timeframe_to_seconds
from freqtrade.exchange.types import OHLCVResponse


logger = logging.getLogger(__name__)


class ExchangeWS:
    def __init__(self, config: Config, ccxt_object: ccxt.Exchange) -> None:
        self.config = config
        self.ccxt_object = ccxt_object
        self._background_tasks: Set[asyncio.Task] = set()

        self._klines_watching: Set[PairWithTimeframe] = set()
        self._klines_scheduled: Set[PairWithTimeframe] = set()
        self.klines_last_refresh: Dict[PairWithTimeframe, float] = {}
        self.klines_last_request: Dict[PairWithTimeframe, float] = {}
        self._thread = Thread(name="ccxt_ws", target=self._start_forever)
        self._thread.start()

    def _start_forever(self) -> None:
        self._loop = asyncio.new_event_loop()
        try:
            self._loop.run_forever()
        finally:
            if self._loop.is_running():
                self._loop.stop()

    def cleanup(self) -> None:
        logger.debug("Cleanup called - stopping")
        self._klines_watching.clear()
        if hasattr(self, '_loop'):
            self._loop.call_soon_threadsafe(self._loop.stop)

        self._thread.join()
        logger.debug("Stopped")

    def cleanup_expired(self) -> None:
        """
        Remove pairs from watchlist if they've not been requested within
        the last timeframe (+ offset)
        """
        for p in list(self._klines_watching):
            _, timeframe, _ = p
            timeframe_s = timeframe_to_seconds(timeframe)
            last_refresh = self.klines_last_request.get(p, 0)
            if last_refresh > 0 and time.time() - last_refresh > timeframe_s + 20:
                logger.info(f"Removing {p} from watchlist")
                self._klines_watching.discard(p)

    async def _schedule_while_true(self) -> None:

        for p in self._klines_watching:
            if p not in self._klines_scheduled:
                self._klines_scheduled.add(p)
                pair, timeframe, candle_type = p
                task = asyncio.create_task(
                    self._continuously_async_watch_ohlcv(pair, timeframe, candle_type))
                self._background_tasks.add(task)
                task.add_done_callback(self._continuous_stopped)

    def _continuous_stopped(self, task: asyncio.Task):
        self._background_tasks.discard(task)
        result = task.result()
        logger.info(f"Task finished {result}")
        # self._pairs_scheduled.discard(pair, timeframe, candle_type)

    async def _continuously_async_watch_ohlcv(
            self, pair: str, timeframe: str, candle_type: CandleType) -> None:
        try:
            while (pair, timeframe, candle_type) in self._klines_watching:
                start = time.time()
                data = await self.ccxt_object.watch_ohlcv(pair, timeframe)
                self.klines_last_refresh[(pair, timeframe, candle_type)] = time.time()
                # logger.info(
                #     f"watch done {pair}, {timeframe}, data {len(data)} "
                #     f"in {time.time() - start:.2f}s")
        except ccxt.BaseError:
            logger.exception("Exception in continuously_async_watch_ohlcv")
        finally:
            self._klines_watching.discard((pair, timeframe, candle_type))

    def schedule_ohlcv(self, pair: str, timeframe: str, candle_type: CandleType) -> None:
        """
        Schedule a pair/timeframe combination to be watched
        """
        self._klines_watching.add((pair, timeframe, candle_type))
        self.klines_last_request[(pair, timeframe, candle_type)] = time.time()
        # asyncio.run_coroutine_threadsafe(self.schedule_schedule(), loop=self._loop)
        asyncio.run_coroutine_threadsafe(self._schedule_while_true(), loop=self._loop)
        self.cleanup_expired()

    async def get_ohlcv(
            self,
            pair: str,
            timeframe: str,
            candle_type: CandleType,
            candle_date: int,
    ) -> OHLCVResponse:
        """
        Returns cached klines from ccxt's "watch" cache.
        :param candle_date: timestamp of the end-time of the candle.
        """
        candles = self.ccxt_object.ohlcvs.get(pair, {}).get(timeframe)
        refresh_date = self.klines_last_refresh[(pair, timeframe, candle_type)]
        drop_hint = False
        if refresh_date > candle_date:
            # Refreshed after candle was complete.
            logger.info(f"{candles[-1][0] // 1000} >= {candle_date}")
            drop_hint = (candles[-1][0] // 1000) >= candle_date
        logger.info(
            f"watch result for {pair}, {timeframe} with length {len(candles)}, "
            f"{datetime.fromtimestamp(candles[-1][0] // 1000)}, "
            f"lref={datetime.fromtimestamp(self.klines_last_refresh[(pair, timeframe, candle_type)])}"
            f"candle_date={datetime.fromtimestamp(candle_date)}, {drop_hint=}"
            )
        return pair, timeframe, candle_type, candles, drop_hint
