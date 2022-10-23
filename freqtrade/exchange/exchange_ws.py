
import asyncio
import logging
import time
from threading import Thread
from typing import Dict, List, Set, Tuple

from freqtrade.constants import Config
from freqtrade.enums.candletype import CandleType


logger = logging.getLogger(__name__)


class ExchangeWS():
    def __init__(self, config: Config, ccxt_object) -> None:
        self.config = config
        self.ccxt_object = ccxt_object
        self._thread = Thread(name="ccxt_ws", target=self.start)
        self._background_tasks = set()
        self._pairs_watching: Set[Tuple[str, str, CandleType]] = set()
        self._pairs_scheduled: Set[Tuple[str, str, CandleType]] = set()
        self.pairs_last_refresh: Dict[Tuple[str, str, CandleType], int] = {}
        self._thread.start()

    def start(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._loop.run_forever()

    def cleanup(self) -> None:
        logger.debug("Cleanup called - stopping")
        self._pairs_watching.clear()
        self._loop.stop()
        self._thread.join()
        logger.debug("Stopped")

# One task per Watch
    # async def schedule_schedule(self) -> None:

    #     for p in self._pairs_watching:
    #         if p not in self._pairs_scheduled:
    #             self._pairs_scheduled.add(p)
    #             await self.schedule_one_task(p[0], p[1], p[2])

    # async def schedule_one_task(self, pair: str, timeframe: str, candle_type: CandleType) -> None:
    #     task = asyncio.create_task(self._async_watch_ohlcv(pair, timeframe, candle_type))

    #     # Add task to the set. This creates a strong reference.
    #     self._background_tasks.add(task)
    #     task.add_done_callback(self.reschedule_or_stop)

    # async def _async_watch_ohlcv(self, pair: str, timeframe: str,
    #                              candle_type: CandleType) -> Tuple[str, str, str, List]:
    #     start = time.time()
    #     data = await self.ccxt_object.watch_ohlcv(pair, timeframe, )
    #     logger.info(f"watch done {pair}, {timeframe}, data {len(data)} in {time.time() - start:.2f}s")
    #     return pair, timeframe, candle_type, data

    # def reschedule_or_stop(self, task: asyncio.Task):
    #     # logger.info(f"Task finished {task}")

    #     self._background_tasks.discard(task)
    #     pair, timeframe, candle_type, data = task.result()

    #     # reschedule
    #     asyncio.run_coroutine_threadsafe(self.schedule_one_task(
    #         pair, timeframe, candle_type), loop=self._loop)

# End one task epr watch

    async def schedule_while_true(self) -> None:

        for p in self._pairs_watching:
            if p not in self._pairs_scheduled:
                self._pairs_scheduled.add(p)
                pair, timeframe, candle_type = p
                task = asyncio.create_task(
                    self.continuously_async_watch_ohlcv(pair, timeframe, candle_type))
                self._background_tasks.add(task)
                task.add_done_callback(self.continuous_stopped)

    def continuous_stopped(self, task: asyncio.Task):
        self._background_tasks.discard(task)
        result = task.result()
        logger.info(f"Task finished {result}")
        # self._pairs_scheduled.discard(pair, timeframe, candle_type)

        logger.info(f"Task finished {task}")

    async def continuously_async_watch_ohlcv(
            self, pair: str, timeframe: str, candle_type: CandleType) -> Tuple[str, str, str, List]:

        while (pair, timeframe, candle_type) in self._pairs_watching:
            logger.info(self._pairs_watching)
            start = time.time()
            data = await self.ccxt_object.watch_ohlcv(pair, timeframe)
            self.pairs_last_refresh[(pair, timeframe, candle_type)] = time.time()
            logger.info(
                f"watch1 done {pair}, {timeframe}, data {len(data)} in {time.time() - start:.2f}s")

    def schedule_ohlcv(self, pair: str, timeframe: str, candle_type: CandleType) -> None:
        self._pairs_watching.add((pair, timeframe, candle_type))
        # asyncio.run_coroutine_threadsafe(self.schedule_schedule(), loop=self._loop)
        asyncio.run_coroutine_threadsafe(self.schedule_while_true(), loop=self._loop)

