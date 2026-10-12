"""Regression coverage for cancellation during connection resets."""

import asyncio
import gc
from unittest.mock import AsyncMock, MagicMock

import pytest

from freqtrade.enums import CandleType
from freqtrade.exchange.exchange_ws import ExchangeWS


@pytest.mark.parametrize("pair_count", [1, 16])
async def test_reset_does_not_create_unwatch_tasks_after_close(mocker, pair_count):
    # Given: real watcher tasks awaiting exchange data on the current event loop.
    loop = asyncio.get_running_loop()
    started = asyncio.Event()
    waiting = []
    destroyed = []
    client = MagicMock()
    client.ohlcvs = {}
    client.has = {"unWatchOHLCVForSymbols": True}

    async def watch(*args):
        future = loop.create_future()
        waiting.append(future)
        if len(waiting) == pair_count:
            started.set()
        await future

    async def close():
        for future in waiting:
            future.cancel()
        await asyncio.gather(*list(ws._background_tasks), return_exceptions=True)

    async def unwatch(*args):
        # An unsubscribe on a closed transport has no peer to acknowledge it.
        await loop.create_future()

    client.watch_ohlcv = AsyncMock(side_effect=watch)
    client.close = AsyncMock(side_effect=close)
    client.un_watch_ohlcv_for_symbols = AsyncMock(side_effect=unwatch)
    mocker.patch.object(ExchangeWS, "_start_forever")
    ws = ExchangeWS({}, client)
    ws._loop = loop
    ws._klines_watching = {(f"PAIR{i}/USDT", "5m", CandleType.SPOT) for i in range(pair_count)}
    previous_handler = loop.get_exception_handler()
    loop.set_exception_handler(lambda _, context: destroyed.append(context["message"]))
    try:
        await ws._schedule_while_true()
        await asyncio.wait_for(started.wait(), timeout=2)

        # When: CCXT closes the connection and cancels its active watch futures.
        await ws._cleanup_async()
        barrier = loop.create_future()
        loop.call_soon(barrier.set_result, None)
        await asyncio.wait_for(barrier, timeout=2)
        gc.collect()

        # Then: no unsubscribe is launched on the dead connection or orphaned.
        assert client.un_watch_ohlcv_for_symbols.await_count == 0
        assert not destroyed
        assert not ws._background_tasks
        assert not ws._klines_scheduled
    finally:
        ws._stopping = True
        remaining = [
            task
            for task in asyncio.all_tasks()
            if task is not asyncio.current_task()
            and task.get_coro().__qualname__ == "ExchangeWS._unwatch_ohlcv"
        ]
        for task in remaining:
            task.cancel()
        await asyncio.gather(*remaining, return_exceptions=True)
        loop.set_exception_handler(previous_handler)
        ws._thread.join(timeout=2)


async def test_expired_watch_still_unsubscribes(mocker):
    # Given: a normally finished watcher on a live connection.
    loop = asyncio.get_running_loop()
    unsubscribed = asyncio.Event()
    client = MagicMock()
    client.ohlcvs = {}
    client.has = {"unWatchOHLCVForSymbols": True}

    async def unwatch(*args):
        unsubscribed.set()

    client.un_watch_ohlcv_for_symbols = AsyncMock(side_effect=unwatch)
    mocker.patch.object(ExchangeWS, "_start_forever")
    ws = ExchangeWS({}, client)
    ws._loop = loop
    pair = ("BTC/USDT", "5m", CandleType.SPOT)
    ws._klines_scheduled.add(pair)

    async def finished_watch():
        return None

    task = asyncio.create_task(finished_watch())
    await task
    ws._background_tasks.add(task)
    try:
        # When: normal expiry completes the watch instead of cancelling it.
        ws._continuous_stopped(task, *pair)

        # Then: the existing unsubscribe behavior is preserved.
        await asyncio.wait_for(unsubscribed.wait(), timeout=2)
        client.un_watch_ohlcv_for_symbols.assert_awaited_once_with([[pair[0], pair[1]]])
        assert pair not in ws._klines_scheduled
        assert task not in ws._background_tasks
    finally:
        ws._stopping = True
        ws._thread.join(timeout=2)
