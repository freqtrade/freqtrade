import asyncio
import logging
import threading
from datetime import timedelta
from time import sleep
from unittest.mock import AsyncMock, MagicMock

import pytest
from ccxt import NotSupported

from freqtrade.enums import CandleType
from freqtrade.exceptions import TemporaryError
from freqtrade.exchange.exchange_ws import ExchangeWS
from freqtrade.util import dt_ts
from ft_client.test_client.test_rest_client import log_has_re


def test_exchangews_init(mocker):
    config = MagicMock()
    ccxt_object = MagicMock()
    mocker.patch("freqtrade.exchange.exchange_ws.ExchangeWS._start_forever", MagicMock())

    exchange_ws = ExchangeWS(config, ccxt_object)
    sleep(0.1)

    assert exchange_ws.config == config
    assert exchange_ws._ccxt_object == ccxt_object
    assert exchange_ws._thread.name == "ccxt_ws"
    assert exchange_ws._background_tasks == set()
    assert exchange_ws._klines_watching == set()
    assert exchange_ws._klines_scheduled == set()
    assert exchange_ws._klines_last_refresh == {}
    assert exchange_ws._klines_last_request == {}
    # Cleanup
    exchange_ws.cleanup()


def test_exchangews_cleanup_error(mocker, caplog):
    config = MagicMock()
    ccxt_object = MagicMock()
    ccxt_object.close = AsyncMock(side_effect=Exception("Test"))
    mocker.patch("freqtrade.exchange.exchange_ws.ExchangeWS._start_forever", MagicMock())

    exchange_ws = ExchangeWS(config, ccxt_object)
    patch_eventloop_threading(exchange_ws)

    sleep(0.1)
    exchange_ws.reset_connections()

    assert log_has_re("Exception in _cleanup_async", caplog)

    exchange_ws.cleanup()


def test_exchangews_reset_connections_timeout_and_exception(mocker, caplog):
    config = MagicMock()
    ccxt_object = MagicMock()
    mocker.patch("freqtrade.exchange.exchange_ws.ExchangeWS._start_forever", MagicMock())

    exchange_ws = ExchangeWS(config, ccxt_object)
    exchange_ws._loop = MagicMock()
    exchange_ws._loop.is_closed.return_value = False

    timeout_future = MagicMock()
    timeout_future.result.side_effect = TimeoutError("timed out")

    error_future = MagicMock()
    error_future.result.side_effect = RuntimeError("broken future")

    def fake_run_coroutine_threadsafe(coro, loop):
        # Avoid coroutine warnings since we don't execute it in this unit test.
        coro.close()
        fake_run_coroutine_threadsafe.calls += 1
        return timeout_future if fake_run_coroutine_threadsafe.calls == 1 else error_future

    fake_run_coroutine_threadsafe.calls = 0

    mock_run = mocker.patch(
        "freqtrade.exchange.exchange_ws.asyncio.run_coroutine_threadsafe",
        side_effect=fake_run_coroutine_threadsafe,
    )

    exchange_ws.reset_connections()
    assert log_has_re("Timed out while resetting websocket connections", caplog)
    assert log_has_re("Resetting exchange WS connections", caplog)
    assert mock_run.call_count == 1

    exchange_ws.reset_connections(cleanup=True)

    assert mock_run.call_count == 2
    assert log_has_re("Exception while resetting websocket connections", caplog)
    assert log_has_re("Cleaning up exchange WS connections", caplog)

    exchange_ws.cleanup()


def test_exchangews_cleanup_thread_timeout_warning(mocker, caplog):
    config = MagicMock()
    ccxt_object = MagicMock()
    mocker.patch("freqtrade.exchange.exchange_ws.ExchangeWS._start_forever", MagicMock())

    exchange_ws = ExchangeWS(config, ccxt_object)
    exchange_ws._loop = MagicMock()
    exchange_ws._loop.is_closed.return_value = True

    thread_mock = MagicMock()
    thread_mock.is_alive.return_value = True
    exchange_ws._thread = thread_mock

    exchange_ws.cleanup()

    thread_mock.join.assert_called_once_with(timeout=5)
    assert log_has_re("Websocket loop thread did not stop within timeout", caplog)


def test_exchangews_schedule_ohlcv_loop_not_ready(mocker, caplog):
    config = MagicMock()
    ccxt_object = MagicMock()
    mocker.patch("freqtrade.exchange.exchange_ws.ExchangeWS._start_forever", MagicMock())
    run_threadsafe = mocker.patch("freqtrade.exchange.exchange_ws.asyncio.run_coroutine_threadsafe")

    exchange_ws = ExchangeWS(config, ccxt_object)
    exchange_ws.schedule_ohlcv("ETH/BTC", "1m", CandleType.SPOT)

    assert exchange_ws._klines_watching == set()
    assert exchange_ws._klines_last_request == {}
    assert run_threadsafe.call_count == 0
    assert log_has_re("Websocket loop not ready. Could not schedule ETH/BTC, 1m", caplog)

    exchange_ws.cleanup()


def patch_eventloop_threading(exchange):
    init_event = threading.Event()

    def thread_func():
        exchange._loop = asyncio.new_event_loop()
        init_event.set()
        exchange._loop.run_forever()

    x = threading.Thread(target=thread_func, daemon=True)
    x.start()
    # Wait for thread to be properly initialized with timeout
    if not init_event.wait(timeout=5.0):
        raise RuntimeError("Failed to initialize event loop thread")


async def test_exchangews_ohlcv(mocker, time_machine, caplog):
    config = MagicMock()
    ccxt_object = MagicMock()
    caplog.set_level(logging.DEBUG)

    async def controlled_sleeper(*args, **kwargs):
        # Sleep to pass control back to the event loop
        await asyncio.sleep(0.1)
        return MagicMock()

    async def wait_for_condition(condition_func, timeout_=5.0, check_interval=0.01):
        """Wait for a condition to be true with timeout."""
        try:
            async with asyncio.timeout(timeout_):
                while True:
                    if condition_func():
                        return True
                    await asyncio.sleep(check_interval)
        except TimeoutError:
            return False

    ccxt_object.un_watch_ohlcv_for_symbols = AsyncMock(side_effect=[NotSupported, ValueError])
    ccxt_object.watch_ohlcv = AsyncMock(side_effect=controlled_sleeper)
    ccxt_object.has = {"unWatchOHLCVForSymbols": True}
    ccxt_object.close = AsyncMock()
    time_machine.move_to("2024-11-01 01:00:02 +00:00")

    mocker.patch("freqtrade.exchange.exchange_ws.ExchangeWS._start_forever", MagicMock())

    exchange_ws = ExchangeWS(config, ccxt_object)
    patch_eventloop_threading(exchange_ws)
    try:
        assert exchange_ws._klines_watching == set()
        assert exchange_ws._klines_scheduled == set()

        exchange_ws.schedule_ohlcv("ETH/BTC", "1m", CandleType.SPOT)
        exchange_ws.schedule_ohlcv("XRP/BTC", "1m", CandleType.SPOT)

        # Wait for both pairs to be properly scheduled and watching
        await wait_for_condition(
            lambda: (
                len(exchange_ws._klines_watching) == 2 and len(exchange_ws._klines_scheduled) == 2
            ),
            timeout_=2.0,
        )

        assert exchange_ws._klines_watching == {
            ("ETH/BTC", "1m", CandleType.SPOT),
            ("XRP/BTC", "1m", CandleType.SPOT),
        }
        assert exchange_ws._klines_scheduled == {
            ("ETH/BTC", "1m", CandleType.SPOT),
            ("XRP/BTC", "1m", CandleType.SPOT),
        }

        # Wait for the expected number of watch calls
        await wait_for_condition(lambda: ccxt_object.watch_ohlcv.call_count >= 6, timeout_=3.0)
        assert ccxt_object.watch_ohlcv.call_count >= 6
        ccxt_object.watch_ohlcv.reset_mock()

        time_machine.shift(timedelta(minutes=5))
        exchange_ws.schedule_ohlcv("ETH/BTC", "1m", CandleType.SPOT)

        # Wait for log message
        await wait_for_condition(
            lambda: log_has_re("un_watch_ohlcv_for_symbols not supported: ", caplog), timeout_=2.0
        )
        assert log_has_re("un_watch_ohlcv_for_symbols not supported: ", caplog)

        # XRP/BTC should be cleaned up.
        assert exchange_ws._klines_watching == {
            ("ETH/BTC", "1m", CandleType.SPOT),
        }

        # Cleanup happened.
        exchange_ws.schedule_ohlcv("ETH/BTC", "1m", CandleType.SPOT)

        # Verify final state
        assert exchange_ws._klines_watching == {
            ("ETH/BTC", "1m", CandleType.SPOT),
        }
        assert exchange_ws._klines_scheduled == {
            ("ETH/BTC", "1m", CandleType.SPOT),
        }

        # Triggers 2nd call to un_watch_ohlcv_for_symbols which raises ValueError
        exchange_ws._klines_watching.discard(("ETH/BTC", "1m", CandleType.SPOT))
        await wait_for_condition(
            lambda: log_has_re("Exception in _unwatch_ohlcv", caplog), timeout_=2.0
        )
        assert log_has_re("Exception in _unwatch_ohlcv", caplog)

    finally:
        # Cleanup
        exchange_ws.cleanup()


async def test_exchangews_get_ohlcv(mocker, caplog):
    config = MagicMock()
    ccxt_object = MagicMock()
    ccxt_object.ohlcvs = {
        "ETH/USDT": {
            "1m": [
                [1635840000000, 100, 200, 300, 400, 500],
                [1635840060000, 101, 201, 301, 401, 501],
                [1635840120000, 102, 202, 302, 402, 502],
            ],
            "5m": [
                [1635840000000, 100, 200, 300, 400, 500],
                [1635840300000, 105, 201, 301, 401, 501],
                [1635840600000, 102, 202, 302, 402, 502],
            ],
        }
    }
    mocker.patch("freqtrade.exchange.exchange_ws.ExchangeWS._start_forever", MagicMock())

    exchange_ws = ExchangeWS(config, ccxt_object)
    exchange_ws._klines_last_refresh = {
        ("ETH/USDT", "1m", CandleType.SPOT): 1635840120000,
        ("ETH/USDT", "5m", CandleType.SPOT): 1635840600000,
    }

    # Matching last candle time - drop hint is true
    resp = await exchange_ws.get_ohlcv("ETH/USDT", "1m", CandleType.SPOT, 1635840120000)
    assert resp[0] == "ETH/USDT"
    assert resp[1] == "1m"
    assert resp[3] == [
        [1635840000000, 100, 200, 300, 400, 500],
        [1635840060000, 101, 201, 301, 401, 501],
        [1635840120000, 102, 202, 302, 402, 502],
    ]
    assert resp[4] is True

    # expected time > last candle time - drop hint is false
    resp = await exchange_ws.get_ohlcv("ETH/USDT", "1m", CandleType.SPOT, 1635840180000)
    assert resp[0] == "ETH/USDT"
    assert resp[1] == "1m"
    assert resp[3] == [
        [1635840000000, 100, 200, 300, 400, 500],
        [1635840060000, 101, 201, 301, 401, 501],
        [1635840120000, 102, 202, 302, 402, 502],
    ]
    assert resp[4] is False

    # Change "received" times to be before the candle starts.
    # This should trigger the "time sync" warning.
    exchange_ws._klines_last_refresh = {
        ("ETH/USDT", "1m", CandleType.SPOT): 1635840110000,
        ("ETH/USDT", "5m", CandleType.SPOT): 1635840600000,
    }
    msg = r".*Candle date > last refresh.*"
    assert not log_has_re(msg, caplog)
    resp = await exchange_ws.get_ohlcv("ETH/USDT", "1m", CandleType.SPOT, 1635840120000)
    assert resp[0] == "ETH/USDT"
    assert resp[1] == "1m"
    assert resp[3] == [
        [1635840000000, 100, 200, 300, 400, 500],
        [1635840060000, 101, 201, 301, 401, 501],
        [1635840120000, 102, 202, 302, 402, 502],
    ]
    assert resp[4] is True

    assert log_has_re(msg, caplog)

    exchange_ws.cleanup()


async def test_exchangews_get_ohlcv_missing_refresh_date(mocker, caplog):
    config = MagicMock()
    ccxt_object = MagicMock()
    ccxt_object.ohlcvs = {
        "ETH/USDT": {
            "1m": [
                [1635840000000, 100, 200, 300, 400, 500],
                [1635840060000, 101, 201, 301, 401, 501],
                [1635840120000, 102, 202, 302, 402, 502],
            ]
        }
    }
    mocker.patch("freqtrade.exchange.exchange_ws.ExchangeWS._start_forever", MagicMock())

    exchange_ws = ExchangeWS(config, ccxt_object)
    exchange_ws._klines_last_refresh = {}

    # No refresh-date entry should not raise KeyError.
    resp = await exchange_ws.get_ohlcv("ETH/USDT", "1m", CandleType.SPOT, 1635840120000)
    assert resp[0] == "ETH/USDT"
    assert resp[1] == "1m"
    assert resp[4] is True
    assert not log_has_re(r".*Candle date > last refresh.*", caplog)

    exchange_ws.cleanup()


def test_exchangews_ohlcvs_deepcopy_and_retry(mocker):
    config = MagicMock()
    ccxt_object = MagicMock()
    ccxt_object.ohlcvs = {
        "ETH/USDT": {
            "1m": [[1, 2, 3, 4, 5, 6]],
        }
    }
    mocker.patch("freqtrade.exchange.exchange_ws.ExchangeWS._start_forever", MagicMock())

    exchange_ws = ExchangeWS(config, ccxt_object)

    call_count = {"count": 0}

    def deepcopy_side_effect(value):
        call_count["count"] += 1
        if call_count["count"] < 3:
            raise RuntimeError("copy failed")
        return [candle.copy() for candle in value]

    mocker.patch("freqtrade.exchange.exchange_ws.deepcopy", deepcopy_side_effect)

    result = exchange_ws.ohlcvs("ETH/USDT", "1m")

    assert call_count["count"] == 3
    assert result == [[1, 2, 3, 4, 5, 6]]
    assert result is not ccxt_object.ohlcvs["ETH/USDT"]["1m"]

    # Fail all the time
    mocker.patch("freqtrade.exchange.exchange_ws.deepcopy", side_effect=RuntimeError("copy failed"))
    with pytest.raises(TemporaryError, match=r"Error deepcopying: copy failed"):
        exchange_ws.ohlcvs("ETH/USDT", "1m")

    exchange_ws.cleanup()


def test_exchangews_get_ohlcv_with_refresh(mocker):
    config = MagicMock()
    ccxt_object = MagicMock()
    mocker.patch("freqtrade.exchange.exchange_ws.ExchangeWS._start_forever", MagicMock())

    exchange_ws = ExchangeWS(config, ccxt_object)
    ohlcvs_mock = mocker.patch.object(
        exchange_ws, "ohlcvs", return_value=[[10, 11, 12, 13, 14, 15]]
    )

    paircomb = ("ETH/USDT", "1m", CandleType.SPOT)
    exchange_ws._klines_last_refresh[paircomb] = 123456789

    candles, refresh = exchange_ws.get_ohlcv_with_refresh("ETH/USDT", "1m", CandleType.SPOT)

    ohlcvs_mock.assert_called_once_with("ETH/USDT", "1m")
    assert candles == [[10, 11, 12, 13, 14, 15]]
    assert refresh == 123456789

    candles, refresh = exchange_ws.get_ohlcv_with_refresh("ETH/USDT", "5m", CandleType.SPOT)
    assert candles == [[10, 11, 12, 13, 14, 15]]
    assert refresh == 0

    exchange_ws.cleanup()


def test_exchangews_get_orderbook(mocker):
    config = MagicMock()
    ccxt_object = MagicMock()
    ob = {
        "bids": [[100.0, 1.0], [99.0, 2.0]],
        "asks": [[101.0, 1.5], [102.0, 3.0]],
        "timestamp": 1635840000000,
        "nonce": 7,
    }
    ccxt_object.orderbooks = {"ETH/USDT": ob}
    mocker.patch("freqtrade.exchange.exchange_ws.ExchangeWS._start_forever")

    exchange_ws = ExchangeWS(config, ccxt_object)

    result = exchange_ws.get_orderbook("ETH/USDT")
    assert result == ob
    # A deep copy is returned - decoupled from the live cache (nested lists copied too).
    assert result is not ob
    assert result["bids"] is not ob["bids"]
    # Mutating the live cache must not affect an already-returned snapshot.
    ccxt_object.orderbooks["ETH/USDT"]["bids"][0][1] = 999
    assert result["bids"][0][1] == 1.0

    # Unknown pair returns an empty dict (caller falls back to REST).
    assert exchange_ws.get_orderbook("BTC/USDT") == {}

    exchange_ws.cleanup()


def test_exchangews_get_orderbook_deepcopy_and_retry(mocker):
    config = MagicMock()
    ccxt_object = MagicMock()
    ccxt_object.orderbooks = {"ETH/USDT": {"bids": [[1, 2]], "asks": [[3, 4]]}}
    mocker.patch("freqtrade.exchange.exchange_ws.ExchangeWS._start_forever")

    exchange_ws = ExchangeWS(config, ccxt_object)

    call_count = {"count": 0}

    def deepcopy_side_effect(value):
        # RuntimeError mimics "collection changed size during iteration" while the
        # ws thread mutates the book mid-copy - the retrier should swallow and retry.
        call_count["count"] += 1
        if call_count["count"] < 3:
            raise RuntimeError("copy failed")
        return {k: [row.copy() for row in v] for k, v in value.items()}

    mocker.patch("freqtrade.exchange.exchange_ws.deepcopy", deepcopy_side_effect)

    result = exchange_ws.get_orderbook("ETH/USDT")

    assert call_count["count"] == 3
    assert result == {"bids": [[1, 2]], "asks": [[3, 4]]}
    assert result is not ccxt_object.orderbooks["ETH/USDT"]

    # Fail every time -> surfaces as TemporaryError once retries are exhausted.
    mocker.patch("freqtrade.exchange.exchange_ws.deepcopy", side_effect=RuntimeError("copy failed"))
    with pytest.raises(TemporaryError, match=r"Error deepcopying: copy failed"):
        exchange_ws.get_orderbook("ETH/USDT")

    exchange_ws.cleanup()


def test_exchangews_orderbook_is_fresh(mocker, time_machine):
    config = MagicMock()
    ccxt_object = MagicMock()
    mocker.patch("freqtrade.exchange.exchange_ws.ExchangeWS._start_forever")
    time_machine.move_to("2024-11-01 01:00:00 +00:00", tick=False)

    exchange_ws = ExchangeWS(config, ccxt_object)
    max_age = 5

    # No refresh recorded yet -> not fresh (fall back to REST).
    assert exchange_ws.orderbook_is_fresh("ETH/BTC", max_age) is False

    exchange_ws._ob_last_refresh["ETH/BTC"] = dt_ts()
    # Just refreshed -> fresh.
    assert exchange_ws.orderbook_is_fresh("ETH/BTC", max_age) is True

    # Still within the max-age window.
    time_machine.shift(timedelta(seconds=max_age - 1))
    assert exchange_ws.orderbook_is_fresh("ETH/BTC", max_age) is True

    # Past the max-age window -> stale (feed likely stuck).
    time_machine.shift(timedelta(seconds=2))
    assert exchange_ws.orderbook_is_fresh("ETH/BTC", max_age) is False

    exchange_ws.cleanup()


def test_exchangews_schedule_orderbook_loop_not_ready(mocker, caplog):
    config = MagicMock()
    ccxt_object = MagicMock()
    mocker.patch("freqtrade.exchange.exchange_ws.ExchangeWS._start_forever")
    run_threadsafe = mocker.patch("freqtrade.exchange.exchange_ws.asyncio.run_coroutine_threadsafe")

    exchange_ws = ExchangeWS(config, ccxt_object)
    exchange_ws.schedule_orderbook("ETH/BTC")

    assert exchange_ws._ob_watching == set()
    assert exchange_ws.ob_last_request == {}
    assert run_threadsafe.call_count == 0
    assert log_has_re("Websocket loop not ready. Could not schedule orderbook for ETH/BTC", caplog)

    exchange_ws.cleanup()


def test_exchangews_orderbook_stopped(mocker):
    config = MagicMock()
    ccxt_object = MagicMock()
    mocker.patch("freqtrade.exchange.exchange_ws.ExchangeWS._start_forever")

    exchange_ws = ExchangeWS(config, ccxt_object)

    task = MagicMock()
    exchange_ws._background_tasks.add(task)
    exchange_ws._ob_scheduled.add("ETH/BTC")

    exchange_ws._orderbook_stopped(task, "ETH/BTC")

    assert task not in exchange_ws._background_tasks
    assert "ETH/BTC" not in exchange_ws._ob_scheduled

    exchange_ws.cleanup()


async def test_exchangews_watch_orderbook(mocker, time_machine, caplog):
    config = MagicMock()
    ccxt_object = MagicMock()
    caplog.set_level(logging.DEBUG)

    async def controlled_sleeper(*args, **kwargs):
        # Sleep to pass control back to the event loop
        await asyncio.sleep(0.1)
        return MagicMock()

    async def wait_for_condition(condition_func, timeout_=5.0, check_interval=0.01):
        """Wait for a condition to be true with timeout."""
        try:
            async with asyncio.timeout(timeout_):
                while True:
                    if condition_func():
                        return True
                    await asyncio.sleep(check_interval)
        except TimeoutError:
            return False

    ccxt_object.watch_order_book = AsyncMock(side_effect=controlled_sleeper)
    ccxt_object.close = AsyncMock()
    time_machine.move_to("2024-11-01 01:00:02 +00:00")

    mocker.patch("freqtrade.exchange.exchange_ws.ExchangeWS._start_forever")

    exchange_ws = ExchangeWS(config, ccxt_object)
    patch_eventloop_threading(exchange_ws)
    try:
        assert exchange_ws._ob_watching == set()
        assert exchange_ws._ob_scheduled == set()

        exchange_ws.schedule_orderbook("ETH/BTC")
        exchange_ws.schedule_orderbook("XRP/BTC")

        # Wait for both pairs to be watching and scheduled.
        assert await wait_for_condition(
            lambda: len(exchange_ws._ob_watching) == 2 and len(exchange_ws._ob_scheduled) == 2,
            timeout_=2.0,
        )
        assert exchange_ws._ob_watching == {"ETH/BTC", "XRP/BTC"}
        assert exchange_ws._ob_scheduled == {"ETH/BTC", "XRP/BTC"}

        # The watch loop keeps calling watch_order_book for both pairs.
        assert await wait_for_condition(
            lambda: ccxt_object.watch_order_book.call_count >= 4, timeout_=3.0
        )
        watched_pairs = {c.args[0] for c in ccxt_object.watch_order_book.call_args_list}
        assert watched_pairs == {"ETH/BTC", "XRP/BTC"}

        # Each applied frame records a refresh time, so the feed reads as fresh.
        assert await wait_for_condition(
            lambda: exchange_ws._ob_last_refresh.keys() == {"ETH/BTC", "XRP/BTC"}, timeout_=2.0
        )
        assert exchange_ws.orderbook_is_fresh("ETH/BTC", 5)

        # Move past the expiry window (timeout + 20s) and trigger the cleanup.
        time_machine.shift(timedelta(seconds=exchange_ws.ob_timeout + 20))
        exchange_ws.cleanup_expired()

        # Both pairs get removed from the watchlist, and the tasks unschedule themselves.
        assert await wait_for_condition(
            lambda: exchange_ws._ob_watching == set() and exchange_ws._ob_scheduled == set(),
            timeout_=2.0,
        )
        assert log_has_re("Removing ETH/BTC from orderbook watchlist", caplog)
        assert log_has_re("Removing XRP/BTC from orderbook watchlist", caplog)
        # Refresh tracking is cleaned up alongside the watchlist entry.
        assert exchange_ws._ob_last_refresh == {}
    finally:
        # Cleanup
        exchange_ws.cleanup()


def test_exchangews_continuous_stopped_task_exception(mocker, caplog):
    config = MagicMock()
    ccxt_object = MagicMock()
    ccxt_object.ohlcvs = {
        "ETH/USDT": {
            "1m": [
                [1635840000000, 100, 200, 300, 400, 500],
                [1635840060000, 101, 201, 301, 401, 501],
                [1635840120000, 102, 202, 302, 402, 502],
            ]
        }
    }
    mocker.patch("freqtrade.exchange.exchange_ws.ExchangeWS._start_forever", MagicMock())

    exchange_ws = ExchangeWS(config, ccxt_object)
    exchange_ws._loop = MagicMock()
    exchange_ws._loop.is_closed.return_value = False

    paircomb = ("ETH/USDT", "1m", CandleType.SPOT)
    exchange_ws._klines_scheduled.add(paircomb)
    exchange_ws._klines_last_refresh[paircomb] = 1

    task = MagicMock()
    task.cancelled.return_value = False
    task.result.side_effect = RuntimeError("unexpected")
    exchange_ws._background_tasks.add(task)

    completed_future = MagicMock()
    completed_future.result.return_value = None

    def side_effect(coro, loop):
        coro.close()
        return completed_future

    run_threadsafe = mocker.patch(
        "freqtrade.exchange.exchange_ws.asyncio.run_coroutine_threadsafe",
        side_effect=side_effect,
    )

    exchange_ws._continuous_stopped(task, "ETH/USDT", "1m", CandleType.SPOT)

    assert task not in exchange_ws._background_tasks
    assert paircomb not in exchange_ws._klines_scheduled
    assert paircomb not in exchange_ws._klines_last_refresh
    assert ccxt_object.ohlcvs["ETH/USDT"].get("1m") is None
    assert run_threadsafe.call_count == 1
    assert log_has_re("Unhandled exception in watch task callback for ETH/USDT, 1m", caplog)

    exchange_ws.cleanup()
