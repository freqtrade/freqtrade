import asyncio
import logging
import threading
from datetime import timedelta
from time import sleep
from unittest.mock import AsyncMock, MagicMock

from ccxt import NotSupported

from freqtrade.enums import CandleType
from freqtrade.exchange.exchange_ws import ExchangeWS
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
    assert exchange_ws.klines_last_refresh == {}
    assert exchange_ws.klines_last_request == {}
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

    ccxt_object.un_watch_ohlcv_for_symbols = AsyncMock(side_effect=NotSupported)
    ccxt_object.watch_ohlcv = AsyncMock(side_effect=controlled_sleeper)
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
        ccxt_object.un_watch_ohlcv_for_symbols = AsyncMock(side_effect=ValueError)
        exchange_ws.schedule_ohlcv("ETH/BTC", "1m", CandleType.SPOT)

        # Verify final state
        assert exchange_ws._klines_watching == {
            ("ETH/BTC", "1m", CandleType.SPOT),
        }
        assert exchange_ws._klines_scheduled == {
            ("ETH/BTC", "1m", CandleType.SPOT),
        }

    finally:
        # Cleanup
        exchange_ws.cleanup()
    assert log_has_re("Exception in _unwatch_ohlcv", caplog)


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
    exchange_ws.klines_last_refresh = {
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
    exchange_ws.klines_last_refresh = {
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
