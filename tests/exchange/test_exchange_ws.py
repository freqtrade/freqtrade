import asyncio
import threading
from time import sleep
from unittest.mock import AsyncMock, MagicMock

from freqtrade.enums import CandleType
from freqtrade.exchange.exchange_ws import ExchangeWS


def test_exchangews_init(mocker):
    config = MagicMock()
    ccxt_object = MagicMock()
    mocker.patch("freqtrade.exchange.exchange_ws.ExchangeWS._start_forever", MagicMock())

    exchange_ws = ExchangeWS(config, ccxt_object)
    sleep(0.1)

    assert exchange_ws.config == config
    assert exchange_ws.ccxt_object == ccxt_object
    assert exchange_ws._thread.name == "ccxt_ws"
    assert exchange_ws._background_tasks == set()
    assert exchange_ws._klines_watching == set()
    assert exchange_ws._klines_scheduled == set()
    assert exchange_ws.klines_last_refresh == {}
    assert exchange_ws.klines_last_request == {}
    # Cleanup
    exchange_ws.cleanup()


def patch_eventloop_threading(exchange):
    is_init = False

    def thread_fuck():
        nonlocal is_init
        exchange._loop = asyncio.new_event_loop()
        is_init = True
        exchange._loop.run_forever()

    x = threading.Thread(target=thread_fuck, daemon=True)
    x.start()
    while not is_init:
        pass


async def test_exchangews_ohlcv(mocker):
    config = MagicMock()
    ccxt_object = MagicMock()
    ccxt_object.watch_ohlcv = AsyncMock()
    ccxt_object.close = AsyncMock()
    mocker.patch("freqtrade.exchange.exchange_ws.ExchangeWS._start_forever", MagicMock())

    exchange_ws = ExchangeWS(config, ccxt_object)
    patch_eventloop_threading(exchange_ws)
    try:
        assert exchange_ws._klines_watching == set()
        assert exchange_ws._klines_scheduled == set()

        exchange_ws.schedule_ohlcv("ETH/BTC", "1m", CandleType.SPOT)
        await asyncio.sleep(0.5)

        assert exchange_ws._klines_watching == {("ETH/BTC", "1m", CandleType.SPOT)}
        assert exchange_ws._klines_scheduled == {("ETH/BTC", "1m", CandleType.SPOT)}
        await asyncio.sleep(0.1)
        assert ccxt_object.watch_ohlcv.call_count == 1
    except Exception as e:
        print(e)
    finally:
        # Cleanup
        exchange_ws.cleanup()
