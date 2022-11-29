

from time import sleep
from unittest.mock import MagicMock

from freqtrade.exchange.exchange_ws import ExchangeWS


def test_exchangews_init(mocker):

    config = MagicMock()
    ccxt_object = MagicMock()
    mocker.patch("freqtrade.exchange.exchange_ws.ExchangeWS._start_forever", MagicMock())

    exchange_ws = ExchangeWS(config, ccxt_object)

    assert exchange_ws.config == config
    assert exchange_ws.ccxt_object == ccxt_object
    assert exchange_ws._thread.name == "ccxt_ws"
    assert exchange_ws._background_tasks == set()
    assert exchange_ws._klines_watching == set()
    assert exchange_ws._klines_scheduled == set()
    assert exchange_ws.klines_last_refresh == {}
    assert exchange_ws.klines_last_request == {}
    assert exchange_ws._ob_watching == set()
    assert exchange_ws._ob_scheduled == set()
    assert exchange_ws.ob_last_request == {}
    sleep(0.1)
    # Cleanup
    exchange_ws.cleanup()
