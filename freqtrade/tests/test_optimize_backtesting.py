# pragma pylint: disable=missing-docstring,W0212


from freqtrade import exchange, optimize
from freqtrade.exchange import Bittrex
from freqtrade.optimize.backtesting import backtest

import pytest

def test_backtest(default_conf, mocker):
    mocker.patch.dict('freqtrade.main._CONF', default_conf)
    exchange._API = Bittrex({'key': '', 'secret': ''})

    data = optimize.load_data(ticker_interval=5, pairs=['BTC_ETH'])
    results = backtest(default_conf, optimize.preprocess(data), 10, True)
    num_resutls = len(results)
    assert num_resutls > 0


def test_1min_ticker_interval(default_conf, mocker):
    mocker.patch.dict('freqtrade.main._CONF', default_conf)
    exchange._API = Bittrex({'key': '', 'secret': ''})

    # Run a backtesting for an exiting 5min ticker_interval
    data = optimize.load_data(ticker_interval=1, pairs=['BTC_UNITEST'])
    results = backtest(default_conf, optimize.preprocess(data), 1, True)
    assert len(results) > 0

    # Run a backtesting for 5min ticker_interval
    with pytest.raises(FileNotFoundError):
        data = optimize.load_data(ticker_interval=5, pairs=['BTC_UNITEST'])
        results = backtest(default_conf, optimize.preprocess(data), 1, True)
