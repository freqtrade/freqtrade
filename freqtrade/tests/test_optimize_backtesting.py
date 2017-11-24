# pragma pylint: disable=missing-docstring,W0212


from freqtrade import exchange
from freqtrade.exchange import Bittrex
from freqtrade.optimize.backtesting import backtest, preprocess
from freqtrade.tests import load_backtesting_data


def test_backtest(backtest_conf, mocker):
    mocker.patch.dict('freqtrade.main._CONF', backtest_conf)
    exchange._API = Bittrex({'key': '', 'secret': ''})

    data = load_backtesting_data(ticker_interval=5, pairs=['BTC_ETH'])
    results = backtest(backtest_conf, preprocess(data), 10, True)
    num_resutls = len(results)
    assert num_resutls > 0

