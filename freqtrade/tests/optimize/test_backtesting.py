# pragma pylint: disable=missing-docstring,W0212

import pandas as pd
from freqtrade import exchange, optimize
from freqtrade.exchange import Bittrex
from freqtrade.optimize.backtesting import backtest, generate_text_table, get_timeframe


def test_generate_text_table():
    results = pd.DataFrame(
        {
            'currency': ['BTC_ETH', 'BTC_ETH'],
            'profit_percent': [0.1, 0.2],
            'profit_BTC': [0.2, 0.4],
            'duration': [10, 30]
        }
    )
    assert generate_text_table({'BTC_ETH': {}}, results, 'BTC', 5) == (
        'pair       buy count  avg profit    total profit      avg duration\n'
        '-------  -----------  ------------  --------------  --------------\n'
        'BTC_ETH            2  15.00%        0.60000000 BTC             100\n'
        'TOTAL              2  15.00%        0.60000000 BTC             100')


def test_get_timeframe():
    data = optimize.load_data(ticker_interval=1, pairs=['BTC_UNITEST'])
    min_date, max_date = get_timeframe(data)
    assert min_date.isoformat() == '2017-11-04T23:02:00+00:00'
    assert max_date.isoformat() == '2017-11-14T22:59:00+00:00'


def test_backtest(default_conf, mocker):
    mocker.patch.dict('freqtrade.main._CONF', default_conf)
    exchange._API = Bittrex({'key': '', 'secret': ''})

    data = optimize.load_data(ticker_interval=5, pairs=['BTC_ETH'])
    results = backtest(default_conf['stake_amount'], optimize.preprocess(data), 10, True)
    assert not results.empty


def test_backtest_1min_ticker_interval(default_conf, mocker):
    mocker.patch.dict('freqtrade.main._CONF', default_conf)
    exchange._API = Bittrex({'key': '', 'secret': ''})

    # Run a backtesting for an exiting 5min ticker_interval
    data = optimize.load_data(ticker_interval=1, pairs=['BTC_UNITEST'])
    results = backtest(default_conf['stake_amount'], optimize.preprocess(data), 1, True)
    assert not results.empty
