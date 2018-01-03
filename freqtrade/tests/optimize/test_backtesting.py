# pragma pylint: disable=missing-docstring,W0212

import math
import pandas as pd
# from unittest.mock import MagicMock
from freqtrade import exchange, optimize
from freqtrade.exchange import Bittrex
from freqtrade.optimize import preprocess
from freqtrade.optimize.backtesting import backtest, generate_text_table, get_timeframe
# import freqtrade.optimize.backtesting as backtesting


def test_generate_text_table():
    results = pd.DataFrame(
        {
            'currency': ['BTC_ETH', 'BTC_ETH'],
            'profit_percent': [0.1, 0.2],
            'profit_BTC': [0.2, 0.4],
            'duration': [10, 30],
            'profit': [2, 0],
            'loss': [0, 0]
        }
    )
    print(generate_text_table({'BTC_ETH': {}}, results, 'BTC', 5))
    assert generate_text_table({'BTC_ETH': {}}, results, 'BTC', 5) == (
        'pair       buy count    avg profit %    total profit BTC    avg duration    profit    loss\n'
        '-------  -----------  --------------  ------------------  --------------  --------  ------\n'
        'BTC_ETH            2           15.00          0.60000000           100.0         2       0\n'
        'TOTAL              2           15.00          0.60000000           100.0         2       0')


def test_get_timeframe():
    data = preprocess(optimize.load_data(ticker_interval=1, pairs=['BTC_UNITEST']))
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


def trim_dictlist(dl, num):
    new = {}
    for pair, pair_data in dl.items():
        # Can't figure out why -num wont work
        new[pair] = pair_data[num:]
    return new


def load_data_test(what):
    data = optimize.load_data(ticker_interval=1, pairs=['BTC_UNITEST'])
    data = trim_dictlist(data, -100)
    pair = data['BTC_UNITEST']
    datalen = len(pair)
    # Depending on the what parameter we now adjust the
    # loaded data looks:
    # pair :: [{'O': 0.123, 'H': 0.123, 'L': 0.123,
    #           'C': 0.123, 'V': 123.123,
    #           'T': '2017-11-04T23:02:00', 'BV': 0.123}]
    base = 0.001
    if what == 'raise':
        return {'BTC_UNITEST':
                [{'T':  pair[x]['T'],  # Keep old dates
                  'V':  pair[x]['V'],  # Keep old volume
                  'BV': pair[x]['BV'],  # keep too
                  'O':  x * base,        # But replace O,H,L,C
                  'H':  x * base + 0.0001,
                  'L':  x * base - 0.0001,
                  'C':  x * base} for x in range(0, datalen)]}
    if what == 'lower':
        return {'BTC_UNITEST':
                [{'T': pair[x]['T'],  # Keep old dates
                  'V': pair[x]['V'],  # Keep old volume
                  'BV': pair[x]['BV'],  # keep too
                  'O': 1 - x * base,        # But replace O,H,L,C
                  'H': 1 - x * base + 0.0001,
                  'L': 1 - x * base - 0.0001,
                  'C': 1 - x * base} for x in range(0, datalen)]}
    if what == 'sine':
        hz = 0.1  # frequency
        return {'BTC_UNITEST':
                [{'T': pair[x]['T'],  # Keep old dates
                  'V': pair[x]['V'],  # Keep old volume
                  'BV': pair[x]['BV'],  # keep too
                  'O': math.sin(x*hz) / 1000 + base,        # But replace O,H,L,C
                  'H': math.sin(x*hz) / 1000 + base + 0.0001,
                  'L': math.sin(x*hz) / 1000 + base - 0.0001,
                  'C': math.sin(x*hz) / 1000 + base} for x in range(0, datalen)]}
    return data


def simple_backtest(config, contour, num_results):
    data = load_data_test(contour)
    processed = optimize.preprocess(data)
    assert isinstance(processed, dict)
    results = backtest(config['stake_amount'], processed, 1, True)
    # results :: <class 'pandas.core.frame.DataFrame'>
    assert len(results) == num_results


# Test backtest on offline data
# loaded by freqdata/optimize/__init__.py::load_data()


def test_backtest2(default_conf, mocker):
    mocker.patch.dict('freqtrade.main._CONF', default_conf)
    data = optimize.load_data(ticker_interval=5, pairs=['BTC_ETH'])
    results = backtest(default_conf['stake_amount'], optimize.preprocess(data), 10, True)
    assert not results.empty


def test_processed(default_conf, mocker):
    mocker.patch.dict('freqtrade.main._CONF', default_conf)
    dict_of_tickerrows = load_data_test('raise')
    dataframes = optimize.preprocess(dict_of_tickerrows)
    dataframe = dataframes['BTC_UNITEST']
    cols = dataframe.columns
    # assert the dataframe got some of the indicator columns
    for col in ['close', 'high', 'low', 'open', 'date',
                'ema50', 'ao', 'macd', 'plus_dm']:
        assert col in cols


def test_backtest_pricecontours(default_conf, mocker):
    mocker.patch.dict('freqtrade.main._CONF', default_conf)
    tests = [['raise', 17], ['lower', 0], ['sine', 17]]
    for [contour, numres] in tests:
        simple_backtest(default_conf, contour, numres)

# Please make this work, the load_config needs to be mocked
# and cleanups.
# def test_backtest_start(default_conf, mocker):
#   default_conf['exchange']['pair_whitelist'] = ['BTC_UNITEST']
#   mocker.patch.dict('freqtrade.main._CONF', default_conf)
#   # see https://pypi.python.org/pypi/pytest-mock/
#   # and http://www.voidspace.org.uk/python/mock/patch.html
#   # No usage example of simple function mocking,
#   # and no documentation of side_effect
#   mocker.patch('freqtrade.misc.load_config', new=lambda s, t: {})
#   args = MagicMock()
#   args.level = 10
#   #load_config('foo')
#   backtesting.start(args)
#
#    Check what sideeffect backtstesting has done.
#    Probably need to capture standard-output and
#    check for the generated report table.
