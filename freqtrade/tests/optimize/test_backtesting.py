import math
import os
import pandas as pd
from freqtrade import exchange, optimize
from freqtrade.exchange import Bittrex
from freqtrade.optimize.backtesting import backtest, generate_text_table, get_timeframe
from freqtrade.optimize.__init__ import testdata_path, download_pairs, download_backtesting_testdata


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


def test_1min_ticker_interval(default_conf, mocker):
    mocker.patch.dict('freqtrade.main._CONF', default_conf)
    exchange._API = Bittrex({'key': '', 'secret': ''})

    # Run a backtesting for an exiting 5min ticker_interval
    data = optimize.load_data(ticker_interval=1, pairs=['BTC_UNITEST'])
    results = backtest(default_conf['stake_amount'], optimize.preprocess(data), 1, True)
    assert not results.empty


def test_backtest_with_new_pair(default_conf, ticker_history, mocker):
    mocker.patch('freqtrade.optimize.get_ticker_history', return_value=ticker_history)
    mocker.patch.dict('freqtrade.main._CONF', default_conf)

    exchange._API = Bittrex({'key': '', 'secret': ''})

    optimize.load_data(ticker_interval=1, pairs=['BTC_MEME'])
    file = 'freqtrade/tests/testdata/BTC_MEME-1.json'
    assert os.path.isfile(file) is True

    # delete file freshly downloaded
    if os.path.isfile(file):
        os.remove(file)


def test_testdata_path():
    assert os.path.join('freqtrade', 'tests', 'testdata') in testdata_path()


def test_download_pairs(default_conf, ticker_history, mocker):
    mocker.patch('freqtrade.optimize.__init__.get_ticker_history', return_value=ticker_history)
    mocker.patch.dict('freqtrade.main._CONF', default_conf)
    exchange._API = Bittrex({'key': '', 'secret': ''})

    file1_1 = 'freqtrade/tests/testdata/BTC_MEME-1.json'
    file1_5 = 'freqtrade/tests/testdata/BTC_MEME-5.json'
    file2_1 = 'freqtrade/tests/testdata/BTC_CFI-1.json'
    file2_5 = 'freqtrade/tests/testdata/BTC_CFI-5.json'

    assert download_pairs(pairs=['BTC-MEME', 'BTC-CFI']) is True

    assert os.path.isfile(file1_1) is True
    assert os.path.isfile(file1_5) is True
    assert os.path.isfile(file2_1) is True
    assert os.path.isfile(file2_5) is True

    # delete files freshly downloaded
    if os.path.isfile(file1_1):
        os.remove(file1_1)

    if os.path.isfile(file1_5):
        os.remove(file1_5)

    if os.path.isfile(file2_1):
        os.remove(file2_1)

    if os.path.isfile(file2_5):
        os.remove(file2_5)


def test_download_backtesting_testdata(default_conf, ticker_history, mocker):
    mocker.patch('freqtrade.optimize.__init__.get_ticker_history', return_value=ticker_history)
    mocker.patch.dict('freqtrade.main._CONF', default_conf)
    exchange._API = Bittrex({'key': '', 'secret': ''})

    # Download a 1 min ticker file
    file1 = 'freqtrade/tests/testdata/BTC_XEL-1.json'
    download_backtesting_testdata(pair="BTC-XEL", interval=1)
    assert os.path.isfile(file1) is True

    if os.path.isfile(file1):
        os.remove(file1)

    # Download a 5 min ticker file
    file2 = 'freqtrade/tests/testdata/BTC_STORJ-5.json'
    download_backtesting_testdata(pair="BTC-STORJ", interval=5)
    assert os.path.isfile(file2) is True

    if os.path.isfile(file2):
        os.remove(file2)


def trim_dataframe(df, num):
    new = dict()
    for pair, pair_data in df.items():
        new[pair] = pair_data[-num:]  # last 50 rows
    return new


def load_data_test(what):
    data = optimize.load_data(ticker_interval=1, pairs=['BTC_UNITEST'])
    data = trim_dataframe(data, -40)
    pair = data['BTC_UNITEST']

    # Depending on the what parameter we now adjust the
    # loaded data:
    # pair :: [{'O': 0.123, 'H': 0.123, 'L': 0.123,
    #           'C': 0.123, 'V': 123.123,
    #           'T': '2017-11-04T23:02:00', 'BV': 0.123}]
    if what == 'raise':
        o = 0.001
        h = 0.001
        ll = 0.001
        c = 0.001
        ll -= 0.0001
        h += 0.0001
        for frame in pair:
            o += 0.0001
            h += 0.0001
            ll += 0.0001
            c += 0.0001
            # save prices rounded to satoshis
            frame['O'] = round(o, 9)
            frame['H'] = round(h, 9)
            frame['L'] = round(ll, 9)
            frame['C'] = round(c, 9)
    if what == 'lower':
        o = 0.001
        h = 0.001
        ll = 0.001
        c = 0.001
        ll -= 0.0001
        h += 0.0001
        for frame in pair:
            o -= 0.0001
            h -= 0.0001
            ll -= 0.0001
            c -= 0.0001
            # save prices rounded to satoshis
            frame['O'] = round(o, 9)
            frame['H'] = round(h, 9)
            frame['L'] = round(ll, 9)
            frame['C'] = round(c, 9)
    if what == 'sine':
        i = 0
        o = (2 + math.sin(i/10)) / 1000
        h = o
        ll = o
        c = o
        h += 0.0001
        ll -= 0.0001
        for frame in pair:
            o = (2 + math.sin(i/10)) / 1000
            h = (2 + math.sin(i/10)) / 1000 + 0.0001
            ll = (2 + math.sin(i/10)) / 1000 - 0.0001
            c = (2 + math.sin(i/10)) / 1000 - 0.000001

            # save prices rounded to satoshis
            frame['O'] = round(o, 9)
            frame['H'] = round(h, 9)
            frame['L'] = round(ll, 9)
            frame['C'] = round(c, 9)
            i += 1
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
    num_resutls = len(results)
    assert num_resutls > 0


def test_processed(default_conf, mocker):
    mocker.patch.dict('freqtrade.main._CONF', default_conf)
    data = load_data_test('raise')
    assert optimize.preprocess(data)


def test_raise(default_conf, mocker):
    mocker.patch.dict('freqtrade.main._CONF', default_conf)
    tests = [['raise', 359], ['lower', 0], ['sine', 1734]]
    for [contour, numres] in tests:
        simple_backtest(default_conf, contour, numres)
