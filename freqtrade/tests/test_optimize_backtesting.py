# pragma pylint: disable=missing-docstring,W0212

from freqtrade import exchange, optimize
from freqtrade.exchange import Bittrex
from freqtrade.optimize.backtesting import backtest
from freqtrade.optimize.__init__ import testdata_path, download_pairs, download_backtesting_testdata
import os


def test_backtest(default_conf, mocker):
    mocker.patch.dict('freqtrade.main._CONF', default_conf)
    exchange._API = Bittrex({'key': '', 'secret': ''})

    data = optimize.load_data(ticker_interval=5, pairs=['BTC_ETH'])
    results = backtest(default_conf['stake_amount'], optimize.preprocess(data), 10, True)
    num_results = len(results)
    assert num_results > 0


def test_1min_ticker_interval(default_conf, mocker):
    mocker.patch.dict('freqtrade.main._CONF', default_conf)
    exchange._API = Bittrex({'key': '', 'secret': ''})

    # Run a backtesting for an exiting 5min ticker_interval
    data = optimize.load_data(ticker_interval=1, pairs=['BTC_UNITEST'])
    results = backtest(default_conf['stake_amount'], optimize.preprocess(data), 1, True)
    assert len(results) > 0


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
    assert str('freqtrade/optimize/../tests/testdata') in testdata_path()


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
