# pragma pylint: disable=missing-docstring,W0212

import os
import logging
from shutil import copyfile
from freqtrade import exchange, optimize
from freqtrade.exchange import Bittrex
from freqtrade.optimize.__init__ import make_testdata_path, download_pairs,\
     download_backtesting_testdata, load_tickerdata_file

# Change this if modifying BTC_UNITEST testdatafile
_btc_unittest_length = 13681


def _backup_file(file: str, copy_file: bool = False) -> None:
    """
    Backup existing file to avoid deleting the user file
    :param file: complete path to the file
    :param touch_file: create an empty file in replacement
    :return: None
    """
    file_swp = file + '.swp'
    if os.path.isfile(file):
        os.rename(file, file_swp)

        if copy_file:
            copyfile(file_swp, file)


def _clean_test_file(file: str) -> None:
    """
    Backup existing file to avoid deleting the user file
    :param file: complete path to the file
    :return: None
    """
    file_swp = file + '.swp'
    # 1. Delete file from the test
    if os.path.isfile(file):
        os.remove(file)

    # 2. Rollback to the initial file
    if os.path.isfile(file_swp):
        os.rename(file_swp, file)


def test_load_data_5min_ticker(default_conf, ticker_history, mocker, caplog):
    mocker.patch('freqtrade.optimize.get_ticker_history', return_value=ticker_history)
    mocker.patch.dict('freqtrade.main._CONF', default_conf)

    exchange._API = Bittrex({'key': '', 'secret': ''})

    file = 'freqtrade/tests/testdata/BTC_ETH-5.json'
    _backup_file(file, copy_file=True)
    optimize.load_data(None, pairs=['BTC_ETH'])
    assert os.path.isfile(file) is True
    assert ('freqtrade.optimize',
            logging.INFO,
            'Download the pair: "BTC_ETH", Interval: 5 min'
            ) not in caplog.record_tuples
    _clean_test_file(file)


def test_load_data_1min_ticker(default_conf, ticker_history, mocker, caplog):
    mocker.patch('freqtrade.optimize.get_ticker_history', return_value=ticker_history)
    mocker.patch.dict('freqtrade.main._CONF', default_conf)

    exchange._API = Bittrex({'key': '', 'secret': ''})

    file = 'freqtrade/tests/testdata/BTC_ETH-1.json'
    _backup_file(file, copy_file=True)
    optimize.load_data(None, ticker_interval=1, pairs=['BTC_ETH'])
    assert os.path.isfile(file) is True
    assert ('freqtrade.optimize',
            logging.INFO,
            'Download the pair: "BTC_ETH", Interval: 1 min'
            ) not in caplog.record_tuples
    _clean_test_file(file)


def test_load_data_with_new_pair_1min(default_conf, ticker_history, mocker, caplog):
    mocker.patch('freqtrade.optimize.get_ticker_history', return_value=ticker_history)
    mocker.patch.dict('freqtrade.main._CONF', default_conf)

    exchange._API = Bittrex({'key': '', 'secret': ''})

    file = 'freqtrade/tests/testdata/BTC_MEME-1.json'
    _backup_file(file)
    optimize.load_data(None, ticker_interval=1, pairs=['BTC_MEME'])
    assert os.path.isfile(file) is True
    assert ('freqtrade.optimize',
            logging.INFO,
            'Download the pair: "BTC_MEME", Interval: 1 min'
            ) in caplog.record_tuples
    _clean_test_file(file)


def test_testdata_path():
    assert os.path.join('freqtrade', 'tests', 'testdata') in make_testdata_path(None)


def test_download_pairs(default_conf, ticker_history, mocker):
    mocker.patch('freqtrade.optimize.__init__.get_ticker_history', return_value=ticker_history)
    mocker.patch.dict('freqtrade.main._CONF', default_conf)
    exchange._API = Bittrex({'key': '', 'secret': ''})

    file1_1 = 'freqtrade/tests/testdata/BTC_MEME-1.json'
    file1_5 = 'freqtrade/tests/testdata/BTC_MEME-5.json'
    file2_1 = 'freqtrade/tests/testdata/BTC_CFI-1.json'
    file2_5 = 'freqtrade/tests/testdata/BTC_CFI-5.json'

    _backup_file(file1_1)
    _backup_file(file1_5)
    _backup_file(file2_1)
    _backup_file(file2_5)

    assert download_pairs(None, pairs=['BTC-MEME', 'BTC-CFI']) is True

    assert os.path.isfile(file1_1) is True
    assert os.path.isfile(file1_5) is True
    assert os.path.isfile(file2_1) is True
    assert os.path.isfile(file2_5) is True

    # clean files freshly downloaded
    _clean_test_file(file1_1)
    _clean_test_file(file1_5)
    _clean_test_file(file2_1)
    _clean_test_file(file2_5)


def test_download_pairs_exception(default_conf, ticker_history, mocker, caplog):
    mocker.patch('freqtrade.optimize.__init__.get_ticker_history', return_value=ticker_history)
    mocker.patch('freqtrade.optimize.__init__.download_backtesting_testdata',
                 side_effect=BaseException('File Error'))
    mocker.patch.dict('freqtrade.main._CONF', default_conf)
    exchange._API = Bittrex({'key': '', 'secret': ''})

    file1_1 = 'freqtrade/tests/testdata/BTC_MEME-1.json'
    file1_5 = 'freqtrade/tests/testdata/BTC_MEME-5.json'
    _backup_file(file1_1)
    _backup_file(file1_5)

    download_pairs(None, pairs=['BTC-MEME'])
    # clean files freshly downloaded
    _clean_test_file(file1_1)
    _clean_test_file(file1_5)
    assert ('freqtrade.optimize.__init__',
            logging.INFO,
            'Failed to download the pair: "BTC-MEME", Interval: 1 min'
            ) in caplog.record_tuples


def test_download_backtesting_testdata(default_conf, ticker_history, mocker):
    mocker.patch('freqtrade.optimize.__init__.get_ticker_history', return_value=ticker_history)
    mocker.patch.dict('freqtrade.main._CONF', default_conf)
    exchange._API = Bittrex({'key': '', 'secret': ''})

    # Download a 1 min ticker file
    file1 = 'freqtrade/tests/testdata/BTC_XEL-1.json'
    _backup_file(file1)
    download_backtesting_testdata(None, pair="BTC-XEL", interval=1)
    assert os.path.isfile(file1) is True
    _clean_test_file(file1)

    # Download a 5 min ticker file
    file2 = 'freqtrade/tests/testdata/BTC_STORJ-5.json'
    _backup_file(file2)

    download_backtesting_testdata(None, pair="BTC-STORJ", interval=5)
    assert os.path.isfile(file2) is True
    _clean_test_file(file2)


def test_load_tickerdata_file():
    assert not load_tickerdata_file(None, 'BTC_UNITEST', 7)
    tickerdata = load_tickerdata_file(None, 'BTC_UNITEST', 1)
    assert _btc_unittest_length == len(tickerdata)


def test_tickerdata_to_dataframe():
    tick = load_tickerdata_file(None, 'BTC_UNITEST', 1)
    tickerlist = {'BTC_UNITEST': tick}
    data = optimize.tickerdata_to_dataframe(tickerlist, timeperiod=-100)
    assert 100 == len(data['BTC_UNITEST'])
