# pragma pylint: disable=missing-docstring, protected-access, C0103

import os
import json
import logging
import uuid
from shutil import copyfile
from freqtrade import exchange, optimize
from freqtrade.exchange import Bittrex
from freqtrade.optimize.__init__ import make_testdata_path, download_pairs,\
    download_backtesting_testdata, load_tickerdata_file, trim_tickerlist, file_dump_json

# Change this if modifying BTC_UNITEST testdatafile
_BTC_UNITTEST_LENGTH = 13681


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


def test_load_data_30min_ticker(default_conf, ticker_history, mocker, caplog):
    caplog.set_level(logging.INFO)
    mocker.patch('freqtrade.optimize.get_ticker_history', return_value=ticker_history)
    mocker.patch.dict('freqtrade.main._CONF', default_conf)

    exchange._API = Bittrex({'key': '', 'secret': ''})

    file = 'freqtrade/tests/testdata/BTC_UNITTEST-30.json'
    _backup_file(file, copy_file=True)
    optimize.load_data(None, pairs=['BTC_UNITTEST'], ticker_interval=30)
    assert os.path.isfile(file) is True
    assert ('freqtrade.optimize',
            logging.INFO,
            'Download the pair: "BTC_ETH", Interval: 30 min') not in caplog.record_tuples
    _clean_test_file(file)


def test_load_data_5min_ticker(default_conf, ticker_history, mocker, caplog):
    caplog.set_level(logging.INFO)
    mocker.patch('freqtrade.optimize.get_ticker_history', return_value=ticker_history)
    mocker.patch.dict('freqtrade.main._CONF', default_conf)

    exchange._API = Bittrex({'key': '', 'secret': ''})

    file = 'freqtrade/tests/testdata/BTC_ETH-5.json'
    _backup_file(file, copy_file=True)
    optimize.load_data(None, pairs=['BTC_ETH'], ticker_interval=5)
    assert os.path.isfile(file) is True
    assert ('freqtrade.optimize',
            logging.INFO,
            'Download the pair: "BTC_ETH", Interval: 5 min') not in caplog.record_tuples
    _clean_test_file(file)


def test_load_data_1min_ticker(default_conf, ticker_history, mocker, caplog):
    caplog.set_level(logging.INFO)
    mocker.patch('freqtrade.optimize.get_ticker_history', return_value=ticker_history)
    mocker.patch.dict('freqtrade.main._CONF', default_conf)

    exchange._API = Bittrex({'key': '', 'secret': ''})

    file = 'freqtrade/tests/testdata/BTC_ETH-1.json'
    _backup_file(file, copy_file=True)
    optimize.load_data(None, ticker_interval=1, pairs=['BTC_ETH'])
    assert os.path.isfile(file) is True
    assert ('freqtrade.optimize',
            logging.INFO,
            'Download the pair: "BTC_ETH", Interval: 1 min') not in caplog.record_tuples
    _clean_test_file(file)


def test_load_data_with_new_pair_1min(default_conf, ticker_history, mocker, caplog):
    caplog.set_level(logging.INFO)
    mocker.patch('freqtrade.optimize.get_ticker_history', return_value=ticker_history)
    mocker.patch.dict('freqtrade.main._CONF', default_conf)

    exchange._API = Bittrex({'key': '', 'secret': ''})

    file = 'freqtrade/tests/testdata/BTC_MEME-1.json'
    _backup_file(file)
    optimize.load_data(None, ticker_interval=1, pairs=['BTC_MEME'])
    assert os.path.isfile(file) is True
    assert ('freqtrade.optimize',
            logging.INFO,
            'Download the pair: "BTC_MEME", Interval: 1 min') in caplog.record_tuples
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

    assert os.path.isfile(file1_1) is False
    assert os.path.isfile(file2_1) is False

    assert download_pairs(None, pairs=['BTC-MEME', 'BTC-CFI'], ticker_interval=1) is True

    assert os.path.isfile(file1_1) is True
    assert os.path.isfile(file2_1) is True

    # clean files freshly downloaded
    _clean_test_file(file1_1)
    _clean_test_file(file2_1)

    assert os.path.isfile(file1_5) is False
    assert os.path.isfile(file2_5) is False

    assert download_pairs(None, pairs=['BTC-MEME', 'BTC-CFI'], ticker_interval=5) is True

    assert os.path.isfile(file1_5) is True
    assert os.path.isfile(file2_5) is True

    # clean files freshly downloaded
    _clean_test_file(file1_5)
    _clean_test_file(file2_5)


def test_download_pairs_exception(default_conf, ticker_history, mocker, caplog):
    caplog.set_level(logging.INFO)
    mocker.patch('freqtrade.optimize.__init__.get_ticker_history', return_value=ticker_history)
    mocker.patch('freqtrade.optimize.__init__.download_backtesting_testdata',
                 side_effect=BaseException('File Error'))
    mocker.patch.dict('freqtrade.main._CONF', default_conf)
    exchange._API = Bittrex({'key': '', 'secret': ''})

    file1_1 = 'freqtrade/tests/testdata/BTC_MEME-1.json'
    file1_5 = 'freqtrade/tests/testdata/BTC_MEME-5.json'
    _backup_file(file1_1)
    _backup_file(file1_5)

    download_pairs(None, pairs=['BTC-MEME'], ticker_interval=1)
    # clean files freshly downloaded
    _clean_test_file(file1_1)
    _clean_test_file(file1_5)
    assert ('freqtrade.optimize.__init__',
            logging.INFO,
            'Failed to download the pair: "BTC-MEME", Interval: 1 min') in caplog.record_tuples


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


def test_download_backtesting_testdata2(mocker):
    tick = [{'T': 'bar'}, {'T': 'foo'}]
    mocker.patch('freqtrade.misc.file_dump_json', return_value=None)
    mocker.patch('freqtrade.optimize.__init__.get_ticker_history', return_value=tick)
    assert download_backtesting_testdata(None, pair="BTC-UNITEST", interval=1)
    assert download_backtesting_testdata(None, pair="BTC-UNITEST", interval=3)


def test_load_tickerdata_file():
    # 7 does not exist in either format.
    assert not load_tickerdata_file(None, 'BTC_UNITEST', 7)
    # 1 exists only as a .json
    tickerdata = load_tickerdata_file(None, 'BTC_UNITEST', 1)
    assert _BTC_UNITTEST_LENGTH == len(tickerdata)
    # 8 .json is empty and will fail if it's loaded. .json.gz is a copy of 1.json
    tickerdata = load_tickerdata_file(None, 'BTC_UNITEST', 8)
    assert _BTC_UNITTEST_LENGTH == len(tickerdata)


def test_init(default_conf, mocker):
    conf = {'exchange': {'pair_whitelist': []}}
    mocker.patch('freqtrade.optimize.hyperopt_optimize_conf', return_value=conf)
    assert {} == optimize.load_data('', pairs=[], refresh_pairs=True,
                                    ticker_interval=int(default_conf['ticker_interval']))


def test_tickerdata_to_dataframe():
    timerange = ((None, 'line'), None, -100)
    tick = load_tickerdata_file(None, 'BTC_UNITEST', 1, timerange=timerange)
    tickerlist = {'BTC_UNITEST': tick}
    data = optimize.tickerdata_to_dataframe(tickerlist)
    assert len(data['BTC_UNITEST']) == 100


def test_trim_tickerlist():
    with open('freqtrade/tests/testdata/BTC_ETH-1.json') as data_file:
        ticker_list = json.load(data_file)
    ticker_list_len = len(ticker_list)

    # Test the pattern ^(-\d+)$
    # This pattern remove X element from the beginning
    timerange = ((None, 'line'), None, 5)
    ticker = trim_tickerlist(ticker_list, timerange)
    ticker_len = len(ticker)

    assert ticker_list_len == ticker_len + 5
    assert ticker_list[0] is not ticker[0]  # The first element should be different
    assert ticker_list[-1] is ticker[-1]  # The last element must be the same

    # Test the pattern ^(\d+)-$
    # This pattern keep X element from the end
    timerange = (('line', None), 5, None)
    ticker = trim_tickerlist(ticker_list, timerange)
    ticker_len = len(ticker)

    assert ticker_len == 5
    assert ticker_list[0] is ticker[0]  # The first element must be the same
    assert ticker_list[-1] is not ticker[-1]  # The last element should be different

    # Test the pattern ^(\d+)-(\d+)$
    # This pattern extract a window
    timerange = (('index', 'index'), 5, 10)
    ticker = trim_tickerlist(ticker_list, timerange)
    ticker_len = len(ticker)

    assert ticker_len == 5
    assert ticker_list[0] is not ticker[0]  # The first element should be different
    assert ticker_list[5] is ticker[0]  # The list starts at the index 5
    assert ticker_list[9] is ticker[-1]  # The list ends at the index 9 (5 elements)

    # Test a wrong pattern
    # This pattern must return the list unchanged
    timerange = ((None, None), None, 5)
    ticker = trim_tickerlist(ticker_list, timerange)
    ticker_len = len(ticker)

    assert ticker_list_len == ticker_len


def test_file_dump_json():

    file = 'freqtrade/tests/testdata/test_{id}.json'.format(id=str(uuid.uuid4()))
    data = {'bar': 'foo'}

    # check the file we will create does not exist
    assert os.path.isfile(file) is False

    # Create the Json file
    file_dump_json(file, data)

    # Check the file was create
    assert os.path.isfile(file) is True

    # Open the Json file created and test the data is in it
    with open(file) as data_file:
        json_from_file = json.load(data_file)

    assert 'bar' in json_from_file
    assert json_from_file['bar'] == 'foo'

    # Remove the file
    _clean_test_file(file)
