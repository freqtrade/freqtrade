# pragma pylint: disable=missing-docstring, protected-access, C0103

import json
import os
import uuid
import arrow
from shutil import copyfile

from freqtrade import optimize
from freqtrade.misc import file_dump_json
from freqtrade.optimize.__init__ import make_testdata_path, download_pairs, \
    download_backtesting_testdata, load_tickerdata_file, trim_tickerlist, \
    get_start_ts_from_timerange
from freqtrade.tests.conftest import log_has

# Change this if modifying UNITTEST/BTC testdatafile
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


def test_load_data_30min_ticker(ticker_history, mocker, caplog) -> None:
    """
    Test load_data() with 30 min ticker
    """
    mocker.patch('freqtrade.optimize.get_ticker_history', return_value=ticker_history)

    file = os.path.join(os.path.dirname(__file__), '..', 'testdata', 'UNITTEST_BTC-30m.json')
    _backup_file(file, copy_file=True)
    optimize.load_data(None, pairs=['UNITTEST/BTC'], ticker_interval='30m')
    assert os.path.isfile(file) is True
    assert not log_has('Download the pair: "UNITTEST/BTC", Interval: 30m', caplog.record_tuples)
    _clean_test_file(file)


def test_load_data_5min_ticker(ticker_history, mocker, caplog) -> None:
    """
    Test load_data() with 5 min ticker
    """
    mocker.patch('freqtrade.optimize.get_ticker_history', return_value=ticker_history)

    file = os.path.join(os.path.dirname(__file__), '..', 'testdata', 'UNITTEST_BTC-5m.json')
    _backup_file(file, copy_file=True)
    optimize.load_data(None, pairs=['UNITTEST/BTC'], ticker_interval='5m')
    assert os.path.isfile(file) is True
    assert not log_has('Download the pair: "UNITTEST/BTC", Interval: 5m', caplog.record_tuples)
    _clean_test_file(file)


def test_load_data_1min_ticker(ticker_history, mocker, caplog) -> None:
    """
    Test load_data() with 1 min ticker
    """
    mocker.patch('freqtrade.optimize.get_ticker_history', return_value=ticker_history)

    file = os.path.join(os.path.dirname(__file__), '..', 'testdata', 'UNITTEST_BTC-1m.json')
    _backup_file(file, copy_file=True)
    optimize.load_data(None, ticker_interval='1m', pairs=['UNITTEST/BTC'])
    assert os.path.isfile(file) is True
    assert not log_has('Download the pair: "UNITTEST/BTC", Interval: 1m', caplog.record_tuples)
    _clean_test_file(file)


def test_load_data_with_new_pair_1min(ticker_history, mocker, caplog) -> None:
    """
    Test load_data() with 1 min ticker
    """
    mocker.patch('freqtrade.optimize.get_ticker_history', return_value=ticker_history)

    file = os.path.join(os.path.dirname(__file__), '..', 'testdata', 'MEME_BTC-1m.json')

    _backup_file(file)
    optimize.load_data(None, ticker_interval='1m', pairs=['MEME/BTC'])
    assert os.path.isfile(file) is True
    assert log_has('Download the pair: "MEME/BTC", Interval: 1m', caplog.record_tuples)
    _clean_test_file(file)


def test_testdata_path() -> None:
    assert os.path.join('freqtrade', 'tests', 'testdata') in make_testdata_path(None)


def test_download_pairs(ticker_history, mocker) -> None:
    mocker.patch('freqtrade.optimize.__init__.get_ticker_history', return_value=ticker_history)

    file1_1 = os.path.join(os.path.dirname(__file__), '..', 'testdata', 'MEME_BTC-1m.json')
    file1_5 = os.path.join(os.path.dirname(__file__), '..', 'testdata', 'MEME_BTC-5m.json')
    file2_1 = os.path.join(os.path.dirname(__file__), '..', 'testdata', 'CFI_BTC-1m.json')
    file2_5 = os.path.join(os.path.dirname(__file__), '..', 'testdata', 'CFI_BTC-5m.json')

    _backup_file(file1_1)
    _backup_file(file1_5)
    _backup_file(file2_1)
    _backup_file(file2_5)

    assert os.path.isfile(file1_1) is False
    assert os.path.isfile(file2_1) is False

    assert download_pairs(None, pairs=['MEME/BTC', 'CFI/BTC'], ticker_interval='1m') is True

    assert os.path.isfile(file1_1) is True
    assert os.path.isfile(file2_1) is True

    # clean files freshly downloaded
    _clean_test_file(file1_1)
    _clean_test_file(file2_1)

    assert os.path.isfile(file1_5) is False
    assert os.path.isfile(file2_5) is False

    assert download_pairs(None, pairs=['MEME/BTC', 'CFI/BTC'], ticker_interval='5m') is True

    assert os.path.isfile(file1_5) is True
    assert os.path.isfile(file2_5) is True

    # clean files freshly downloaded
    _clean_test_file(file1_5)
    _clean_test_file(file2_5)


def test_get_start_ts_from_timerange(mocker) -> None:
    start = get_start_ts_from_timerange(None, '1m')
    assert start is None

    # check 'date'
    start = get_start_ts_from_timerange((('date', 'date'), 1000, 2000), '1m')
    assert start == 1000 * 1000

    start = get_start_ts_from_timerange((('date', 'date'), 1000, 2000), '5m')
    assert start == 1000 * 1000

    # check 'line'
    mock_now = arrow.get(1367900664)
    mocker.patch('arrow.utcnow', return_value=mock_now)

    start = get_start_ts_from_timerange(((None, 'line'), None, -200), '1m')
    assert start == (1367900664 - 200 * 60) * 1000

    start = get_start_ts_from_timerange(((None, 'line'), None, -200), '5m')
    assert start == (1367900664 - 5 * 200 * 60) * 1000
    

def test_download_pairs_exception(ticker_history, mocker, caplog) -> None:
    mocker.patch('freqtrade.optimize.__init__.get_ticker_history', return_value=ticker_history)
    mocker.patch('freqtrade.optimize.__init__.download_backtesting_testdata',
                 side_effect=BaseException('File Error'))

    file1_1 = os.path.join(os.path.dirname(__file__), '..', 'testdata', 'MEME_BTC-1m.json')
    file1_5 = os.path.join(os.path.dirname(__file__), '..', 'testdata', 'MEME_BTC-5m.json')
    _backup_file(file1_1)
    _backup_file(file1_5)

    download_pairs(None, pairs=['MEME/BTC'], ticker_interval='1m')
    # clean files freshly downloaded
    _clean_test_file(file1_1)
    _clean_test_file(file1_5)
    assert log_has('Failed to download the pair: "MEME/BTC", Interval: 1m', caplog.record_tuples)


def test_download_backtesting_testdata(ticker_history, mocker) -> None:
    mocker.patch('freqtrade.optimize.__init__.get_ticker_history', return_value=ticker_history)

    # Download a 1 min ticker file
    file1 = os.path.join(os.path.dirname(__file__), '..', 'testdata', 'XEL_BTC-1m.json')
    _backup_file(file1)
    download_backtesting_testdata(None, pair="XEL/BTC", interval='1m')
    assert os.path.isfile(file1) is True
    _clean_test_file(file1)

    # Download a 5 min ticker file
    file2 = os.path.join(os.path.dirname(__file__), '..', 'testdata', 'STORJ_BTC-5m.json')
    _backup_file(file2)

    download_backtesting_testdata(None, pair="STORJ/BTC", interval='5m')
    assert os.path.isfile(file2) is True
    _clean_test_file(file2)


def test_download_backtesting_testdata2(mocker) -> None:
    tick = [
        [1509836520000, 0.00162008, 0.00162008, 0.00162008, 0.00162008, 108.14853839],
        [1509836580000, 0.00161, 0.00161, 0.00161, 0.00161, 82.390199]
    ]
    mocker.patch('freqtrade.misc.file_dump_json', return_value=None)
    mocker.patch('freqtrade.optimize.__init__.get_ticker_history', return_value=tick)
    assert download_backtesting_testdata(None, pair="UNITTEST/BTC", interval='1m')
    assert download_backtesting_testdata(None, pair="UNITTEST/BTC", interval='3m')


def test_load_tickerdata_file() -> None:
    # 7 does not exist in either format.
    assert not load_tickerdata_file(None, 'UNITTEST/BTC', '7m')
    # 1 exists only as a .json
    tickerdata = load_tickerdata_file(None, 'UNITTEST/BTC', '1m')
    assert _BTC_UNITTEST_LENGTH == len(tickerdata)
    # 8 .json is empty and will fail if it's loaded. .json.gz is a copy of 1.json
    tickerdata = load_tickerdata_file(None, 'UNITTEST/BTC', '8m')
    assert _BTC_UNITTEST_LENGTH == len(tickerdata)


def test_init(default_conf, mocker) -> None:
    conf = {'exchange': {'pair_whitelist': []}}
    mocker.patch('freqtrade.optimize.hyperopt_optimize_conf', return_value=conf)
    assert {} == optimize.load_data(
        '',
        pairs=[],
        refresh_pairs=True,
        ticker_interval=default_conf['ticker_interval']
    )


def test_trim_tickerlist() -> None:
    file = os.path.join(os.path.dirname(__file__), '..', 'testdata', 'UNITTEST_BTC-1m.json')
    with open(file) as data_file:
        ticker_list = json.load(data_file)
    ticker_list_len = len(ticker_list)

    # Test the pattern ^(-\d+)$
    # This pattern uses the latest N elements
    timerange = ((None, 'line'), None, -5)
    ticker = trim_tickerlist(ticker_list, timerange)
    ticker_len = len(ticker)

    assert ticker_len == 5
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

    # Test the pattern ^(\d{8})-(\d{8})$
    # This pattern extract a window between the dates
    timerange = (('date', 'date'), ticker_list[5][0] / 1000, ticker_list[10][0] / 1000 - 1)
    ticker = trim_tickerlist(ticker_list, timerange)
    ticker_len = len(ticker)

    assert ticker_len == 5
    assert ticker_list[0] is not ticker[0]  # The first element should be different
    assert ticker_list[5] is ticker[0]  # The list starts at the index 5
    assert ticker_list[9] is ticker[-1]  # The list ends at the index 9 (5 elements)

    # Test the pattern ^-(\d{8})$
    # This pattern extracts elements from the start to the date
    timerange = ((None, 'date'), None, ticker_list[10][0] / 1000 - 1)
    ticker = trim_tickerlist(ticker_list, timerange)
    ticker_len = len(ticker)

    assert ticker_len == 10
    assert ticker_list[0] is ticker[0]  # The start of the list is included
    assert ticker_list[9] is ticker[-1]  # The element 10 is not included

    # Test the pattern ^(\d{8})-$
    # This pattern extracts elements from the date to now
    timerange = (('date', None), ticker_list[10][0] / 1000 - 1, None)
    ticker = trim_tickerlist(ticker_list, timerange)
    ticker_len = len(ticker)

    assert ticker_len == ticker_list_len - 10
    assert ticker_list[10] is ticker[0]  # The first element is element #10
    assert ticker_list[-1] is ticker[-1]  # The last element is the same

    # Test a wrong pattern
    # This pattern must return the list unchanged
    timerange = ((None, None), None, 5)
    ticker = trim_tickerlist(ticker_list, timerange)
    ticker_len = len(ticker)

    assert ticker_list_len == ticker_len


def test_file_dump_json() -> None:
    """
    Test file_dump_json()
    :return: None
    """
    file = os.path.join(os.path.dirname(__file__), '..', 'testdata',
                        'test_{id}.json'.format(id=str(uuid.uuid4())))
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
