# pragma pylint: disable=missing-docstring, protected-access, C0103

import json
import os
from pathlib import Path
import uuid
from shutil import copyfile

import arrow
from pandas import DataFrame
import pytest

from freqtrade import OperationalException
from freqtrade.arguments import TimeRange
from freqtrade.data import history
from freqtrade.data.history import (download_backtesting_testdata,
                                    load_cached_data_for_updating,
                                    load_tickerdata_file,
                                    make_testdata_path,
                                    trim_tickerlist)
from freqtrade.misc import file_dump_json
from freqtrade.tests.conftest import get_patched_exchange, log_has

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


def test_load_data_30min_ticker(mocker, caplog, default_conf) -> None:
    ld = history.load_pair_history(pair='UNITTEST/BTC', ticker_interval='30m', datadir=None)
    assert isinstance(ld, DataFrame)
    assert not log_has('Download the pair: "UNITTEST/BTC", Interval: 30m', caplog.record_tuples)


def test_load_data_7min_ticker(mocker, caplog, default_conf) -> None:
    ld = history.load_pair_history(pair='UNITTEST/BTC', ticker_interval='7m', datadir=None)
    assert not isinstance(ld, DataFrame)
    assert ld is None
    assert log_has(
        'No data for pair: "UNITTEST/BTC", Interval: 7m. '
        'Use --refresh-pairs-cached to download the data', caplog.record_tuples)


def test_load_data_1min_ticker(ticker_history, mocker, caplog) -> None:
    mocker.patch('freqtrade.exchange.Exchange.get_history', return_value=ticker_history)
    file = os.path.join(os.path.dirname(__file__), '..', 'testdata', 'UNITTEST_BTC-1m.json')
    _backup_file(file, copy_file=True)
    history.load_data(datadir=None, ticker_interval='1m', pairs=['UNITTEST/BTC'])
    assert os.path.isfile(file) is True
    assert not log_has('Download the pair: "UNITTEST/BTC", Interval: 1m', caplog.record_tuples)
    _clean_test_file(file)


def test_load_data_with_new_pair_1min(ticker_history_list, mocker, caplog, default_conf) -> None:
    """
    Test load_pair_history() with 1 min ticker
    """
    mocker.patch('freqtrade.exchange.Exchange.get_history', return_value=ticker_history_list)
    exchange = get_patched_exchange(mocker, default_conf)
    file = os.path.join(os.path.dirname(__file__), '..', 'testdata', 'MEME_BTC-1m.json')

    _backup_file(file)
    # do not download a new pair if refresh_pairs isn't set
    history.load_pair_history(datadir=None,
                              ticker_interval='1m',
                              refresh_pairs=False,
                              pair='MEME/BTC')
    assert os.path.isfile(file) is False
    assert log_has('No data for pair: "MEME/BTC", Interval: 1m. '
                   'Use --refresh-pairs-cached to download the data',
                   caplog.record_tuples)

    # download a new pair if refresh_pairs is set
    history.load_pair_history(datadir=None,
                              ticker_interval='1m',
                              refresh_pairs=True,
                              exchange=exchange,
                              pair='MEME/BTC')
    assert os.path.isfile(file) is True
    assert log_has('Download the pair: "MEME/BTC", Interval: 1m', caplog.record_tuples)
    with pytest.raises(OperationalException, match=r'Exchange needs to be initialized when.*'):
        history.load_pair_history(datadir=None,
                                  ticker_interval='1m',
                                  refresh_pairs=True,
                                  exchange=None,
                                  pair='MEME/BTC')
    _clean_test_file(file)


def test_testdata_path() -> None:
    assert str(Path('freqtrade') / 'tests' / 'testdata') in str(make_testdata_path(None))


def test_load_cached_data_for_updating(mocker) -> None:
    datadir = Path(__file__).parent.parent.joinpath('testdata')

    test_data = None
    test_filename = datadir.joinpath('UNITTEST_BTC-1m.json')
    with open(test_filename, "rt") as file:
        test_data = json.load(file)

    # change now time to test 'line' cases
    # now = last cached item + 1 hour
    now_ts = test_data[-1][0] / 1000 + 60 * 60
    mocker.patch('arrow.utcnow', return_value=arrow.get(now_ts))

    # timeframe starts earlier than the cached data
    # should fully update data
    timerange = TimeRange('date', None, test_data[0][0] / 1000 - 1, 0)
    data, start_ts = load_cached_data_for_updating(test_filename,
                                                   '1m',
                                                   timerange)
    assert data == []
    assert start_ts == test_data[0][0] - 1000

    # same with 'line' timeframe
    num_lines = (test_data[-1][0] - test_data[1][0]) / 1000 / 60 + 120
    data, start_ts = load_cached_data_for_updating(test_filename,
                                                   '1m',
                                                   TimeRange(None, 'line', 0, -num_lines))
    assert data == []
    assert start_ts < test_data[0][0] - 1

    # timeframe starts in the center of the cached data
    # should return the chached data w/o the last item
    timerange = TimeRange('date', None, test_data[0][0] / 1000 + 1, 0)
    data, start_ts = load_cached_data_for_updating(test_filename,
                                                   '1m',
                                                   timerange)
    assert data == test_data[:-1]
    assert test_data[-2][0] < start_ts < test_data[-1][0]

    # same with 'line' timeframe
    num_lines = (test_data[-1][0] - test_data[1][0]) / 1000 / 60 + 30
    timerange = TimeRange(None, 'line', 0, -num_lines)
    data, start_ts = load_cached_data_for_updating(test_filename,
                                                   '1m',
                                                   timerange)
    assert data == test_data[:-1]
    assert test_data[-2][0] < start_ts < test_data[-1][0]

    # timeframe starts after the chached data
    # should return the chached data w/o the last item
    timerange = TimeRange('date', None, test_data[-1][0] / 1000 + 1, 0)
    data, start_ts = load_cached_data_for_updating(test_filename,
                                                   '1m',
                                                   timerange)
    assert data == test_data[:-1]
    assert test_data[-2][0] < start_ts < test_data[-1][0]

    # same with 'line' timeframe
    num_lines = 30
    timerange = TimeRange(None, 'line', 0, -num_lines)
    data, start_ts = load_cached_data_for_updating(test_filename,
                                                   '1m',
                                                   timerange)
    assert data == test_data[:-1]
    assert test_data[-2][0] < start_ts < test_data[-1][0]

    # no timeframe is set
    # should return the chached data w/o the last item
    num_lines = 30
    timerange = TimeRange(None, 'line', 0, -num_lines)
    data, start_ts = load_cached_data_for_updating(test_filename,
                                                   '1m',
                                                   timerange)
    assert data == test_data[:-1]
    assert test_data[-2][0] < start_ts < test_data[-1][0]

    # no datafile exist
    # should return timestamp start time
    timerange = TimeRange('date', None, now_ts - 10000, 0)
    data, start_ts = load_cached_data_for_updating(test_filename.with_name('unexist'),
                                                   '1m',
                                                   timerange)
    assert data == []
    assert start_ts == (now_ts - 10000) * 1000

    # same with 'line' timeframe
    num_lines = 30
    timerange = TimeRange(None, 'line', 0, -num_lines)
    data, start_ts = load_cached_data_for_updating(test_filename.with_name('unexist'),
                                                   '1m',
                                                   timerange)
    assert data == []
    assert start_ts == (now_ts - num_lines * 60) * 1000

    # no datafile exist, no timeframe is set
    # should return an empty array and None
    data, start_ts = load_cached_data_for_updating(test_filename.with_name('unexist'),
                                                   '1m',
                                                   None)
    assert data == []
    assert start_ts is None


def test_download_backtesting_testdata(ticker_history_list, mocker, default_conf) -> None:
    mocker.patch('freqtrade.exchange.Exchange.get_history', return_value=ticker_history_list)
    exchange = get_patched_exchange(mocker, default_conf)
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

    assert download_backtesting_testdata(datadir=None, exchange=exchange,
                                         pair='MEME/BTC',
                                         tick_interval='1m')
    assert download_backtesting_testdata(datadir=None, exchange=exchange,
                                         pair='CFI/BTC',
                                         tick_interval='1m')
    assert not exchange._pairs_last_refresh_time
    assert os.path.isfile(file1_1) is True
    assert os.path.isfile(file2_1) is True

    # clean files freshly downloaded
    _clean_test_file(file1_1)
    _clean_test_file(file2_1)

    assert os.path.isfile(file1_5) is False
    assert os.path.isfile(file2_5) is False

    assert download_backtesting_testdata(datadir=None, exchange=exchange,
                                         pair='MEME/BTC',
                                         tick_interval='5m')
    assert download_backtesting_testdata(datadir=None, exchange=exchange,
                                         pair='CFI/BTC',
                                         tick_interval='5m')
    assert not exchange._pairs_last_refresh_time
    assert os.path.isfile(file1_5) is True
    assert os.path.isfile(file2_5) is True

    # clean files freshly downloaded
    _clean_test_file(file1_5)
    _clean_test_file(file2_5)


def test_download_backtesting_testdata2(mocker, default_conf) -> None:
    tick = [
        [1509836520000, 0.00162008, 0.00162008, 0.00162008, 0.00162008, 108.14853839],
        [1509836580000, 0.00161, 0.00161, 0.00161, 0.00161, 82.390199]
    ]
    json_dump_mock = mocker.patch('freqtrade.misc.file_dump_json', return_value=None)
    mocker.patch('freqtrade.exchange.Exchange.get_history', return_value=tick)
    exchange = get_patched_exchange(mocker, default_conf)
    download_backtesting_testdata(None, exchange, pair="UNITTEST/BTC", tick_interval='1m')
    download_backtesting_testdata(None, exchange, pair="UNITTEST/BTC", tick_interval='3m')
    assert json_dump_mock.call_count == 2


def test_download_backtesting_data_exception(ticker_history, mocker, caplog, default_conf) -> None:
    mocker.patch('freqtrade.exchange.Exchange.get_history',
                 side_effect=BaseException('File Error'))

    exchange = get_patched_exchange(mocker, default_conf)

    file1_1 = os.path.join(os.path.dirname(__file__), '..', 'testdata', 'MEME_BTC-1m.json')
    file1_5 = os.path.join(os.path.dirname(__file__), '..', 'testdata', 'MEME_BTC-5m.json')
    _backup_file(file1_1)
    _backup_file(file1_5)

    assert not download_backtesting_testdata(datadir=None, exchange=exchange,
                                             pair='MEME/BTC',
                                             tick_interval='1m')
    # clean files freshly downloaded
    _clean_test_file(file1_1)
    _clean_test_file(file1_5)
    assert log_has('Failed to download the pair: "MEME/BTC", Interval: 1m', caplog.record_tuples)


def test_load_tickerdata_file() -> None:
    # 7 does not exist in either format.
    assert not load_tickerdata_file(None, 'UNITTEST/BTC', '7m')
    # 1 exists only as a .json
    tickerdata = load_tickerdata_file(None, 'UNITTEST/BTC', '1m')
    assert _BTC_UNITTEST_LENGTH == len(tickerdata)
    # 8 .json is empty and will fail if it's loaded. .json.gz is a copy of 1.json
    tickerdata = load_tickerdata_file(None, 'UNITTEST/BTC', '8m')
    assert _BTC_UNITTEST_LENGTH == len(tickerdata)


def test_load_partial_missing(caplog) -> None:
    # Make sure we start fresh - test missing data at start
    start = arrow.get('2018-01-01T00:00:00')
    end = arrow.get('2018-01-11T00:00:00')
    tickerdata = history.load_data(None, '5m', ['UNITTEST/BTC'],
                                   refresh_pairs=False,
                                   timerange=TimeRange('date', 'date',
                                                       start.timestamp, end.timestamp))
    # timedifference in 5 minutes
    td = ((end - start).total_seconds() // 60 // 5) + 1
    assert td != len(tickerdata['UNITTEST/BTC'])
    start_real = tickerdata['UNITTEST/BTC'].iloc[0, 0]
    assert log_has(f'Missing data at start for pair '
                   f'UNITTEST/BTC, data starts at {start_real.strftime("%Y-%m-%d %H:%M:%S")}',
                   caplog.record_tuples)
    # Make sure we start fresh - test missing data at end
    caplog.clear()
    start = arrow.get('2018-01-10T00:00:00')
    end = arrow.get('2018-02-20T00:00:00')
    tickerdata = history.load_data(datadir=None, ticker_interval='5m',
                                   pairs=['UNITTEST/BTC'], refresh_pairs=False,
                                   timerange=TimeRange('date', 'date',
                                                       start.timestamp, end.timestamp))
    # timedifference in 5 minutes
    td = ((end - start).total_seconds() // 60 // 5) + 1
    assert td != len(tickerdata['UNITTEST/BTC'])
    # Shift endtime with +5 - as last candle is dropped (partial candle)
    end_real = arrow.get(tickerdata['UNITTEST/BTC'].iloc[-1, 0]).shift(minutes=5)
    assert log_has(f'Missing data at end for pair '
                   f'UNITTEST/BTC, data ends at {end_real.strftime("%Y-%m-%d %H:%M:%S")}',
                   caplog.record_tuples)


def test_init(default_conf, mocker) -> None:
    exchange = get_patched_exchange(mocker, default_conf)
    assert {} == history.load_data(
        datadir='',
        exchange=exchange,
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
    timerange = TimeRange(None, 'line', 0, -5)
    ticker = trim_tickerlist(ticker_list, timerange)
    ticker_len = len(ticker)

    assert ticker_len == 5
    assert ticker_list[0] is not ticker[0]  # The first element should be different
    assert ticker_list[-1] is ticker[-1]  # The last element must be the same

    # Test the pattern ^(\d+)-$
    # This pattern keep X element from the end
    timerange = TimeRange('line', None, 5, 0)
    ticker = trim_tickerlist(ticker_list, timerange)
    ticker_len = len(ticker)

    assert ticker_len == 5
    assert ticker_list[0] is ticker[0]  # The first element must be the same
    assert ticker_list[-1] is not ticker[-1]  # The last element should be different

    # Test the pattern ^(\d+)-(\d+)$
    # This pattern extract a window
    timerange = TimeRange('index', 'index', 5, 10)
    ticker = trim_tickerlist(ticker_list, timerange)
    ticker_len = len(ticker)

    assert ticker_len == 5
    assert ticker_list[0] is not ticker[0]  # The first element should be different
    assert ticker_list[5] is ticker[0]  # The list starts at the index 5
    assert ticker_list[9] is ticker[-1]  # The list ends at the index 9 (5 elements)

    # Test the pattern ^(\d{8})-(\d{8})$
    # This pattern extract a window between the dates
    timerange = TimeRange('date', 'date', ticker_list[5][0] / 1000, ticker_list[10][0] / 1000 - 1)
    ticker = trim_tickerlist(ticker_list, timerange)
    ticker_len = len(ticker)

    assert ticker_len == 5
    assert ticker_list[0] is not ticker[0]  # The first element should be different
    assert ticker_list[5] is ticker[0]  # The list starts at the index 5
    assert ticker_list[9] is ticker[-1]  # The list ends at the index 9 (5 elements)

    # Test the pattern ^-(\d{8})$
    # This pattern extracts elements from the start to the date
    timerange = TimeRange(None, 'date', 0, ticker_list[10][0] / 1000 - 1)
    ticker = trim_tickerlist(ticker_list, timerange)
    ticker_len = len(ticker)

    assert ticker_len == 10
    assert ticker_list[0] is ticker[0]  # The start of the list is included
    assert ticker_list[9] is ticker[-1]  # The element 10 is not included

    # Test the pattern ^(\d{8})-$
    # This pattern extracts elements from the date to now
    timerange = TimeRange('date', None, ticker_list[10][0] / 1000 - 1, None)
    ticker = trim_tickerlist(ticker_list, timerange)
    ticker_len = len(ticker)

    assert ticker_len == ticker_list_len - 10
    assert ticker_list[10] is ticker[0]  # The first element is element #10
    assert ticker_list[-1] is ticker[-1]  # The last element is the same

    # Test a wrong pattern
    # This pattern must return the list unchanged
    timerange = TimeRange(None, None, None, 5)
    ticker = trim_tickerlist(ticker_list, timerange)
    ticker_len = len(ticker)

    assert ticker_list_len == ticker_len

    # Test invalid timerange (start after stop)
    timerange = TimeRange('index', 'index', 10, 5)
    with pytest.raises(ValueError, match=r'The timerange .* is incorrect'):
        trim_tickerlist(ticker_list, timerange)

    assert ticker_list_len == ticker_len

    # passing empty list
    timerange = TimeRange(None, None, None, 5)
    ticker = trim_tickerlist([], timerange)
    assert 0 == len(ticker)
    assert not ticker


def test_file_dump_json() -> None:
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
