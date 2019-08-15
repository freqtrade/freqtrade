# pragma pylint: disable=missing-docstring, protected-access, C0103

import json
import uuid
from pathlib import Path
from shutil import copyfile
from unittest.mock import MagicMock, PropertyMock

import arrow
import pytest
from pandas import DataFrame

from freqtrade import OperationalException
from freqtrade.configuration import TimeRange
from freqtrade.data import history
from freqtrade.data.history import (download_pair_history,
                                    _load_cached_data_for_updating,
                                    refresh_backtest_ohlcv_data,
                                    load_tickerdata_file, pair_data_filename,
                                    pair_trades_filename,
                                    trim_tickerlist)
from freqtrade.exchange import timeframe_to_minutes
from freqtrade.misc import file_dump_json
from freqtrade.strategy.default_strategy import DefaultStrategy
from tests.conftest import (get_patched_exchange, log_has, log_has_re,
                            patch_exchange)

# Change this if modifying UNITTEST/BTC testdatafile
_BTC_UNITTEST_LENGTH = 13681


def _backup_file(file: Path, copy_file: bool = False) -> None:
    """
    Backup existing file to avoid deleting the user file
    :param file: complete path to the file
    :param touch_file: create an empty file in replacement
    :return: None
    """
    file_swp = str(file) + '.swp'
    if file.is_file():
        file.rename(file_swp)

        if copy_file:
            copyfile(file_swp, file)


def _clean_test_file(file: Path) -> None:
    """
    Backup existing file to avoid deleting the user file
    :param file: complete path to the file
    :return: None
    """
    file_swp = Path(str(file) + '.swp')
    # 1. Delete file from the test
    if file.is_file():
        file.unlink()

    # 2. Rollback to the initial file
    if file_swp.is_file():
        file_swp.rename(file)


def test_load_data_30min_ticker(mocker, caplog, default_conf, testdatadir) -> None:
    ld = history.load_pair_history(pair='UNITTEST/BTC', ticker_interval='30m', datadir=testdatadir)
    assert isinstance(ld, DataFrame)
    assert not log_has(
        'Download history data for pair: "UNITTEST/BTC", interval: 30m '
        'and store in None.', caplog
    )


def test_load_data_7min_ticker(mocker, caplog, default_conf, testdatadir) -> None:
    ld = history.load_pair_history(pair='UNITTEST/BTC', ticker_interval='7m', datadir=testdatadir)
    assert not isinstance(ld, DataFrame)
    assert ld is None
    assert log_has(
        'No history data for pair: "UNITTEST/BTC", interval: 7m. '
        'Use `freqtrade download-data` to download the data', caplog
    )


def test_load_data_1min_ticker(ticker_history, mocker, caplog, testdatadir) -> None:
    mocker.patch('freqtrade.exchange.Exchange.get_historic_ohlcv', return_value=ticker_history)
    file = testdatadir / 'UNITTEST_BTC-1m.json'
    _backup_file(file, copy_file=True)
    history.load_data(datadir=testdatadir, ticker_interval='1m', pairs=['UNITTEST/BTC'])
    assert file.is_file()
    assert not log_has(
        'Download history data for pair: "UNITTEST/BTC", interval: 1m '
        'and store in None.', caplog
    )
    _clean_test_file(file)


def test_load_data_with_new_pair_1min(ticker_history_list, mocker, caplog,
                                      default_conf, testdatadir) -> None:
    """
    Test load_pair_history() with 1 min ticker
    """
    mocker.patch('freqtrade.exchange.Exchange.get_historic_ohlcv', return_value=ticker_history_list)
    exchange = get_patched_exchange(mocker, default_conf)
    file = testdatadir / 'MEME_BTC-1m.json'

    _backup_file(file)
    # do not download a new pair if refresh_pairs isn't set
    history.load_pair_history(datadir=testdatadir,
                              ticker_interval='1m',
                              pair='MEME/BTC')
    assert not file.is_file()
    assert log_has(
        'No history data for pair: "MEME/BTC", interval: 1m. '
        'Use `freqtrade download-data` to download the data', caplog
    )

    # download a new pair if refresh_pairs is set
    history.load_pair_history(datadir=testdatadir,
                              ticker_interval='1m',
                              refresh_pairs=True,
                              exchange=exchange,
                              pair='MEME/BTC')
    assert file.is_file()
    assert log_has_re(
        'Download history data for pair: "MEME/BTC", interval: 1m '
        'and store in .*', caplog
    )
    with pytest.raises(OperationalException, match=r'Exchange needs to be initialized when.*'):
        history.load_pair_history(datadir=testdatadir,
                                  ticker_interval='1m',
                                  refresh_pairs=True,
                                  exchange=None,
                                  pair='MEME/BTC')
    _clean_test_file(file)


def test_testdata_path(testdatadir) -> None:
    assert str(Path('tests') / 'testdata') in str(testdatadir)


def test_pair_data_filename():
    fn = pair_data_filename(Path('freqtrade/hello/world'), 'ETH/BTC', '5m')
    assert isinstance(fn, Path)
    assert fn == Path('freqtrade/hello/world/ETH_BTC-5m.json')


def test_pair_trades_filename():
    fn = pair_trades_filename(Path('freqtrade/hello/world'), 'ETH/BTC')
    assert isinstance(fn, Path)
    assert fn == Path('freqtrade/hello/world/ETH_BTC-trades.json')


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
    data, start_ts = _load_cached_data_for_updating(datadir, 'UNITTEST/BTC', '1m', timerange)
    assert data == []
    assert start_ts == test_data[0][0] - 1000

    # same with 'line' timeframe
    num_lines = (test_data[-1][0] - test_data[1][0]) / 1000 / 60 + 120
    data, start_ts = _load_cached_data_for_updating(datadir, 'UNITTEST/BTC', '1m',
                                                    TimeRange(None, 'line', 0, -num_lines))
    assert data == []
    assert start_ts < test_data[0][0] - 1

    # timeframe starts in the center of the cached data
    # should return the chached data w/o the last item
    timerange = TimeRange('date', None, test_data[0][0] / 1000 + 1, 0)
    data, start_ts = _load_cached_data_for_updating(datadir, 'UNITTEST/BTC', '1m', timerange)
    assert data == test_data[:-1]
    assert test_data[-2][0] < start_ts < test_data[-1][0]

    # same with 'line' timeframe
    num_lines = (test_data[-1][0] - test_data[1][0]) / 1000 / 60 + 30
    timerange = TimeRange(None, 'line', 0, -num_lines)
    data, start_ts = _load_cached_data_for_updating(datadir, 'UNITTEST/BTC', '1m', timerange)
    assert data == test_data[:-1]
    assert test_data[-2][0] < start_ts < test_data[-1][0]

    # timeframe starts after the chached data
    # should return the chached data w/o the last item
    timerange = TimeRange('date', None, test_data[-1][0] / 1000 + 1, 0)
    data, start_ts = _load_cached_data_for_updating(datadir, 'UNITTEST/BTC', '1m', timerange)
    assert data == test_data[:-1]
    assert test_data[-2][0] < start_ts < test_data[-1][0]

    # Try loading last 30 lines.
    # Not supported by _load_cached_data_for_updating, we always need to get the full data.
    num_lines = 30
    timerange = TimeRange(None, 'line', 0, -num_lines)
    data, start_ts = _load_cached_data_for_updating(datadir, 'UNITTEST/BTC', '1m', timerange)
    assert data == test_data[:-1]
    assert test_data[-2][0] < start_ts < test_data[-1][0]

    # no timeframe is set
    # should return the chached data w/o the last item
    num_lines = 30
    timerange = TimeRange(None, 'line', 0, -num_lines)
    data, start_ts = _load_cached_data_for_updating(datadir, 'UNITTEST/BTC', '1m', timerange)
    assert data == test_data[:-1]
    assert test_data[-2][0] < start_ts < test_data[-1][0]

    # no datafile exist
    # should return timestamp start time
    timerange = TimeRange('date', None, now_ts - 10000, 0)
    data, start_ts = _load_cached_data_for_updating(datadir, 'NONEXIST/BTC', '1m', timerange)
    assert data == []
    assert start_ts == (now_ts - 10000) * 1000

    # same with 'line' timeframe
    num_lines = 30
    timerange = TimeRange(None, 'line', 0, -num_lines)
    data, start_ts = _load_cached_data_for_updating(datadir, 'NONEXIST/BTC', '1m', timerange)
    assert data == []
    assert start_ts == (now_ts - num_lines * 60) * 1000

    # no datafile exist, no timeframe is set
    # should return an empty array and None
    data, start_ts = _load_cached_data_for_updating(datadir, 'NONEXIST/BTC', '1m', None)
    assert data == []
    assert start_ts is None


def test_download_pair_history(ticker_history_list, mocker, default_conf, testdatadir) -> None:
    mocker.patch('freqtrade.exchange.Exchange.get_historic_ohlcv', return_value=ticker_history_list)
    exchange = get_patched_exchange(mocker, default_conf)
    file1_1 = testdatadir / 'MEME_BTC-1m.json'
    file1_5 = testdatadir / 'MEME_BTC-5m.json'
    file2_1 = testdatadir / 'CFI_BTC-1m.json'
    file2_5 = testdatadir / 'CFI_BTC-5m.json'

    _backup_file(file1_1)
    _backup_file(file1_5)
    _backup_file(file2_1)
    _backup_file(file2_5)

    assert not file1_1.is_file()
    assert not file2_1.is_file()

    assert download_pair_history(datadir=testdatadir, exchange=exchange,
                                 pair='MEME/BTC',
                                 ticker_interval='1m')
    assert download_pair_history(datadir=testdatadir, exchange=exchange,
                                 pair='CFI/BTC',
                                 ticker_interval='1m')
    assert not exchange._pairs_last_refresh_time
    assert file1_1.is_file()
    assert file2_1.is_file()

    # clean files freshly downloaded
    _clean_test_file(file1_1)
    _clean_test_file(file2_1)

    assert not file1_5.is_file()
    assert not file2_5.is_file()

    assert download_pair_history(datadir=testdatadir, exchange=exchange,
                                 pair='MEME/BTC',
                                 ticker_interval='5m')
    assert download_pair_history(datadir=testdatadir, exchange=exchange,
                                 pair='CFI/BTC',
                                 ticker_interval='5m')
    assert not exchange._pairs_last_refresh_time
    assert file1_5.is_file()
    assert file2_5.is_file()

    # clean files freshly downloaded
    _clean_test_file(file1_5)
    _clean_test_file(file2_5)


def test_download_pair_history2(mocker, default_conf, testdatadir) -> None:
    tick = [
        [1509836520000, 0.00162008, 0.00162008, 0.00162008, 0.00162008, 108.14853839],
        [1509836580000, 0.00161, 0.00161, 0.00161, 0.00161, 82.390199]
    ]
    json_dump_mock = mocker.patch('freqtrade.misc.file_dump_json', return_value=None)
    mocker.patch('freqtrade.exchange.Exchange.get_historic_ohlcv', return_value=tick)
    exchange = get_patched_exchange(mocker, default_conf)
    download_pair_history(testdatadir, exchange, pair="UNITTEST/BTC", ticker_interval='1m')
    download_pair_history(testdatadir, exchange, pair="UNITTEST/BTC", ticker_interval='3m')
    assert json_dump_mock.call_count == 2


def test_download_backtesting_data_exception(ticker_history, mocker, caplog,
                                             default_conf, testdatadir) -> None:
    mocker.patch('freqtrade.exchange.Exchange.get_historic_ohlcv',
                 side_effect=Exception('File Error'))

    exchange = get_patched_exchange(mocker, default_conf)

    file1_1 = testdatadir / 'MEME_BTC-1m.json'
    file1_5 = testdatadir / 'MEME_BTC-5m.json'
    _backup_file(file1_1)
    _backup_file(file1_5)

    assert not download_pair_history(datadir=testdatadir, exchange=exchange,
                                     pair='MEME/BTC',
                                     ticker_interval='1m')
    # clean files freshly downloaded
    _clean_test_file(file1_1)
    _clean_test_file(file1_5)
    assert log_has(
        'Failed to download history data for pair: "MEME/BTC", interval: 1m. '
        'Error: File Error', caplog
    )


def test_load_tickerdata_file(testdatadir) -> None:
    # 7 does not exist in either format.
    assert not load_tickerdata_file(testdatadir, 'UNITTEST/BTC', '7m')
    # 1 exists only as a .json
    tickerdata = load_tickerdata_file(testdatadir, 'UNITTEST/BTC', '1m')
    assert _BTC_UNITTEST_LENGTH == len(tickerdata)
    # 8 .json is empty and will fail if it's loaded. .json.gz is a copy of 1.json
    tickerdata = load_tickerdata_file(testdatadir, 'UNITTEST/BTC', '8m')
    assert _BTC_UNITTEST_LENGTH == len(tickerdata)


def test_load_partial_missing(testdatadir, caplog) -> None:
    # Make sure we start fresh - test missing data at start
    start = arrow.get('2018-01-01T00:00:00')
    end = arrow.get('2018-01-11T00:00:00')
    tickerdata = history.load_data(testdatadir, '5m', ['UNITTEST/BTC'],
                                   timerange=TimeRange('date', 'date',
                                                       start.timestamp, end.timestamp))
    # timedifference in 5 minutes
    td = ((end - start).total_seconds() // 60 // 5) + 1
    assert td != len(tickerdata['UNITTEST/BTC'])
    start_real = tickerdata['UNITTEST/BTC'].iloc[0, 0]
    assert log_has(f'Missing data at start for pair '
                   f'UNITTEST/BTC, data starts at {start_real.strftime("%Y-%m-%d %H:%M:%S")}',
                   caplog)
    # Make sure we start fresh - test missing data at end
    caplog.clear()
    start = arrow.get('2018-01-10T00:00:00')
    end = arrow.get('2018-02-20T00:00:00')
    tickerdata = history.load_data(datadir=testdatadir, ticker_interval='5m',
                                   pairs=['UNITTEST/BTC'],
                                   timerange=TimeRange('date', 'date',
                                                       start.timestamp, end.timestamp))
    # timedifference in 5 minutes
    td = ((end - start).total_seconds() // 60 // 5) + 1
    assert td != len(tickerdata['UNITTEST/BTC'])
    # Shift endtime with +5 - as last candle is dropped (partial candle)
    end_real = arrow.get(tickerdata['UNITTEST/BTC'].iloc[-1, 0]).shift(minutes=5)
    assert log_has(f'Missing data at end for pair '
                   f'UNITTEST/BTC, data ends at {end_real.strftime("%Y-%m-%d %H:%M:%S")}',
                   caplog)


def test_init(default_conf, mocker) -> None:
    exchange = get_patched_exchange(mocker, default_conf)
    assert {} == history.load_data(
        datadir='',
        exchange=exchange,
        pairs=[],
        refresh_pairs=True,
        ticker_interval=default_conf['ticker_interval']
    )


def test_trim_tickerlist(testdatadir) -> None:
    file = testdatadir / 'UNITTEST_BTC-1m.json'
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


def test_file_dump_json_tofile(testdatadir) -> None:
    file = testdatadir / 'test_{id}.json'.format(id=str(uuid.uuid4()))
    data = {'bar': 'foo'}

    # check the file we will create does not exist
    assert not file.is_file()

    # Create the Json file
    file_dump_json(file, data)

    # Check the file was create
    assert file.is_file()

    # Open the Json file created and test the data is in it
    with file.open() as data_file:
        json_from_file = json.load(data_file)

    assert 'bar' in json_from_file
    assert json_from_file['bar'] == 'foo'

    # Remove the file
    _clean_test_file(file)


def test_get_timeframe(default_conf, mocker, testdatadir) -> None:
    patch_exchange(mocker)
    strategy = DefaultStrategy(default_conf)

    data = strategy.tickerdata_to_dataframe(
        history.load_data(
            datadir=testdatadir,
            ticker_interval='1m',
            pairs=['UNITTEST/BTC']
        )
    )
    min_date, max_date = history.get_timeframe(data)
    assert min_date.isoformat() == '2017-11-04T23:02:00+00:00'
    assert max_date.isoformat() == '2017-11-14T22:58:00+00:00'


def test_validate_backtest_data_warn(default_conf, mocker, caplog, testdatadir) -> None:
    patch_exchange(mocker)
    strategy = DefaultStrategy(default_conf)

    data = strategy.tickerdata_to_dataframe(
        history.load_data(
            datadir=testdatadir,
            ticker_interval='1m',
            pairs=['UNITTEST/BTC'],
            fill_up_missing=False
        )
    )
    min_date, max_date = history.get_timeframe(data)
    caplog.clear()
    assert history.validate_backtest_data(data['UNITTEST/BTC'], 'UNITTEST/BTC',
                                          min_date, max_date, timeframe_to_minutes('1m'))
    assert len(caplog.record_tuples) == 1
    assert log_has(
        "UNITTEST/BTC has missing frames: expected 14396, got 13680, that's 716 missing values",
        caplog)


def test_validate_backtest_data(default_conf, mocker, caplog, testdatadir) -> None:
    patch_exchange(mocker)
    strategy = DefaultStrategy(default_conf)

    timerange = TimeRange('index', 'index', 200, 250)
    data = strategy.tickerdata_to_dataframe(
        history.load_data(
            datadir=testdatadir,
            ticker_interval='5m',
            pairs=['UNITTEST/BTC'],
            timerange=timerange
        )
    )

    min_date, max_date = history.get_timeframe(data)
    caplog.clear()
    assert not history.validate_backtest_data(data['UNITTEST/BTC'], 'UNITTEST/BTC',
                                              min_date, max_date, timeframe_to_minutes('5m'))
    assert len(caplog.record_tuples) == 0


def test_refresh_backtest_ohlcv_data(mocker, default_conf, markets, caplog, testdatadir):
    dl_mock = mocker.patch('freqtrade.data.history.download_pair_history', MagicMock())
    mocker.patch(
        'freqtrade.exchange.Exchange.markets', PropertyMock(return_value=markets)
    )
    mocker.patch.object(Path, "exists", MagicMock(return_value=True))
    mocker.patch.object(Path, "unlink", MagicMock())

    ex = get_patched_exchange(mocker, default_conf)
    timerange = TimeRange.parse_timerange("20190101-20190102")
    refresh_backtest_ohlcv_data(exchange=ex, pairs=["ETH/BTC", "XRP/BTC"],
                                timeframes=["1m", "5m"], dl_path=testdatadir,
                                timerange=timerange, erase=True
                                )

    assert dl_mock.call_count == 4
    assert dl_mock.call_args[1]['timerange'].starttype == 'date'

    assert log_has("Downloading pair ETH/BTC, interval 1m.", caplog)


def test_download_data_no_markets(mocker, default_conf, caplog, testdatadir):
    dl_mock = mocker.patch('freqtrade.data.history.download_pair_history', MagicMock())
    mocker.patch(
        'freqtrade.exchange.Exchange.markets', PropertyMock(return_value={})
    )
    ex = get_patched_exchange(mocker, default_conf)
    timerange = TimeRange.parse_timerange("20190101-20190102")
    unav_pairs = refresh_backtest_ohlcv_data(exchange=ex, pairs=["ETH/BTC", "XRP/BTC"],
                                             timeframes=["1m", "5m"],
                                             dl_path=testdatadir,
                                             timerange=timerange, erase=False
                                             )

    assert dl_mock.call_count == 0
    assert "ETH/BTC" in unav_pairs
    assert "XRP/BTC" in unav_pairs
    assert log_has("Skipping pair ETH/BTC...", caplog)
