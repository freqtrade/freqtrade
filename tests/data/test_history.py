# pragma pylint: disable=missing-docstring, protected-access, C0103

import json
import uuid
from pathlib import Path
from shutil import copyfile
from unittest.mock import MagicMock, PropertyMock

import arrow
import pytest
from pandas import DataFrame
from pandas.testing import assert_frame_equal

from freqtrade.configuration import TimeRange
from freqtrade.data.converter import parse_ticker_dataframe
from freqtrade.data.datahandlers import get_datahandler, get_datahandlerclass
from freqtrade.data.datahandlers.idatahandler import IDataHandler
from freqtrade.data.datahandlers.jsondatahandler import (JsonDataHandler,
                                                         JsonGzDataHandler)
from freqtrade.data.history import (_download_pair_history,
                                    _download_trades_history,
                                    _load_cached_data_for_updating,
                                    convert_trades_to_ohlcv, get_timerange,
                                    load_data, load_pair_history,
                                    refresh_backtest_ohlcv_data,
                                    refresh_backtest_trades_data, refresh_data,
                                    validate_backtest_data)
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
    ld = load_pair_history(pair='UNITTEST/BTC', timeframe='30m', datadir=testdatadir)
    assert isinstance(ld, DataFrame)
    assert not log_has(
        'Download history data for pair: "UNITTEST/BTC", timeframe: 30m '
        'and store in None.', caplog
    )


def test_load_data_7min_ticker(mocker, caplog, default_conf, testdatadir) -> None:
    ld = load_pair_history(pair='UNITTEST/BTC', timeframe='7m', datadir=testdatadir)
    assert isinstance(ld, DataFrame)
    assert ld.empty
    assert log_has(
        'No history data for pair: "UNITTEST/BTC", timeframe: 7m. '
        'Use `freqtrade download-data` to download the data', caplog
    )


def test_load_data_1min_ticker(ticker_history, mocker, caplog, testdatadir) -> None:
    mocker.patch('freqtrade.exchange.Exchange.get_historic_ohlcv', return_value=ticker_history)
    file = testdatadir / 'UNITTEST_BTC-1m.json'
    _backup_file(file, copy_file=True)
    load_data(datadir=testdatadir, timeframe='1m', pairs=['UNITTEST/BTC'])
    assert file.is_file()
    assert not log_has(
        'Download history data for pair: "UNITTEST/BTC", interval: 1m '
        'and store in None.', caplog
    )
    _clean_test_file(file)


def test_load_data_startup_candles(mocker, caplog, default_conf, testdatadir) -> None:
    ltfmock = mocker.patch(
        'freqtrade.data.datahandlers.jsondatahandler.JsonDataHandler._ohlcv_load',
        MagicMock(return_value=DataFrame()))
    timerange = TimeRange('date', None, 1510639620, 0)
    load_pair_history(pair='UNITTEST/BTC', timeframe='1m',
                      datadir=testdatadir, timerange=timerange,
                      startup_candles=20,)

    assert ltfmock.call_count == 1
    assert ltfmock.call_args_list[0][1]['timerange'] != timerange
    # startts is 20 minutes earlier
    assert ltfmock.call_args_list[0][1]['timerange'].startts == timerange.startts - 20 * 60


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
    load_pair_history(datadir=testdatadir, timeframe='1m', pair='MEME/BTC')
    assert not file.is_file()
    assert log_has(
        'No history data for pair: "MEME/BTC", timeframe: 1m. '
        'Use `freqtrade download-data` to download the data', caplog
    )

    # download a new pair if refresh_pairs is set
    refresh_data(datadir=testdatadir, timeframe='1m', pairs=['MEME/BTC'],
                 exchange=exchange)
    load_pair_history(datadir=testdatadir, timeframe='1m', pair='MEME/BTC')
    assert file.is_file()
    assert log_has_re(
        'Download history data for pair: "MEME/BTC", timeframe: 1m '
        'and store in .*', caplog
    )
    _clean_test_file(file)


def test_testdata_path(testdatadir) -> None:
    assert str(Path('tests') / 'testdata') in str(testdatadir)


def test_json_pair_data_filename():
    fn = JsonDataHandler._pair_data_filename(Path('freqtrade/hello/world'), 'ETH/BTC', '5m')
    assert isinstance(fn, Path)
    assert fn == Path('freqtrade/hello/world/ETH_BTC-5m.json')
    fn = JsonGzDataHandler._pair_data_filename(Path('freqtrade/hello/world'), 'ETH/BTC', '5m')
    assert isinstance(fn, Path)
    assert fn == Path('freqtrade/hello/world/ETH_BTC-5m.json.gz')


def test_json_pair_trades_filename():
    fn = JsonDataHandler._pair_trades_filename(Path('freqtrade/hello/world'), 'ETH/BTC')
    assert isinstance(fn, Path)
    assert fn == Path('freqtrade/hello/world/ETH_BTC-trades.json')

    fn = JsonGzDataHandler._pair_trades_filename(Path('freqtrade/hello/world'), 'ETH/BTC')
    assert isinstance(fn, Path)
    assert fn == Path('freqtrade/hello/world/ETH_BTC-trades.json.gz')


def test_load_cached_data_for_updating(mocker, testdatadir) -> None:

    data_handler = get_datahandler(testdatadir, 'json')

    test_data = None
    test_filename = testdatadir.joinpath('UNITTEST_BTC-1m.json')
    with open(test_filename, "rt") as file:
        test_data = json.load(file)

    test_data_df = parse_ticker_dataframe(test_data, '1m', 'UNITTEST/BTC',
                                          fill_missing=False, drop_incomplete=False)
    # now = last cached item + 1 hour
    now_ts = test_data[-1][0] / 1000 + 60 * 60
    mocker.patch('arrow.utcnow', return_value=arrow.get(now_ts))

    # timeframe starts earlier than the cached data
    # should fully update data
    timerange = TimeRange('date', None, test_data[0][0] / 1000 - 1, 0)
    data, start_ts = _load_cached_data_for_updating('UNITTEST/BTC', '1m', timerange, data_handler)
    assert data.empty
    assert start_ts == test_data[0][0] - 1000

    # timeframe starts in the center of the cached data
    # should return the chached data w/o the last item
    timerange = TimeRange('date', None, test_data[0][0] / 1000 + 1, 0)
    data, start_ts = _load_cached_data_for_updating('UNITTEST/BTC', '1m', timerange, data_handler)

    assert_frame_equal(data, test_data_df.iloc[:-1])
    assert test_data[-2][0] <= start_ts < test_data[-1][0]

    # timeframe starts after the chached data
    # should return the chached data w/o the last item
    timerange = TimeRange('date', None, test_data[-1][0] / 1000 + 100, 0)
    data, start_ts = _load_cached_data_for_updating('UNITTEST/BTC', '1m', timerange, data_handler)
    assert_frame_equal(data, test_data_df.iloc[:-1])
    assert test_data[-2][0] <= start_ts < test_data[-1][0]

    # no datafile exist
    # should return timestamp start time
    timerange = TimeRange('date', None, now_ts - 10000, 0)
    data, start_ts = _load_cached_data_for_updating('NONEXIST/BTC', '1m', timerange, data_handler)
    assert data.empty
    assert start_ts == (now_ts - 10000) * 1000

    # no datafile exist, no timeframe is set
    # should return an empty array and None
    data, start_ts = _load_cached_data_for_updating('NONEXIST/BTC', '1m', None, data_handler)
    assert data.empty
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

    assert _download_pair_history(datadir=testdatadir, exchange=exchange,
                                  pair='MEME/BTC',
                                  timeframe='1m')
    assert _download_pair_history(datadir=testdatadir, exchange=exchange,
                                  pair='CFI/BTC',
                                  timeframe='1m')
    assert not exchange._pairs_last_refresh_time
    assert file1_1.is_file()
    assert file2_1.is_file()

    # clean files freshly downloaded
    _clean_test_file(file1_1)
    _clean_test_file(file2_1)

    assert not file1_5.is_file()
    assert not file2_5.is_file()

    assert _download_pair_history(datadir=testdatadir, exchange=exchange,
                                  pair='MEME/BTC',
                                  timeframe='5m')
    assert _download_pair_history(datadir=testdatadir, exchange=exchange,
                                  pair='CFI/BTC',
                                  timeframe='5m')
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
    json_dump_mock = mocker.patch(
        'freqtrade.data.datahandlers.jsondatahandler.JsonDataHandler.ohlcv_store',
        return_value=None)
    mocker.patch('freqtrade.exchange.Exchange.get_historic_ohlcv', return_value=tick)
    exchange = get_patched_exchange(mocker, default_conf)
    _download_pair_history(testdatadir, exchange, pair="UNITTEST/BTC", timeframe='1m')
    _download_pair_history(testdatadir, exchange, pair="UNITTEST/BTC", timeframe='3m')
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

    assert not _download_pair_history(datadir=testdatadir, exchange=exchange,
                                      pair='MEME/BTC',
                                      timeframe='1m')
    # clean files freshly downloaded
    _clean_test_file(file1_1)
    _clean_test_file(file1_5)
    assert log_has(
        'Failed to download history data for pair: "MEME/BTC", timeframe: 1m. '
        'Error: File Error', caplog
    )


def test_load_partial_missing(testdatadir, caplog) -> None:
    # Make sure we start fresh - test missing data at start
    start = arrow.get('2018-01-01T00:00:00')
    end = arrow.get('2018-01-11T00:00:00')
    tickerdata = load_data(testdatadir, '5m', ['UNITTEST/BTC'], startup_candles=20,
                           timerange=TimeRange('date', 'date', start.timestamp, end.timestamp))
    assert log_has(
        'Using indicator startup period: 20 ...', caplog
    )
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
    tickerdata = load_data(datadir=testdatadir, timeframe='5m', pairs=['UNITTEST/BTC'],
                           timerange=TimeRange('date', 'date', start.timestamp, end.timestamp))
    # timedifference in 5 minutes
    td = ((end - start).total_seconds() // 60 // 5) + 1
    assert td != len(tickerdata['UNITTEST/BTC'])

    # Shift endtime with +5 - as last candle is dropped (partial candle)
    end_real = arrow.get(tickerdata['UNITTEST/BTC'].iloc[-1, 0]).shift(minutes=5)
    assert log_has(f'Missing data at end for pair '
                   f'UNITTEST/BTC, data ends at {end_real.strftime("%Y-%m-%d %H:%M:%S")}',
                   caplog)


def test_init(default_conf, mocker) -> None:
    assert {} == load_data(
        datadir='',
        pairs=[],
        timeframe=default_conf['ticker_interval']
    )


def test_init_with_refresh(default_conf, mocker) -> None:
    exchange = get_patched_exchange(mocker, default_conf)
    refresh_data(
        datadir='',
        pairs=[],
        timeframe=default_conf['ticker_interval'],
        exchange=exchange
    )
    assert {} == load_data(
        datadir='',
        pairs=[],
        timeframe=default_conf['ticker_interval']
    )


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


def test_get_timerange(default_conf, mocker, testdatadir) -> None:
    patch_exchange(mocker)
    strategy = DefaultStrategy(default_conf)

    data = strategy.tickerdata_to_dataframe(
        load_data(
            datadir=testdatadir,
            timeframe='1m',
            pairs=['UNITTEST/BTC']
        )
    )
    min_date, max_date = get_timerange(data)
    assert min_date.isoformat() == '2017-11-04T23:02:00+00:00'
    assert max_date.isoformat() == '2017-11-14T22:58:00+00:00'


def test_validate_backtest_data_warn(default_conf, mocker, caplog, testdatadir) -> None:
    patch_exchange(mocker)
    strategy = DefaultStrategy(default_conf)

    data = strategy.tickerdata_to_dataframe(
        load_data(
            datadir=testdatadir,
            timeframe='1m',
            pairs=['UNITTEST/BTC'],
            fill_up_missing=False
        )
    )
    min_date, max_date = get_timerange(data)
    caplog.clear()
    assert validate_backtest_data(data['UNITTEST/BTC'], 'UNITTEST/BTC',
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
        load_data(
            datadir=testdatadir,
            timeframe='5m',
            pairs=['UNITTEST/BTC'],
            timerange=timerange
        )
    )

    min_date, max_date = get_timerange(data)
    caplog.clear()
    assert not validate_backtest_data(data['UNITTEST/BTC'], 'UNITTEST/BTC',
                                      min_date, max_date, timeframe_to_minutes('5m'))
    assert len(caplog.record_tuples) == 0


def test_refresh_backtest_ohlcv_data(mocker, default_conf, markets, caplog, testdatadir):
    dl_mock = mocker.patch('freqtrade.data.history._download_pair_history', MagicMock())
    mocker.patch(
        'freqtrade.exchange.Exchange.markets', PropertyMock(return_value=markets)
    )
    mocker.patch.object(Path, "exists", MagicMock(return_value=True))
    mocker.patch.object(Path, "unlink", MagicMock())

    ex = get_patched_exchange(mocker, default_conf)
    timerange = TimeRange.parse_timerange("20190101-20190102")
    refresh_backtest_ohlcv_data(exchange=ex, pairs=["ETH/BTC", "XRP/BTC"],
                                timeframes=["1m", "5m"], datadir=testdatadir,
                                timerange=timerange, erase=True
                                )

    assert dl_mock.call_count == 4
    assert dl_mock.call_args[1]['timerange'].starttype == 'date'

    assert log_has("Downloading pair ETH/BTC, interval 1m.", caplog)


def test_download_data_no_markets(mocker, default_conf, caplog, testdatadir):
    dl_mock = mocker.patch('freqtrade.data.history._download_pair_history', MagicMock())

    ex = get_patched_exchange(mocker, default_conf)
    mocker.patch(
        'freqtrade.exchange.Exchange.markets', PropertyMock(return_value={})
    )
    timerange = TimeRange.parse_timerange("20190101-20190102")
    unav_pairs = refresh_backtest_ohlcv_data(exchange=ex, pairs=["BTT/BTC", "LTC/USDT"],
                                             timeframes=["1m", "5m"],
                                             datadir=testdatadir,
                                             timerange=timerange, erase=False
                                             )

    assert dl_mock.call_count == 0
    assert "BTT/BTC" in unav_pairs
    assert "LTC/USDT" in unav_pairs
    assert log_has("Skipping pair BTT/BTC...", caplog)


def test_refresh_backtest_trades_data(mocker, default_conf, markets, caplog, testdatadir):
    dl_mock = mocker.patch('freqtrade.data.history._download_trades_history', MagicMock())
    mocker.patch(
        'freqtrade.exchange.Exchange.markets', PropertyMock(return_value=markets)
    )
    mocker.patch.object(Path, "exists", MagicMock(return_value=True))
    mocker.patch.object(Path, "unlink", MagicMock())

    ex = get_patched_exchange(mocker, default_conf)
    timerange = TimeRange.parse_timerange("20190101-20190102")
    unavailable_pairs = refresh_backtest_trades_data(exchange=ex,
                                                     pairs=["ETH/BTC", "XRP/BTC", "XRP/ETH"],
                                                     datadir=testdatadir,
                                                     timerange=timerange, erase=True
                                                     )

    assert dl_mock.call_count == 2
    assert dl_mock.call_args[1]['timerange'].starttype == 'date'

    assert log_has("Downloading trades for pair ETH/BTC.", caplog)
    assert unavailable_pairs == ["XRP/ETH"]
    assert log_has("Skipping pair XRP/ETH...", caplog)


def test_download_trades_history(trades_history, mocker, default_conf, testdatadir, caplog) -> None:

    ght_mock = MagicMock(side_effect=lambda pair, *args, **kwargs: (pair, trades_history))
    mocker.patch('freqtrade.exchange.Exchange.get_historic_trades',
                 ght_mock)
    exchange = get_patched_exchange(mocker, default_conf)
    file1 = testdatadir / 'ETH_BTC-trades.json.gz'
    data_handler = get_datahandler(testdatadir, data_format='jsongz')
    _backup_file(file1)

    assert not file1.is_file()

    assert _download_trades_history(data_handler=data_handler, exchange=exchange,
                                    pair='ETH/BTC')
    assert log_has("New Amount of trades: 5", caplog)
    assert file1.is_file()

    # clean files freshly downloaded
    _clean_test_file(file1)

    mocker.patch('freqtrade.exchange.Exchange.get_historic_trades',
                 MagicMock(side_effect=ValueError))

    assert not _download_trades_history(data_handler=data_handler, exchange=exchange,
                                        pair='ETH/BTC')
    assert log_has_re('Failed to download historic trades for pair: "ETH/BTC".*', caplog)


def test_convert_trades_to_ohlcv(mocker, default_conf, testdatadir, caplog):

    pair = 'XRP/ETH'
    file1 = testdatadir / 'XRP_ETH-1m.json'
    file5 = testdatadir / 'XRP_ETH-5m.json'
    # Compare downloaded dataset with converted dataset
    dfbak_1m = load_pair_history(datadir=testdatadir, timeframe="1m", pair=pair)
    dfbak_5m = load_pair_history(datadir=testdatadir, timeframe="5m", pair=pair)

    _backup_file(file1, copy_file=True)
    _backup_file(file5)

    tr = TimeRange.parse_timerange('20191011-20191012')

    convert_trades_to_ohlcv([pair], timeframes=['1m', '5m'],
                            datadir=testdatadir, timerange=tr, erase=True)

    assert log_has("Deleting existing data for pair XRP/ETH, interval 1m.", caplog)
    # Load new data
    df_1m = load_pair_history(datadir=testdatadir, timeframe="1m", pair=pair)
    df_5m = load_pair_history(datadir=testdatadir, timeframe="5m", pair=pair)

    assert df_1m.equals(dfbak_1m)
    assert df_5m.equals(dfbak_5m)

    _clean_test_file(file1)
    _clean_test_file(file5)


def test_jsondatahandler_ohlcv_get_pairs(testdatadir):
    pairs = JsonDataHandler.ohlcv_get_pairs(testdatadir, '5m')
    # Convert to set to avoid failures due to sorting
    assert set(pairs) == {'UNITTEST/BTC', 'XLM/BTC', 'ETH/BTC', 'TRX/BTC', 'LTC/BTC',
                          'XMR/BTC', 'ZEC/BTC', 'ADA/BTC', 'ETC/BTC', 'NXT/BTC',
                          'DASH/BTC', 'XRP/ETH'}

    pairs = JsonGzDataHandler.ohlcv_get_pairs(testdatadir, '8m')
    assert set(pairs) == {'UNITTEST/BTC'}


def test_jsondatahandler_trades_get_pairs(testdatadir):
    pairs = JsonGzDataHandler.trades_get_pairs(testdatadir)
    # Convert to set to avoid failures due to sorting
    assert set(pairs) == {'XRP/ETH'}


def test_jsondatahandler_ohlcv_purge(mocker, testdatadir):
    mocker.patch.object(Path, "exists", MagicMock(return_value=False))
    mocker.patch.object(Path, "unlink", MagicMock())
    dh = JsonGzDataHandler(testdatadir)
    assert not dh.ohlcv_purge('UNITTEST/NONEXIST', '5m')

    mocker.patch.object(Path, "exists", MagicMock(return_value=True))
    assert dh.ohlcv_purge('UNITTEST/NONEXIST', '5m')


def test_jsondatahandler_trades_purge(mocker, testdatadir):
    mocker.patch.object(Path, "exists", MagicMock(return_value=False))
    mocker.patch.object(Path, "unlink", MagicMock())
    dh = JsonGzDataHandler(testdatadir)
    assert not dh.trades_purge('UNITTEST/NONEXIST')

    mocker.patch.object(Path, "exists", MagicMock(return_value=True))
    assert dh.trades_purge('UNITTEST/NONEXIST')


def test_jsondatahandler_ohlcv_append(testdatadir):
    dh = JsonGzDataHandler(testdatadir)
    with pytest.raises(NotImplementedError):
        dh.ohlcv_append('UNITTEST/ETH', '5m', DataFrame())


def test_jsondatahandler_trades_append(testdatadir):
    dh = JsonGzDataHandler(testdatadir)
    with pytest.raises(NotImplementedError):
        dh.trades_append('UNITTEST/ETH', [])


def test_gethandlerclass():
    cl = get_datahandlerclass('json')
    assert cl == JsonDataHandler
    assert issubclass(cl, IDataHandler)
    cl = get_datahandlerclass('jsongz')
    assert cl == JsonGzDataHandler
    assert issubclass(cl, IDataHandler)
    assert issubclass(cl, JsonDataHandler)
    with pytest.raises(ValueError, match=r"No datahandler for .*"):
        get_datahandlerclass('DeadBeef')


def test_get_datahandler(testdatadir):
    dh = get_datahandler(testdatadir, 'json')
    assert type(dh) == JsonDataHandler
    dh = get_datahandler(testdatadir, 'jsongz')
    assert type(dh) == JsonGzDataHandler
    dh1 = get_datahandler(testdatadir, 'jsongz', dh)
    assert id(dh1) == id(dh)
