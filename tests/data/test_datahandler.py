# pragma pylint: disable=missing-docstring, protected-access, C0103

import re
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from pandas import DataFrame

from freqtrade.configuration import TimeRange
from freqtrade.constants import AVAILABLE_DATAHANDLERS
from freqtrade.data.history.featherdatahandler import FeatherDataHandler
from freqtrade.data.history.hdf5datahandler import HDF5DataHandler
from freqtrade.data.history.idatahandler import IDataHandler, get_datahandler, get_datahandlerclass
from freqtrade.data.history.jsondatahandler import JsonDataHandler, JsonGzDataHandler
from freqtrade.data.history.parquetdatahandler import ParquetDataHandler
from freqtrade.enums import CandleType, TradingMode
from tests.conftest import log_has, log_has_re


def test_datahandler_ohlcv_get_pairs(testdatadir):
    pairs = JsonDataHandler.ohlcv_get_pairs(testdatadir, '5m', candle_type=CandleType.SPOT)
    # Convert to set to avoid failures due to sorting
    assert set(pairs) == {'UNITTEST/BTC', 'XLM/BTC', 'ETH/BTC', 'TRX/BTC', 'LTC/BTC',
                          'XMR/BTC', 'ZEC/BTC', 'ADA/BTC', 'ETC/BTC', 'NXT/BTC',
                          'DASH/BTC', 'XRP/ETH'}

    pairs = JsonGzDataHandler.ohlcv_get_pairs(testdatadir, '8m', candle_type=CandleType.SPOT)
    assert set(pairs) == {'UNITTEST/BTC'}

    pairs = HDF5DataHandler.ohlcv_get_pairs(testdatadir, '5m', candle_type=CandleType.SPOT)
    assert set(pairs) == {'UNITTEST/BTC'}

    pairs = JsonDataHandler.ohlcv_get_pairs(testdatadir, '1h', candle_type=CandleType.MARK)
    assert set(pairs) == {'UNITTEST/USDT:USDT', 'XRP/USDT:USDT'}

    pairs = JsonGzDataHandler.ohlcv_get_pairs(testdatadir, '1h', candle_type=CandleType.FUTURES)
    assert set(pairs) == {'XRP/USDT:USDT'}

    pairs = HDF5DataHandler.ohlcv_get_pairs(testdatadir, '1h', candle_type=CandleType.MARK)
    assert set(pairs) == {'UNITTEST/USDT:USDT'}


@pytest.mark.parametrize('filename,pair,timeframe,candletype', [
    ('XMR_BTC-5m.json', 'XMR_BTC', '5m', ''),
    ('XMR_USDT-1h.h5', 'XMR_USDT', '1h', ''),
    ('BTC-PERP-1h.h5', 'BTC-PERP', '1h', ''),
    ('BTC_USDT-2h.jsongz', 'BTC_USDT', '2h', ''),
    ('BTC_USDT-2h-mark.jsongz', 'BTC_USDT', '2h', 'mark'),
    ('XMR_USDT-1h-mark.h5', 'XMR_USDT', '1h', 'mark'),
    ('XMR_USDT-1h-random.h5', 'XMR_USDT', '1h', 'random'),
    ('BTC-PERP-1h-index.h5', 'BTC-PERP', '1h', 'index'),
    ('XMR_USDT_USDT-1h-mark.h5', 'XMR_USDT_USDT', '1h', 'mark'),
])
def test_datahandler_ohlcv_regex(filename, pair, timeframe, candletype):
    regex = JsonDataHandler._OHLCV_REGEX

    match = re.search(regex, filename)
    assert len(match.groups()) > 1
    assert match[1] == pair
    assert match[2] == timeframe
    assert match[3] == candletype


@pytest.mark.parametrize('input,expected', [
    ('XMR_USDT', 'XMR/USDT'),
    ('BTC_USDT', 'BTC/USDT'),
    ('USDT_BUSD', 'USDT/BUSD'),
    ('BTC_USDT_USDT', 'BTC/USDT:USDT'),  # Futures
    ('XRP_USDT_USDT', 'XRP/USDT:USDT'),  # futures
    ('BTC-PERP', 'BTC-PERP'),
    ('BTC-PERP_USDT', 'BTC-PERP:USDT'),
    ('UNITTEST_USDT', 'UNITTEST/USDT'),
])
def test_rebuild_pair_from_filename(input, expected):

    assert IDataHandler.rebuild_pair_from_filename(input) == expected


def test_datahandler_ohlcv_get_available_data(testdatadir):
    paircombs = JsonDataHandler.ohlcv_get_available_data(testdatadir, TradingMode.SPOT)
    # Convert to set to avoid failures due to sorting
    assert set(paircombs) == {
        ('UNITTEST/BTC', '5m', CandleType.SPOT),
        ('ETH/BTC', '5m', CandleType.SPOT),
        ('XLM/BTC', '5m', CandleType.SPOT),
        ('TRX/BTC', '5m', CandleType.SPOT),
        ('LTC/BTC', '5m', CandleType.SPOT),
        ('XMR/BTC', '5m', CandleType.SPOT),
        ('ZEC/BTC', '5m', CandleType.SPOT),
        ('UNITTEST/BTC', '1m', CandleType.SPOT),
        ('ADA/BTC', '5m', CandleType.SPOT),
        ('ETC/BTC', '5m', CandleType.SPOT),
        ('NXT/BTC', '5m', CandleType.SPOT),
        ('DASH/BTC', '5m', CandleType.SPOT),
        ('XRP/ETH', '1m', CandleType.SPOT),
        ('XRP/ETH', '5m', CandleType.SPOT),
        ('UNITTEST/BTC', '30m', CandleType.SPOT),
        ('UNITTEST/BTC', '8m', CandleType.SPOT),
        ('NOPAIR/XXX', '4m', CandleType.SPOT),
    }

    paircombs = JsonDataHandler.ohlcv_get_available_data(testdatadir, TradingMode.FUTURES)
    # Convert to set to avoid failures due to sorting
    assert set(paircombs) == {
        ('UNITTEST/USDT:USDT', '1h', 'mark'),
        ('XRP/USDT:USDT', '5m', 'futures'),
        ('XRP/USDT:USDT', '1h', 'futures'),
        ('XRP/USDT:USDT', '1h', 'mark'),
        ('XRP/USDT:USDT', '8h', 'mark'),
        ('XRP/USDT:USDT', '8h', 'funding_rate'),
    }

    paircombs = JsonGzDataHandler.ohlcv_get_available_data(testdatadir, TradingMode.SPOT)
    assert set(paircombs) == {('UNITTEST/BTC', '8m', CandleType.SPOT)}
    paircombs = HDF5DataHandler.ohlcv_get_available_data(testdatadir, TradingMode.SPOT)
    assert set(paircombs) == {('UNITTEST/BTC', '5m', CandleType.SPOT)}


def test_jsondatahandler_trades_get_pairs(testdatadir):
    pairs = JsonGzDataHandler.trades_get_pairs(testdatadir)
    # Convert to set to avoid failures due to sorting
    assert set(pairs) == {'XRP/ETH', 'XRP/OLD'}


def test_jsondatahandler_ohlcv_purge(mocker, testdatadir):
    mocker.patch.object(Path, "exists", MagicMock(return_value=False))
    unlinkmock = mocker.patch.object(Path, "unlink", MagicMock())
    dh = JsonGzDataHandler(testdatadir)
    assert not dh.ohlcv_purge('UNITTEST/NONEXIST', '5m', '')
    assert not dh.ohlcv_purge('UNITTEST/NONEXIST', '5m', candle_type='mark')
    assert unlinkmock.call_count == 0

    mocker.patch.object(Path, "exists", MagicMock(return_value=True))
    assert dh.ohlcv_purge('UNITTEST/NONEXIST', '5m', '')
    assert dh.ohlcv_purge('UNITTEST/NONEXIST', '5m', candle_type='mark')
    assert unlinkmock.call_count == 2


def test_jsondatahandler_ohlcv_load(testdatadir, caplog):
    dh = JsonDataHandler(testdatadir)
    df = dh.ohlcv_load('XRP/ETH', '5m', 'spot')
    assert len(df) == 712

    df_mark = dh.ohlcv_load('UNITTEST/USDT:USDT', '1h', candle_type="mark")
    assert len(df_mark) == 100

    df_no_mark = dh.ohlcv_load('UNITTEST/USDT', '1h', 'spot')
    assert len(df_no_mark) == 0

    # Failure case (empty array)
    df1 = dh.ohlcv_load('NOPAIR/XXX', '4m', 'spot')
    assert len(df1) == 0
    assert log_has("Could not load data for NOPAIR/XXX.", caplog)
    assert df.columns.equals(df1.columns)


def test_datahandler_ohlcv_data_min_max(testdatadir):
    dh = JsonDataHandler(testdatadir)
    min_max = dh.ohlcv_data_min_max('UNITTEST/BTC', '5m', 'spot')
    assert len(min_max) == 2

    # Empty pair
    min_max = dh.ohlcv_data_min_max('UNITTEST/BTC', '8m', 'spot')
    assert len(min_max) == 2
    assert min_max[0] == datetime.fromtimestamp(0, tz=timezone.utc)
    assert min_max[0] == min_max[1]
    # Empty pair2
    min_max = dh.ohlcv_data_min_max('NOPAIR/XXX', '4m', 'spot')
    assert len(min_max) == 2
    assert min_max[0] == datetime.fromtimestamp(0, tz=timezone.utc)
    assert min_max[0] == min_max[1]


def test_datahandler__check_empty_df(testdatadir, caplog):
    dh = JsonDataHandler(testdatadir)
    expected_text = r"Price jump in UNITTEST/USDT, 1h, spot between"
    df = DataFrame([
        [
            1511686200000,  # 8:50:00
            8.794,  # open
            8.948,  # high
            8.794,  # low
            8.88,  # close
            2255,  # volume (in quote currency)
        ],
        [
            1511686500000,  # 8:55:00
            8.88,
            8.942,
            8.88,
            8.893,
            9911,
        ],
        [
            1511687100000,  # 9:05:00
            8.891,
            8.893,
            8.875,
            8.877,
            2251
        ],
        [
            1511687400000,  # 9:10:00
            8.877,
            8.883,
            8.895,
            8.817,
            123551
        ]
    ], columns=['date', 'open', 'high', 'low', 'close', 'volume'])

    dh._check_empty_df(df, 'UNITTEST/USDT', '1h', CandleType.SPOT, True, True)
    assert not log_has_re(expected_text, caplog)
    df = DataFrame([
        [
            1511686200000,  # 8:50:00
            8.794,  # open
            8.948,  # high
            8.794,  # low
            8.88,  # close
            2255,  # volume (in quote currency)
        ],
        [
            1511686500000,  # 8:55:00
            8.88,
            8.942,
            8.88,
            8.893,
            9911,
        ],
        [
            1511687100000,  # 9:05:00
            889.1,   # Price jump by several decimals
            889.3,
            887.5,
            887.7,
            2251
        ],
        [
            1511687400000,  # 9:10:00
            8.877,
            8.883,
            8.895,
            8.817,
            123551
        ]
    ], columns=['date', 'open', 'high', 'low', 'close', 'volume'])

    dh._check_empty_df(df, 'UNITTEST/USDT', '1h', CandleType.SPOT, True, True)
    assert log_has_re(expected_text, caplog)


@pytest.mark.parametrize('datahandler', ['feather', 'parquet'])
def test_datahandler_trades_not_supported(datahandler, testdatadir, ):
    dh = get_datahandler(testdatadir, datahandler)
    with pytest.raises(NotImplementedError):
        dh.trades_load('UNITTEST/ETH')
    with pytest.raises(NotImplementedError):
        dh.trades_store('UNITTEST/ETH', MagicMock())


def test_jsondatahandler_trades_load(testdatadir, caplog):
    dh = JsonGzDataHandler(testdatadir)
    logmsg = "Old trades format detected - converting"
    dh.trades_load('XRP/ETH')
    assert not log_has(logmsg, caplog)

    # Test conversation is happening
    dh.trades_load('XRP/OLD')
    assert log_has(logmsg, caplog)


def test_jsondatahandler_trades_purge(mocker, testdatadir):
    mocker.patch.object(Path, "exists", MagicMock(return_value=False))
    unlinkmock = mocker.patch.object(Path, "unlink", MagicMock())
    dh = JsonGzDataHandler(testdatadir)
    assert not dh.trades_purge('UNITTEST/NONEXIST')
    assert unlinkmock.call_count == 0

    mocker.patch.object(Path, "exists", MagicMock(return_value=True))
    assert dh.trades_purge('UNITTEST/NONEXIST')
    assert unlinkmock.call_count == 1


@pytest.mark.parametrize('datahandler', AVAILABLE_DATAHANDLERS)
def test_datahandler_ohlcv_append(datahandler, testdatadir, ):
    dh = get_datahandler(testdatadir, datahandler)
    with pytest.raises(NotImplementedError):
        dh.ohlcv_append('UNITTEST/ETH', '5m', DataFrame(), CandleType.SPOT)
    with pytest.raises(NotImplementedError):
        dh.ohlcv_append('UNITTEST/ETH', '5m', DataFrame(), CandleType.MARK)


@pytest.mark.parametrize('datahandler', AVAILABLE_DATAHANDLERS)
def test_datahandler_trades_append(datahandler, testdatadir):
    dh = get_datahandler(testdatadir, datahandler)
    with pytest.raises(NotImplementedError):
        dh.trades_append('UNITTEST/ETH', [])


def test_hdf5datahandler_trades_get_pairs(testdatadir):
    pairs = HDF5DataHandler.trades_get_pairs(testdatadir)
    # Convert to set to avoid failures due to sorting
    assert set(pairs) == {'XRP/ETH'}


def test_hdf5datahandler_trades_load(testdatadir):
    dh = get_datahandler(testdatadir, 'hdf5')
    trades = dh.trades_load('XRP/ETH')
    assert isinstance(trades, list)

    trades1 = dh.trades_load('UNITTEST/NONEXIST')
    assert trades1 == []
    # data goes from 2019-10-11 - 2019-10-13
    timerange = TimeRange.parse_timerange('20191011-20191012')

    trades2 = dh._trades_load('XRP/ETH', timerange)
    assert len(trades) > len(trades2)
    # Check that ID is None (If it's nan, it's wrong)
    assert trades2[0][2] is None

    # unfiltered load has trades before starttime
    assert len([t for t in trades if t[0] < timerange.startts * 1000]) >= 0
    # filtered list does not have trades before starttime
    assert len([t for t in trades2 if t[0] < timerange.startts * 1000]) == 0
    # unfiltered load has trades after endtime
    assert len([t for t in trades if t[0] > timerange.stopts * 1000]) > 0
    # filtered list does not have trades after endtime
    assert len([t for t in trades2 if t[0] > timerange.stopts * 1000]) == 0


def test_hdf5datahandler_trades_store(testdatadir, tmpdir):
    tmpdir1 = Path(tmpdir)
    dh = get_datahandler(testdatadir, 'hdf5')
    trades = dh.trades_load('XRP/ETH')

    dh1 = get_datahandler(tmpdir1, 'hdf5')
    dh1.trades_store('XRP/NEW', trades)
    file = tmpdir1 / 'XRP_NEW-trades.h5'
    assert file.is_file()
    # Load trades back
    trades_new = dh1.trades_load('XRP/NEW')

    assert len(trades_new) == len(trades)
    assert trades[0][0] == trades_new[0][0]
    assert trades[0][1] == trades_new[0][1]
    # assert trades[0][2] == trades_new[0][2]  # This is nan - so comparison does not make sense
    assert trades[0][3] == trades_new[0][3]
    assert trades[0][4] == trades_new[0][4]
    assert trades[0][5] == trades_new[0][5]
    assert trades[0][6] == trades_new[0][6]
    assert trades[-1][0] == trades_new[-1][0]
    assert trades[-1][1] == trades_new[-1][1]
    # assert trades[-1][2] == trades_new[-1][2]  # This is nan - so comparison does not make sense
    assert trades[-1][3] == trades_new[-1][3]
    assert trades[-1][4] == trades_new[-1][4]
    assert trades[-1][5] == trades_new[-1][5]
    assert trades[-1][6] == trades_new[-1][6]


def test_hdf5datahandler_trades_purge(mocker, testdatadir):
    mocker.patch.object(Path, "exists", MagicMock(return_value=False))
    unlinkmock = mocker.patch.object(Path, "unlink", MagicMock())
    dh = get_datahandler(testdatadir, 'hdf5')
    assert not dh.trades_purge('UNITTEST/NONEXIST')
    assert unlinkmock.call_count == 0

    mocker.patch.object(Path, "exists", MagicMock(return_value=True))
    assert dh.trades_purge('UNITTEST/NONEXIST')
    assert unlinkmock.call_count == 1


@pytest.mark.parametrize('pair,timeframe,candle_type,candle_append,startdt,enddt', [
    # Data goes from 2018-01-10 - 2018-01-30
    ('UNITTEST/BTC', '5m', 'spot',  '', '2018-01-15', '2018-01-19'),
    # Mark data goes from to 2021-11-15 2021-11-19
    ('UNITTEST/USDT:USDT', '1h', 'mark', '-mark', '2021-11-16', '2021-11-18'),
])
def test_hdf5datahandler_ohlcv_load_and_resave(
    testdatadir,
    tmpdir,
    pair,
    timeframe,
    candle_type,
    candle_append,
    startdt, enddt
):
    tmpdir1 = Path(tmpdir)
    tmpdir2 = tmpdir1
    if candle_type not in ('', 'spot'):
        tmpdir2 = tmpdir1 / 'futures'
        tmpdir2.mkdir()
    dh = get_datahandler(testdatadir, 'hdf5')
    ohlcv = dh._ohlcv_load(pair, timeframe, None, candle_type=candle_type)
    assert isinstance(ohlcv, DataFrame)
    assert len(ohlcv) > 0

    file = tmpdir2 / f"UNITTEST_NEW-{timeframe}{candle_append}.h5"
    assert not file.is_file()

    dh1 = get_datahandler(tmpdir1, 'hdf5')
    dh1.ohlcv_store('UNITTEST/NEW', timeframe, ohlcv, candle_type=candle_type)
    assert file.is_file()

    assert not ohlcv[ohlcv['date'] < startdt].empty

    timerange = TimeRange.parse_timerange(f"{startdt.replace('-', '')}-{enddt.replace('-', '')}")

    # Call private function to ensure timerange is filtered in hdf5
    ohlcv = dh._ohlcv_load(pair, timeframe, timerange, candle_type=candle_type)
    ohlcv1 = dh1._ohlcv_load('UNITTEST/NEW', timeframe, timerange, candle_type=candle_type)
    assert len(ohlcv) == len(ohlcv1)
    assert ohlcv.equals(ohlcv1)
    assert ohlcv[ohlcv['date'] < startdt].empty
    assert ohlcv[ohlcv['date'] > enddt].empty

    # Try loading inexisting file
    ohlcv = dh.ohlcv_load('UNITTEST/NONEXIST', timeframe, candle_type=candle_type)
    assert ohlcv.empty


@pytest.mark.parametrize('pair,timeframe,candle_type,candle_append,startdt,enddt', [
    # Data goes from 2018-01-10 - 2018-01-30
    ('UNITTEST/BTC', '5m', 'spot',  '', '2018-01-15', '2018-01-19'),
    # Mark data goes from to 2021-11-15 2021-11-19
    ('UNITTEST/USDT:USDT', '1h', 'mark', '-mark', '2021-11-16', '2021-11-18'),
])
@pytest.mark.parametrize('datahandler', ['hdf5', 'feather', 'parquet'])
def test_generic_datahandler_ohlcv_load_and_resave(
    datahandler,
    testdatadir,
    tmpdir,
    pair,
    timeframe,
    candle_type,
    candle_append,
    startdt, enddt
):
    tmpdir1 = Path(tmpdir)
    tmpdir2 = tmpdir1
    if candle_type not in ('', 'spot'):
        tmpdir2 = tmpdir1 / 'futures'
        tmpdir2.mkdir()
    # Load data from one common file
    dhbase = get_datahandler(testdatadir, 'json')
    ohlcv = dhbase._ohlcv_load(pair, timeframe, None, candle_type=candle_type)
    assert isinstance(ohlcv, DataFrame)
    assert len(ohlcv) > 0

    # Get data to test
    dh = get_datahandler(testdatadir, datahandler)

    file = tmpdir2 / f"UNITTEST_NEW-{timeframe}{candle_append}.{dh._get_file_extension()}"
    assert not file.is_file()

    dh1 = get_datahandler(tmpdir1, datahandler)
    dh1.ohlcv_store('UNITTEST/NEW', timeframe, ohlcv, candle_type=candle_type)
    assert file.is_file()

    assert not ohlcv[ohlcv['date'] < startdt].empty

    timerange = TimeRange.parse_timerange(f"{startdt.replace('-', '')}-{enddt.replace('-', '')}")

    ohlcv = dhbase.ohlcv_load(pair, timeframe, timerange=timerange, candle_type=candle_type)
    if datahandler == 'hdf5':
        ohlcv1 = dh1._ohlcv_load('UNITTEST/NEW', timeframe, timerange, candle_type=candle_type)
        if candle_type == 'mark':
            ohlcv1['volume'] = 0.0
    else:
        ohlcv1 = dh1.ohlcv_load('UNITTEST/NEW', timeframe,
                                timerange=timerange, candle_type=candle_type)

    assert len(ohlcv) == len(ohlcv1)
    assert ohlcv.equals(ohlcv1)
    assert ohlcv[ohlcv['date'] < startdt].empty
    assert ohlcv[ohlcv['date'] > enddt].empty

    # Try loading inexisting file
    ohlcv = dh.ohlcv_load('UNITTEST/NONEXIST', timeframe, candle_type=candle_type)
    assert ohlcv.empty


def test_hdf5datahandler_ohlcv_purge(mocker, testdatadir):
    mocker.patch.object(Path, "exists", MagicMock(return_value=False))
    unlinkmock = mocker.patch.object(Path, "unlink", MagicMock())
    dh = get_datahandler(testdatadir, 'hdf5')
    assert not dh.ohlcv_purge('UNITTEST/NONEXIST', '5m', '')
    assert not dh.ohlcv_purge('UNITTEST/NONEXIST', '5m', candle_type='mark')
    assert unlinkmock.call_count == 0

    mocker.patch.object(Path, "exists", MagicMock(return_value=True))
    assert dh.ohlcv_purge('UNITTEST/NONEXIST', '5m', '')
    assert dh.ohlcv_purge('UNITTEST/NONEXIST', '5m', candle_type='mark')
    assert unlinkmock.call_count == 2


def test_gethandlerclass():
    cl = get_datahandlerclass('json')
    assert cl == JsonDataHandler
    assert issubclass(cl, IDataHandler)

    cl = get_datahandlerclass('jsongz')
    assert cl == JsonGzDataHandler
    assert issubclass(cl, IDataHandler)
    assert issubclass(cl, JsonDataHandler)

    cl = get_datahandlerclass('hdf5')
    assert cl == HDF5DataHandler
    assert issubclass(cl, IDataHandler)

    cl = get_datahandlerclass('feather')
    assert cl == FeatherDataHandler
    assert issubclass(cl, IDataHandler)

    cl = get_datahandlerclass('parquet')
    assert cl == ParquetDataHandler
    assert issubclass(cl, IDataHandler)

    with pytest.raises(ValueError, match=r"No datahandler for .*"):
        get_datahandlerclass('DeadBeef')


def test_get_datahandler(testdatadir):
    dh = get_datahandler(testdatadir, 'json')
    assert type(dh) == JsonDataHandler
    dh = get_datahandler(testdatadir, 'jsongz')
    assert type(dh) == JsonGzDataHandler
    dh1 = get_datahandler(testdatadir, 'jsongz', dh)
    assert id(dh1) == id(dh)

    dh = get_datahandler(testdatadir, 'hdf5')
    assert type(dh) == HDF5DataHandler
