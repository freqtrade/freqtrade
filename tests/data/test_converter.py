# pragma pylint: disable=missing-docstring, C0103
import logging
from pathlib import Path
from shutil import copyfile

import numpy as np
import pytest

from freqtrade.configuration.timerange import TimeRange
from freqtrade.data.converter import (convert_ohlcv_format, convert_trades_format,
                                      ohlcv_fill_up_missing_data, ohlcv_to_dataframe,
                                      reduce_dataframe_footprint, trades_dict_to_list,
                                      trades_remove_duplicates, trades_to_ohlcv, trim_dataframe)
from freqtrade.data.history import (get_timerange, load_data, load_pair_history,
                                    validate_backtest_data)
from freqtrade.data.history.idatahandler import IDataHandler
from freqtrade.enums import CandleType
from tests.conftest import generate_test_data, log_has, log_has_re
from tests.data.test_history import _clean_test_file


def test_dataframe_correct_columns(dataframe_1m):
    assert dataframe_1m.columns.tolist() == ['date', 'open', 'high', 'low', 'close', 'volume']


def test_ohlcv_to_dataframe(ohlcv_history_list, caplog):
    columns = ['date', 'open', 'high', 'low', 'close', 'volume']

    caplog.set_level(logging.DEBUG)
    # Test file with BV data
    dataframe = ohlcv_to_dataframe(ohlcv_history_list, '5m', pair="UNITTEST/BTC",
                                   fill_missing=True)
    assert dataframe.columns.tolist() == columns
    assert log_has('Converting candle (OHLCV) data to dataframe for pair UNITTEST/BTC.', caplog)


def test_trades_to_ohlcv(ohlcv_history_list, caplog):

    caplog.set_level(logging.DEBUG)
    with pytest.raises(ValueError, match="Trade-list empty."):
        trades_to_ohlcv([], '1m')

    trades = [
        [1570752011620, "13519807", None, "sell", 0.00141342, 23.0, 0.03250866],
        [1570752011620, "13519808", None, "sell", 0.00141266, 54.0, 0.07628364],
        [1570752017964, "13519809", None, "sell", 0.00141266, 8.0, 0.01130128]]

    df = trades_to_ohlcv(trades, '1m')
    assert not df.empty
    assert len(df) == 1
    assert 'open' in df.columns
    assert 'high' in df.columns
    assert 'low' in df.columns
    assert 'close' in df.columns
    assert df.loc[:, 'high'][0] == 0.00141342
    assert df.loc[:, 'low'][0] == 0.00141266


def test_ohlcv_fill_up_missing_data(testdatadir, caplog):
    data = load_pair_history(datadir=testdatadir,
                             timeframe='1m',
                             pair='UNITTEST/BTC',
                             fill_up_missing=False)
    caplog.set_level(logging.DEBUG)
    data2 = ohlcv_fill_up_missing_data(data, '1m', 'UNITTEST/BTC')
    assert len(data2) > len(data)
    # Column names should not change
    assert (data.columns == data2.columns).all()

    assert log_has_re(f"Missing data fillup for UNITTEST/BTC: before: "
                      f"{len(data)} - after: {len(data2)}.*", caplog)

    # Test fillup actually fixes invalid backtest data
    min_date, max_date = get_timerange({'UNITTEST/BTC': data})
    assert validate_backtest_data(data, 'UNITTEST/BTC', min_date, max_date, 1)
    assert not validate_backtest_data(data2, 'UNITTEST/BTC', min_date, max_date, 1)


def test_ohlcv_fill_up_missing_data2(caplog):
    timeframe = '5m'
    ticks = [
        [
            1511686200000,  # 8:50:00
            8.794e-05,  # open
            8.948e-05,  # high
            8.794e-05,  # low
            8.88e-05,  # close
            2255,  # volume (in quote currency)
        ],
        [
            1511686500000,  # 8:55:00
            8.88e-05,
            8.942e-05,
            8.88e-05,
            8.893e-05,
            9911,
        ],
        [
            1511687100000,  # 9:05:00
            8.891e-05,
            8.893e-05,
            8.875e-05,
            8.877e-05,
            2251
        ],
        [
            1511687400000,  # 9:10:00
            8.877e-05,
            8.883e-05,
            8.895e-05,
            8.817e-05,
            123551
        ]
    ]

    # Generate test-data without filling missing
    data = ohlcv_to_dataframe(ticks, timeframe, pair="UNITTEST/BTC",
                              fill_missing=False)
    assert len(data) == 3
    caplog.set_level(logging.DEBUG)
    data2 = ohlcv_fill_up_missing_data(data, timeframe, "UNITTEST/BTC")
    assert len(data2) == 4
    # 3rd candle has been filled
    row = data2.loc[2, :]
    assert row['volume'] == 0
    # close should match close of previous candle
    assert row['close'] == data.loc[1, 'close']
    assert row['open'] == row['close']
    assert row['high'] == row['close']
    assert row['low'] == row['close']
    # Column names should not change
    assert (data.columns == data2.columns).all()

    assert log_has_re(f"Missing data fillup for UNITTEST/BTC: before: "
                      f"{len(data)} - after: {len(data2)}.*", caplog)


def test_ohlcv_drop_incomplete(caplog):
    timeframe = '1d'
    ticks = [
        [
            1559750400000,  # 2019-06-04
            8.794e-05,  # open
            8.948e-05,  # high
            8.794e-05,  # low
            8.88e-05,  # close
            2255,  # volume (in quote currency)
        ],
        [
            1559836800000,  # 2019-06-05
            8.88e-05,
            8.942e-05,
            8.88e-05,
            8.893e-05,
            9911,
        ],
        [
            1559923200000,  # 2019-06-06
            8.891e-05,
            8.893e-05,
            8.875e-05,
            8.877e-05,
            2251
        ],
        [
            1560009600000,  # 2019-06-07
            8.877e-05,
            8.883e-05,
            8.895e-05,
            8.817e-05,
            123551
        ]
    ]
    caplog.set_level(logging.DEBUG)
    data = ohlcv_to_dataframe(ticks, timeframe, pair="UNITTEST/BTC",
                              fill_missing=False, drop_incomplete=False)
    assert len(data) == 4
    assert not log_has("Dropping last candle", caplog)

    # Drop last candle
    data = ohlcv_to_dataframe(ticks, timeframe, pair="UNITTEST/BTC",
                              fill_missing=False, drop_incomplete=True)
    assert len(data) == 3

    assert log_has("Dropping last candle", caplog)


def test_trim_dataframe(testdatadir) -> None:
    data = load_data(
        datadir=testdatadir,
        timeframe='1m',
        pairs=['UNITTEST/BTC']
    )['UNITTEST/BTC']
    min_date = int(data.iloc[0]['date'].timestamp())
    max_date = int(data.iloc[-1]['date'].timestamp())
    data_modify = data.copy()

    # Remove first 30 minutes (1800 s)
    tr = TimeRange('date', None, min_date + 1800, 0)
    data_modify = trim_dataframe(data_modify, tr)
    assert not data_modify.equals(data)
    assert len(data_modify) < len(data)
    assert len(data_modify) == len(data) - 30
    assert all(data_modify.iloc[-1] == data.iloc[-1])
    assert all(data_modify.iloc[0] == data.iloc[30])

    data_modify = data.copy()
    tr = TimeRange('date', None, min_date + 1800, 0)
    # Remove first 20 candles - ignores min date
    data_modify = trim_dataframe(data_modify, tr, startup_candles=20)
    assert not data_modify.equals(data)
    assert len(data_modify) < len(data)
    assert len(data_modify) == len(data) - 20
    assert all(data_modify.iloc[-1] == data.iloc[-1])
    assert all(data_modify.iloc[0] == data.iloc[20])

    data_modify = data.copy()
    # Remove last 30 minutes (1800 s)
    tr = TimeRange(None, 'date', 0, max_date - 1800)
    data_modify = trim_dataframe(data_modify, tr)
    assert not data_modify.equals(data)
    assert len(data_modify) < len(data)
    assert len(data_modify) == len(data) - 30
    assert all(data_modify.iloc[0] == data.iloc[0])
    assert all(data_modify.iloc[-1] == data.iloc[-31])

    data_modify = data.copy()
    # Remove first 25 and last 30 minutes (1800 s)
    tr = TimeRange('date', 'date', min_date + 1500, max_date - 1800)
    data_modify = trim_dataframe(data_modify, tr)
    assert not data_modify.equals(data)
    assert len(data_modify) < len(data)
    assert len(data_modify) == len(data) - 55
    # first row matches 25th original row
    assert all(data_modify.iloc[0] == data.iloc[25])


def test_trades_remove_duplicates(trades_history):
    trades_history1 = trades_history * 3
    assert len(trades_history1) == len(trades_history) * 3
    res = trades_remove_duplicates(trades_history1)
    assert len(res) == len(trades_history)
    for i, t in enumerate(res):
        assert t == trades_history[i]


def test_trades_dict_to_list(fetch_trades_result):
    res = trades_dict_to_list(fetch_trades_result)
    assert isinstance(res, list)
    assert isinstance(res[0], list)
    for i, t in enumerate(res):
        assert t[0] == fetch_trades_result[i]['timestamp']
        assert t[1] == fetch_trades_result[i]['id']
        assert t[2] == fetch_trades_result[i]['type']
        assert t[3] == fetch_trades_result[i]['side']
        assert t[4] == fetch_trades_result[i]['price']
        assert t[5] == fetch_trades_result[i]['amount']
        assert t[6] == fetch_trades_result[i]['cost']


def test_convert_trades_format(default_conf, testdatadir, tmpdir):
    tmpdir1 = Path(tmpdir)
    files = [{'old': tmpdir1 / "XRP_ETH-trades.json.gz",
              'new': tmpdir1 / "XRP_ETH-trades.json"},
             {'old': tmpdir1 / "XRP_OLD-trades.json.gz",
              'new': tmpdir1 / "XRP_OLD-trades.json"},
             ]
    for file in files:
        copyfile(testdatadir / file['old'].name, file['old'])
        assert not file['new'].exists()

    default_conf['datadir'] = tmpdir1

    convert_trades_format(default_conf, convert_from='jsongz',
                          convert_to='json', erase=False)

    for file in files:
        assert file['new'].exists()
        assert file['old'].exists()

        # Remove original file
        file['old'].unlink()
    # Convert back
    convert_trades_format(default_conf, convert_from='json',
                          convert_to='jsongz', erase=True)
    for file in files:
        assert file['old'].exists()
        assert not file['new'].exists()

        _clean_test_file(file['old'])
        if file['new'].exists():
            file['new'].unlink()


@pytest.mark.parametrize('file_base,candletype', [
    (['XRP_ETH-5m', 'XRP_ETH-1m'], CandleType.SPOT),
    (['UNITTEST_USDT_USDT-1h-mark', 'XRP_USDT_USDT-1h-mark'], CandleType.MARK),
    (['XRP_USDT_USDT-1h-futures'], CandleType.FUTURES),
])
def test_convert_ohlcv_format(default_conf, testdatadir, tmpdir, file_base, candletype):
    tmpdir1 = Path(tmpdir)
    prependix = '' if candletype == CandleType.SPOT else 'futures/'
    files_orig = []
    files_temp = []
    files_new = []
    for file in file_base:
        file_orig = testdatadir / f"{prependix}{file}.json"
        file_temp = tmpdir1 / f"{prependix}{file}.json"
        file_new = tmpdir1 / f"{prependix}{file}.json.gz"
        IDataHandler.create_dir_if_needed(file_temp)
        copyfile(file_orig, file_temp)

        files_orig.append(file_orig)
        files_temp.append(file_temp)
        files_new.append(file_new)

    default_conf['datadir'] = tmpdir1
    if candletype == CandleType.SPOT:
        default_conf['pairs'] = ['XRP/ETH', 'XRP/USDT', 'UNITTEST/USDT']
    else:
        default_conf['pairs'] = ['XRP/ETH:ETH', 'XRP/USDT:USDT', 'UNITTEST/USDT:USDT']
    default_conf['timeframes'] = ['1m', '5m', '1h']

    assert not file_new.exists()

    convert_ohlcv_format(
        default_conf,
        convert_from='json',
        convert_to='jsongz',
        erase=False,
        candle_type=candletype
    )
    for file in (files_temp + files_new):
        assert file.exists()

    # Remove original files
    for file in (files_temp):
        file.unlink()
    # Convert back
    convert_ohlcv_format(
        default_conf,
        convert_from='jsongz',
        convert_to='json',
        erase=True,
        candle_type=candletype
    )
    for file in (files_temp):
        assert file.exists()
    for file in (files_new):
        assert not file.exists()


def test_reduce_dataframe_footprint():
    data = generate_test_data('15m', 40)

    data['open_copy'] = data['open']
    data['close_copy'] = data['close']
    data['close_copy'] = data['close']

    assert data['open'].dtype == np.float64
    assert data['open_copy'].dtype == np.float64
    assert data['close_copy'].dtype == np.float64

    df2 = reduce_dataframe_footprint(data)

    # Does not modify original dataframe
    assert data['open'].dtype == np.float64
    assert data['open_copy'].dtype == np.float64
    assert data['close_copy'].dtype == np.float64

    # skips ohlcv columns
    assert df2['open'].dtype == np.float64
    assert df2['high'].dtype == np.float64
    assert df2['low'].dtype == np.float64
    assert df2['close'].dtype == np.float64
    assert df2['volume'].dtype == np.float64

    # Changes dtype of returned dataframe
    assert df2['open_copy'].dtype == np.float32
    assert df2['close_copy'].dtype == np.float32
