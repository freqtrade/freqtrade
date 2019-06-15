# pragma pylint: disable=missing-docstring, C0103
import logging

from freqtrade.data.converter import parse_ticker_dataframe, ohlcv_fill_up_missing_data
from freqtrade.data.history import load_pair_history, validate_backtest_data, get_timeframe
from freqtrade.tests.conftest import log_has


def test_dataframe_correct_columns(result):
    assert result.columns.tolist() == ['date', 'open', 'high', 'low', 'close', 'volume']


def test_parse_ticker_dataframe(ticker_history_list, caplog):
    columns = ['date', 'open', 'high', 'low', 'close', 'volume']

    caplog.set_level(logging.DEBUG)
    # Test file with BV data
    dataframe = parse_ticker_dataframe(ticker_history_list, '5m', fill_missing=True)
    assert dataframe.columns.tolist() == columns
    assert log_has('Parsing tickerlist to dataframe', caplog.record_tuples)


def test_ohlcv_fill_up_missing_data(caplog):
    data = load_pair_history(datadir=None,
                             ticker_interval='1m',
                             refresh_pairs=False,
                             pair='UNITTEST/BTC',
                             fill_up_missing=False)
    caplog.set_level(logging.DEBUG)
    data2 = ohlcv_fill_up_missing_data(data, '1m')
    assert len(data2) > len(data)
    # Column names should not change
    assert (data.columns == data2.columns).all()

    assert log_has(f"Missing data fillup: before: {len(data)} - after: {len(data2)}",
                   caplog.record_tuples)

    # Test fillup actually fixes invalid backtest data
    min_date, max_date = get_timeframe({'UNITTEST/BTC': data})
    assert validate_backtest_data(data, 'UNITTEST/BTC', min_date, max_date, 1)
    assert not validate_backtest_data(data2, 'UNITTEST/BTC', min_date, max_date, 1)


def test_ohlcv_fill_up_missing_data2(caplog):
    ticker_interval = '5m'
    ticks = [[
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
    data = parse_ticker_dataframe(ticks, ticker_interval, fill_missing=False)
    assert len(data) == 3
    caplog.set_level(logging.DEBUG)
    data2 = ohlcv_fill_up_missing_data(data, ticker_interval)
    assert len(data2) == 4
    # 3rd candle has been filled
    row = data2.loc[2, :]
    assert row['volume'] == 0
    # close shoult match close of previous candle
    assert row['close'] == data.loc[1, 'close']
    assert row['open'] == row['close']
    assert row['high'] == row['close']
    assert row['low'] == row['close']
    # Column names should not change
    assert (data.columns == data2.columns).all()

    assert log_has(f"Missing data fillup: before: {len(data)} - after: {len(data2)}",
                   caplog.record_tuples)


def test_ohlcv_drop_incomplete(caplog):
    ticker_interval = '1d'
    ticks = [[
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
    data = parse_ticker_dataframe(ticks, ticker_interval, fill_missing=False, drop_incomplete=False)
    assert len(data) == 4
    assert not log_has("Dropping last candle", caplog.record_tuples)

    # Drop last candle
    data = parse_ticker_dataframe(ticks, ticker_interval, fill_missing=False, drop_incomplete=True)
    assert len(data) == 3

    assert log_has("Dropping last candle", caplog.record_tuples)
