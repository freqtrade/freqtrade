# pragma pylint: disable=missing-docstring, C0103
import logging

from freqtrade.data.converter import parse_ticker_dataframe, ohlcv_fill_up_missing_data
from freqtrade.data.history import load_pair_history
from freqtrade.optimize import validate_backtest_data, get_timeframe
from freqtrade.tests.conftest import log_has


def test_dataframe_correct_length(result):
    dataframe = parse_ticker_dataframe(result)
    assert len(result.index) - 1 == len(dataframe.index)    # last partial candle removed


def test_dataframe_correct_columns(result):
    assert result.columns.tolist() == ['date', 'open', 'high', 'low', 'close', 'volume']


def test_parse_ticker_dataframe(ticker_history, caplog):
    columns = ['date', 'open', 'high', 'low', 'close', 'volume']

    caplog.set_level(logging.DEBUG)
    # Test file with BV data
    dataframe = parse_ticker_dataframe(ticker_history)
    assert dataframe.columns.tolist() == columns
    assert log_has('Parsing tickerlist to dataframe', caplog.record_tuples)


def test_ohlcv_fill_up_missing_data(caplog):
    data = load_pair_history(datadir=None,
                             ticker_interval='1m',
                             refresh_pairs=False,
                             pair='UNITTEST/BTC')
    caplog.set_level(logging.DEBUG)
    data2 = ohlcv_fill_up_missing_data(data, '1m')
    assert len(data2) > len(data)
    # Column names should not change
    assert (data.columns == data2.columns).all()

    assert log_has(f"Missing data fillup: before: {len(data)} - after: {len(data2)}",
                   caplog.record_tuples)

    # Test fillup actually fixes invalid backtest data
    min_date, max_date = get_timeframe({'UNITTEST/BTC': data})
    assert validate_backtest_data({'UNITTEST/BTC': data}, min_date, max_date, 1)
    assert not validate_backtest_data({'UNITTEST/BTC': data2}, min_date, max_date, 1)
