# pragma pylint: disable=missing-docstring, C0103

"""
Unit test file for analyse.py
"""

from freqtrade.analyze import parse_ticker_dataframe


def test_dataframe_correct_length(result):
    dataframe = parse_ticker_dataframe(result)
    assert len(result.index) - 1 == len(dataframe.index)    # last partial candle removed


def test_dataframe_correct_columns(result):
    assert result.columns.tolist() == \
        ['date', 'open', 'high', 'low', 'close', 'volume']


def test_parse_ticker_dataframe(ticker_history):
    columns = ['date', 'open', 'high', 'low', 'close', 'volume']

    # Test file with BV data
    dataframe = parse_ticker_dataframe(ticker_history)
    assert dataframe.columns.tolist() == columns
