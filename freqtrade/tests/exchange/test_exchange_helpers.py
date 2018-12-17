# pragma pylint: disable=missing-docstring, C0103
import logging

from freqtrade.exchange.exchange_helpers import parse_ticker_dataframe
from freqtrade.tests.conftest import log_has


def test_dataframe_correct_length(result):
    dataframe = parse_ticker_dataframe(result)
    assert len(result.index) - 1 == len(dataframe.index)    # last partial candle removed


def test_dataframe_correct_columns(result):
    assert result.columns.tolist() == \
        ['date', 'open', 'high', 'low', 'close', 'volume']


def test_parse_ticker_dataframe(ticker_history, caplog):
    columns = ['date', 'open', 'high', 'low', 'close', 'volume']

    caplog.set_level(logging.DEBUG)
    # Test file with BV data
    dataframe = parse_ticker_dataframe(ticker_history)
    assert dataframe.columns.tolist() == columns
    assert log_has('Parsing tickerlist to dataframe', caplog.record_tuples)
