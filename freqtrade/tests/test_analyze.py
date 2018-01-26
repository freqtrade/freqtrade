# pragma pylint: disable=missing-docstring,W0621
import datetime
import json
from unittest.mock import MagicMock

import arrow
import pytest
from pandas import DataFrame

import freqtrade.tests.conftest as tt  # test tools
from freqtrade.analyze import (get_signal, parse_ticker_dataframe,
                               populate_buy_trend, populate_indicators,
                               populate_sell_trend)
from freqtrade.strategy.strategy import Strategy


@pytest.fixture
def result():
    with open('freqtrade/tests/testdata/BTC_ETH-1.json') as data_file:
        return parse_ticker_dataframe(json.load(data_file))


def test_dataframe_correct_columns(result):
    assert result.columns.tolist() == \
        ['close', 'high', 'low', 'open', 'date', 'volume']


def test_dataframe_correct_length(result):
    assert len(result.index) == 14395


def test_populates_buy_trend(result):
    # Load the default strategy for the unit test, because this logic is done in main.py
    Strategy().init({'strategy': 'default_strategy'})

    dataframe = populate_buy_trend(populate_indicators(result))
    assert 'buy' in dataframe.columns


def test_populates_sell_trend(result):
    # Load the default strategy for the unit test, because this logic is done in main.py
    Strategy().init({'strategy': 'default_strategy'})

    dataframe = populate_sell_trend(populate_indicators(result))
    assert 'sell' in dataframe.columns


def test_returns_latest_buy_signal(mocker):
    mocker.patch('freqtrade.analyze.get_ticker_history', return_value=MagicMock())
    mocker.patch(
        'freqtrade.analyze.analyze_ticker',
        return_value=DataFrame([{'buy': 1, 'sell': 0, 'date': arrow.utcnow()}])
    )
    assert get_signal('BTC-ETH', 5) == (True, False)

    mocker.patch(
        'freqtrade.analyze.analyze_ticker',
        return_value=DataFrame([{'buy': 0, 'sell': 1, 'date': arrow.utcnow()}])
    )
    assert get_signal('BTC-ETH', 5) == (False, True)


def test_returns_latest_sell_signal(mocker):
    mocker.patch('freqtrade.analyze.get_ticker_history', return_value=MagicMock())
    mocker.patch(
        'freqtrade.analyze.analyze_ticker',
        return_value=DataFrame([{'sell': 1, 'buy': 0, 'date': arrow.utcnow()}])
    )
    assert get_signal('BTC-ETH', 5) == (False, True)

    mocker.patch(
        'freqtrade.analyze.analyze_ticker',
        return_value=DataFrame([{'sell': 0, 'buy': 1, 'date': arrow.utcnow()}])
    )
    assert get_signal('BTC-ETH', 5) == (True, False)


def test_get_signal_empty(default_conf, mocker, caplog):
    mocker.patch('freqtrade.analyze.get_ticker_history', return_value=None)
    assert (False, False) == get_signal('foo', int(default_conf['ticker_interval']))
    assert tt.log_has('Empty ticker history for pair foo',
                      caplog.record_tuples)


def test_get_signal_exception_valueerror(default_conf, mocker, caplog):
    mocker.patch('freqtrade.analyze.get_ticker_history', return_value=1)
    mocker.patch('freqtrade.analyze.analyze_ticker',
                 side_effect=ValueError('xyz'))
    assert (False, False) == get_signal('foo', int(default_conf['ticker_interval']))
    assert tt.log_has('Unable to analyze ticker for pair foo: xyz',
                      caplog.record_tuples)


def test_get_signal_empty_dataframe(default_conf, mocker, caplog):
    mocker.patch('freqtrade.analyze.get_ticker_history', return_value=1)
    mocker.patch('freqtrade.analyze.analyze_ticker', return_value=DataFrame([]))
    assert (False, False) == get_signal('xyz', int(default_conf['ticker_interval']))
    assert tt.log_has('Empty dataframe for pair xyz',
                      caplog.record_tuples)


def test_get_signal_old_dataframe(default_conf, mocker, caplog):
    mocker.patch('freqtrade.analyze.get_ticker_history', return_value=1)
    # FIX: The get_signal function has hardcoded 10, which we must inturn hardcode
    oldtime = arrow.utcnow() - datetime.timedelta(minutes=11)
    ticks = DataFrame([{'buy': 1, 'date': oldtime}])
    mocker.patch('freqtrade.analyze.analyze_ticker', return_value=DataFrame(ticks))
    assert (False, False) == get_signal('xyz', int(default_conf['ticker_interval']))
    assert tt.log_has('Too old dataframe for pair xyz',
                      caplog.record_tuples)


def test_get_signal_handles_exceptions(mocker):
    mocker.patch('freqtrade.analyze.get_ticker_history', return_value=MagicMock())
    mocker.patch('freqtrade.analyze.analyze_ticker',
                 side_effect=Exception('invalid ticker history '))

    assert get_signal('BTC-ETH', 5) == (False, False)


def test_parse_ticker_dataframe(ticker_history, ticker_history_without_bv):

    columns = ['close', 'high', 'low', 'open', 'date', 'volume']

    # Test file with BV data
    dataframe = parse_ticker_dataframe(ticker_history)
    assert dataframe.columns.tolist() == columns

    # Test file without BV data
    dataframe = parse_ticker_dataframe(ticker_history_without_bv)
    assert dataframe.columns.tolist() == columns
