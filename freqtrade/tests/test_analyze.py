# pragma pylint: disable=missing-docstring, C0103

"""
Unit test file for analyse.py
"""

import datetime
import logging
from unittest.mock import MagicMock

import arrow
from pandas import DataFrame

from freqtrade.analyze import Analyze, SignalType
from freqtrade.optimize.__init__ import load_tickerdata_file
from freqtrade.tests.conftest import log_has

# Avoid to reinit the same object again and again
_ANALYZE = Analyze({'strategy': 'default_strategy'})


def test_signaltype_object() -> None:
    """
    Test the SignalType object has the mandatory Constants
    :return: None
    """
    assert hasattr(SignalType, 'BUY')
    assert hasattr(SignalType, 'SELL')


def test_analyze_object() -> None:
    """
    Test the Analyze object has the mandatory methods
    :return: None
    """
    assert hasattr(Analyze, 'parse_ticker_dataframe')
    assert hasattr(Analyze, 'populate_indicators')
    assert hasattr(Analyze, 'populate_buy_trend')
    assert hasattr(Analyze, 'populate_sell_trend')
    assert hasattr(Analyze, 'analyze_ticker')
    assert hasattr(Analyze, 'get_signal')
    assert hasattr(Analyze, 'should_sell')
    assert hasattr(Analyze, 'min_roi_reached')


def test_dataframe_correct_length(result):
    dataframe = Analyze.parse_ticker_dataframe(result)
    assert len(result.index) == len(dataframe.index)


def test_dataframe_correct_columns(result):
    assert result.columns.tolist() == \
        ['date', 'open', 'high', 'low', 'close', 'volume']


def test_populates_buy_trend(result):
    # Load the default strategy for the unit test, because this logic is done in main.py
    dataframe = _ANALYZE.populate_buy_trend(_ANALYZE.populate_indicators(result))
    assert 'buy' in dataframe.columns


def test_populates_sell_trend(result):
    # Load the default strategy for the unit test, because this logic is done in main.py
    dataframe = _ANALYZE.populate_sell_trend(_ANALYZE.populate_indicators(result))
    assert 'sell' in dataframe.columns


def test_returns_latest_buy_signal(mocker):
    mocker.patch('freqtrade.analyze.get_ticker_history', return_value=MagicMock())

    mocker.patch.multiple(
        'freqtrade.analyze.Analyze',
        analyze_ticker=MagicMock(
            return_value=DataFrame([{'buy': 1, 'sell': 0, 'date': arrow.utcnow()}])
        )
    )
    assert _ANALYZE.get_signal('ETH/BTC', '5m') == (True, False)

    mocker.patch.multiple(
        'freqtrade.analyze.Analyze',
        analyze_ticker=MagicMock(
            return_value=DataFrame([{'buy': 0, 'sell': 1, 'date': arrow.utcnow()}])
        )
    )
    assert _ANALYZE.get_signal('ETH/BTC', '5m') == (False, True)


def test_returns_latest_sell_signal(mocker):
    mocker.patch('freqtrade.analyze.get_ticker_history', return_value=MagicMock())
    mocker.patch.multiple(
        'freqtrade.analyze.Analyze',
        analyze_ticker=MagicMock(
            return_value=DataFrame([{'sell': 1, 'buy': 0, 'date': arrow.utcnow()}])
        )
    )

    assert _ANALYZE.get_signal('ETH/BTC', '5m') == (False, True)

    mocker.patch.multiple(
        'freqtrade.analyze.Analyze',
        analyze_ticker=MagicMock(
            return_value=DataFrame([{'sell': 0, 'buy': 1, 'date': arrow.utcnow()}])
        )
    )
    assert _ANALYZE.get_signal('ETH/BTC', '5m') == (True, False)


def test_get_signal_empty(default_conf, mocker, caplog):
    caplog.set_level(logging.INFO)
    mocker.patch('freqtrade.analyze.get_ticker_history', return_value=None)
    assert (False, False) == _ANALYZE.get_signal('foo', default_conf['ticker_interval'])
    assert log_has('Empty ticker history for pair foo', caplog.record_tuples)


def test_get_signal_exception_valueerror(default_conf, mocker, caplog):
    caplog.set_level(logging.INFO)
    mocker.patch('freqtrade.analyze.get_ticker_history', return_value=1)
    mocker.patch.multiple(
        'freqtrade.analyze.Analyze',
        analyze_ticker=MagicMock(
            side_effect=ValueError('xyz')
        )
    )
    assert (False, False) == _ANALYZE.get_signal('foo', default_conf['ticker_interval'])
    assert log_has('Unable to analyze ticker for pair foo: xyz', caplog.record_tuples)


def test_get_signal_empty_dataframe(default_conf, mocker, caplog):
    caplog.set_level(logging.INFO)
    mocker.patch('freqtrade.analyze.get_ticker_history', return_value=1)
    mocker.patch.multiple(
        'freqtrade.analyze.Analyze',
        analyze_ticker=MagicMock(
            return_value=DataFrame([])
        )
    )
    assert (False, False) == _ANALYZE.get_signal('xyz', default_conf['ticker_interval'])
    assert log_has('Empty dataframe for pair xyz', caplog.record_tuples)


def test_get_signal_old_dataframe(default_conf, mocker, caplog):
    caplog.set_level(logging.INFO)
    mocker.patch('freqtrade.analyze.get_ticker_history', return_value=1)
    # FIX: The get_signal function has hardcoded 10, which we must inturn hardcode
    oldtime = arrow.utcnow() - datetime.timedelta(minutes=11)
    ticks = DataFrame([{'buy': 1, 'date': oldtime}])
    mocker.patch.multiple(
        'freqtrade.analyze.Analyze',
        analyze_ticker=MagicMock(
            return_value=DataFrame(ticks)
        )
    )
    assert (False, False) == _ANALYZE.get_signal('xyz', default_conf['ticker_interval'])
    assert log_has(
        'Outdated history for pair xyz. Last tick is 11 minutes old',
        caplog.record_tuples
    )


def test_get_signal_handles_exceptions(mocker):
    mocker.patch('freqtrade.analyze.get_ticker_history', return_value=MagicMock())
    mocker.patch.multiple(
        'freqtrade.analyze.Analyze',
        analyze_ticker=MagicMock(
            side_effect=Exception('invalid ticker history ')
        )
    )

    assert _ANALYZE.get_signal('ETH/BTC', '5m') == (False, False)


def test_parse_ticker_dataframe(ticker_history):
    columns = ['date', 'open', 'high', 'low', 'close', 'volume']

    # Test file with BV data
    dataframe = Analyze.parse_ticker_dataframe(ticker_history)
    assert dataframe.columns.tolist() == columns


def test_tickerdata_to_dataframe(default_conf) -> None:
    """
    Test Analyze.tickerdata_to_dataframe() method
    """
    analyze = Analyze(default_conf)

    timerange = ((None, 'line'), None, -100)
    tick = load_tickerdata_file(None, 'UNITTEST/BTC', '1m', timerange=timerange)
    tickerlist = {'UNITTEST/BTC': tick}
    data = analyze.tickerdata_to_dataframe(tickerlist)
    assert len(data['UNITTEST/BTC']) == 100
