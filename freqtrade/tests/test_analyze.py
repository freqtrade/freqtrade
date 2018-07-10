# pragma pylint: disable=missing-docstring, C0103

"""
Unit test file for analyse.py
"""

import logging
from unittest.mock import MagicMock

import arrow
from pandas import DataFrame

from freqtrade.analyze import Analyze, parse_ticker_dataframe
from freqtrade.arguments import TimeRange
from freqtrade.optimize.__init__ import load_tickerdata_file
from freqtrade.tests.conftest import get_patched_exchange, log_has
from freqtrade.strategy.default_strategy import DefaultStrategy

# Avoid to reinit the same object again and again
_ANALYZE = Analyze({}, DefaultStrategy())


def test_dataframe_correct_length(result):
    dataframe = parse_ticker_dataframe(result)
    assert len(result.index) - 1 == len(dataframe.index)    # last partial candle removed


def test_dataframe_correct_columns(result):
    assert result.columns.tolist() == \
        ['date', 'open', 'high', 'low', 'close', 'volume']


def test_returns_latest_buy_signal(mocker, default_conf):
    mocker.patch('freqtrade.exchange.Exchange.get_ticker_history', return_value=MagicMock())
    exchange = get_patched_exchange(mocker, default_conf)
    mocker.patch.multiple(
        'freqtrade.analyze.Analyze',
        analyze_ticker=MagicMock(
            return_value=DataFrame([{'buy': 1, 'sell': 0, 'date': arrow.utcnow()}])
        )
    )
    assert _ANALYZE.get_signal(exchange, 'ETH/BTC', '5m') == (True, False)

    mocker.patch.multiple(
        'freqtrade.analyze.Analyze',
        analyze_ticker=MagicMock(
            return_value=DataFrame([{'buy': 0, 'sell': 1, 'date': arrow.utcnow()}])
        )
    )
    assert _ANALYZE.get_signal(exchange, 'ETH/BTC', '5m') == (False, True)


def test_returns_latest_sell_signal(mocker, default_conf):
    mocker.patch('freqtrade.exchange.Exchange.get_ticker_history', return_value=MagicMock())
    exchange = get_patched_exchange(mocker, default_conf)
    mocker.patch.multiple(
        'freqtrade.analyze.Analyze',
        analyze_ticker=MagicMock(
            return_value=DataFrame([{'sell': 1, 'buy': 0, 'date': arrow.utcnow()}])
        )
    )

    assert _ANALYZE.get_signal(exchange, 'ETH/BTC', '5m') == (False, True)

    mocker.patch.multiple(
        'freqtrade.analyze.Analyze',
        analyze_ticker=MagicMock(
            return_value=DataFrame([{'sell': 0, 'buy': 1, 'date': arrow.utcnow()}])
        )
    )
    assert _ANALYZE.get_signal(exchange, 'ETH/BTC', '5m') == (True, False)


def test_get_signal_empty(default_conf, mocker, caplog):
    caplog.set_level(logging.INFO)
    mocker.patch('freqtrade.exchange.Exchange.get_ticker_history', return_value=None)
    exchange = get_patched_exchange(mocker, default_conf)
    assert (False, False) == _ANALYZE.get_signal(exchange, 'foo', default_conf['ticker_interval'])
    assert log_has('Empty ticker history for pair foo', caplog.record_tuples)


def test_get_signal_exception_valueerror(default_conf, mocker, caplog):
    caplog.set_level(logging.INFO)
    mocker.patch('freqtrade.exchange.Exchange.get_ticker_history', return_value=1)
    exchange = get_patched_exchange(mocker, default_conf)
    mocker.patch.multiple(
        'freqtrade.analyze.Analyze',
        analyze_ticker=MagicMock(
            side_effect=ValueError('xyz')
        )
    )
    assert (False, False) == _ANALYZE.get_signal(exchange, 'foo', default_conf['ticker_interval'])
    assert log_has('Unable to analyze ticker for pair foo: xyz', caplog.record_tuples)


def test_get_signal_empty_dataframe(default_conf, mocker, caplog):
    caplog.set_level(logging.INFO)
    mocker.patch('freqtrade.exchange.Exchange.get_ticker_history', return_value=1)
    exchange = get_patched_exchange(mocker, default_conf)
    mocker.patch.multiple(
        'freqtrade.analyze.Analyze',
        analyze_ticker=MagicMock(
            return_value=DataFrame([])
        )
    )
    assert (False, False) == _ANALYZE.get_signal(exchange, 'xyz', default_conf['ticker_interval'])
    assert log_has('Empty dataframe for pair xyz', caplog.record_tuples)


def test_get_signal_old_dataframe(default_conf, mocker, caplog):
    caplog.set_level(logging.INFO)
    mocker.patch('freqtrade.exchange.Exchange.get_ticker_history', return_value=1)
    exchange = get_patched_exchange(mocker, default_conf)
    # default_conf defines a 5m interval. we check interval * 2 + 5m
    # this is necessary as the last candle is removed (partial candles) by default
    oldtime = arrow.utcnow().shift(minutes=-16)
    ticks = DataFrame([{'buy': 1, 'date': oldtime}])
    mocker.patch.multiple(
        'freqtrade.analyze.Analyze',
        analyze_ticker=MagicMock(
            return_value=DataFrame(ticks)
        )
    )
    assert (False, False) == _ANALYZE.get_signal(exchange, 'xyz', default_conf['ticker_interval'])
    assert log_has(
        'Outdated history for pair xyz. Last tick is 16 minutes old',
        caplog.record_tuples
    )


def test_get_signal_handles_exceptions(mocker, default_conf):
    mocker.patch('freqtrade.exchange.Exchange.get_ticker_history', return_value=MagicMock())
    exchange = get_patched_exchange(mocker, default_conf)
    mocker.patch.multiple(
        'freqtrade.analyze.Analyze',
        analyze_ticker=MagicMock(
            side_effect=Exception('invalid ticker history ')
        )
    )

    assert _ANALYZE.get_signal(exchange, 'ETH/BTC', '5m') == (False, False)


def test_parse_ticker_dataframe(ticker_history):
    columns = ['date', 'open', 'high', 'low', 'close', 'volume']

    # Test file with BV data
    dataframe = parse_ticker_dataframe(ticker_history)
    assert dataframe.columns.tolist() == columns


def test_tickerdata_to_dataframe(default_conf) -> None:
    """
    Test Analyze.tickerdata_to_dataframe() method
    """
    analyze = Analyze(default_conf, DefaultStrategy())

    timerange = TimeRange(None, 'line', 0, -100)
    tick = load_tickerdata_file(None, 'UNITTEST/BTC', '1m', timerange=timerange)
    tickerlist = {'UNITTEST/BTC': tick}
    data = analyze.tickerdata_to_dataframe(tickerlist)
    assert len(data['UNITTEST/BTC']) == 99       # partial candle was removed
