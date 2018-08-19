# pragma pylint: disable=missing-docstring, C0103

import logging
from unittest.mock import MagicMock

import arrow
from pandas import DataFrame

from freqtrade.arguments import TimeRange
from freqtrade.optimize.__init__ import load_tickerdata_file
from freqtrade.tests.conftest import get_patched_exchange, log_has
from freqtrade.strategy.default_strategy import DefaultStrategy

# Avoid to reinit the same object again and again
_STRATEGY = DefaultStrategy(config={})


def test_returns_latest_buy_signal(mocker, default_conf):
    mocker.patch.object(
        _STRATEGY, 'analyze_ticker',
        return_value=DataFrame([{'buy': 1, 'sell': 0, 'date': arrow.utcnow()}])
    )
    assert _STRATEGY.get_signal('ETH/BTC', '5m', MagicMock()) == (True, False)

    mocker.patch.object(
        _STRATEGY, 'analyze_ticker',
        return_value=DataFrame([{'buy': 0, 'sell': 1, 'date': arrow.utcnow()}])
    )
    assert _STRATEGY.get_signal('ETH/BTC', '5m', MagicMock()) == (False, True)


def test_returns_latest_sell_signal(mocker, default_conf):
    mocker.patch.object(
        _STRATEGY, 'analyze_ticker',
        return_value=DataFrame([{'sell': 1, 'buy': 0, 'date': arrow.utcnow()}])
    )

    assert _STRATEGY.get_signal('ETH/BTC', '5m', MagicMock()) == (False, True)

    mocker.patch.object(
        _STRATEGY, 'analyze_ticker',
        return_value=DataFrame([{'sell': 0, 'buy': 1, 'date': arrow.utcnow()}])
    )
    assert _STRATEGY.get_signal('ETH/BTC', '5m', MagicMock()) == (True, False)


def test_get_signal_empty(default_conf, mocker, caplog):
    assert (False, False) == _STRATEGY.get_signal('foo', default_conf['ticker_interval'],
                                                  None)
    assert log_has('Empty ticker history for pair foo', caplog.record_tuples)


def test_get_signal_exception_valueerror(default_conf, mocker, caplog):
    caplog.set_level(logging.INFO)
    mocker.patch.object(
        _STRATEGY, 'analyze_ticker',
        side_effect=ValueError('xyz')
    )
    assert (False, False) == _STRATEGY.get_signal('foo', default_conf['ticker_interval'], 1)
    assert log_has('Unable to analyze ticker for pair foo: xyz', caplog.record_tuples)


def test_get_signal_empty_dataframe(default_conf, mocker, caplog):
    caplog.set_level(logging.INFO)
    mocker.patch.object(
        _STRATEGY, 'analyze_ticker',
        return_value=DataFrame([])
    )
    assert (False, False) == _STRATEGY.get_signal('xyz', default_conf['ticker_interval'], 1)
    assert log_has('Empty dataframe for pair xyz', caplog.record_tuples)


def test_get_signal_old_dataframe(default_conf, mocker, caplog):
    caplog.set_level(logging.INFO)
    # default_conf defines a 5m interval. we check interval * 2 + 5m
    # this is necessary as the last candle is removed (partial candles) by default
    oldtime = arrow.utcnow().shift(minutes=-16)
    ticks = DataFrame([{'buy': 1, 'date': oldtime}])
    mocker.patch.object(
        _STRATEGY, 'analyze_ticker',
        return_value=DataFrame(ticks)
    )
    assert (False, False) == _STRATEGY.get_signal('xyz', default_conf['ticker_interval'], 1)
    assert log_has(
        'Outdated history for pair xyz. Last tick is 16 minutes old',
        caplog.record_tuples
    )


def test_get_signal_handles_exceptions(mocker, default_conf):
    exchange = get_patched_exchange(mocker, default_conf)
    mocker.patch.object(
        _STRATEGY, 'analyze_ticker',
        side_effect=Exception('invalid ticker history ')
    )
    assert _STRATEGY.get_signal(exchange, 'ETH/BTC', '5m') == (False, False)


def test_tickerdata_to_dataframe(default_conf) -> None:
    strategy = DefaultStrategy(default_conf)

    timerange = TimeRange(None, 'line', 0, -100)
    tick = load_tickerdata_file(None, 'UNITTEST/BTC', '1m', timerange=timerange)
    tickerlist = {'UNITTEST/BTC': tick}
    data = strategy.tickerdata_to_dataframe(tickerlist)
    assert len(data['UNITTEST/BTC']) == 99       # partial candle was removed
