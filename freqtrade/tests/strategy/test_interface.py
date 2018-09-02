# pragma pylint: disable=missing-docstring, C0103

import logging
from unittest.mock import MagicMock

import arrow
from pandas import DataFrame

from freqtrade.arguments import TimeRange
from freqtrade.optimize.__init__ import load_tickerdata_file
from freqtrade.persistence import Trade
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
    mocker.patch('freqtrade.exchange.Exchange.get_candle_history', return_value=MagicMock())
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


def test_min_roi_reached(default_conf, fee) -> None:
    strategy = DefaultStrategy(default_conf)
    strategy.minimal_roi = {0: 0.1, 20: 0.05, 55: 0.01}
    trade = Trade(
        pair='ETH/BTC',
        stake_amount=0.001,
        open_date=arrow.utcnow().shift(hours=-1).datetime,
        fee_open=fee.return_value,
        fee_close=fee.return_value,
        exchange='bittrex',
        open_rate=1,
    )

    assert not strategy.min_roi_reached(trade, 0.01, arrow.utcnow().shift(minutes=-55).datetime)
    assert strategy.min_roi_reached(trade, 0.12, arrow.utcnow().shift(minutes=-55).datetime)

    assert not strategy.min_roi_reached(trade, 0.04, arrow.utcnow().shift(minutes=-39).datetime)
    assert strategy.min_roi_reached(trade, 0.06, arrow.utcnow().shift(minutes=-39).datetime)

    assert not strategy.min_roi_reached(trade, -0.01, arrow.utcnow().shift(minutes=-1).datetime)
    assert strategy.min_roi_reached(trade, 0.02, arrow.utcnow().shift(minutes=-1).datetime)


def test_analyze_ticker_default(ticker_history, mocker, caplog) -> None:
    caplog.set_level(logging.DEBUG)
    ind_mock = MagicMock(side_effect=lambda x, meta: x)
    buy_mock = MagicMock(side_effect=lambda x, meta: x)
    sell_mock = MagicMock(side_effect=lambda x, meta: x)
    mocker.patch.multiple(
        'freqtrade.strategy.interface.IStrategy',
        advise_indicators=ind_mock,
        advise_buy=buy_mock,
        advise_sell=sell_mock,

    )
    strategy = DefaultStrategy({})
    strategy.analyze_ticker(ticker_history, {'pair': 'ETH/BTC'})
    assert ind_mock.call_count == 1
    assert buy_mock.call_count == 1
    assert buy_mock.call_count == 1

    assert log_has('TA Analysis Launched', caplog.record_tuples)
    assert not log_has('Skippinig TA Analysis for already analyzed candle',
                       caplog.record_tuples)
    caplog.clear()

    strategy.analyze_ticker(ticker_history, {'pair': 'ETH/BTC'})
    # No analysis happens as process_only_new_candles is true
    assert ind_mock.call_count == 2
    assert buy_mock.call_count == 2
    assert buy_mock.call_count == 2
    assert log_has('TA Analysis Launched', caplog.record_tuples)
    assert not log_has('Skippinig TA Analysis for already analyzed candle',
                       caplog.record_tuples)


def test_analyze_ticker_skip_analyze(ticker_history, mocker, caplog) -> None:
    caplog.set_level(logging.DEBUG)
    ind_mock = MagicMock(side_effect=lambda x, meta: x)
    buy_mock = MagicMock(side_effect=lambda x, meta: x)
    sell_mock = MagicMock(side_effect=lambda x, meta: x)
    mocker.patch.multiple(
        'freqtrade.strategy.interface.IStrategy',
        advise_indicators=ind_mock,
        advise_buy=buy_mock,
        advise_sell=sell_mock,

    )
    strategy = DefaultStrategy({})
    strategy.process_only_new_candles = True

    ret = strategy.analyze_ticker(ticker_history, {'pair': 'ETH/BTC'})
    assert ind_mock.call_count == 1
    assert buy_mock.call_count == 1
    assert buy_mock.call_count == 1
    assert log_has('TA Analysis Launched', caplog.record_tuples)
    assert not log_has('Skippinig TA Analysis for already analyzed candle',
                       caplog.record_tuples)
    caplog.clear()

    ret = strategy.analyze_ticker(ticker_history, {'pair': 'ETH/BTC'})
    # No analysis happens as process_only_new_candles is true
    assert ind_mock.call_count == 1
    assert buy_mock.call_count == 1
    assert buy_mock.call_count == 1
    # only skipped analyze adds buy and sell columns, otherwise it's all mocked
    assert 'buy' in ret
    assert 'sell' in ret
    assert ret['buy'].sum() == 0
    assert ret['sell'].sum() == 0
    assert not log_has('TA Analysis Launched', caplog.record_tuples)
    assert log_has('Skippinig TA Analysis for already analyzed candle',
                   caplog.record_tuples)
