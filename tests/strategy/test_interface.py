# pragma pylint: disable=missing-docstring, C0103

import logging
from unittest.mock import MagicMock

import arrow
from pandas import DataFrame

from freqtrade.configuration import TimeRange
from freqtrade.data.converter import parse_ticker_dataframe
from freqtrade.data.history import load_tickerdata_file
from freqtrade.persistence import Trade
from tests.conftest import get_patched_exchange, log_has
from freqtrade.strategy.default_strategy import DefaultStrategy

# Avoid to reinit the same object again and again
_STRATEGY = DefaultStrategy(config={})


def test_returns_latest_buy_signal(mocker, default_conf, ticker_history):
    mocker.patch.object(
        _STRATEGY, '_analyze_ticker_internal',
        return_value=DataFrame([{'buy': 1, 'sell': 0, 'date': arrow.utcnow()}])
    )
    assert _STRATEGY.get_signal('ETH/BTC', '5m', ticker_history) == (True, False)

    mocker.patch.object(
        _STRATEGY, '_analyze_ticker_internal',
        return_value=DataFrame([{'buy': 0, 'sell': 1, 'date': arrow.utcnow()}])
    )
    assert _STRATEGY.get_signal('ETH/BTC', '5m', ticker_history) == (False, True)


def test_returns_latest_sell_signal(mocker, default_conf, ticker_history):
    mocker.patch.object(
        _STRATEGY, '_analyze_ticker_internal',
        return_value=DataFrame([{'sell': 1, 'buy': 0, 'date': arrow.utcnow()}])
    )

    assert _STRATEGY.get_signal('ETH/BTC', '5m', ticker_history) == (False, True)

    mocker.patch.object(
        _STRATEGY, '_analyze_ticker_internal',
        return_value=DataFrame([{'sell': 0, 'buy': 1, 'date': arrow.utcnow()}])
    )
    assert _STRATEGY.get_signal('ETH/BTC', '5m', ticker_history) == (True, False)


def test_get_signal_empty(default_conf, mocker, caplog):
    assert (False, False) == _STRATEGY.get_signal('foo', default_conf['ticker_interval'],
                                                  DataFrame())
    assert log_has('Empty ticker history for pair foo', caplog)
    caplog.clear()

    assert (False, False) == _STRATEGY.get_signal('bar', default_conf['ticker_interval'],
                                                  [])
    assert log_has('Empty ticker history for pair bar', caplog)


def test_get_signal_exception_valueerror(default_conf, mocker, caplog, ticker_history):
    caplog.set_level(logging.INFO)
    mocker.patch.object(
        _STRATEGY, '_analyze_ticker_internal',
        side_effect=ValueError('xyz')
    )
    assert (False, False) == _STRATEGY.get_signal('foo', default_conf['ticker_interval'],
                                                  ticker_history)
    assert log_has('Unable to analyze ticker for pair foo: xyz', caplog)


def test_get_signal_empty_dataframe(default_conf, mocker, caplog, ticker_history):
    caplog.set_level(logging.INFO)
    mocker.patch.object(
        _STRATEGY, '_analyze_ticker_internal',
        return_value=DataFrame([])
    )
    assert (False, False) == _STRATEGY.get_signal('xyz', default_conf['ticker_interval'],
                                                  ticker_history)
    assert log_has('Empty dataframe for pair xyz', caplog)


def test_get_signal_old_dataframe(default_conf, mocker, caplog, ticker_history):
    caplog.set_level(logging.INFO)
    # default_conf defines a 5m interval. we check interval * 2 + 5m
    # this is necessary as the last candle is removed (partial candles) by default
    oldtime = arrow.utcnow().shift(minutes=-16)
    ticks = DataFrame([{'buy': 1, 'date': oldtime}])
    mocker.patch.object(
        _STRATEGY, '_analyze_ticker_internal',
        return_value=DataFrame(ticks)
    )
    assert (False, False) == _STRATEGY.get_signal('xyz', default_conf['ticker_interval'],
                                                  ticker_history)
    assert log_has('Outdated history for pair xyz. Last tick is 16 minutes old', caplog)


def test_get_signal_handles_exceptions(mocker, default_conf):
    exchange = get_patched_exchange(mocker, default_conf)
    mocker.patch.object(
        _STRATEGY, 'analyze_ticker',
        side_effect=Exception('invalid ticker history ')
    )
    assert _STRATEGY.get_signal(exchange, 'ETH/BTC', '5m') == (False, False)


def test_tickerdata_to_dataframe(default_conf, testdatadir) -> None:
    strategy = DefaultStrategy(default_conf)

    timerange = TimeRange.parse_timerange('1510694220-1510700340')
    tick = load_tickerdata_file(testdatadir, 'UNITTEST/BTC', '1m', timerange=timerange)
    tickerlist = {'UNITTEST/BTC': parse_ticker_dataframe(tick, '1m', pair="UNITTEST/BTC",
                                                         fill_missing=True)}
    data = strategy.tickerdata_to_dataframe(tickerlist)
    assert len(data['UNITTEST/BTC']) == 102       # partial candle was removed


def test_min_roi_reached(default_conf, fee) -> None:

    # Use list to confirm sequence does not matter
    min_roi_list = [{20: 0.05, 55: 0.01, 0: 0.1},
                    {0: 0.1, 20: 0.05, 55: 0.01}]
    for roi in min_roi_list:
        strategy = DefaultStrategy(default_conf)
        strategy.minimal_roi = roi
        trade = Trade(
            pair='ETH/BTC',
            stake_amount=0.001,
            amount=5,
            open_date=arrow.utcnow().shift(hours=-1).datetime,
            fee_open=fee.return_value,
            fee_close=fee.return_value,
            exchange='bittrex',
            open_rate=1,
        )

        assert not strategy.min_roi_reached(trade, 0.02, arrow.utcnow().shift(minutes=-56).datetime)
        assert strategy.min_roi_reached(trade, 0.12, arrow.utcnow().shift(minutes=-56).datetime)

        assert not strategy.min_roi_reached(trade, 0.04, arrow.utcnow().shift(minutes=-39).datetime)
        assert strategy.min_roi_reached(trade, 0.06, arrow.utcnow().shift(minutes=-39).datetime)

        assert not strategy.min_roi_reached(trade, -0.01, arrow.utcnow().shift(minutes=-1).datetime)
        assert strategy.min_roi_reached(trade, 0.02, arrow.utcnow().shift(minutes=-1).datetime)


def test_min_roi_reached2(default_conf, fee) -> None:

    # test with ROI raising after last interval
    min_roi_list = [{20: 0.07,
                     30: 0.05,
                     55: 0.30,
                     0: 0.1
                     },
                    {0: 0.1,
                     20: 0.07,
                     30: 0.05,
                     55: 0.30
                     },
                    ]
    for roi in min_roi_list:
        strategy = DefaultStrategy(default_conf)
        strategy.minimal_roi = roi
        trade = Trade(
            pair='ETH/BTC',
            stake_amount=0.001,
            amount=5,
            open_date=arrow.utcnow().shift(hours=-1).datetime,
            fee_open=fee.return_value,
            fee_close=fee.return_value,
            exchange='bittrex',
            open_rate=1,
        )

        assert not strategy.min_roi_reached(trade, 0.02, arrow.utcnow().shift(minutes=-56).datetime)
        assert strategy.min_roi_reached(trade, 0.12, arrow.utcnow().shift(minutes=-56).datetime)

        assert not strategy.min_roi_reached(trade, 0.04, arrow.utcnow().shift(minutes=-39).datetime)
        assert strategy.min_roi_reached(trade, 0.071, arrow.utcnow().shift(minutes=-39).datetime)

        assert not strategy.min_roi_reached(trade, 0.04, arrow.utcnow().shift(minutes=-26).datetime)
        assert strategy.min_roi_reached(trade, 0.06, arrow.utcnow().shift(minutes=-26).datetime)

        # Should not trigger with 20% profit since after 55 minutes only 30% is active.
        assert not strategy.min_roi_reached(trade, 0.20, arrow.utcnow().shift(minutes=-2).datetime)
        assert strategy.min_roi_reached(trade, 0.31, arrow.utcnow().shift(minutes=-2).datetime)


def test_min_roi_reached3(default_conf, fee) -> None:

    # test for issue #1948
    min_roi = {20: 0.07,
               30: 0.05,
               55: 0.30,
               }
    strategy = DefaultStrategy(default_conf)
    strategy.minimal_roi = min_roi
    trade = Trade(
            pair='ETH/BTC',
            stake_amount=0.001,
            amount=5,
            open_date=arrow.utcnow().shift(hours=-1).datetime,
            fee_open=fee.return_value,
            fee_close=fee.return_value,
            exchange='bittrex',
            open_rate=1,
    )

    assert not strategy.min_roi_reached(trade, 0.02, arrow.utcnow().shift(minutes=-56).datetime)
    assert not strategy.min_roi_reached(trade, 0.12, arrow.utcnow().shift(minutes=-56).datetime)

    assert not strategy.min_roi_reached(trade, 0.04, arrow.utcnow().shift(minutes=-39).datetime)
    assert strategy.min_roi_reached(trade, 0.071, arrow.utcnow().shift(minutes=-39).datetime)

    assert not strategy.min_roi_reached(trade, 0.04, arrow.utcnow().shift(minutes=-26).datetime)
    assert strategy.min_roi_reached(trade, 0.06, arrow.utcnow().shift(minutes=-26).datetime)

    # Should not trigger with 20% profit since after 55 minutes only 30% is active.
    assert not strategy.min_roi_reached(trade, 0.20, arrow.utcnow().shift(minutes=-2).datetime)
    assert strategy.min_roi_reached(trade, 0.31, arrow.utcnow().shift(minutes=-2).datetime)


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

    assert log_has('TA Analysis Launched', caplog)
    assert not log_has('Skipping TA Analysis for already analyzed candle', caplog)
    caplog.clear()

    strategy.analyze_ticker(ticker_history, {'pair': 'ETH/BTC'})
    # No analysis happens as process_only_new_candles is true
    assert ind_mock.call_count == 2
    assert buy_mock.call_count == 2
    assert buy_mock.call_count == 2
    assert log_has('TA Analysis Launched', caplog)
    assert not log_has('Skipping TA Analysis for already analyzed candle', caplog)


def test__analyze_ticker_internal_skip_analyze(ticker_history, mocker, caplog) -> None:
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

    ret = strategy._analyze_ticker_internal(ticker_history, {'pair': 'ETH/BTC'})
    assert 'high' in ret.columns
    assert 'low' in ret.columns
    assert 'close' in ret.columns
    assert isinstance(ret, DataFrame)
    assert ind_mock.call_count == 1
    assert buy_mock.call_count == 1
    assert buy_mock.call_count == 1
    assert log_has('TA Analysis Launched', caplog)
    assert not log_has('Skipping TA Analysis for already analyzed candle', caplog)
    caplog.clear()

    ret = strategy._analyze_ticker_internal(ticker_history, {'pair': 'ETH/BTC'})
    # No analysis happens as process_only_new_candles is true
    assert ind_mock.call_count == 1
    assert buy_mock.call_count == 1
    assert buy_mock.call_count == 1
    # only skipped analyze adds buy and sell columns, otherwise it's all mocked
    assert 'buy' in ret.columns
    assert 'sell' in ret.columns
    assert ret['buy'].sum() == 0
    assert ret['sell'].sum() == 0
    assert not log_has('TA Analysis Launched', caplog)
    assert log_has('Skipping TA Analysis for already analyzed candle', caplog)


def test_is_pair_locked(default_conf):
    strategy = DefaultStrategy(default_conf)
    # dict should be empty
    assert not strategy._pair_locked_until

    pair = 'ETH/BTC'
    assert not strategy.is_pair_locked(pair)
    strategy.lock_pair(pair, arrow.utcnow().shift(minutes=4).datetime)
    # ETH/BTC locked for 4 minutes
    assert strategy.is_pair_locked(pair)

    # XRP/BTC should not be locked now
    pair = 'XRP/BTC'
    assert not strategy.is_pair_locked(pair)
