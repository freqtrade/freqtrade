# pragma pylint: disable=missing-docstring, C0103
from freqtrade.strategy.interface import SellCheckTuple, SellType
import logging
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import arrow
import pytest
from pandas import DataFrame

from freqtrade.configuration import TimeRange
from freqtrade.data.dataprovider import DataProvider
from freqtrade.data.history import load_data
from freqtrade.exceptions import OperationalException, StrategyError
from freqtrade.persistence import PairLocks, Trade
from freqtrade.resolvers import StrategyResolver
from freqtrade.strategy.strategy_wrapper import strategy_safe_wrapper
from tests.conftest import log_has, log_has_re

from .strats.default_strategy import DefaultStrategy


# Avoid to reinit the same object again and again
_STRATEGY = DefaultStrategy(config={})
_STRATEGY.dp = DataProvider({}, None, None)


def test_returns_latest_signal(mocker, default_conf, ohlcv_history):
    ohlcv_history.loc[1, 'date'] = arrow.utcnow()
    # Take a copy to correctly modify the call
    mocked_history = ohlcv_history.copy()
    mocked_history['sell'] = 0
    mocked_history['buy'] = 0
    mocked_history.loc[1, 'sell'] = 1

    assert _STRATEGY.get_signal('ETH/BTC', '5m', mocked_history) == (False, True)
    mocked_history.loc[1, 'sell'] = 0
    mocked_history.loc[1, 'buy'] = 1

    assert _STRATEGY.get_signal('ETH/BTC', '5m', mocked_history) == (True, False)
    mocked_history.loc[1, 'sell'] = 0
    mocked_history.loc[1, 'buy'] = 0

    assert _STRATEGY.get_signal('ETH/BTC', '5m', mocked_history) == (False, False)


def test_analyze_pair_empty(default_conf, mocker, caplog, ohlcv_history):
    mocker.patch.object(_STRATEGY.dp, 'ohlcv', return_value=ohlcv_history)
    mocker.patch.object(
        _STRATEGY, '_analyze_ticker_internal',
        return_value=DataFrame([])
    )
    mocker.patch.object(_STRATEGY, 'assert_df')

    _STRATEGY.analyze_pair('ETH/BTC')

    assert log_has('Empty dataframe for pair ETH/BTC', caplog)


def test_get_signal_empty(default_conf, mocker, caplog):
    assert (False, False) == _STRATEGY.get_signal('foo', default_conf['timeframe'], DataFrame())
    assert log_has('Empty candle (OHLCV) data for pair foo', caplog)
    caplog.clear()

    assert (False, False) == _STRATEGY.get_signal('bar', default_conf['timeframe'], None)
    assert log_has('Empty candle (OHLCV) data for pair bar', caplog)
    caplog.clear()

    assert (False, False) == _STRATEGY.get_signal('baz', default_conf['timeframe'], DataFrame([]))
    assert log_has('Empty candle (OHLCV) data for pair baz', caplog)


def test_get_signal_exception_valueerror(default_conf, mocker, caplog, ohlcv_history):
    caplog.set_level(logging.INFO)
    mocker.patch.object(_STRATEGY.dp, 'ohlcv', return_value=ohlcv_history)
    mocker.patch.object(
        _STRATEGY, '_analyze_ticker_internal',
        side_effect=ValueError('xyz')
    )
    _STRATEGY.analyze_pair('foo')
    assert log_has_re(r'Strategy caused the following exception: xyz.*', caplog)
    caplog.clear()

    mocker.patch.object(
        _STRATEGY, 'analyze_ticker',
        side_effect=Exception('invalid ticker history ')
    )
    _STRATEGY.analyze_pair('foo')
    assert log_has_re(r'Strategy caused the following exception: xyz.*', caplog)


def test_get_signal_old_dataframe(default_conf, mocker, caplog, ohlcv_history):
    # default_conf defines a 5m interval. we check interval * 2 + 5m
    # this is necessary as the last candle is removed (partial candles) by default
    ohlcv_history.loc[1, 'date'] = arrow.utcnow().shift(minutes=-16)
    # Take a copy to correctly modify the call
    mocked_history = ohlcv_history.copy()
    mocked_history['sell'] = 0
    mocked_history['buy'] = 0
    mocked_history.loc[1, 'buy'] = 1

    caplog.set_level(logging.INFO)
    mocker.patch.object(_STRATEGY, 'assert_df')

    assert (False, False) == _STRATEGY.get_signal('xyz', default_conf['timeframe'], mocked_history)
    assert log_has('Outdated history for pair xyz. Last tick is 16 minutes old', caplog)


def test_assert_df_raise(mocker, caplog, ohlcv_history):
    ohlcv_history.loc[1, 'date'] = arrow.utcnow().shift(minutes=-16)
    # Take a copy to correctly modify the call
    mocked_history = ohlcv_history.copy()
    mocked_history['sell'] = 0
    mocked_history['buy'] = 0
    mocked_history.loc[1, 'buy'] = 1

    caplog.set_level(logging.INFO)
    mocker.patch.object(_STRATEGY.dp, 'ohlcv', return_value=ohlcv_history)
    mocker.patch.object(_STRATEGY.dp, 'get_analyzed_dataframe', return_value=(mocked_history, 0))
    mocker.patch.object(
        _STRATEGY, 'assert_df',
        side_effect=StrategyError('Dataframe returned...')
    )
    _STRATEGY.analyze_pair('xyz')
    assert log_has('Unable to analyze candle (OHLCV) data for pair xyz: Dataframe returned...',
                   caplog)


def test_assert_df(ohlcv_history, caplog):
    df_len = len(ohlcv_history) - 1
    # Ensure it's running when passed correctly
    _STRATEGY.assert_df(ohlcv_history, len(ohlcv_history),
                        ohlcv_history.loc[df_len, 'close'], ohlcv_history.loc[df_len, 'date'])

    with pytest.raises(StrategyError, match=r"Dataframe returned from strategy.*length\."):
        _STRATEGY.assert_df(ohlcv_history, len(ohlcv_history) + 1,
                            ohlcv_history.loc[df_len, 'close'], ohlcv_history.loc[df_len, 'date'])

    with pytest.raises(StrategyError,
                       match=r"Dataframe returned from strategy.*last close price\."):
        _STRATEGY.assert_df(ohlcv_history, len(ohlcv_history),
                            ohlcv_history.loc[df_len, 'close'] + 0.01,
                            ohlcv_history.loc[df_len, 'date'])
    with pytest.raises(StrategyError,
                       match=r"Dataframe returned from strategy.*last date\."):
        _STRATEGY.assert_df(ohlcv_history, len(ohlcv_history),
                            ohlcv_history.loc[df_len, 'close'], ohlcv_history.loc[0, 'date'])

    _STRATEGY.disable_dataframe_checks = True
    caplog.clear()
    _STRATEGY.assert_df(ohlcv_history, len(ohlcv_history),
                        ohlcv_history.loc[2, 'close'], ohlcv_history.loc[0, 'date'])
    assert log_has_re(r"Dataframe returned from strategy.*last date\.", caplog)
    # reset to avoid problems in other tests due to test leakage
    _STRATEGY.disable_dataframe_checks = False


def test_ohlcvdata_to_dataframe(default_conf, testdatadir) -> None:
    default_conf.update({'strategy': 'DefaultStrategy'})
    strategy = StrategyResolver.load_strategy(default_conf)

    timerange = TimeRange.parse_timerange('1510694220-1510700340')
    data = load_data(testdatadir, '1m', ['UNITTEST/BTC'], timerange=timerange,
                     fill_up_missing=True)
    processed = strategy.ohlcvdata_to_dataframe(data)
    assert len(processed['UNITTEST/BTC']) == 102  # partial candle was removed


def test_ohlcvdata_to_dataframe_copy(mocker, default_conf, testdatadir) -> None:
    default_conf.update({'strategy': 'DefaultStrategy'})
    strategy = StrategyResolver.load_strategy(default_conf)
    aimock = mocker.patch('freqtrade.strategy.interface.IStrategy.advise_indicators')
    timerange = TimeRange.parse_timerange('1510694220-1510700340')
    data = load_data(testdatadir, '1m', ['UNITTEST/BTC'], timerange=timerange,
                     fill_up_missing=True)
    strategy.ohlcvdata_to_dataframe(data)
    assert aimock.call_count == 1
    # Ensure that a copy of the dataframe is passed to advice_indicators
    assert aimock.call_args_list[0][0][0] is not data


def test_min_roi_reached(default_conf, fee) -> None:

    # Use list to confirm sequence does not matter
    min_roi_list = [{20: 0.05, 55: 0.01, 0: 0.1},
                    {0: 0.1, 20: 0.05, 55: 0.01}]
    for roi in min_roi_list:
        default_conf.update({'strategy': 'DefaultStrategy'})
        strategy = StrategyResolver.load_strategy(default_conf)
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
        default_conf.update({'strategy': 'DefaultStrategy'})
        strategy = StrategyResolver.load_strategy(default_conf)
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
    default_conf.update({'strategy': 'DefaultStrategy'})
    strategy = StrategyResolver.load_strategy(default_conf)
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


@pytest.mark.parametrize(
    'profit,adjusted,expected,trailing,custom,profit2,adjusted2,expected2,custom_stop', [
        # Profit, adjusted stoploss(absolute), profit for 2nd call, enable trailing,
        #   enable custom stoploss, expected after 1st call, expected after 2nd call
        (0.2, 0.9, SellType.NONE, False, False, 0.3, 0.9, SellType.NONE, None),
        (0.2, 0.9, SellType.NONE, False, False, -0.2, 0.9, SellType.STOP_LOSS, None),
        (0.2, 1.14, SellType.NONE, True, False, 0.05, 1.14, SellType.TRAILING_STOP_LOSS, None),
        (0.01, 0.96, SellType.NONE, True, False, 0.05, 1, SellType.NONE, None),
        (0.05, 1, SellType.NONE, True, False, -0.01, 1, SellType.TRAILING_STOP_LOSS, None),
        # Default custom case - trails with 10%
        (0.05, 0.95, SellType.NONE, False, True, -0.02, 0.95, SellType.NONE, None),
        (0.05, 0.95, SellType.NONE, False, True, -0.06, 0.95, SellType.TRAILING_STOP_LOSS, None),
        (0.05, 1, SellType.NONE, False, True, -0.06, 1, SellType.TRAILING_STOP_LOSS,
         lambda **kwargs: -0.05),
        (0.05, 1, SellType.NONE, False, True, 0.09, 1.04, SellType.NONE,
         lambda **kwargs: -0.05),
        (0.05, 0.95, SellType.NONE, False, True, 0.09, 0.98, SellType.NONE,
         lambda current_profit, **kwargs: -0.1 if current_profit < 0.6 else -(current_profit * 2)),
        # Error case - static stoploss in place
        (0.05, 0.9, SellType.NONE, False, True, 0.09, 0.9, SellType.NONE,
         lambda **kwargs: None),
    ])
def test_stop_loss_reached(default_conf, fee, profit, adjusted, expected, trailing, custom,
                           profit2, adjusted2, expected2, custom_stop) -> None:

    default_conf.update({'strategy': 'DefaultStrategy'})

    strategy = StrategyResolver.load_strategy(default_conf)
    trade = Trade(
        pair='ETH/BTC',
        stake_amount=0.01,
        amount=1,
        open_date=arrow.utcnow().shift(hours=-1).datetime,
        fee_open=fee.return_value,
        fee_close=fee.return_value,
        exchange='bittrex',
        open_rate=1,
    )
    trade.adjust_min_max_rates(trade.open_rate)
    strategy.trailing_stop = trailing
    strategy.trailing_stop_positive = -0.05
    strategy.use_custom_stoploss = custom
    original_stopvalue = strategy.stoploss_value
    if custom_stop:
        strategy.stoploss_value = custom_stop

    now = arrow.utcnow().datetime
    sl_flag = strategy.stop_loss_reached(current_rate=trade.open_rate * (1 + profit), trade=trade,
                                         current_time=now, current_profit=profit,
                                         force_stoploss=0, high=None)
    assert isinstance(sl_flag, SellCheckTuple)
    assert sl_flag.sell_type == expected
    if expected == SellType.NONE:
        assert sl_flag.sell_flag is False
    else:
        assert sl_flag.sell_flag is True
    assert round(trade.stop_loss, 2) == adjusted

    sl_flag = strategy.stop_loss_reached(current_rate=trade.open_rate * (1 + profit2), trade=trade,
                                         current_time=now, current_profit=profit2,
                                         force_stoploss=0, high=None)
    assert sl_flag.sell_type == expected2
    if expected2 == SellType.NONE:
        assert sl_flag.sell_flag is False
    else:
        assert sl_flag.sell_flag is True
    assert round(trade.stop_loss, 2) == adjusted2

    strategy.stoploss_value = original_stopvalue



def test_analyze_ticker_default(ohlcv_history, mocker, caplog) -> None:
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
    strategy.analyze_ticker(ohlcv_history, {'pair': 'ETH/BTC'})
    assert ind_mock.call_count == 1
    assert buy_mock.call_count == 1
    assert buy_mock.call_count == 1

    assert log_has('TA Analysis Launched', caplog)
    assert not log_has('Skipping TA Analysis for already analyzed candle', caplog)
    caplog.clear()

    strategy.analyze_ticker(ohlcv_history, {'pair': 'ETH/BTC'})
    # No analysis happens as process_only_new_candles is true
    assert ind_mock.call_count == 2
    assert buy_mock.call_count == 2
    assert buy_mock.call_count == 2
    assert log_has('TA Analysis Launched', caplog)
    assert not log_has('Skipping TA Analysis for already analyzed candle', caplog)


def test__analyze_ticker_internal_skip_analyze(ohlcv_history, mocker, caplog) -> None:
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
    strategy.dp = DataProvider({}, None, None)
    strategy.process_only_new_candles = True

    ret = strategy._analyze_ticker_internal(ohlcv_history, {'pair': 'ETH/BTC'})
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

    ret = strategy._analyze_ticker_internal(ohlcv_history, {'pair': 'ETH/BTC'})
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


@pytest.mark.usefixtures("init_persistence")
def test_is_pair_locked(default_conf):
    default_conf.update({'strategy': 'DefaultStrategy'})
    PairLocks.timeframe = default_conf['timeframe']
    strategy = StrategyResolver.load_strategy(default_conf)
    # No lock should be present
    assert len(PairLocks.get_pair_locks(None)) == 0

    pair = 'ETH/BTC'
    assert not strategy.is_pair_locked(pair)
    strategy.lock_pair(pair, arrow.now(timezone.utc).shift(minutes=4).datetime)
    # ETH/BTC locked for 4 minutes
    assert strategy.is_pair_locked(pair)

    # XRP/BTC should not be locked now
    pair = 'XRP/BTC'
    assert not strategy.is_pair_locked(pair)

    # Unlocking a pair that's not locked should not raise an error
    strategy.unlock_pair(pair)

    # Unlock original pair
    pair = 'ETH/BTC'
    strategy.unlock_pair(pair)
    assert not strategy.is_pair_locked(pair)

    pair = 'BTC/USDT'
    # Lock until 14:30
    lock_time = datetime(2020, 5, 1, 14, 30, 0, tzinfo=timezone.utc)
    # Subtract 2 seconds, as locking rounds up to the next candle.
    strategy.lock_pair(pair, lock_time - timedelta(seconds=2))

    assert not strategy.is_pair_locked(pair)
    # latest candle is from 14:20, lock goes to 14:30
    assert strategy.is_pair_locked(pair, lock_time + timedelta(minutes=-10))
    assert strategy.is_pair_locked(pair, lock_time + timedelta(minutes=-50))

    # latest candle is from 14:25 (lock should be lifted)
    # Since this is the "new candle" available at 14:30
    assert not strategy.is_pair_locked(pair, lock_time + timedelta(minutes=-4))

    # Should not be locked after time expired
    assert not strategy.is_pair_locked(pair, lock_time + timedelta(minutes=10))

    # Change timeframe to 15m
    strategy.timeframe = '15m'
    # Candle from 14:14 - lock goes until 14:30
    assert strategy.is_pair_locked(pair, lock_time + timedelta(minutes=-16))
    assert strategy.is_pair_locked(pair, lock_time + timedelta(minutes=-15, seconds=-2))
    # Candle from 14:15 - lock goes until 14:30
    assert not strategy.is_pair_locked(pair, lock_time + timedelta(minutes=-15))


def test_is_informative_pairs_callback(default_conf):
    default_conf.update({'strategy': 'TestStrategyLegacy'})
    strategy = StrategyResolver.load_strategy(default_conf)
    # Should return empty
    # Uses fallback to base implementation
    assert [] == strategy.informative_pairs()


@pytest.mark.parametrize('error', [
    ValueError, KeyError, Exception,
])
def test_strategy_safe_wrapper_error(caplog, error):
    def failing_method():
        raise error('This is an error.')

    def working_method(argumentpassedin):
        return argumentpassedin

    with pytest.raises(StrategyError, match=r'This is an error.'):
        strategy_safe_wrapper(failing_method, message='DeadBeef')()

    assert log_has_re(r'DeadBeef.*', caplog)
    ret = strategy_safe_wrapper(failing_method, message='DeadBeef', default_retval=True)()

    assert isinstance(ret, bool)
    assert ret

    caplog.clear()
    # Test supressing error
    ret = strategy_safe_wrapper(failing_method, message='DeadBeef', supress_error=True)()
    assert log_has_re(r'DeadBeef.*', caplog)


@pytest.mark.parametrize('value', [
    1, 22, 55, True, False, {'a': 1, 'b': '112'},
    [1, 2, 3, 4], (4, 2, 3, 6)
])
def test_strategy_safe_wrapper(value):

    def working_method(argumentpassedin):
        return argumentpassedin

    ret = strategy_safe_wrapper(working_method, message='DeadBeef')(value)

    assert type(ret) == type(value)
    assert ret == value
