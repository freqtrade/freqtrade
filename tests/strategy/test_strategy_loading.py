# pragma pylint: disable=missing-docstring, protected-access, C0103
import logging
import warnings
from base64 import urlsafe_b64encode
from pathlib import Path

import pytest
from pandas import DataFrame

from freqtrade.exceptions import OperationalException
from freqtrade.resolvers import StrategyResolver
from freqtrade.strategy.interface import IStrategy
from tests.conftest import log_has, log_has_re


def test_search_strategy():
    default_location = Path(__file__).parent / 'strats'

    s, _ = StrategyResolver._search_object(
        directory=default_location,
        object_name='StrategyTestV2',
        add_source=True,
    )
    assert issubclass(s, IStrategy)

    s, _ = StrategyResolver._search_object(
        directory=default_location,
        object_name='NotFoundStrategy',
        add_source=True,
    )
    assert s is None


def test_search_all_strategies_no_failed():
    directory = Path(__file__).parent / "strats"
    strategies = StrategyResolver.search_all_objects(directory, enum_failed=False)
    assert isinstance(strategies, list)
    assert len(strategies) == 4
    assert isinstance(strategies[0], dict)


def test_search_all_strategies_with_failed():
    directory = Path(__file__).parent / "strats"
    strategies = StrategyResolver.search_all_objects(directory, enum_failed=True)
    assert isinstance(strategies, list)
    assert len(strategies) == 5
    # with enum_failed=True search_all_objects() shall find 2 good strategies
    # and 1 which fails to load
    assert len([x for x in strategies if x['class'] is not None]) == 4
    assert len([x for x in strategies if x['class'] is None]) == 1


def test_load_strategy(default_conf, result):
    default_conf.update({'strategy': 'SampleStrategy',
                         'strategy_path': str(Path(__file__).parents[2] / 'freqtrade/templates')
                         })
    strategy = StrategyResolver.load_strategy(default_conf)
    assert isinstance(strategy.__source__, str)
    assert 'class SampleStrategy' in strategy.__source__
    assert isinstance(strategy.__file__, str)
    assert 'rsi' in strategy.advise_indicators(result, {'pair': 'ETH/BTC'})


def test_load_strategy_base64(result, caplog, default_conf):
    filepath = Path(__file__).parents[2] / 'freqtrade/templates/sample_strategy.py'
    encoded_string = urlsafe_b64encode(filepath.read_bytes()).decode("utf-8")
    default_conf.update({'strategy': 'SampleStrategy:{}'.format(encoded_string)})

    strategy = StrategyResolver.load_strategy(default_conf)
    assert 'rsi' in strategy.advise_indicators(result, {'pair': 'ETH/BTC'})
    # Make sure strategy was loaded from base64 (using temp directory)!!
    assert log_has_re(r"Using resolved strategy SampleStrategy from '"
                      r".*(/|\\).*(/|\\)SampleStrategy\.py'\.\.\.", caplog)


def test_load_strategy_invalid_directory(result, caplog, default_conf):
    default_conf['strategy'] = 'StrategyTestV2'
    extra_dir = Path.cwd() / 'some/path'
    with pytest.raises(OperationalException):
        StrategyResolver._load_strategy('StrategyTestV2', config=default_conf,
                                        extra_dir=extra_dir)

    assert log_has_re(r'Path .*' + r'some.*path.*' + r'.* does not exist', caplog)


def test_load_not_found_strategy(default_conf):
    default_conf['strategy'] = 'NotFoundStrategy'
    with pytest.raises(OperationalException,
                       match=r"Impossible to load Strategy 'NotFoundStrategy'. "
                             r"This class does not exist or contains Python code errors."):
        StrategyResolver.load_strategy(default_conf)


def test_load_strategy_noname(default_conf):
    default_conf['strategy'] = ''
    with pytest.raises(OperationalException,
                       match="No strategy set. Please use `--strategy` to specify "
                             "the strategy class to use."):
        StrategyResolver.load_strategy(default_conf)


def test_strategy(result, default_conf):
    default_conf.update({'strategy': 'StrategyTestV2'})

    strategy = StrategyResolver.load_strategy(default_conf)
    metadata = {'pair': 'ETH/BTC'}
    assert strategy.minimal_roi[0] == 0.04
    assert default_conf["minimal_roi"]['0'] == 0.04

    assert strategy.stoploss == -0.10
    assert default_conf['stoploss'] == -0.10

    assert strategy.timeframe == '5m'
    assert strategy.ticker_interval == '5m'
    assert default_conf['timeframe'] == '5m'

    df_indicators = strategy.advise_indicators(result, metadata=metadata)
    assert 'adx' in df_indicators

    dataframe = strategy.advise_buy(df_indicators, metadata=metadata)
    assert 'buy' in dataframe.columns

    dataframe = strategy.advise_sell(df_indicators, metadata=metadata)
    assert 'sell' in dataframe.columns


def test_strategy_override_minimal_roi(caplog, default_conf):
    caplog.set_level(logging.INFO)
    default_conf.update({
        'strategy': 'StrategyTestV2',
        'minimal_roi': {
            "20": 0.1,
            "0": 0.5
        }
    })
    strategy = StrategyResolver.load_strategy(default_conf)

    assert strategy.minimal_roi[0] == 0.5
    assert log_has(
        "Override strategy 'minimal_roi' with value in config file: {'20': 0.1, '0': 0.5}.",
        caplog)


def test_strategy_override_stoploss(caplog, default_conf):
    caplog.set_level(logging.INFO)
    default_conf.update({
        'strategy': 'StrategyTestV2',
        'stoploss': -0.5
    })
    strategy = StrategyResolver.load_strategy(default_conf)

    assert strategy.stoploss == -0.5
    assert log_has("Override strategy 'stoploss' with value in config file: -0.5.", caplog)


def test_strategy_override_trailing_stop(caplog, default_conf):
    caplog.set_level(logging.INFO)
    default_conf.update({
        'strategy': 'StrategyTestV2',
        'trailing_stop': True
    })
    strategy = StrategyResolver.load_strategy(default_conf)

    assert strategy.trailing_stop
    assert isinstance(strategy.trailing_stop, bool)
    assert log_has("Override strategy 'trailing_stop' with value in config file: True.", caplog)


def test_strategy_override_trailing_stop_positive(caplog, default_conf):
    caplog.set_level(logging.INFO)
    default_conf.update({
        'strategy': 'StrategyTestV2',
        'trailing_stop_positive': -0.1,
        'trailing_stop_positive_offset': -0.2

    })
    strategy = StrategyResolver.load_strategy(default_conf)

    assert strategy.trailing_stop_positive == -0.1
    assert log_has("Override strategy 'trailing_stop_positive' with value in config file: -0.1.",
                   caplog)

    assert strategy.trailing_stop_positive_offset == -0.2
    assert log_has("Override strategy 'trailing_stop_positive' with value in config file: -0.1.",
                   caplog)


def test_strategy_override_timeframe(caplog, default_conf):
    caplog.set_level(logging.INFO)

    default_conf.update({
        'strategy': 'StrategyTestV2',
        'timeframe': 60,
        'stake_currency': 'ETH'
    })
    strategy = StrategyResolver.load_strategy(default_conf)

    assert strategy.timeframe == 60
    assert strategy.stake_currency == 'ETH'
    assert log_has("Override strategy 'timeframe' with value in config file: 60.",
                   caplog)


def test_strategy_override_process_only_new_candles(caplog, default_conf):
    caplog.set_level(logging.INFO)

    default_conf.update({
        'strategy': 'StrategyTestV2',
        'process_only_new_candles': True
    })
    strategy = StrategyResolver.load_strategy(default_conf)

    assert strategy.process_only_new_candles
    assert log_has("Override strategy 'process_only_new_candles' with value in config file: True.",
                   caplog)


def test_strategy_override_order_types(caplog, default_conf):
    caplog.set_level(logging.INFO)

    order_types = {
        'buy': 'market',
        'sell': 'limit',
        'stoploss': 'limit',
        'stoploss_on_exchange': True,
    }
    default_conf.update({
        'strategy': 'StrategyTestV2',
        'order_types': order_types
    })
    strategy = StrategyResolver.load_strategy(default_conf)

    assert strategy.order_types
    for method in ['buy', 'sell', 'stoploss', 'stoploss_on_exchange']:
        assert strategy.order_types[method] == order_types[method]

    assert log_has("Override strategy 'order_types' with value in config file:"
                   " {'buy': 'market', 'sell': 'limit', 'stoploss': 'limit',"
                   " 'stoploss_on_exchange': True}.", caplog)

    default_conf.update({
        'strategy': 'StrategyTestV2',
        'order_types': {'buy': 'market'}
    })
    # Raise error for invalid configuration
    with pytest.raises(ImportError,
                       match=r"Impossible to load Strategy 'StrategyTestV2'. "
                             r"Order-types mapping is incomplete."):
        StrategyResolver.load_strategy(default_conf)


def test_strategy_override_order_tif(caplog, default_conf):
    caplog.set_level(logging.INFO)

    order_time_in_force = {
        'buy': 'fok',
        'sell': 'gtc',
    }

    default_conf.update({
        'strategy': 'StrategyTestV2',
        'order_time_in_force': order_time_in_force
    })
    strategy = StrategyResolver.load_strategy(default_conf)

    assert strategy.order_time_in_force
    for method in ['buy', 'sell']:
        assert strategy.order_time_in_force[method] == order_time_in_force[method]

    assert log_has("Override strategy 'order_time_in_force' with value in config file:"
                   " {'buy': 'fok', 'sell': 'gtc'}.", caplog)

    default_conf.update({
        'strategy': 'StrategyTestV2',
        'order_time_in_force': {'buy': 'fok'}
    })
    # Raise error for invalid configuration
    with pytest.raises(ImportError,
                       match=r"Impossible to load Strategy 'StrategyTestV2'. "
                             r"Order-time-in-force mapping is incomplete."):
        StrategyResolver.load_strategy(default_conf)


def test_strategy_override_use_sell_signal(caplog, default_conf):
    caplog.set_level(logging.INFO)
    default_conf.update({
        'strategy': 'StrategyTestV2',
    })
    strategy = StrategyResolver.load_strategy(default_conf)
    assert strategy.use_sell_signal
    assert isinstance(strategy.use_sell_signal, bool)
    # must be inserted to configuration
    assert 'use_sell_signal' in default_conf
    assert default_conf['use_sell_signal']

    default_conf.update({
        'strategy': 'StrategyTestV2',
        'use_sell_signal': False,
    })
    strategy = StrategyResolver.load_strategy(default_conf)

    assert not strategy.use_sell_signal
    assert isinstance(strategy.use_sell_signal, bool)
    assert log_has("Override strategy 'use_sell_signal' with value in config file: False.", caplog)


def test_strategy_override_use_sell_profit_only(caplog, default_conf):
    caplog.set_level(logging.INFO)
    default_conf.update({
        'strategy': 'StrategyTestV2',
    })
    strategy = StrategyResolver.load_strategy(default_conf)
    assert not strategy.sell_profit_only
    assert isinstance(strategy.sell_profit_only, bool)
    # must be inserted to configuration
    assert 'sell_profit_only' in default_conf
    assert not default_conf['sell_profit_only']

    default_conf.update({
        'strategy': 'StrategyTestV2',
        'sell_profit_only': True,
    })
    strategy = StrategyResolver.load_strategy(default_conf)

    assert strategy.sell_profit_only
    assert isinstance(strategy.sell_profit_only, bool)
    assert log_has("Override strategy 'sell_profit_only' with value in config file: True.", caplog)


@pytest.mark.filterwarnings("ignore:deprecated")
def test_deprecate_populate_indicators(result, default_conf):
    default_location = Path(__file__).parent / "strats"
    default_conf.update({'strategy': 'TestStrategyLegacyV1',
                         'strategy_path': default_location})
    strategy = StrategyResolver.load_strategy(default_conf)
    with warnings.catch_warnings(record=True) as w:
        # Cause all warnings to always be triggered.
        warnings.simplefilter("always")
        indicators = strategy.advise_indicators(result, {'pair': 'ETH/BTC'})
        assert len(w) == 1
        assert issubclass(w[-1].category, DeprecationWarning)
        assert "deprecated - check out the Sample strategy to see the current function headers!" \
            in str(w[-1].message)

    with warnings.catch_warnings(record=True) as w:
        # Cause all warnings to always be triggered.
        warnings.simplefilter("always")
        strategy.advise_buy(indicators, {'pair': 'ETH/BTC'})
        assert len(w) == 1
        assert issubclass(w[-1].category, DeprecationWarning)
        assert "deprecated - check out the Sample strategy to see the current function headers!" \
            in str(w[-1].message)

    with warnings.catch_warnings(record=True) as w:
        # Cause all warnings to always be triggered.
        warnings.simplefilter("always")
        strategy.advise_sell(indicators, {'pair': 'ETH_BTC'})
        assert len(w) == 1
        assert issubclass(w[-1].category, DeprecationWarning)
        assert "deprecated - check out the Sample strategy to see the current function headers!" \
            in str(w[-1].message)


@pytest.mark.filterwarnings("ignore:deprecated")
def test_call_deprecated_function(result, monkeypatch, default_conf, caplog):
    default_location = Path(__file__).parent / "strats"
    del default_conf['timeframe']
    default_conf.update({'strategy': 'TestStrategyLegacyV1',
                         'strategy_path': default_location})
    strategy = StrategyResolver.load_strategy(default_conf)
    metadata = {'pair': 'ETH/BTC'}

    # Make sure we are using a legacy function
    assert strategy._populate_fun_len == 2
    assert strategy._buy_fun_len == 2
    assert strategy._sell_fun_len == 2
    assert strategy.INTERFACE_VERSION == 1
    assert strategy.timeframe == '5m'
    assert strategy.ticker_interval == '5m'

    indicator_df = strategy.advise_indicators(result, metadata=metadata)
    assert isinstance(indicator_df, DataFrame)
    assert 'adx' in indicator_df.columns

    buydf = strategy.advise_buy(result, metadata=metadata)
    assert isinstance(buydf, DataFrame)
    assert 'buy' in buydf.columns

    selldf = strategy.advise_sell(result, metadata=metadata)
    assert isinstance(selldf, DataFrame)
    assert 'sell' in selldf

    assert log_has("DEPRECATED: Please migrate to using 'timeframe' instead of 'ticker_interval'.",
                   caplog)


def test_strategy_interface_versioning(result, monkeypatch, default_conf):
    default_conf.update({'strategy': 'StrategyTestV2'})
    strategy = StrategyResolver.load_strategy(default_conf)
    metadata = {'pair': 'ETH/BTC'}

    # Make sure we are using a legacy function
    assert strategy._populate_fun_len == 3
    assert strategy._buy_fun_len == 3
    assert strategy._sell_fun_len == 3
    assert strategy.INTERFACE_VERSION == 2

    indicator_df = strategy.advise_indicators(result, metadata=metadata)
    assert isinstance(indicator_df, DataFrame)
    assert 'adx' in indicator_df.columns

    buydf = strategy.advise_buy(result, metadata=metadata)
    assert isinstance(buydf, DataFrame)
    assert 'buy' in buydf.columns

    selldf = strategy.advise_sell(result, metadata=metadata)
    assert isinstance(selldf, DataFrame)
    assert 'sell' in selldf
