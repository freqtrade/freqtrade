# pragma pylint: disable=missing-docstring, protected-access, C0103
import logging
import tempfile
import warnings
from base64 import urlsafe_b64encode
from os import path
from pathlib import Path
from unittest.mock import Mock

import pytest
from pandas import DataFrame

from freqtrade import OperationalException
from freqtrade.resolvers import StrategyResolver
from freqtrade.strategy import import_strategy
from freqtrade.strategy.default_strategy import DefaultStrategy
from freqtrade.strategy.interface import IStrategy
from freqtrade.tests.conftest import log_has, log_has_re


def test_import_strategy(caplog):
    caplog.set_level(logging.DEBUG)
    default_config = {}

    strategy = DefaultStrategy(default_config)
    strategy.some_method = lambda *args, **kwargs: 42

    assert strategy.__module__ == 'freqtrade.strategy.default_strategy'
    assert strategy.some_method() == 42

    imported_strategy = import_strategy(strategy, default_config)

    assert dir(strategy) == dir(imported_strategy)

    assert imported_strategy.__module__ == 'freqtrade.strategy'
    assert imported_strategy.some_method() == 42

    assert log_has('Imported strategy freqtrade.strategy.default_strategy.DefaultStrategy '
                   'as freqtrade.strategy.DefaultStrategy', caplog)


def test_search_strategy():
    default_config = {}
    default_location = Path(__file__).parent.parent.parent.joinpath('strategy').resolve()

    s, _ = StrategyResolver._search_object(
        directory=default_location,
        object_type=IStrategy,
        kwargs={'config': default_config},
        object_name='DefaultStrategy'
    )
    assert isinstance(s, IStrategy)

    s, _ = StrategyResolver._search_object(
        directory=default_location,
        object_type=IStrategy,
        kwargs={'config': default_config},
        object_name='NotFoundStrategy'
    )
    assert s is None


def test_load_strategy(default_conf, result):
    default_conf.update({'strategy': 'TestStrategy'})
    resolver = StrategyResolver(default_conf)
    assert 'adx' in resolver.strategy.advise_indicators(result, {'pair': 'ETH/BTC'})


def test_load_strategy_base64(result, caplog, default_conf):
    with open("user_data/strategies/test_strategy.py", "rb") as file:
        encoded_string = urlsafe_b64encode(file.read()).decode("utf-8")
    default_conf.update({'strategy': 'TestStrategy:{}'.format(encoded_string)})

    resolver = StrategyResolver(default_conf)
    assert 'adx' in resolver.strategy.advise_indicators(result, {'pair': 'ETH/BTC'})
    # Make sure strategy was loaded from base64 (using temp directory)!!
    assert log_has_re(r"Using resolved strategy TestStrategy from '"
                      + tempfile.gettempdir() + r"/.*/TestStrategy\.py'\.\.\.", caplog)


def test_load_strategy_invalid_directory(result, caplog, default_conf):
    resolver = StrategyResolver(default_conf)
    extra_dir = Path.cwd() / 'some/path'
    resolver._load_strategy('TestStrategy', config=default_conf, extra_dir=extra_dir)

    assert log_has_re(r'Path .*' + r'some.*path.*' + r'.* does not exist', caplog)

    assert 'adx' in resolver.strategy.advise_indicators(result, {'pair': 'ETH/BTC'})


def test_load_not_found_strategy(default_conf):
    strategy = StrategyResolver(default_conf)
    with pytest.raises(OperationalException,
                       match=r"Impossible to load Strategy 'NotFoundStrategy'. "
                             r"This class does not exist or contains Python code errors."):
        strategy._load_strategy(strategy_name='NotFoundStrategy', config=default_conf)


def test_load_staticmethod_importerror(mocker, caplog, default_conf):
    mocker.patch("freqtrade.resolvers.strategy_resolver.import_strategy", Mock(
        side_effect=TypeError("can't pickle staticmethod objects")))
    with pytest.raises(OperationalException,
                       match=r"Impossible to load Strategy 'DefaultStrategy'. "
                             r"This class does not exist or contains Python code errors."):
        StrategyResolver(default_conf)
    assert log_has_re(r".*Error: can't pickle staticmethod objects", caplog)


def test_strategy(result, default_conf):
    default_conf.update({'strategy': 'DefaultStrategy'})

    resolver = StrategyResolver(default_conf)
    metadata = {'pair': 'ETH/BTC'}
    assert resolver.strategy.minimal_roi[0] == 0.04
    assert default_conf["minimal_roi"]['0'] == 0.04

    assert resolver.strategy.stoploss == -0.10
    assert default_conf['stoploss'] == -0.10

    assert resolver.strategy.ticker_interval == '5m'
    assert default_conf['ticker_interval'] == '5m'

    df_indicators = resolver.strategy.advise_indicators(result, metadata=metadata)
    assert 'adx' in df_indicators

    dataframe = resolver.strategy.advise_buy(df_indicators, metadata=metadata)
    assert 'buy' in dataframe.columns

    dataframe = resolver.strategy.advise_sell(df_indicators, metadata=metadata)
    assert 'sell' in dataframe.columns


def test_strategy_override_minimal_roi(caplog, default_conf):
    caplog.set_level(logging.INFO)
    default_conf.update({
        'strategy': 'DefaultStrategy',
        'minimal_roi': {
            "0": 0.5
        }
    })
    resolver = StrategyResolver(default_conf)

    assert resolver.strategy.minimal_roi[0] == 0.5
    assert log_has("Override strategy 'minimal_roi' with value in config file: {'0': 0.5}.", caplog)


def test_strategy_override_stoploss(caplog, default_conf):
    caplog.set_level(logging.INFO)
    default_conf.update({
        'strategy': 'DefaultStrategy',
        'stoploss': -0.5
    })
    resolver = StrategyResolver(default_conf)

    assert resolver.strategy.stoploss == -0.5
    assert log_has("Override strategy 'stoploss' with value in config file: -0.5.", caplog)


def test_strategy_override_trailing_stop(caplog, default_conf):
    caplog.set_level(logging.INFO)
    default_conf.update({
        'strategy': 'DefaultStrategy',
        'trailing_stop': True
    })
    resolver = StrategyResolver(default_conf)

    assert resolver.strategy.trailing_stop
    assert isinstance(resolver.strategy.trailing_stop, bool)
    assert log_has("Override strategy 'trailing_stop' with value in config file: True.", caplog)


def test_strategy_override_trailing_stop_positive(caplog, default_conf):
    caplog.set_level(logging.INFO)
    default_conf.update({
        'strategy': 'DefaultStrategy',
        'trailing_stop_positive': -0.1,
        'trailing_stop_positive_offset': -0.2

    })
    resolver = StrategyResolver(default_conf)

    assert resolver.strategy.trailing_stop_positive == -0.1
    assert log_has("Override strategy 'trailing_stop_positive' with value in config file: -0.1.",
                   caplog)

    assert resolver.strategy.trailing_stop_positive_offset == -0.2
    assert log_has("Override strategy 'trailing_stop_positive' with value in config file: -0.1.",
                   caplog)


def test_strategy_override_ticker_interval(caplog, default_conf):
    caplog.set_level(logging.INFO)

    default_conf.update({
        'strategy': 'DefaultStrategy',
        'ticker_interval': 60,
        'stake_currency': 'ETH'
    })
    resolver = StrategyResolver(default_conf)

    assert resolver.strategy.ticker_interval == 60
    assert resolver.strategy.stake_currency == 'ETH'
    assert log_has("Override strategy 'ticker_interval' with value in config file: 60.",
                   caplog)


def test_strategy_override_process_only_new_candles(caplog, default_conf):
    caplog.set_level(logging.INFO)

    default_conf.update({
        'strategy': 'DefaultStrategy',
        'process_only_new_candles': True
    })
    resolver = StrategyResolver(default_conf)

    assert resolver.strategy.process_only_new_candles
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
        'strategy': 'DefaultStrategy',
        'order_types': order_types
    })
    resolver = StrategyResolver(default_conf)

    assert resolver.strategy.order_types
    for method in ['buy', 'sell', 'stoploss', 'stoploss_on_exchange']:
        assert resolver.strategy.order_types[method] == order_types[method]

    assert log_has("Override strategy 'order_types' with value in config file:"
                   " {'buy': 'market', 'sell': 'limit', 'stoploss': 'limit',"
                   " 'stoploss_on_exchange': True}.", caplog)

    default_conf.update({
        'strategy': 'DefaultStrategy',
        'order_types': {'buy': 'market'}
    })
    # Raise error for invalid configuration
    with pytest.raises(ImportError,
                       match=r"Impossible to load Strategy 'DefaultStrategy'. "
                             r"Order-types mapping is incomplete."):
        StrategyResolver(default_conf)


def test_strategy_override_order_tif(caplog, default_conf):
    caplog.set_level(logging.INFO)

    order_time_in_force = {
        'buy': 'fok',
        'sell': 'gtc',
    }

    default_conf.update({
        'strategy': 'DefaultStrategy',
        'order_time_in_force': order_time_in_force
    })
    resolver = StrategyResolver(default_conf)

    assert resolver.strategy.order_time_in_force
    for method in ['buy', 'sell']:
        assert resolver.strategy.order_time_in_force[method] == order_time_in_force[method]

    assert log_has("Override strategy 'order_time_in_force' with value in config file:"
                   " {'buy': 'fok', 'sell': 'gtc'}.", caplog)

    default_conf.update({
        'strategy': 'DefaultStrategy',
        'order_time_in_force': {'buy': 'fok'}
    })
    # Raise error for invalid configuration
    with pytest.raises(ImportError,
                       match=r"Impossible to load Strategy 'DefaultStrategy'. "
                             r"Order-time-in-force mapping is incomplete."):
        StrategyResolver(default_conf)


def test_strategy_override_use_sell_signal(caplog, default_conf):
    caplog.set_level(logging.INFO)
    default_conf.update({
        'strategy': 'DefaultStrategy',
    })
    resolver = StrategyResolver(default_conf)
    assert not resolver.strategy.use_sell_signal
    assert isinstance(resolver.strategy.use_sell_signal, bool)
    # must be inserted to configuration
    assert 'use_sell_signal' in default_conf['experimental']
    assert not default_conf['experimental']['use_sell_signal']

    default_conf.update({
        'strategy': 'DefaultStrategy',
        'experimental': {
            'use_sell_signal': True,
        },
    })
    resolver = StrategyResolver(default_conf)

    assert resolver.strategy.use_sell_signal
    assert isinstance(resolver.strategy.use_sell_signal, bool)
    assert log_has("Override strategy 'use_sell_signal' with value in config file: True.", caplog)


def test_strategy_override_use_sell_profit_only(caplog, default_conf):
    caplog.set_level(logging.INFO)
    default_conf.update({
        'strategy': 'DefaultStrategy',
    })
    resolver = StrategyResolver(default_conf)
    assert not resolver.strategy.sell_profit_only
    assert isinstance(resolver.strategy.sell_profit_only, bool)
    # must be inserted to configuration
    assert 'sell_profit_only' in default_conf['experimental']
    assert not default_conf['experimental']['sell_profit_only']

    default_conf.update({
        'strategy': 'DefaultStrategy',
        'experimental': {
            'sell_profit_only': True,
        },
    })
    resolver = StrategyResolver(default_conf)

    assert resolver.strategy.sell_profit_only
    assert isinstance(resolver.strategy.sell_profit_only, bool)
    assert log_has("Override strategy 'sell_profit_only' with value in config file: True.", caplog)


@pytest.mark.filterwarnings("ignore:deprecated")
def test_deprecate_populate_indicators(result, default_conf):
    default_location = path.join(path.dirname(path.realpath(__file__)))
    default_conf.update({'strategy': 'TestStrategyLegacy',
                         'strategy_path': default_location})
    resolver = StrategyResolver(default_conf)
    with warnings.catch_warnings(record=True) as w:
        # Cause all warnings to always be triggered.
        warnings.simplefilter("always")
        indicators = resolver.strategy.advise_indicators(result, {'pair': 'ETH/BTC'})
        assert len(w) == 1
        assert issubclass(w[-1].category, DeprecationWarning)
        assert "deprecated - check out the Sample strategy to see the current function headers!" \
            in str(w[-1].message)

    with warnings.catch_warnings(record=True) as w:
        # Cause all warnings to always be triggered.
        warnings.simplefilter("always")
        resolver.strategy.advise_buy(indicators, {'pair': 'ETH/BTC'})
        assert len(w) == 1
        assert issubclass(w[-1].category, DeprecationWarning)
        assert "deprecated - check out the Sample strategy to see the current function headers!" \
            in str(w[-1].message)

    with warnings.catch_warnings(record=True) as w:
        # Cause all warnings to always be triggered.
        warnings.simplefilter("always")
        resolver.strategy.advise_sell(indicators, {'pair': 'ETH_BTC'})
        assert len(w) == 1
        assert issubclass(w[-1].category, DeprecationWarning)
        assert "deprecated - check out the Sample strategy to see the current function headers!" \
            in str(w[-1].message)


@pytest.mark.filterwarnings("ignore:deprecated")
def test_call_deprecated_function(result, monkeypatch, default_conf):
    default_location = path.join(path.dirname(path.realpath(__file__)))
    default_conf.update({'strategy': 'TestStrategyLegacy',
                         'strategy_path': default_location})
    resolver = StrategyResolver(default_conf)
    metadata = {'pair': 'ETH/BTC'}

    # Make sure we are using a legacy function
    assert resolver.strategy._populate_fun_len == 2
    assert resolver.strategy._buy_fun_len == 2
    assert resolver.strategy._sell_fun_len == 2
    assert resolver.strategy.strategy_version == 1

    indicator_df = resolver.strategy.advise_indicators(result, metadata=metadata)
    assert isinstance(indicator_df, DataFrame)
    assert 'adx' in indicator_df.columns

    buydf = resolver.strategy.advise_buy(result, metadata=metadata)
    assert isinstance(buydf, DataFrame)
    assert 'buy' in buydf.columns

    selldf = resolver.strategy.advise_sell(result, metadata=metadata)
    assert isinstance(selldf, DataFrame)
    assert 'sell' in selldf


def test_strategy_versioning(result, monkeypatch, default_conf):
    default_conf.update({'strategy': 'DefaultStrategy'})
    resolver = StrategyResolver(default_conf)
    metadata = {'pair': 'ETH/BTC'}

    # Make sure we are using a legacy function
    assert resolver.strategy._populate_fun_len == 3
    assert resolver.strategy._buy_fun_len == 3
    assert resolver.strategy._sell_fun_len == 3
    assert resolver.strategy.strategy_version == 2

    indicator_df = resolver.strategy.advise_indicators(result, metadata=metadata)
    assert isinstance(indicator_df, DataFrame)
    assert 'adx' in indicator_df.columns

    buydf = resolver.strategy.advise_buy(result, metadata=metadata)
    assert isinstance(buydf, DataFrame)
    assert 'buy' in buydf.columns

    selldf = resolver.strategy.advise_sell(result, metadata=metadata)
    assert isinstance(selldf, DataFrame)
    assert 'sell' in selldf
