# pragma pylint: disable=missing-docstring, protected-access, C0103
import logging
from base64 import urlsafe_b64encode
from os import path
from pathlib import Path
import warnings

import pytest
from pandas import DataFrame

from freqtrade.strategy import import_strategy
from freqtrade.strategy.default_strategy import DefaultStrategy
from freqtrade.strategy.interface import IStrategy
from freqtrade.resolvers import StrategyResolver


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

    assert (
        'freqtrade.strategy',
        logging.DEBUG,
        'Imported strategy freqtrade.strategy.default_strategy.DefaultStrategy '
        'as freqtrade.strategy.DefaultStrategy',
    ) in caplog.record_tuples


def test_search_strategy():
    default_config = {}
    default_location = Path(__file__).parent.parent.joinpath('strategy').resolve()
    assert isinstance(
        StrategyResolver._search_object(
            directory=default_location,
            object_type=IStrategy,
            kwargs={'config': default_config},
            object_name='DefaultStrategy'
        ),
        IStrategy
    )
    assert StrategyResolver._search_object(
        directory=default_location,
        object_type=IStrategy,
        kwargs={'config': default_config},
        object_name='NotFoundStrategy'
    ) is None


def test_load_strategy(result):
    resolver = StrategyResolver({'strategy': 'TestStrategy'})
    metadata = {'pair': 'ETH/BTC'}
    assert 'adx' in resolver.strategy.advise_indicators(result, metadata=metadata)


def test_load_strategy_byte64(result):
    with open("freqtrade/tests/strategy/test_strategy.py", "r") as file:
        encoded_string = urlsafe_b64encode(file.read().encode("utf-8")).decode("utf-8")
    resolver = StrategyResolver({'strategy': 'TestStrategy:{}'.format(encoded_string)})
    assert 'adx' in resolver.strategy.advise_indicators(result, 'ETH/BTC')


def test_load_strategy_invalid_directory(result, caplog):
    resolver = StrategyResolver()
    extra_dir = path.join('some', 'path')
    resolver._load_strategy('TestStrategy', config={}, extra_dir=extra_dir)

    assert (
        'freqtrade.resolvers.strategy_resolver',
        logging.WARNING,
        'Path "{}" does not exist'.format(extra_dir),
    ) in caplog.record_tuples

    assert 'adx' in resolver.strategy.advise_indicators(result, {'pair': 'ETH/BTC'})


def test_load_not_found_strategy():
    strategy = StrategyResolver()
    with pytest.raises(ImportError,
                       match=r"Impossible to load Strategy 'NotFoundStrategy'."
                             r" This class does not exist or contains Python code errors"):
        strategy._load_strategy(strategy_name='NotFoundStrategy', config={})


def test_strategy(result):
    config = {'strategy': 'DefaultStrategy'}

    resolver = StrategyResolver(config)
    metadata = {'pair': 'ETH/BTC'}
    assert resolver.strategy.minimal_roi[0] == 0.04
    assert config["minimal_roi"]['0'] == 0.04

    assert resolver.strategy.stoploss == -0.10
    assert config['stoploss'] == -0.10

    assert resolver.strategy.ticker_interval == '5m'
    assert config['ticker_interval'] == '5m'

    df_indicators = resolver.strategy.advise_indicators(result, metadata=metadata)
    assert 'adx' in df_indicators

    dataframe = resolver.strategy.advise_buy(df_indicators, metadata=metadata)
    assert 'buy' in dataframe.columns

    dataframe = resolver.strategy.advise_sell(df_indicators, metadata=metadata)
    assert 'sell' in dataframe.columns


def test_strategy_override_minimal_roi(caplog):
    caplog.set_level(logging.INFO)
    config = {
        'strategy': 'DefaultStrategy',
        'minimal_roi': {
            "0": 0.5
        }
    }
    resolver = StrategyResolver(config)

    assert resolver.strategy.minimal_roi[0] == 0.5
    assert ('freqtrade.resolvers.strategy_resolver',
            logging.INFO,
            "Override strategy 'minimal_roi' with value in config file: {'0': 0.5}."
            ) in caplog.record_tuples


def test_strategy_override_stoploss(caplog):
    caplog.set_level(logging.INFO)
    config = {
        'strategy': 'DefaultStrategy',
        'stoploss': -0.5
    }
    resolver = StrategyResolver(config)

    assert resolver.strategy.stoploss == -0.5
    assert ('freqtrade.resolvers.strategy_resolver',
            logging.INFO,
            "Override strategy 'stoploss' with value in config file: -0.5."
            ) in caplog.record_tuples


def test_strategy_override_ticker_interval(caplog):
    caplog.set_level(logging.INFO)

    config = {
        'strategy': 'DefaultStrategy',
        'ticker_interval': 60
    }
    resolver = StrategyResolver(config)

    assert resolver.strategy.ticker_interval == 60
    assert ('freqtrade.resolvers.strategy_resolver',
            logging.INFO,
            "Override strategy 'ticker_interval' with value in config file: 60."
            ) in caplog.record_tuples


def test_strategy_override_process_only_new_candles(caplog):
    caplog.set_level(logging.INFO)

    config = {
        'strategy': 'DefaultStrategy',
        'process_only_new_candles': True
    }
    resolver = StrategyResolver(config)

    assert resolver.strategy.process_only_new_candles
    assert ('freqtrade.resolvers.strategy_resolver',
            logging.INFO,
            "Override process_only_new_candles 'process_only_new_candles' "
            "with value in config file: True."
            ) in caplog.record_tuples


def test_strategy_override_order_types(caplog):
    caplog.set_level(logging.INFO)

    order_types = {
        'buy': 'market',
        'sell': 'limit',
        'stoploss': 'limit'
    }

    config = {
        'strategy': 'DefaultStrategy',
        'order_types': order_types
    }
    resolver = StrategyResolver(config)

    assert resolver.strategy.order_types
    for method in ['buy', 'sell', 'stoploss']:
        assert resolver.strategy.order_types[method] == order_types[method]

    assert ('freqtrade.resolvers.strategy_resolver',
            logging.INFO,
            "Override strategy 'order_types' with value in config file:"
            " {'buy': 'market', 'sell': 'limit', 'stoploss': 'limit'}."
            ) in caplog.record_tuples

    config = {
        'strategy': 'DefaultStrategy',
        'order_types': {'buy': 'market'}
    }
    # Raise error for invalid configuration
    with pytest.raises(ImportError,
                       match=r"Impossible to load Strategy 'DefaultStrategy'. "
                             r"Order-types mapping is incomplete."):
        StrategyResolver(config)


def test_deprecate_populate_indicators(result):
    default_location = path.join(path.dirname(path.realpath(__file__)))
    resolver = StrategyResolver({'strategy': 'TestStrategyLegacy',
                                 'strategy_path': default_location})
    with warnings.catch_warnings(record=True) as w:
        # Cause all warnings to always be triggered.
        warnings.simplefilter("always")
        indicators = resolver.strategy.advise_indicators(result, 'ETH/BTC')
        assert len(w) == 1
        assert issubclass(w[-1].category, DeprecationWarning)
        assert "deprecated - check out the Sample strategy to see the current function headers!" \
            in str(w[-1].message)

    with warnings.catch_warnings(record=True) as w:
            # Cause all warnings to always be triggered.
        warnings.simplefilter("always")
        resolver.strategy.advise_buy(indicators, 'ETH/BTC')
        assert len(w) == 1
        assert issubclass(w[-1].category, DeprecationWarning)
        assert "deprecated - check out the Sample strategy to see the current function headers!" \
            in str(w[-1].message)

    with warnings.catch_warnings(record=True) as w:
        # Cause all warnings to always be triggered.
        warnings.simplefilter("always")
        resolver.strategy.advise_sell(indicators, 'ETH_BTC')
        assert len(w) == 1
        assert issubclass(w[-1].category, DeprecationWarning)
        assert "deprecated - check out the Sample strategy to see the current function headers!" \
            in str(w[-1].message)


def test_call_deprecated_function(result, monkeypatch):
    default_location = path.join(path.dirname(path.realpath(__file__)))
    resolver = StrategyResolver({'strategy': 'TestStrategyLegacy',
                                 'strategy_path': default_location})
    metadata = {'pair': 'ETH/BTC'}

    # Make sure we are using a legacy function
    assert resolver.strategy._populate_fun_len == 2
    assert resolver.strategy._buy_fun_len == 2
    assert resolver.strategy._sell_fun_len == 2

    indicator_df = resolver.strategy.advise_indicators(result, metadata=metadata)
    assert type(indicator_df) is DataFrame
    assert 'adx' in indicator_df.columns

    buydf = resolver.strategy.advise_buy(result, metadata=metadata)
    assert type(buydf) is DataFrame
    assert 'buy' in buydf.columns

    selldf = resolver.strategy.advise_sell(result, metadata=metadata)
    assert type(selldf) is DataFrame
    assert 'sell' in selldf
