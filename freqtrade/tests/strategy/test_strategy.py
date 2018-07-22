# pragma pylint: disable=missing-docstring, protected-access, C0103
import logging
from os import path
import warnings

import pytest
from pandas import DataFrame

from freqtrade.strategy import import_strategy
from freqtrade.strategy.default_strategy import DefaultStrategy
from freqtrade.strategy.interface import IStrategy
from freqtrade.strategy.resolver import StrategyResolver


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
    default_location = path.join(path.dirname(
        path.realpath(__file__)), '..', '..', 'strategy'
    )
    assert isinstance(
        StrategyResolver._search_strategy(
            default_location,
            config=default_config,
            strategy_name='DefaultStrategy'
        ),
        IStrategy
    )
    assert StrategyResolver._search_strategy(
        default_location,
        config=default_config,
        strategy_name='NotFoundStrategy'
    ) is None


def test_load_strategy(result):
    resolver = StrategyResolver({'strategy': 'TestStrategy'})
    pair = 'ETH/BTC'
    assert len(resolver.strategy.populate_indicators.__annotations__) == 3
    assert 'dataframe' in resolver.strategy.populate_indicators.__annotations__
    assert 'pair' in resolver.strategy.populate_indicators.__annotations__
    assert 'adx' in resolver.strategy.advise_indicators(result, pair=pair)


def test_load_strategy_invalid_directory(result, caplog):
    resolver = StrategyResolver()
    extra_dir = path.join('some', 'path')
    resolver._load_strategy('TestStrategy', config={}, extra_dir=extra_dir)

    assert (
        'freqtrade.strategy.resolver',
        logging.WARNING,
        'Path "{}" does not exist'.format(extra_dir),
    ) in caplog.record_tuples

    assert 'adx' in resolver.strategy.advise_indicators(result, 'ETH/BTC')


def test_load_not_found_strategy():
    strategy = StrategyResolver()
    with pytest.raises(ImportError,
                       match=r'Impossible to load Strategy \'NotFoundStrategy\'.'
                             r' This class does not exist or contains Python code errors'):
        strategy._load_strategy(strategy_name='NotFoundStrategy', config={})


def test_strategy(result):
    config = {'strategy': 'DefaultStrategy'}

    resolver = StrategyResolver(config)
    pair = 'ETH/BTC'
    assert resolver.strategy.minimal_roi[0] == 0.04
    assert config["minimal_roi"]['0'] == 0.04

    assert resolver.strategy.stoploss == -0.10
    assert config['stoploss'] == -0.10

    assert resolver.strategy.ticker_interval == '5m'
    assert config['ticker_interval'] == '5m'

    df_indicators = resolver.strategy.advise_indicators(result, pair=pair)
    assert 'adx' in df_indicators

    dataframe = resolver.strategy.advise_buy(df_indicators, pair=pair)
    assert 'buy' in dataframe.columns

    dataframe = resolver.strategy.advise_sell(df_indicators, pair='ETH/BTC')
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
    assert ('freqtrade.strategy.resolver',
            logging.INFO,
            'Override strategy \'minimal_roi\' with value in config file.'
            ) in caplog.record_tuples


def test_strategy_override_stoploss(caplog):
    caplog.set_level(logging.INFO)
    config = {
        'strategy': 'DefaultStrategy',
        'stoploss': -0.5
    }
    resolver = StrategyResolver(config)

    assert resolver.strategy.stoploss == -0.5
    assert ('freqtrade.strategy.resolver',
            logging.INFO,
            'Override strategy \'stoploss\' with value in config file: -0.5.'
            ) in caplog.record_tuples


def test_strategy_override_ticker_interval(caplog):
    caplog.set_level(logging.INFO)

    config = {
        'strategy': 'DefaultStrategy',
        'ticker_interval': 60
    }
    resolver = StrategyResolver(config)

    assert resolver.strategy.ticker_interval == 60
    assert ('freqtrade.strategy.resolver',
            logging.INFO,
            'Override strategy \'ticker_interval\' with value in config file: 60.'
            ) in caplog.record_tuples


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
    pair = 'ETH/BTC'

    # Make sure we are using a legacy function
    assert len(resolver.strategy.populate_indicators.__annotations__) == 2
    assert 'dataframe' in resolver.strategy.populate_indicators.__annotations__
    assert 'pair' not in resolver.strategy.populate_indicators.__annotations__
    assert len(resolver.strategy.populate_buy_trend.__annotations__) == 2
    assert 'dataframe' in resolver.strategy.populate_buy_trend.__annotations__
    assert 'pair' not in resolver.strategy.populate_buy_trend.__annotations__
    assert len(resolver.strategy.populate_sell_trend.__annotations__) == 2
    assert 'dataframe' in resolver.strategy.populate_sell_trend.__annotations__
    assert 'pair' not in resolver.strategy.populate_sell_trend.__annotations__

    indicator_df = resolver.strategy.advise_indicators(result, pair=pair)
    assert type(indicator_df) is DataFrame
    assert 'adx' in indicator_df.columns

    buydf = resolver.strategy.advise_buy(result, pair=pair)
    assert type(buydf) is DataFrame
    assert 'buy' in buydf.columns

    selldf = resolver.strategy.advise_sell(result, pair=pair)
    assert type(selldf) is DataFrame
    assert 'sell' in selldf
