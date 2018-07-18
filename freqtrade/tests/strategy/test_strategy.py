# pragma pylint: disable=missing-docstring, protected-access, C0103
import logging
import os
import warnings

import pytest

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
    default_location = os.path.join(os.path.dirname(
        os.path.realpath(__file__)), '..', '..', 'strategy'
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
    assert 'adx' in resolver.strategy.advise_indicators(result, pair=pair)


def test_load_strategy_invalid_directory(result, caplog):
    resolver = StrategyResolver()
    extra_dir = os.path.join('some', 'path')
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
    resolver = StrategyResolver({'strategy': 'TestStrategy'})
    with warnings.catch_warnings(record=True) as w:
        # Cause all warnings to always be triggered.
        warnings.simplefilter("always")
        resolver.strategy.populate_indicators(result)
        assert len(w) == 1
        assert issubclass(w[-1].category, DeprecationWarning)
        assert "deprecated - please replace this method with advise_indicators!" in str(
            w[-1].message)


def test_deprecate_populate_buy_trend(result):
    resolver = StrategyResolver({'strategy': 'TestStrategy'})
    with warnings.catch_warnings(record=True) as w:
        # Cause all warnings to always be triggered.
        warnings.simplefilter("always")
        resolver.strategy.populate_buy_trend(result)
        assert len(w) == 1
        assert issubclass(w[-1].category, DeprecationWarning)
        assert "deprecated - please replace this method with advise_buy!" in str(
            w[-1].message)


def test_deprecate_populate_sell_trend(result):
    resolver = StrategyResolver({'strategy': 'TestStrategy'})
    with warnings.catch_warnings(record=True) as w:
        # Cause all warnings to always be triggered.
        warnings.simplefilter("always")
        resolver.strategy.populate_sell_trend(result)
        assert len(w) == 1
        assert issubclass(w[-1].category, DeprecationWarning)
        assert "deprecated - please replace this method with advise_sell!" in str(
            w[-1].message)
