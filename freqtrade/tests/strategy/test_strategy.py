# pragma pylint: disable=missing-docstring, protected-access, C0103
import logging
import os

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
    assert hasattr(resolver.strategy, 'populate_indicators')
    assert 'adx' in resolver.strategy.populate_indicators(result)


def test_load_strategy_invalid_directory(result, caplog):
    resolver = StrategyResolver()
    extra_dir = os.path.join('some', 'path')
    resolver._load_strategy('TestStrategy', extra_dir)

    assert (
        'freqtrade.strategy.resolver',
        logging.WARNING,
        'Path "{}" does not exist'.format(extra_dir),
    ) in caplog.record_tuples

    assert hasattr(resolver.strategy, 'populate_indicators')
    assert 'adx' in resolver.strategy.populate_indicators(result)


def test_load_not_found_strategy():
    strategy = StrategyResolver()
    with pytest.raises(ImportError,
                       match=r'Impossible to load Strategy \'NotFoundStrategy\'.'
                             r' This class does not exist or contains Python code errors'):
        strategy._load_strategy(strategy_name='NotFoundStrategy', config={})


def test_strategy(result):
    resolver = StrategyResolver({'strategy': 'DefaultStrategy'})

    assert hasattr(resolver.strategy, 'minimal_roi')
    assert resolver.strategy.minimal_roi[0] == 0.04

    assert hasattr(resolver.strategy, 'stoploss')
    assert resolver.strategy.stoploss == -0.10

    assert hasattr(resolver.strategy, 'populate_indicators')
    assert 'adx' in resolver.strategy.populate_indicators(result)

    assert hasattr(resolver.strategy, 'populate_buy_trend')
    dataframe = resolver.strategy.populate_buy_trend(resolver.strategy.populate_indicators(result))
    assert 'buy' in dataframe.columns

    assert hasattr(resolver.strategy, 'populate_sell_trend')
    dataframe = resolver.strategy.populate_sell_trend(resolver.strategy.populate_indicators(result))
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

    assert hasattr(resolver.strategy, 'minimal_roi')
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

    assert hasattr(resolver.strategy, 'stoploss')
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

    assert hasattr(resolver.strategy, 'ticker_interval')
    assert resolver.strategy.ticker_interval == 60
    assert ('freqtrade.strategy.resolver',
            logging.INFO,
            'Override strategy \'ticker_interval\' with value in config file: 60.'
            ) in caplog.record_tuples
