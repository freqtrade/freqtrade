# pragma pylint: disable=missing-docstring, protected-access, C0103

import logging

import os

from freqtrade.strategy.interface import IStrategy
from freqtrade.strategy.resolver import StrategyResolver


def test_search_strategy():
    default_location = os.path.join(os.path.dirname(
        os.path.realpath(__file__)), '..', '..', 'strategy'
    )
    assert isinstance(
        StrategyResolver._search_strategy(default_location, 'DefaultStrategy'), IStrategy
    )
    assert StrategyResolver._search_strategy(default_location, 'NotFoundStrategy') is None


def test_load_strategy(result):
    strategy = StrategyResolver()
    strategy.logger = logging.getLogger(__name__)

    assert not hasattr(StrategyResolver, 'custom_strategy')
    strategy._load_strategy('TestStrategy')

    assert not hasattr(StrategyResolver, 'custom_strategy')

    assert hasattr(strategy.custom_strategy, 'populate_indicators')
    assert 'adx' in strategy.populate_indicators(result)


def test_load_not_found_strategy(caplog):
    strategy = StrategyResolver()
    strategy.logger = logging.getLogger(__name__)

    assert not hasattr(StrategyResolver, 'custom_strategy')
    strategy._load_strategy('NotFoundStrategy')

    error_msg = "Impossible to load Strategy '{}'. This class does not " \
                "exist or contains Python code errors".format('NotFoundStrategy')
    assert ('freqtrade.strategy.resolver', logging.ERROR, error_msg) in caplog.record_tuples


def test_strategy(result):
    strategy = StrategyResolver({'strategy': 'DefaultStrategy'})

    assert hasattr(strategy.custom_strategy, 'minimal_roi')
    assert strategy.minimal_roi[0] == 0.04

    assert hasattr(strategy.custom_strategy, 'stoploss')
    assert strategy.stoploss == -0.10

    assert hasattr(strategy.custom_strategy, 'populate_indicators')
    assert 'adx' in strategy.populate_indicators(result)

    assert hasattr(strategy.custom_strategy, 'populate_buy_trend')
    dataframe = strategy.populate_buy_trend(strategy.populate_indicators(result))
    assert 'buy' in dataframe.columns

    assert hasattr(strategy.custom_strategy, 'populate_sell_trend')
    dataframe = strategy.populate_sell_trend(strategy.populate_indicators(result))
    assert 'sell' in dataframe.columns


def test_strategy_override_minimal_roi(caplog):
    caplog.set_level(logging.INFO)
    config = {
        'strategy': 'DefaultStrategy',
        'minimal_roi': {
            "0": 0.5
        }
    }
    strategy = StrategyResolver(config)

    assert hasattr(strategy.custom_strategy, 'minimal_roi')
    assert strategy.minimal_roi[0] == 0.5
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
    strategy = StrategyResolver(config)

    assert hasattr(strategy.custom_strategy, 'stoploss')
    assert strategy.stoploss == -0.5
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
    strategy = StrategyResolver(config)

    assert hasattr(strategy.custom_strategy, 'ticker_interval')
    assert strategy.ticker_interval == 60
    assert ('freqtrade.strategy.resolver',
            logging.INFO,
            'Override strategy \'ticker_interval\' with value in config file: 60.'
            ) in caplog.record_tuples


def test_strategy_fallback_default_strategy():
    strategy = StrategyResolver()
    strategy.logger = logging.getLogger(__name__)

    assert not hasattr(StrategyResolver, 'custom_strategy')
    strategy._load_strategy('../../super_duper')
    assert not hasattr(StrategyResolver, 'custom_strategy')


def test_strategy_singleton():
    strategy1 = StrategyResolver({'strategy': 'DefaultStrategy'})

    assert hasattr(strategy1.custom_strategy, 'minimal_roi')
    assert strategy1.minimal_roi[0] == 0.04

    strategy2 = StrategyResolver()
    assert hasattr(strategy2.custom_strategy, 'minimal_roi')
    assert strategy2.minimal_roi[0] == 0.04
