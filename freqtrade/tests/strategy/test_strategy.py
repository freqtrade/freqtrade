# pragma pylint: disable=missing-docstring, protected-access, C0103

import logging
from freqtrade.strategy.strategy import Strategy


def test_sanitize_module_name():
    assert Strategy._sanitize_module_name('default_strategy') == 'default_strategy'
    assert Strategy._sanitize_module_name('default_strategy.py') == 'default_strategy'
    assert Strategy._sanitize_module_name('../default_strategy.py') == 'default_strategy'
    assert Strategy._sanitize_module_name('../default_strategy') == 'default_strategy'
    assert Strategy._sanitize_module_name('.default_strategy') == '.default_strategy'
    assert Strategy._sanitize_module_name('foo-bar') == 'foo-bar'
    assert Strategy._sanitize_module_name('foo/bar') == 'bar'


def test_search_strategy():
    assert Strategy._search_strategy('default_strategy') == '.'
    assert Strategy._search_strategy('test_strategy') == 'user_data.strategies.'
    assert Strategy._search_strategy('super_duper') is None


def test_strategy_structure():
    assert hasattr(Strategy, 'init')
    assert hasattr(Strategy, 'populate_indicators')
    assert hasattr(Strategy, 'populate_buy_trend')
    assert hasattr(Strategy, 'populate_sell_trend')


def test_load_strategy(result):
    strategy = Strategy()
    strategy.logger = logging.getLogger(__name__)

    assert not hasattr(Strategy, 'custom_strategy')
    strategy._load_strategy('test_strategy')

    assert not hasattr(Strategy, 'custom_strategy')

    assert hasattr(strategy.custom_strategy, 'populate_indicators')
    assert 'adx' in strategy.populate_indicators(result)


def test_load_not_found_strategy(caplog):
    strategy = Strategy()
    strategy.logger = logging.getLogger(__name__)

    assert not hasattr(Strategy, 'custom_strategy')
    strategy._load_strategy('NotFoundStrategy')

    error_msg = "Impossible to load Strategy 'user_data/strategies/{}.py'. This file does not " \
                "exist or contains Python code errors".format('NotFoundStrategy')
    assert ('test_strategy', logging.ERROR, error_msg) in caplog.record_tuples


def test_strategy(result):
    strategy = Strategy()
    strategy.init({'strategy': 'default_strategy'})

    assert hasattr(strategy.custom_strategy, 'minimal_roi')
    assert strategy.minimal_roi['0'] == 0.04

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
        'strategy': 'default_strategy',
        'minimal_roi': {
            "0": 0.5
        }
    }
    strategy = Strategy()
    strategy.init(config)

    assert hasattr(strategy.custom_strategy, 'minimal_roi')
    assert strategy.minimal_roi['0'] == 0.5
    assert ('freqtrade.strategy.strategy',
            logging.INFO,
            'Override strategy \'minimal_roi\' with value in config file.'
            ) in caplog.record_tuples


def test_strategy_override_stoploss(caplog):
    caplog.set_level(logging.INFO)
    config = {
        'strategy': 'default_strategy',
        'stoploss': -0.5
    }
    strategy = Strategy()
    strategy.init(config)

    assert hasattr(strategy.custom_strategy, 'stoploss')
    assert strategy.stoploss == -0.5
    assert ('freqtrade.strategy.strategy',
            logging.INFO,
            'Override strategy \'stoploss\' with value in config file: -0.5.'
            ) in caplog.record_tuples


def test_strategy_override_ticker_interval(caplog):
    caplog.set_level(logging.INFO)

    config = {
        'strategy': 'default_strategy',
        'ticker_interval': 60
    }
    strategy = Strategy()
    strategy.init(config)

    assert hasattr(strategy.custom_strategy, 'ticker_interval')
    assert strategy.ticker_interval == 60
    assert ('freqtrade.strategy.strategy',
            logging.INFO,
            'Override strategy \'ticker_interval\' with value in config file: 60.'
            ) in caplog.record_tuples


def test_strategy_fallback_default_strategy():
    strategy = Strategy()
    strategy.logger = logging.getLogger(__name__)

    assert not hasattr(Strategy, 'custom_strategy')
    strategy._load_strategy('../../super_duper')
    assert not hasattr(Strategy, 'custom_strategy')


def test_strategy_singleton():
    strategy1 = Strategy()
    strategy1.init({'strategy': 'default_strategy'})

    assert hasattr(strategy1.custom_strategy, 'minimal_roi')
    assert strategy1.minimal_roi['0'] == 0.04

    strategy2 = Strategy()
    assert hasattr(strategy2.custom_strategy, 'minimal_roi')
    assert strategy2.minimal_roi['0'] == 0.04
