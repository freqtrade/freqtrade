import json
import pytest
from pandas import DataFrame
from freqtrade.strategy.default_strategy import DefaultStrategy, class_name
from freqtrade.analyze import parse_ticker_dataframe


@pytest.fixture
def result():
    with open('freqtrade/tests/testdata/BTC_ETH-1.json') as data_file:
        return parse_ticker_dataframe(json.load(data_file))


def test_default_strategy_class_name():
    assert class_name == DefaultStrategy.__name__


def test_default_strategy_structure():
    assert hasattr(DefaultStrategy, 'minimal_roi')
    assert hasattr(DefaultStrategy, 'stoploss')
    assert hasattr(DefaultStrategy, 'populate_indicators')
    assert hasattr(DefaultStrategy, 'populate_buy_trend')
    assert hasattr(DefaultStrategy, 'populate_sell_trend')
    assert hasattr(DefaultStrategy, 'hyperopt_space')
    assert hasattr(DefaultStrategy, 'buy_strategy_generator')


def test_default_strategy(result):
    strategy = DefaultStrategy()

    assert type(strategy.minimal_roi) is dict
    assert type(strategy.stoploss) is float
    indicators = strategy.populate_indicators(result)
    assert type(indicators) is DataFrame
    assert type(strategy.populate_buy_trend(indicators)) is DataFrame
    assert type(strategy.populate_sell_trend(indicators)) is DataFrame
    assert type(strategy.hyperopt_space()) is dict
    assert callable(strategy.buy_strategy_generator({}))
