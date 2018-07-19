import json

import pytest
from pandas import DataFrame

from freqtrade.exchange.exchange_helpers import parse_ticker_dataframe
from freqtrade.strategy.default_strategy import DefaultStrategy


@pytest.fixture
def result():
    with open('freqtrade/tests/testdata/ETH_BTC-1m.json') as data_file:
        return parse_ticker_dataframe(json.load(data_file))


def test_default_strategy_structure():
    assert hasattr(DefaultStrategy, 'minimal_roi')
    assert hasattr(DefaultStrategy, 'stoploss')
    assert hasattr(DefaultStrategy, 'ticker_interval')
    assert hasattr(DefaultStrategy, 'populate_indicators')
    assert hasattr(DefaultStrategy, 'populate_buy_trend')
    assert hasattr(DefaultStrategy, 'populate_sell_trend')


def test_default_strategy(result):
    strategy = DefaultStrategy({})

    assert type(strategy.minimal_roi) is dict
    assert type(strategy.stoploss) is float
    assert type(strategy.ticker_interval) is str
    indicators = strategy.populate_indicators(result)
    assert type(indicators) is DataFrame
    assert type(strategy.populate_buy_trend(indicators)) is DataFrame
    assert type(strategy.populate_sell_trend(indicators)) is DataFrame
