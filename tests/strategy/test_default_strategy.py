from pandas import DataFrame

from .strats.default_strategy import DefaultStrategy


def test_default_strategy_structure():
    assert hasattr(DefaultStrategy, 'minimal_roi')
    assert hasattr(DefaultStrategy, 'stoploss')
    assert hasattr(DefaultStrategy, 'timeframe')
    assert hasattr(DefaultStrategy, 'populate_indicators')
    assert hasattr(DefaultStrategy, 'populate_buy_trend')
    assert hasattr(DefaultStrategy, 'populate_sell_trend')


def test_default_strategy(result):
    strategy = DefaultStrategy({})

    metadata = {'pair': 'ETH/BTC'}
    assert type(strategy.minimal_roi) is dict
    assert type(strategy.stoploss) is float
    assert type(strategy.timeframe) is str
    indicators = strategy.populate_indicators(result, metadata)
    assert type(indicators) is DataFrame
    assert type(strategy.populate_buy_trend(indicators, metadata)) is DataFrame
    assert type(strategy.populate_sell_trend(indicators, metadata)) is DataFrame
