# pragma pylint: disable=missing-docstring, C0103

import pandas

from freqtrade.optimize import load_data
from freqtrade.strategy.resolver import StrategyResolver

_pairs = ['ETH/BTC']


def load_dataframe_pair(pairs, strategy):
    ld = load_data(None, ticker_interval='5m', pairs=pairs)
    assert isinstance(ld, dict)
    assert isinstance(pairs[0], str)
    dataframe = ld[pairs[0]]

    dataframe = strategy.analyze_ticker(dataframe)
    return dataframe


def test_dataframe_load():
    strategy = StrategyResolver({'strategy': 'DefaultStrategy'}).strategy
    dataframe = load_dataframe_pair(_pairs, strategy)
    assert isinstance(dataframe, pandas.core.frame.DataFrame)


def test_dataframe_columns_exists():
    strategy = StrategyResolver({'strategy': 'DefaultStrategy'}).strategy
    dataframe = load_dataframe_pair(_pairs, strategy)
    assert 'high' in dataframe.columns
    assert 'low' in dataframe.columns
    assert 'close' in dataframe.columns
