# pragma pylint: disable=missing-docstring, C0103

import pandas
from freqtrade.optimize import load_data
from freqtrade.analyze import Analyze

_pairs = ['BTC_ETH']


def load_dataframe_pair(pairs):
    ld = load_data(None, ticker_interval=5, pairs=pairs)
    assert isinstance(ld, dict)
    assert isinstance(pairs[0], str)
    dataframe = ld[pairs[0]]

    analyze = Analyze({'strategy': 'default_strategy'})
    dataframe = analyze.analyze_ticker(dataframe)
    return dataframe


def test_dataframe_load():
    dataframe = load_dataframe_pair(_pairs)
    assert isinstance(dataframe, pandas.core.frame.DataFrame)


def test_dataframe_columns_exists():
    dataframe = load_dataframe_pair(_pairs)
    assert 'high' in dataframe.columns
    assert 'low' in dataframe.columns
    assert 'close' in dataframe.columns
