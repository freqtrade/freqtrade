import pandas

from freqtrade import analyze
import freqtrade.optimize

_pairs = ['BTC_ETH']


def load_dataframe_pair(pairs):
    ld = freqtrade.optimize.load_data(None, ticker_interval=5, pairs=pairs)
    assert isinstance(ld, dict)
    assert isinstance(pairs[0], str)
    dataframe = ld[pairs[0]]
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
