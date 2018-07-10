import pandas as pd

from freqtrade.indicator_helpers import went_down, went_up


def test_went_up():
    series = pd.Series([1, 2, 3, 1])
    assert went_up(series).equals(pd.Series([False, True, True, False]))


def test_went_down():
    series = pd.Series([1, 2, 3, 1])
    assert went_down(series).equals(pd.Series([False, False, False, True]))
