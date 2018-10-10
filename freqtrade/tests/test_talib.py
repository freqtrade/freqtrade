import math
import talib.abstract as ta
import pandas as pd


def test_talib_bollingerbands_near_zero_values():
    inputs = pd.DataFrame([
        {'close': 0.00000010},
        {'close': 0.00000011},
        {'close': 0.00000012},
        {'close': 0.00000013},
        {'close': 0.00000014}
    ])
    bollinger = ta.BBANDS(inputs, matype=0, timeperiod=2)
    upper_band = bollinger['upperband'][3]
    middle_band = bollinger['middleband'][3]
    assert not math.isclose(upper_band, middle_band)
