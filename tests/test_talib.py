import pandas as pd
import talib.abstract as ta


def test_talib_bollingerbands_near_zero_values():
    inputs = pd.DataFrame(
        [
            {"close": 0.00000010},
            {"close": 0.00000011},
            {"close": 0.00000012},
            {"close": 0.00000013},
            {"close": 0.00000014},
        ]
    )
    bollinger = ta.BBANDS(inputs, matype=0, timeperiod=2)
    assert bollinger["upperband"][3] != bollinger["middleband"][3]
