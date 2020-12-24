import numpy as np
import pandas as pd

import freqtrade.vendor.qtpylib.indicators as qtpylib


def test_crossed_numpy_types():
    """
    This test is only present since this method currently diverges from the qtpylib implementation.
    And we must ensure to not break this again once we update from the original source.
    """
    series = pd.Series([56, 97, 19, 76, 65, 25, 87, 91, 79, 79])
    expected_result = pd.Series([False, True, False, True, False, False, True, False, False, False])

    assert qtpylib.crossed_above(series, 60).equals(expected_result)
    assert qtpylib.crossed_above(series, 60.0).equals(expected_result)
    assert qtpylib.crossed_above(series, np.int32(60)).equals(expected_result)
    assert qtpylib.crossed_above(series, np.int64(60)).equals(expected_result)
    assert qtpylib.crossed_above(series, np.float64(60.0)).equals(expected_result)
