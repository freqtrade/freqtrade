import importlib
import warnings

import numpy as np
import pandas as pd
import pytest


def test_crossed_numpy_types():
    """
    This test is only present since this method currently diverges from the qtpylib implementation.
    And we must ensure to not break this again once we update from the original source.
    """
    series = pd.Series([56, 97, 19, 76, 65, 25, 87, 91, 79, 79])
    expected_result = pd.Series([False, True, False, True, False, False, True, False, False, False])

    # freqtrade.vendor.qtpylib is a deprecated shim around technical and emits a
    # FutureWarning on import - suppress it here, it's asserted below.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        import freqtrade.vendor.qtpylib.indicators as qtpylib

        assert qtpylib.crossed_above(series, 60).equals(expected_result)
        assert qtpylib.crossed_above(series, 60.0).equals(expected_result)
        assert qtpylib.crossed_above(series, np.int32(60)).equals(expected_result)
        assert qtpylib.crossed_above(series, np.int64(60)).equals(expected_result)
        assert qtpylib.crossed_above(series, np.float64(60.0)).equals(expected_result)


@pytest.mark.filterwarnings("ignore:freqtrade.vendor.qtpylib.indicators' is deprecated")
def test_qtpylib_deprecation():
    """freqtrade.vendor.qtpylib.indicators only re-exports technical now and is deprecated."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        import freqtrade.vendor.qtpylib.indicators as qtpylib

    with pytest.warns(FutureWarning, match="freqtrade.vendor.qtpylib.indicators' is deprecated"):
        importlib.reload(qtpylib)
