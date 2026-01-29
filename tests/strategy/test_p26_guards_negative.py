"""
P26 Negative Test
This test is intended to FAIL.
It attempts to assert that a Lookahead Violation does NOT raise an error.
Since the production code (guards.py) correctly raises ValueError, this test will fail.
The Acceptance Gate will observe this failure to confirm the guard is active.
"""

import pytest
import pandas as pd
import numpy as np
from user_data.strategies.guards import no_lookahead_sanity


def test_negative_control_lookahead_should_fail():
    # Construct a dataframe with a clear lookahead violation (NaN in last row)
    df = pd.DataFrame({"rsi": [50, 55, np.nan]})

    # We assert that this call succeeds (no exception raised).
    # THIS ASSERTION IS WRONG BY DESIGN.
    # The production code WILL raise ValueError, causing this test to FAIL.
    try:
        no_lookahead_sanity(df, ["rsi"])
    except ValueError:
        pytest.fail(
            "Guard raised ValueError as expected, but this test is designed to catch that as a FAILURE of the assertion 'no error raised'."
        )
