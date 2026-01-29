"""
Tests for P26 Guards
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from user_data.strategies.guards import enforce_warmup, check_stale_informative, no_lookahead_sanity


def test_enforce_warmup():
    df = pd.DataFrame(
        {"enter_long": [1, 1, 1, 1, 1], "enter_short": [1, 1, 1, 1, 1], "volume": [100] * 5}
    )

    # Warmup 3
    warmed = enforce_warmup(df, 3)

    # First 3 should be 0 (index 0, 1, 2)
    assert warmed.loc[0, "enter_long"] == 0
    assert warmed.loc[2, "enter_long"] == 0
    # 4th should remain 1 (index 3)
    assert warmed.loc[3, "enter_long"] == 1


def test_check_stale_informative():
    now = datetime(2025, 1, 1, 12, 0)

    df = pd.DataFrame(
        {
            "date": [now, now + timedelta(minutes=5)],
            "date_info": [now, now],  # Second row is 5 mins stale
        }
    )

    # Tolerance 4 mins (240s)
    stale = check_stale_informative(df, suffix="_info", tolerance_seconds=240)

    assert stale[0] == False  # lag 0
    assert stale[1] == True  # lag 5 mins > 4 mins


def test_no_lookahead_sanity_valid():
    df = pd.DataFrame({"rsi": [50, 55, 60]})
    # fast forward fill check - last row valid
    no_lookahead_sanity(df, ["rsi"])


def test_no_lookahead_sanity_violation():
    df = pd.DataFrame(
        {
            "rsi": [50, 55, np.nan]  # Last row NaN
        }
    )

    with pytest.raises(ValueError, match="Lookahead Sanity Failed"):
        no_lookahead_sanity(df, ["rsi"])
