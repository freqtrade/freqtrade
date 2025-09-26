from __future__ import annotations

import numpy as np
import pandas as pd

from freqtrade_ext.feature_store import _derive_features, _minute_last_right_open, _safe_shift


def _make_sec_index(start: str, seconds: int) -> pd.DatetimeIndex:
    return pd.date_range(start=start, periods=seconds, freq="S", tz="UTC")


def test_minute_last_right_open_with_embargo():
    # Create 120 seconds across 2 minutes
    idx = _make_sec_index("2025-01-01 00:00:00+00:00", 120)
    df = pd.DataFrame({"spread": np.arange(len(idx)), "mid": 100.0}, index=idx)
    out = _minute_last_right_open(df, embargo_secs=5)
    # For each minute, last usable second is 54s and 1m54s
    assert out.index[0].minute == 0
    assert out.index[1].minute == 1
    # Check values correspond to second=54 and 1m54 -> values 54 and 114
    assert out.iloc[0]["spread"] == 54
    assert out.iloc[1]["spread"] == 114


def test_safe_shift_applies_to_all_columns():
    idx = _make_sec_index("2025-01-01 00:00:00+00:00", 3)
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}, index=idx)
    shifted = _safe_shift(df, ["a", "b"], n=1)
    assert pd.isna(shifted.iloc[0]["a"]) and pd.isna(shifted.iloc[0]["b"])  # first row NaN
    assert shifted.iloc[2]["a"] == 2 and shifted.iloc[2]["b"] == 5


def test_derive_features_basic():
    idx = _make_sec_index("2025-01-01 00:00:00+00:00", 3)
    df = pd.DataFrame(
        {
            "best_bid": [99, 100, 101],
            "best_ask": [101, 102, 103],
            "spread": [2, 2, 2],
            "mid": [100, 101, 102],
            "imb_200": [0.5, 0.6, 0.7],
            "depth_delta_200": [1.0, -0.5, 0.1],
            "top_bid_qty": [10, 11, 12],
            "top_ask_qty": [9, 8, 7],
        },
        index=idx,
    )
    out = _derive_features(df, depth=200)
    # New columns exist
    for col in [
        "rel_spread",
        "spread_bps",
        "ob_imbalance",
        "ob_depth_delta",
        "microprice",
        "ofi_top",
    ]:
        assert col in out.columns
    # No inf values
    assert not np.isinf(out.replace({pd.NA: np.nan}).select_dtypes(float).to_numpy()).any()
