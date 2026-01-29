"""
Data Integrity Guard (FR-202)
Pure logic validator for OHLCV dataframes.
"""

import pandas as pd
from typing import List


def validate_ohlcv_integrity(df: pd.DataFrame) -> List[str]:
    """
    Checks dataframe for basic integrity violations.
    Returns list of error codes.
    """
    violations = []

    if df.empty:
        return violations

    # 1. Monotonic Time
    if not df["date"].is_monotonic_increasing:
        violations.append("NON_MONOTONIC_TIME")

    # 2. Duplicate Timestamps
    if df["date"].duplicated().any():
        violations.append("DUP_TIMESTAMP")

    # 3. Negative Volume
    if (df["volume"] < 0).any():
        violations.append("NEG_VOLUME")

    # 4. OHLC Sanity
    # High must be >= Open, Close, Low
    # Low must be <= Open, Close, High
    # ( vectorized checks )

    high_violation = (
        (df["high"] < df["open"]) | (df["high"] < df["close"]) | (df["high"] < df["low"])
    )
    if high_violation.any():
        violations.append("OHLC_SANITY_HIGH")

    low_violation = (df["low"] > df["open"]) | (df["low"] > df["close"]) | (df["low"] > df["high"])
    if low_violation.any():
        violations.append("OHLC_SANITY_LOW")

    return violations
