"""
P27 Data Integrity Tests
"""

import pytest
import pandas as pd
from datetime import datetime
from user_data.strategies.data_integrity_fr202 import validate_ohlcv_integrity


def test_valid_df():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-01 10:00", "2025-01-01 10:05"]),
            "open": [100, 101],
            "high": [102, 103],
            "low": [99, 100],
            "close": [101, 102],
            "volume": [1000, 2000],
        }
    )
    violations = validate_ohlcv_integrity(df)
    assert len(violations) == 0


def test_non_monotonic():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-01 10:05", "2025-01-01 10:00"]),
            "volume": [1, 1],
            "open": [1, 1],
            "high": [1, 1],
            "low": [1, 1],
            "close": [1, 1],
        }
    )
    violations = validate_ohlcv_integrity(df)
    assert "NON_MONOTONIC_TIME" in violations


def test_ohlc_sanity():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-01 10:00"]),
            "open": [100],
            "high": [90],
            "low": [99],
            "close": [101],  # High < Open!
            "volume": [100],
        }
    )
    violations = validate_ohlcv_integrity(df)
    assert "OHLC_SANITY_HIGH" in violations
