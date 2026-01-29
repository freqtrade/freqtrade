"""
Strategy Guards
Enforces data integrity, stale checks, and no-lookahead rules.
"""

import pandas as pd
import numpy as np


def enforce_warmup(df: pd.DataFrame, startup_candle_count: int) -> pd.DataFrame:
    """
    Ensures entry signals are blocked during the warmup period.
    """
    if df.empty or len(df) < startup_candle_count:
        return df

    # Force 0 for first N candles
    # We perform this modification in place or return copy. Returning copy for safety.
    df = df.copy()

    # Columns to zero out if they exist
    cols = ["enter_long", "enter_short"]
    for col in cols:
        if col in df.columns:
            df.loc[: startup_candle_count - 1, col] = 0

    return df


def check_stale_informative(
    df: pd.DataFrame, suffix: str = "", tolerance_seconds: int = 600
) -> pd.Series:
    """
    Checks if informative data is stale compared to main dataframe timestamp.
    Returns a boolean Series (True if STALE).
    """
    if df.empty:
        return pd.Series(dtype=bool)

    date_col = "date"
    info_date_col = f"date{suffix}" if suffix else "date"

    if info_date_col not in df.columns:
        # If informative date is missing, assume safe (or risky? usually safe in single timeframe)
        return pd.Series([False] * len(df), index=df.index)

    # Calculate lag in seconds
    # Assuming datetime objects or compatible
    lag = (df[date_col] - df[info_date_col]).dt.total_seconds()

    # Stale if lag exceeds tolerance
    is_stale = lag > tolerance_seconds
    return is_stale


def no_lookahead_sanity(df: pd.DataFrame, required_cols: list) -> None:
    """
    Ensures required columns do not have NaNs in the last row if sufficient data exists.
    Checks for forward-fill violations (naively, by checking for NaNs that shouldn't be there
    or logic that implies future knowledge - here we check basic availability).

    Actually, a common lookahead issue is accessing .shift(-1).
    We can't easily detect that statically here without AST.

    However, the requirement says:
    "last row must not be NaN for required cols if len(df) >= warmup"
    """
    if df.empty:
        return

    # Assume warmup is passed or implicit?
    # Contract says: "last row must not be NaN for required cols if len(df) >= warmup"
    # But we don't know warmup here easily. Let's assume if data exists at the end, it should be valid.

    last_row = df.iloc[-1]

    for col in required_cols:
        if col not in df.columns:
            continue  # Should rely on other validation for existence

        if pd.isna(last_row[col]):
            raise ValueError(f"Lookahead Sanity Failed: Column '{col}' is NaN in the last row.")
