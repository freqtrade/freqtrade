"""TDD tests for EmaTrendFollow, run directly against real indicator computation.

Expected signal locations are derived programmatically from the strategy's own computed
EMA columns (never hand-calculated), then compared row-for-row against the strategy's
actual enter_long/exit_long columns. Each fixture also asserts it actually exercises the
case it claims to (a nonzero count of crossings) so a passing test can't be vacuous.
"""

import sys
from pathlib import Path

import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from EmaTrendFollow import EmaTrendFollow


def _levels_to_df(levels: list[float]) -> pd.DataFrame:
    n = len(levels)
    dates = pd.date_range(start="2020-01-01", periods=n, freq="1h", tz="UTC")
    close = pd.Series(levels, dtype=float)
    return pd.DataFrame(
        {
            "date": dates,
            "open": close,
            "high": close + 0.01,
            "low": close - 0.01,
            "close": close,
            "volume": 1000.0,
        }
    )


def _flat(level: float, n: int) -> list[float]:
    return [level] * n


def _ramp(start: float, end: float, n: int) -> list[float]:
    step = (end - start) / n
    return [start + step * i for i in range(1, n + 1)]


def _run_strategy(levels: list[float]) -> pd.DataFrame:
    strategy = EmaTrendFollow({"stake_currency": "USDT", "strategy": "EmaTrendFollow"})
    dataframe = _levels_to_df(levels)
    metadata = {"pair": "BTC/USDT"}
    dataframe = strategy.populate_indicators(dataframe, metadata)
    dataframe = strategy.populate_entry_trend(dataframe, metadata)
    dataframe = strategy.populate_exit_trend(dataframe, metadata)
    # populate_entry_trend/populate_exit_trend only set 1 where a signal fires (matching
    # freqtrade's own strategy convention, see StrategyTestV3) -- everywhere else is NaN,
    # which freqtrade's own `dataframe["enter_long"] == 1` checks treat identically to 0
    # ("no signal"). Normalize to int(0/1) here so the test can compare directly.
    for col in ("enter_long", "exit_long"):
        dataframe[col] = dataframe[col].fillna(0).astype(int)
    return dataframe


def _expected_cross_up(df: pd.DataFrame) -> pd.Series:
    fast, slow = df["ema_fast"], df["ema_slow"]
    return (fast > slow) & (fast.shift(1) <= slow.shift(1))


def _expected_cross_down(df: pd.DataFrame) -> pd.Series:
    fast, slow = df["ema_fast"], df["ema_slow"]
    return (fast < slow) & (fast.shift(1) >= slow.shift(1))


def test_enters_on_ema_crossover_above_trend_filter_and_exits_on_reverse_crossover():
    # Long flat warmup so EMA200 fully settles at 100, then a sustained rally (fast EMA
    # crosses above slow EMA while price is already above the barely-moved EMA200), then
    # a symmetric decline back down (fast EMA crosses back below slow EMA).
    levels = _flat(100.0, 250) + _ramp(100.0, 180.0, 40) + _ramp(180.0, 100.0, 40)
    df = _run_strategy(levels)

    expected_up = _expected_cross_up(df)
    expected_down = _expected_cross_down(df)
    expected_entry = expected_up & (df["close"] > df["ema_trend"])

    assert expected_up.sum() >= 1, "fixture must contain at least one bullish EMA crossover"
    assert expected_down.sum() >= 1, "fixture must contain at least one bearish EMA crossover"

    assert (df["enter_long"] == expected_entry.astype(int)).all()
    assert (df["exit_long"] == expected_down.astype(int)).all()
    assert df["enter_long"].sum() >= 1
    assert df["exit_long"].sum() >= 1


def test_crossover_below_trend_filter_does_not_enter():
    # High flat warmup (EMA200 settles at 300), a long decline, then a modest local
    # bounce -- enough to flip the fast/slow EMA crossover, but EMA200 (barely moved from
    # 300 over this short a decline) stays far above the bounce's price level.
    levels = _flat(300.0, 250) + _ramp(300.0, 120.0, 60) + _ramp(120.0, 220.0, 40)
    df = _run_strategy(levels)

    expected_up = _expected_cross_up(df)
    expected_entry = expected_up & (df["close"] > df["ema_trend"])

    assert expected_up.sum() >= 1, "fixture must contain a bullish EMA crossover to be blocked"
    assert (df.loc[expected_up, "close"] <= df.loc[expected_up, "ema_trend"]).all(), (
        "fixture must place the crossover below the trend filter to test blocking"
    )

    assert (df["enter_long"] == expected_entry.astype(int)).all()
    assert df["enter_long"].sum() == 0
