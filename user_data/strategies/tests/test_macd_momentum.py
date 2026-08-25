"""TDD tests for MacdMomentum, run directly against real indicator computation.

Expected signal locations are derived programmatically from the strategy's own computed
macd/macdsignal columns (never hand-calculated), then compared row-for-row against the
strategy's actual enter_long/exit_long columns. Each fixture also asserts it actually
exercises the case it claims to, so a passing test can't be vacuous. Mirrors
test_ema_trend_follow.py's established pattern.
"""

import sys
from pathlib import Path

import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from MacdMomentum import MacdMomentum


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
    strategy = MacdMomentum({"stake_currency": "USDT", "strategy": "MacdMomentum"})
    dataframe = _levels_to_df(levels)
    metadata = {"pair": "BTC/USDT"}
    dataframe = strategy.populate_indicators(dataframe, metadata)
    dataframe = strategy.populate_entry_trend(dataframe, metadata)
    dataframe = strategy.populate_exit_trend(dataframe, metadata)
    # populate_entry_trend/populate_exit_trend only set 1 where a signal fires (matching
    # freqtrade's own strategy convention) -- everywhere else is NaN. Normalize to
    # int(0/1) here so the test can compare directly.
    for col in ("enter_long", "exit_long"):
        dataframe[col] = dataframe[col].fillna(0).astype(int)
    return dataframe


def _expected_cross_up(df: pd.DataFrame) -> pd.Series:
    macd, signal = df["macd"], df["macdsignal"]
    return (macd > signal) & (macd.shift(1) <= signal.shift(1))


def _expected_cross_down(df: pd.DataFrame) -> pd.Series:
    macd, signal = df["macd"], df["macdsignal"]
    return (macd < signal) & (macd.shift(1) >= signal.shift(1))


def test_enters_on_macd_cross_above_signal_and_exits_on_cross_below():
    # Long flat warmup so MACD/signal both settle near zero, then a sustained rally
    # (MACD line rises above its signal line), then a symmetric decline back down (MACD
    # crosses back below signal).
    levels = _flat(100.0, 60) + _ramp(100.0, 180.0, 40) + _ramp(180.0, 100.0, 40)
    df = _run_strategy(levels)

    expected_up = _expected_cross_up(df)
    expected_down = _expected_cross_down(df)

    assert expected_up.sum() >= 1, "fixture must contain at least one bullish MACD crossover"
    assert expected_down.sum() >= 1, "fixture must contain at least one bearish MACD crossover"

    assert (df["enter_long"] == expected_up.astype(int)).all()
    assert (df["exit_long"] == expected_down.astype(int)).all()
    assert df["enter_long"].sum() >= 1
    assert df["exit_long"].sum() >= 1


def test_flat_price_produces_no_crossovers_or_signals():
    # A perfectly flat series never moves MACD/signal apart -- no crossovers, no
    # signals. Proves the strategy doesn't fire spuriously on a dataframe with zero real
    # momentum.
    levels = _flat(100.0, 80)
    df = _run_strategy(levels)

    assert df["enter_long"].sum() == 0
    assert df["exit_long"].sum() == 0
