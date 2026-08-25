"""TDD tests for BandtasticMeanReversion, run directly against real indicator
computation.

Expected signal locations are derived programmatically from the strategy's own computed
rsi/bb_lowerband/bb_upperband columns (never hand-calculated), then compared row-for-row
against the strategy's actual enter_long/exit_long columns. Each fixture also asserts it
actually exercises the case it claims to, so a passing test can't be vacuous. Mirrors
test_ema_trend_follow.py's established pattern.
"""

import sys
from pathlib import Path

import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from BandtasticMeanReversion import BandtasticMeanReversion


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


def _sine_chop(level: float, amplitude: float, n: int) -> list[float]:
    import math

    return [level + amplitude * math.sin(i * 0.9) for i in range(n)]


def _run_strategy(levels: list[float]) -> pd.DataFrame:
    strategy = BandtasticMeanReversion(
        {"stake_currency": "USDT", "strategy": "BandtasticMeanReversion"}
    )
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


def test_enters_on_oversold_rsi_below_lower_band_and_exits_on_overbought_above_upper_band():
    # Flat warmup settles the Bollinger band, then a sharp decline drives RSI down and
    # price below the lower band (entry), then a sharp rally drives RSI up and price
    # above the upper band (exit).
    levels = _flat(100.0, 60) + _ramp(100.0, 60.0, 15) + _ramp(60.0, 140.0, 15)
    df = _run_strategy(levels)
    rsi_buy = BandtasticMeanReversion.rsi_buy.value
    rsi_sell = BandtasticMeanReversion.rsi_sell.value

    expected_entry = (df["rsi"] < rsi_buy) & (df["close"] < df["bb_lowerband"]) & (df["volume"] > 0)
    expected_exit = (df["rsi"] > rsi_sell) & (df["close"] > df["bb_upperband"]) & (df["volume"] > 0)

    assert expected_entry.sum() >= 1, "fixture must contain at least one oversold entry"
    assert expected_exit.sum() >= 1, "fixture must contain at least one overbought exit"

    assert (df["enter_long"] == expected_entry.astype(int)).all()
    assert (df["exit_long"] == expected_exit.astype(int)).all()


def test_oversold_rsi_alone_without_lower_band_breach_does_not_enter():
    # Gentle chop keeps RSI oscillating (occasionally below rsi_buy) but never moves
    # price far enough to breach the (wider, volatility-scaled) Bollinger lower band --
    # proves the AND-gate between RSI and the band actually filters signals, not just
    # RSI alone.
    levels = _flat(100.0, 60) + _sine_chop(100.0, 3.0, 80)
    df = _run_strategy(levels)
    rsi_buy = BandtasticMeanReversion.rsi_buy.value

    rsi_alone = df["rsi"] < rsi_buy
    expected_entry = rsi_alone & (df["close"] < df["bb_lowerband"]) & (df["volume"] > 0)

    assert rsi_alone.sum() >= 1, "fixture must dip RSI below rsi_buy at least once"
    assert rsi_alone.sum() > expected_entry.sum(), (
        "fixture must have at least one RSI-oversold row where the band doesn't also "
        "breach, or this test doesn't actually exercise the AND-gate"
    )

    assert (df["enter_long"] == expected_entry.astype(int)).all()
