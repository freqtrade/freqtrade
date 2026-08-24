# research/regime.py
"""Regime breakdown: labels each walk-forward window's out-of-sample test period by the
market conditions it actually occurred in (Trend x Volatility), and aggregates a gate's
window-level results by that label. Purely informational -- never changes
GateResult.passed. See docs/superpowers/specs/2026-08-24-regime-breakdown-design.md for
full reasoning, including why this classifier is NOT safe to reuse for a live/production
regime-switching signal: it ranks each window against the full-sample median of every
window in this run, including ones chronologically after it -- fine for a one-shot,
post-hoc report generated after the whole backtest already ran, not causal.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from freqtrade.configuration import TimeRange
from freqtrade.data import history
from research.walkforward import Window, WindowResult


def classify_regimes(
    pair: str,
    timeframe: str,
    datadir: Path,
    windows: list[Window],
    # ponytail: starting default, not empirically derived -- adjust based on real usage
    # once this runs against real strategies.
    trend_threshold: float = 0.05,
) -> list[str]:
    """Label each window's test period "{Trend}/{Volatility}", e.g. "Bull/High".

    Trend: "Bull" if the test period's total return exceeds `trend_threshold`, "Bear" if
    below -`trend_threshold`, "Sideways" otherwise.

    Volatility: "High" if this window's realized volatility (std of per-candle pct
    change) is strictly above the median realized volatility across every window in
    `windows`, "Low" otherwise (a window sitting exactly on the median is "Low" by
    construction -- ">", not ">=").

    Self-referential to this run's own windows (no external volatility index needed) --
    honest about what it is: "more/less volatile than this backtest's other periods," not
    "objectively high/low."

    A window whose test period has fewer than 2 candles can't compute a return or
    volatility; it fails closed to total_return=0.0, realized_vol=0.0 (classified
    "Sideways" on the trend axis by construction, since 0.0 is inside
    [-trend_threshold, trend_threshold]).

    Returns one label per window, same order as `windows`.
    """
    if not windows:
        raise ValueError("windows must not be empty")

    returns: list[float] = []
    vols: list[float] = []
    for window in windows:
        timerange = TimeRange(
            "date", "date", int(window.test_start.timestamp()), int(window.test_end.timestamp())
        )
        data = history.load_data(
            datadir=datadir, timeframe=timeframe, pairs=[pair], timerange=timerange
        )
        # freqtrade's history.load_data only inserts a pair into the returned dict when
        # the loaded frame is non-empty -- a window whose test period contains zero
        # candles leaves `pair` absent entirely, so guard before indexing rather than
        # let a KeyError bypass the `len(close) < 2` fail-closed path below.
        close = data[pair]["close"] if pair in data else pd.Series(dtype=float)
        if len(close) < 2:
            returns.append(0.0)
            vols.append(0.0)
            continue
        returns.append(float(close.iloc[-1] / close.iloc[0] - 1))
        pct_changes = close.pct_change().dropna()
        # A single pct-change value (i.e. exactly 2 candles) has an undefined sample std
        # (pandas' default ddof=1 divides by n-1=0, producing NaN) -- fail closed to 0.0
        # rather than let a NaN silently poison np.median(vols) for the whole run.
        vols.append(float(pct_changes.std()) if len(pct_changes) >= 2 else 0.0)

    median_vol = float(np.median(vols))

    labels = []
    for total_return, realized_vol in zip(returns, vols, strict=True):
        if total_return > trend_threshold:
            trend = "Bull"
        elif total_return < -trend_threshold:
            trend = "Bear"
        else:
            trend = "Sideways"
        volatility = "High" if realized_vol > median_vol else "Low"
        labels.append(f"{trend}/{volatility}")
    return labels


def regime_report(window_results: list[WindowResult], labels: list[str]) -> dict[str, dict]:
    """Group `window_results` by their parallel `labels` entry and aggregate each group.

    Raises ValueError if len(window_results) != len(labels) -- mismatched parallel lists
    is a caller-contract violation, not a data problem.

    Returns {label: {"n_windows": int, "n_trades": int, "mean_test_sharpe": float,
    "total_return": float}}. `total_return` is the plain arithmetic sum of every trade's
    fractional return across the group's windows -- NOT a geometrically compounded
    return, a rough same-units aggregate for comparing regime buckets against each
    other, not a claim about realized account growth.

    WindowResult.test_sharpe is never NaN for a zero-trade window (calculate_sharpe
    returns the plain sentinel 0), so np.mean over any group is always well-defined --
    no NaN-guard needed here.

    Caution when reading `mean_test_sharpe`: freqtrade's `calculate_sharpe` returns a
    sentinel of -100.0 whenever a window's return standard deviation is zero -- notably
    true for any window with exactly one trade. That sentinel is diluted across many
    windows in the gate's overall mean_test_sharpe, but a regime bucket here often
    contains only 1-3 windows, so a single thin-window's undefined-variance sentinel can
    dominate the bucket's mean and look like a real (very bad) market finding when it is
    actually a sentinel artifact.
    """
    if len(window_results) != len(labels):
        raise ValueError(
            f"window_results and labels must be the same length, got "
            f"{len(window_results)} and {len(labels)}"
        )

    grouped: dict[str, list[WindowResult]] = defaultdict(list)
    for wr, label in zip(window_results, labels, strict=True):
        grouped[label].append(wr)

    report: dict[str, dict] = {}
    for label, group in grouped.items():
        report[label] = {
            "n_windows": len(group),
            "n_trades": sum(wr.test_n_trades for wr in group),
            "mean_test_sharpe": float(np.mean([wr.test_sharpe for wr in group])),
            "total_return": float(sum(r for wr in group for r in wr.test_returns)),
        }
    return report
