from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class CalibratedThresholds:
    long_threshold: float
    short_threshold: float
    long_trades: int
    short_trades: int
    long_expectation: float
    short_expectation: float


def best_proba_thresholds(
    df: pd.DataFrame,
    *,
    long_col: str,
    short_col: str,
    target_col: str = "&-target",
    dp_col: str = "do_predict",
    grid: Iterable[float] | None = None,
    min_trades: int = 50,
) -> CalibratedThresholds | None:
    """
    Calibrate probability thresholds by maximizing simple expected return on the
    available dataset. For long, expectation is mean(target) where long_col>=thr.
    For short, expectation is mean(-target) where short_col>=thr.

    Returns None if columns are missing or insufficient data.
    """
    if df is None or df.empty:
        return None
    for c in [long_col, short_col, target_col]:
        if c not in df.columns:
            return None
    mask = pd.Series(True, index=df.index)
    if dp_col in df.columns:
        mask &= df[dp_col] == 1
    d = df.loc[mask, [long_col, short_col, target_col]].dropna()
    if d.empty:
        return None

    gl = list(grid) if grid is not None else list(np.round(np.linspace(0.50, 0.80, 7), 3))
    best_l = (0.5, -np.inf, 0)
    best_s = (0.5, -np.inf, 0)

    for thr in gl:
        sel_l = d[d[long_col] >= thr]
        if len(sel_l) >= min_trades:
            exp_l = float(sel_l[target_col].mean())
            if exp_l > best_l[1]:
                best_l = (thr, exp_l, len(sel_l))
        sel_s = d[d[short_col] >= thr]
        if len(sel_s) >= min_trades:
            # Short expectation = mean(-target)
            exp_s = float((-sel_s[target_col]).mean())
            if exp_s > best_s[1]:
                best_s = (thr, exp_s, len(sel_s))

    if best_l[2] == 0 and best_s[2] == 0:
        return None
    return CalibratedThresholds(
        long_threshold=best_l[0],
        short_threshold=best_s[0],
        long_trades=best_l[2],
        short_trades=best_s[2],
        long_expectation=best_l[1] if best_l[2] else 0.0,
        short_expectation=best_s[1] if best_s[2] else 0.0,
    )


def best_abs_threshold(
    df: pd.DataFrame,
    *,
    pred_col: str = "&-target",
    dp_col: str = "do_predict",
    grid: Iterable[float] | None = None,
    min_trades: int = 50,
) -> tuple[float, float, int, int] | None:
    """
    Calibrate absolute prediction threshold for regression by maximizing
    expected return for long (pred>=thr) and short (pred<=-thr) separately.
    Returns (thr_long, thr_short, n_long, n_short) or None.
    """
    if df is None or df.empty or pred_col not in df.columns:
        return None
    mask = pd.Series(True, index=df.index)
    if dp_col in df.columns:
        mask &= df[dp_col] == 1
    d = df.loc[mask, [pred_col]].dropna()
    if d.empty:
        return None
    gl = list(grid) if grid is not None else list(np.round(np.linspace(0.0005, 0.0040, 8), 4))
    best_l = (gl[0], -np.inf, 0)
    best_s = (gl[0], -np.inf, 0)
    # Use realized return proxy as pred itself for calibration (simple approach)
    # In more advanced flow, pass realized target column.
    tgt = d[pred_col]
    for thr in gl:
        sel_l = tgt[tgt >= thr]
        if len(sel_l) >= min_trades:
            exp_l = float(sel_l.mean())
            if exp_l > best_l[1]:
                best_l = (thr, exp_l, len(sel_l))
        sel_s = tgt[tgt <= -thr]
        if len(sel_s) >= min_trades:
            exp_s = float((-sel_s).mean())
            if exp_s > best_s[1]:
                best_s = (thr, exp_s, len(sel_s))
    if best_l[2] == 0 and best_s[2] == 0:
        return None
    return best_l[0], best_s[0], best_l[2], best_s[2]
