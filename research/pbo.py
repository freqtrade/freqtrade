from __future__ import annotations

import itertools

import numpy as np


def choose_n_splits(n_periods: int, max_splits: int = 16) -> int:
    """Largest even number <= min(max_splits, n_periods), so CSCV's IS/OOS blocks can
    be split into two equal-sized HALVES (of blocks, not necessarily equal-sized
    periods within each block -- see below). Only evenness and s >= 2 matter here: an
    exact divisor of n_periods was never actually required, and requiring one was a
    real bug -- a prime n_periods (e.g. 17 walk-forward windows) has no even divisor at
    all, silently forcing every PBO computation on that data to fail closed to a
    maximally pessimistic 1.0 regardless of what the data actually showed."""
    s = min(max_splits, n_periods)
    s = s if s % 2 == 0 else s - 1
    return max(2, s)


def probability_of_backtest_overfitting(
    returns_matrix: np.ndarray, n_splits: int | None = None
) -> dict:
    """Combinatorially Symmetric Cross-Validation (Bailey, Borwein, Lopez de Prado,
    Zhu 2014). Splits the n_periods columns into n_splits contiguous blocks (via
    np.array_split, which distributes any remainder across the first few blocks --
    blocks need not be perfectly equal-sized, only reasonably close, which CSCV only
    ever required), and for every way of picking half the blocks as in-sample: finds
    the variant with the best in-sample Sharpe, then checks how that same variant
    ranks out-of-sample. PBO is the fraction of splits where the in-sample winner
    ranked in the OOS-worse half — logit of the OOS relative rank <= 0.
    """
    returns_matrix = np.asarray(returns_matrix, dtype=float)
    if returns_matrix.ndim != 2:
        return {"pbo": 1.0, "n_splits": 0, "n_combinations": 0, "logits": []}

    n_variants, n_periods = returns_matrix.shape
    if n_variants < 2 or n_periods < 4:
        return {"pbo": 1.0, "n_splits": 0, "n_combinations": 0, "logits": []}

    s = n_splits or choose_n_splits(n_periods)
    # s must be even (for a symmetric half-IS/half-OOS split) and >= 2 -- an odd s here
    # only ever comes from an explicitly-passed n_splits (choose_n_splits never returns
    # one), a caller error worth failing closed on. s need NOT evenly divide n_periods:
    # np.array_split below handles a remainder gracefully.
    if s < 2 or s % 2 != 0:
        return {"pbo": 1.0, "n_splits": 0, "n_combinations": 0, "logits": []}

    blocks = np.array_split(returns_matrix, s, axis=1)
    logits: list[float] = []

    for is_block_idx in itertools.combinations(range(s), s // 2):
        oos_block_idx = [i for i in range(s) if i not in is_block_idx]
        is_returns = np.concatenate([blocks[i] for i in is_block_idx], axis=1)
        oos_returns = np.concatenate([blocks[i] for i in oos_block_idx], axis=1)

        is_sharpe = is_returns.mean(axis=1) / (is_returns.std(axis=1) + 1e-12)
        oos_sharpe = oos_returns.mean(axis=1) / (oos_returns.std(axis=1) + 1e-12)

        best_variant = int(np.argmax(is_sharpe))
        # relative rank of the IS-best variant's OOS performance, in (0, 1)
        rank = int((oos_sharpe < oos_sharpe[best_variant]).sum()) + 1
        omega = rank / (n_variants + 1)
        omega = min(max(omega, 1e-6), 1 - 1e-6)
        logits.append(float(np.log(omega / (1 - omega))))

    pbo = float(np.mean([1.0 if lg <= 0 else 0.0 for lg in logits]))
    return {"pbo": pbo, "n_splits": s, "n_combinations": len(logits), "logits": logits}
