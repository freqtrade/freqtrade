# research/parameter_stability.py
"""Parameter stability: does a candidate's edge hold across a region of its parameter
grid, not just one lucky combination (CRYPTO_STRATEGY_DISCOVERY_PROPOSAL.md Sec 12).
Reuses the variant_matrix research.gate.run_promotion_gate already builds for PBO --
train-period-only, zero new backtests. See
docs/superpowers/specs/2026-08-24-parameter-stability-design.md for why this is
in-sample only (a.k.a. "Parameter Plateau Analysis" in quant research) rather than
re-backtesting every grid variant's out-of-sample performance.
"""

from __future__ import annotations

import numpy as np


def parameter_stability(variant_matrix: np.ndarray) -> float:
    """Fraction of grid variants (rows of `variant_matrix`) whose mean train-period
    return across windows (columns) is positive. Always in [0, 1].

    `variant_matrix` is the exact n_variants x n_windows array
    research.gate.run_promotion_gate already builds for
    research.pbo.probability_of_backtest_overfitting -- no new data, no new backtests.

    Fails open to 1.0 for a single-variant grid (variant_matrix.shape[0] == 1): there's
    no region to be unstable across, so this isn't evidence against the candidate --
    same fail-open convention as scoring.robustness_score's cost_sensitivity component.

    Raises ValueError if variant_matrix isn't 2-D or has zero rows -- a caller-contract
    violation, not a data condition this function should silently paper over.
    """
    variant_matrix = np.asarray(variant_matrix, dtype=float)
    if variant_matrix.ndim != 2 or variant_matrix.shape[0] == 0:
        raise ValueError(
            f"variant_matrix must be 2-D with at least one row, got shape {variant_matrix.shape}"
        )
    if variant_matrix.shape[0] == 1:
        return 1.0

    row_means = variant_matrix.mean(axis=1)
    return float((row_means > 0).sum() / len(row_means))
