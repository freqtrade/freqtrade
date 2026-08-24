"""
Statistical functions for validating trading strategies before promotion.

Ported and adapted from `C:\\dev\\MarketMind\\backend\\backtest\\enhanced\\statistics.py`
(lines 14-104 for `benjamini_hochberg`/`permutation_test`, lines 168-228 for
`deflated_sharpe_ratio`; see FREQTRADE_RESEARCH_ARCHITECTURE.md §15). Two adaptations from
the source were applied: the hand-rolled normal CDF/PPF helpers (used there to avoid a new
dependency) are replaced with `scipy.stats.norm`, since SciPy is already a freqtrade
dependency; and the default `periods_per_year` is 365 rather than 252, since crypto trades
every calendar day rather than only equities trading days.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import norm


_EULER_MASCHERONI = 0.5772156649015329


def benjamini_hochberg(p_values: list[float], q: float = 0.05) -> list[bool]:
    """Benjamini-Hochberg step-up FDR control over a batch of p-values.

    Returns a reject mask aligned with the INPUT order: True where the null is rejected
    (a discovery), such that the expected false-discovery rate among rejections is <= q.

    This is the batch-level multiple-testing control the funnel needs. A naive "keep
    p < q" cut passes ~q*N candidates from a pure-noise batch by construction; BH passes
    ~0, because it raises the bar with the number of comparisons. Step-up: find the largest
    rank k (1-indexed, ascending p) with p_(k) <= (k/m)*q, then reject EVERY candidate at
    rank <= k -- including ones that fail their own critical value but sit below the cutoff.
    """
    m = len(p_values)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: p_values[i])  # ascending p, original indices
    max_k = 0
    for rank, idx in enumerate(order, start=1):
        if p_values[idx] <= (rank / m) * q:
            max_k = rank  # largest passing rank so far (step-up keeps the last one)
    reject = [False] * m
    for rank, idx in enumerate(order, start=1):
        if rank <= max_k:
            reject[idx] = True
    return reject


def permutation_test(
    observed_stat: float,
    returns: np.ndarray,
    n_permutations: int = 1000,
    seed: int | None = None,
) -> float:
    """Sign-flip randomization test for the significance of an observed statistic.

    Null hypothesis: the returns carry no directional edge -- the *sign* of each return
    is arbitrary. Each iteration randomly flips the sign of every return and recomputes
    the null statistic (mean/std, in the same units as `observed_stat`), building a null
    distribution; the p-value is the fraction of null statistics that match or beat the
    observed one.

    NOTE: shuffling the *order* of the returns (as an earlier, buggy version did) is
    meaningless -- mean/std is order-invariant, so every reordering yields the same
    statistic and the p-value would be floating-point noise (~1.0 even for a strongly
    trending series). Randomizing signs is what actually tests whether the observed drift
    is distinguishable from chance.

    Args:
        observed_stat: The statistic computed on the real (unpermuted) returns, e.g.
            `returns.mean() / returns.std()`. Compared directly against each permutation's
            null statistic, so it must be in the same units (no implicit annualization is
            applied here).
        returns: Array of periodic returns.
        n_permutations: Number of sign-flip resamples.
        seed: Optional seed for reproducibility.

    Returns:
        p-value in [0, 1]: the fraction of null statistics >= observed_stat.
    """
    r = np.asarray(returns, dtype=float)
    if len(r) < 2:
        return 1.0

    std_r = np.std(r, ddof=1)
    if std_r == 0:
        return 1.0

    rng = np.random.default_rng(seed)
    ge, counted = 0, 0
    for _ in range(n_permutations):
        flipped = r * rng.choice((-1.0, 1.0), size=len(r))
        s = np.std(flipped, ddof=1)
        if s == 0:
            continue
        null_stat = np.mean(flipped) / s
        counted += 1
        if null_stat >= observed_stat:
            ge += 1

    return 1.0 if counted == 0 else ge / counted


def deflated_sharpe_ratio(
    sharpe_ratio: float,
    n_obs: int,
    n_trials: int = 1,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
    periods_per_year: int = 365,
) -> float:
    """Bailey-Lopez de Prado (2014) Deflated Sharpe Ratio. Returns a PROBABILITY in [0, 1].

        DSR = Phi[ ((SR - SR*) * sqrt(n-1)) / sqrt(1 - g3*SR + ((g4-1)/4)*SR^2) ]

    i.e. the probability the observed Sharpe beats the benchmark SR*, given the returns'
    non-normality and how many trials were searched. At n_trials=1 there is no multiple
    testing, SR* = 0, and this reduces to the Probabilistic Sharpe Ratio against zero.

    Args:
        sharpe_ratio: ANNUALIZED Sharpe. De-annualized internally: the formula needs SR
            and n at the same frequency, and n counts per-period observations.
        n_obs: Number of return observations. Without this term, a 6-trade and a
            10,000-trade backtest would deflate identically -- a DSR that can't tell them
            apart is just a sign test wearing a rigor costume.
        n_trials: Parameter combinations searched.
        skewness / kurtosis: Standardized 3rd/4th moments of the returns (g3, g4).
        periods_per_year: Annualization factor used to recover the per-period SR. Default
            365 (crypto trades every calendar day, unlike equities' 252 trading days).

    Returns:
        Probability in [0, 1]. Fails CLOSED to 0.0 on degenerate input (n_obs < 2,
        non-finite SR, non-positive variance term) -- no data is not a coin flip.
    """
    if not math.isfinite(sharpe_ratio) or n_obs is None or n_obs < 2:
        return 0.0

    # Per-period SR: n counts per-period observations, so SR must match their frequency.
    sr = float(sharpe_ratio) / math.sqrt(periods_per_year)

    # V[SR] -- the variance of the Sharpe estimator under non-normality. Used both to
    # scale SR* and (via its square root) as this estimator's own standard error.
    variance_term = 1.0 - skewness * sr + (kurtosis - 1.0) / 4.0 * sr**2
    if not math.isfinite(variance_term) or variance_term <= 0:
        return 0.0  # fail closed rather than take a root of a negative number

    # SR*: expected MAXIMUM Sharpe under the null across n_trials independent trials.
    # n_trials=1 -> no search -> SR* = 0 (norm.ppf(1 - 1/1) is -inf and must not leak in).
    sr_star = 0.0
    if n_trials > 1:
        v_sr = math.sqrt(variance_term / (n_obs - 1))
        g = _EULER_MASCHERONI
        sr_star = v_sr * (
            (1 - g) * norm.ppf(1 - 1.0 / n_trials) + g * norm.ppf(1 - 1.0 / (n_trials * math.e))
        )

    z = ((sr - sr_star) * math.sqrt(n_obs - 1)) / math.sqrt(variance_term)
    if not math.isfinite(z):
        return 0.0
    return float(norm.cdf(z))
