"""
Combinatorially Purged Cross-Validation (CPCV) + Probability of Backtest Overfitting (PBO)

Implements the CPCV framework from Bailey, Borwein, López de Prado & Zhu (2017):
"Backtest Overfitting" (Journal of Computational Finance).

Key concepts:
1. **CPCV**: Instead of standard k-fold CV (which has information leakage from
   serial correlation and strategy selection), CPCV:
   - Enumerates all $\\binom{N}{N/2}$ train/test splits of N time blocks
   - Purges samples near the train/test boundary to prevent leakage
   - Embargoes samples after each purged region to account for lagged features

2. **PBO (Probability of Backtest Overfitting)**: Given the CPCV results, PBO
   quantifies the probability that the strategy selected as "best" in-sample
   is actually below-median out-of-sample. A PBO > 0.5 indicates overfitting
   is likely.

Usage in GA context:
- CPCV is expensive (exponential in N), so it's used as an **opt-in** validation
  step on final strategies, not during evolution.
- PBO can be estimated efficiently with a Monte Carlo subset of paths.
- The `cpcv_penalty()` provides a multiplicative fitness penalty similar to DSR.

Configuration (ga_config.yaml):
    cpcv:
        enabled: false         # Expensive — opt-in for final validation
        n_groups: 6            # Number of time blocks (N)
        n_test_groups: 2       # Test size (S) — paths = C(N, S)
        purge_pct: 0.01        # % of total data to purge at boundaries
        embargo_pct: 0.01      # % of total data to embargo after purge
        max_paths: 100         # Max combinatorial paths to evaluate (Monte Carlo subset)
        pbo_threshold: 0.5     # PBO above this triggers penalty
        penalty_weight: 0.20   # Max fitness penalty from PBO
"""

import logging
import math
from itertools import combinations
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

logger = logging.getLogger(__name__)


def generate_cpcv_paths(
    n_groups: int,
    n_test_groups: int,
    max_paths: Optional[int] = None,
    random_state: int = 42,
) -> List[Tuple[List[int], List[int]]]:
    """
    Generate all combinatorial train/test split paths.

    Each path consists of:
    - test_groups: subset of S group indices used as test set
    - train_groups: remaining N-S group indices used as training set

    For N=6, S=2: C(6,2) = 15 paths, which is tractable.
    For N=10, S=5: C(10,5) = 252 paths — use max_paths to subsample.

    Args:
        n_groups: Total number of time blocks (N)
        n_test_groups: Number of blocks used as test set (S)
        max_paths: Maximum paths to return (random subset if exceeded)
        random_state: Random seed for subset selection

    Returns:
        List of (train_group_indices, test_group_indices) tuples
    """
    all_groups = list(range(n_groups))
    all_paths = []

    for test_combo in combinations(all_groups, n_test_groups):
        test_groups = list(test_combo)
        train_groups = [g for g in all_groups if g not in test_groups]
        all_paths.append((train_groups, test_groups))

    n_total = len(all_paths)
    logger.debug(f"CPCV: Generated C({n_groups},{n_test_groups}) = {n_total} paths")

    if max_paths and n_total > max_paths:
        rng = np.random.RandomState(random_state)
        indices = rng.choice(n_total, size=max_paths, replace=False)
        all_paths = [all_paths[i] for i in sorted(indices)]
        logger.info(f"CPCV: Subsampled {max_paths}/{n_total} paths")

    return all_paths


def create_time_blocks(
    n_samples: int,
    n_groups: int,
    purge_pct: float = 0.01,
    embargo_pct: float = 0.01,
) -> Dict[str, Any]:
    """
    Divide a time series of n_samples into n_groups contiguous blocks,
    with purge and embargo zones between them.

    Args:
        n_samples: Total number of observations
        n_groups: Number of groups to divide into
        purge_pct: Fraction of samples to purge at boundaries
        embargo_pct: Fraction of samples to embargo after purge

    Returns:
        Dict with 'blocks' (list of (start, end) index tuples),
        'purge_size', 'embargo_size'
    """
    purge_size = max(1, int(n_samples * purge_pct))
    embargo_size = max(1, int(n_samples * embargo_pct))
    block_size = n_samples // n_groups

    blocks = []
    for i in range(n_groups):
        start = i * block_size
        end = (i + 1) * block_size if i < n_groups - 1 else n_samples
        blocks.append((start, end))

    return {
        'blocks': blocks,
        'purge_size': purge_size,
        'embargo_size': embargo_size,
        'block_size': block_size,
        'n_samples': n_samples,
    }


def get_train_test_indices(
    block_info: Dict[str, Any],
    train_groups: List[int],
    test_groups: List[int],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get purged and embargoed train/test sample indices for a CPCV path.

    Purging removes samples from the training set that are within
    `purge_size` of any test block boundary. Embargo removes additional
    samples after each purge zone to account for lagged serial correlation.

    Args:
        block_info: Output from create_time_blocks()
        train_groups: Group indices for training
        test_groups: Group indices for testing

    Returns:
        Tuple of (train_indices, test_indices) as sorted numpy arrays
    """
    blocks = block_info['blocks']
    purge_size = block_info['purge_size']
    embargo_size = block_info['embargo_size']
    n_samples = block_info['n_samples']

    # Collect raw test indices
    test_indices = set()
    for g in test_groups:
        start, end = blocks[g]
        test_indices.update(range(start, end))

    # Collect raw train indices
    train_indices = set()
    for g in train_groups:
        start, end = blocks[g]
        train_indices.update(range(start, end))

    # Purge: remove train samples near test boundaries
    purge_set = set()
    for g in test_groups:
        test_start, test_end = blocks[g]
        # Purge before test block
        purge_set.update(range(max(0, test_start - purge_size), test_start))
        # Purge after test block
        purge_set.update(range(test_end, min(n_samples, test_end + purge_size)))

    # Embargo: additional removal after purge zones
    embargo_set = set()
    for g in test_groups:
        test_start, test_end = blocks[g]
        embargo_start = min(n_samples, test_end + purge_size)
        embargo_end = min(n_samples, embargo_start + embargo_size)
        embargo_set.update(range(embargo_start, embargo_end))

    # Remove purged and embargoed samples from training set
    train_indices -= purge_set
    train_indices -= embargo_set
    train_indices -= test_indices  # Safety: no overlap

    return np.array(sorted(train_indices)), np.array(sorted(test_indices))


def compute_pbo(
    in_sample_performances: np.ndarray,
    out_of_sample_performances: np.ndarray,
) -> Tuple[float, Dict[str, float]]:
    """
    Compute the Probability of Backtest Overfitting (PBO).

    For each CPCV path:
    1. Rank strategies by in-sample performance
    2. Select the best in-sample strategy (n*)
    3. Check if n*'s out-of-sample performance is below median OOS

    PBO = fraction of paths where the in-sample winner is below-median OOS.

    A PBO > 0.5 means the "best" strategy in backtesting is more likely
    than not to underperform out-of-sample — clear evidence of overfitting.

    Args:
        in_sample_performances: Shape (n_paths, n_strategies) — IS performance per path
        out_of_sample_performances: Shape (n_paths, n_strategies) — OOS performance per path

    Returns:
        Tuple of (pbo_score, details_dict)
        pbo_score: float in [0, 1], higher = more overfitting
    """
    n_paths, n_strategies = in_sample_performances.shape

    if n_paths == 0 or n_strategies < 2:
        return 0.0, {'n_paths': n_paths, 'n_strategies': n_strategies,
                      'error': 'insufficient_data'}

    overfit_count = 0
    oos_ranks = []
    logit_values = []

    for path_idx in range(n_paths):
        is_perfs = in_sample_performances[path_idx]
        oos_perfs = out_of_sample_performances[path_idx]

        # Find best in-sample strategy
        best_is_idx = np.argmax(is_perfs)

        # Get OOS performance of IS winner
        best_oos = oos_perfs[best_is_idx]

        # Median OOS performance
        median_oos = np.median(oos_perfs)

        # Is best IS strategy below median OOS?
        if best_oos < median_oos:
            overfit_count += 1

        # OOS rank of IS winner (0 = worst, 1 = best)
        rank = np.mean(oos_perfs <= best_oos)
        oos_ranks.append(rank)

        # Logit of rank for statistical testing
        rank_clipped = np.clip(rank, 0.01, 0.99)
        logit_values.append(math.log(rank_clipped / (1 - rank_clipped)))

    pbo = overfit_count / n_paths

    details = {
        'pbo': pbo,
        'n_paths': n_paths,
        'n_strategies': n_strategies,
        'overfit_count': overfit_count,
        'mean_oos_rank': float(np.mean(oos_ranks)),
        'median_oos_rank': float(np.median(oos_ranks)),
        'mean_logit': float(np.mean(logit_values)) if logit_values else 0.0,
    }

    return pbo, details


def cpcv_penalty(
    pbo: float,
    pbo_threshold: float = 0.5,
    penalty_weight: float = 0.20,
) -> float:
    """
    Compute multiplicative fitness penalty from PBO score.

    Penalty curve:
    - PBO < 0.3 → no penalty (1.0)
    - PBO = threshold → moderate penalty
    - PBO > 0.8 → maximum penalty (1.0 - penalty_weight)

    The transition is smooth (sigmoid-shaped).

    Args:
        pbo: Probability of Backtest Overfitting [0, 1]
        pbo_threshold: PBO level where penalty kicks in
        penalty_weight: Maximum penalty (0.20 = 20% fitness reduction)

    Returns:
        Penalty multiplier in [1 - penalty_weight, 1.0]
    """
    if pbo < 0.2:
        return 1.0  # No penalty for low PBO

    # Sigmoid centered at threshold
    steepness = 8.0
    try:
        logistic = 1.0 / (1.0 + math.exp(-steepness * (pbo - pbo_threshold)))
    except OverflowError:
        logistic = 1.0 if pbo > pbo_threshold else 0.0

    # Map: high PBO → low multiplier
    multiplier = 1.0 - penalty_weight * logistic

    return multiplier


class CPCVValidator:
    """
    Orchestrates CPCV validation and PBO computation for a set of strategies.

    This is designed as a post-evolution validation step:
    1. Take the top K strategies from the GA
    2. Run CPCV backtests across all paths
    3. Compute PBO to quantify overfitting risk
    4. Optionally apply penalty to fitness scores

    Expensive by design — use only for final validation, not during evolution.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        cpcv_config = self.config.get('cpcv', {})

        self.enabled = cpcv_config.get('enabled', False)
        self.n_groups = cpcv_config.get('n_groups', 6)
        self.n_test_groups = cpcv_config.get('n_test_groups', 2)
        self.purge_pct = cpcv_config.get('purge_pct', 0.01)
        self.embargo_pct = cpcv_config.get('embargo_pct', 0.01)
        self.max_paths = cpcv_config.get('max_paths', 100)
        self.pbo_threshold = cpcv_config.get('pbo_threshold', 0.5)
        self.penalty_weight = cpcv_config.get('penalty_weight', 0.20)

        logger.info(
            f"CPCVValidator initialized: enabled={self.enabled}, "
            f"groups={self.n_groups}, test_groups={self.n_test_groups}, "
            f"max_paths={self.max_paths}"
        )

    def validate_strategies(
        self,
        strategy_results: Dict[str, np.ndarray],
        timerange: str = '',
    ) -> Dict[str, Any]:
        """
        Run CPCV validation on a set of strategy backtest results.

        This is the high-level entry point. It assumes you've already collected
        per-block performance metrics for each strategy.

        Args:
            strategy_results: Dict mapping strategy_id -> array of per-block
                             performance metrics (shape: (n_groups,))
            timerange: Original timerange for logging

        Returns:
            Dict with 'pbo', 'details', 'per_strategy_oos_rank', 'penalty'
        """
        if not self.enabled:
            return {'pbo': 0.0, 'skipped': True, 'reason': 'disabled'}

        n_strategies = len(strategy_results)
        if n_strategies < 2:
            return {'pbo': 0.0, 'skipped': True, 'reason': 'need_2+_strategies'}

        strategy_ids = list(strategy_results.keys())
        # Collect performance arrays: shape (n_strategies, n_groups)
        perf_matrix = np.array([strategy_results[sid] for sid in strategy_ids])

        if perf_matrix.shape[1] != self.n_groups:
            logger.warning(
                f"CPCV: strategy results have {perf_matrix.shape[1]} blocks, "
                f"expected {self.n_groups}. Adjusting n_groups."
            )
            self.n_groups = perf_matrix.shape[1]

        # Generate CPCV paths
        paths = generate_cpcv_paths(
            n_groups=self.n_groups,
            n_test_groups=self.n_test_groups,
            max_paths=self.max_paths,
        )

        if not paths:
            return {'pbo': 0.0, 'skipped': True, 'reason': 'no_valid_paths'}

        # Compute IS and OOS performance for each path
        is_perfs = np.zeros((len(paths), n_strategies))
        oos_perfs = np.zeros((len(paths), n_strategies))

        for path_idx, (train_groups, test_groups) in enumerate(paths):
            for s_idx in range(n_strategies):
                # IS performance = mean of training blocks
                is_perfs[path_idx, s_idx] = perf_matrix[s_idx, train_groups].mean()
                # OOS performance = mean of test blocks
                oos_perfs[path_idx, s_idx] = perf_matrix[s_idx, test_groups].mean()

        # Compute PBO
        pbo, details = compute_pbo(is_perfs, oos_perfs)

        # Compute per-strategy OOS rank
        per_strategy_oos = {}
        for s_idx, sid in enumerate(strategy_ids):
            oos_vals = oos_perfs[:, s_idx]
            per_strategy_oos[sid] = {
                'mean_oos': float(np.mean(oos_vals)),
                'std_oos': float(np.std(oos_vals)),
                'min_oos': float(np.min(oos_vals)),
            }

        # Compute penalty
        penalty = cpcv_penalty(pbo, self.pbo_threshold, self.penalty_weight)

        result = {
            'pbo': pbo,
            'penalty': penalty,
            'details': details,
            'per_strategy_oos': per_strategy_oos,
            'n_paths': len(paths),
            'n_strategies': n_strategies,
            'timerange': timerange,
            'skipped': False,
        }

        logger.info(
            f"[CPCV] PBO = {pbo:.3f}, penalty = {penalty:.3f}, "
            f"paths = {len(paths)}, strategies = {n_strategies}"
        )

        return result

    def quick_pbo_estimate(
        self,
        in_sample_fitness: List[float],
        out_of_sample_fitness: List[float],
    ) -> float:
        """
        Quick PBO estimate from a single train/test split.

        Not as rigorous as full CPCV, but useful as a lightweight check
        during evolution.

        Args:
            in_sample_fitness: IS fitness values for each strategy
            out_of_sample_fitness: OOS fitness values for each strategy

        Returns:
            Estimated PBO (0-1)
        """
        if len(in_sample_fitness) < 2:
            return 0.0

        is_arr = np.array(in_sample_fitness)
        oos_arr = np.array(out_of_sample_fitness)

        # Best IS strategy
        best_is_idx = np.argmax(is_arr)
        best_oos = oos_arr[best_is_idx]

        # Is it below median OOS?
        median_oos = np.median(oos_arr)

        # Quick rank-based estimate
        rank = np.mean(oos_arr <= best_oos)

        # PBO estimate: if rank < 0.5, the IS winner is below median OOS
        return 1.0 - rank
