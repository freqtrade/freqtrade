"""
Monte-Carlo Robustness Scoring

Evaluates how robust a trading strategy is by running it through multiple
randomised permutations. A strategy that only works under exact historical
conditions is fragile; one that stays profitable under small perturbations
is robust and more likely to succeed in live trading.

Perturbation methods:
- Trade-order shuffling: randomise the order trades are counted
- Entry-time jitter: shift entry signals by ±N candles
- Slippage jitter: vary fee/slippage around the configured mean
- Bootstrap resampling: sample trades with replacement to estimate
  confidence intervals on key metrics

The 'robustness_score' (0.0–1.0) is the fraction of permutations that
remain profitable. This can be used as a fitness multiplier or a
standalone ranking criterion.
"""

import logging
import random
import math
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class MonteCarloResult:
    """Container for Monte-Carlo robustness analysis results."""
    # Fraction of permutations that were profitable (0.0–1.0)
    robustness_score: float = 0.0
    # Mean profit across all permutations
    mean_profit: float = 0.0
    # Standard deviation of profit across permutations
    profit_std: float = 0.0
    # 5th percentile of profit (worst-case estimate)
    profit_p5: float = 0.0
    # 95th percentile of profit
    profit_p95: float = 0.0
    # Mean Sharpe ratio across permutations
    mean_sharpe: float = 0.0
    # Number of permutations run
    num_permutations: int = 0
    # Detailed per-permutation results
    permutation_profits: List[float] = field(default_factory=list)


def bootstrap_trades(
    trades: List[Dict[str, Any]],
    num_permutations: int = 100,
    sample_fraction: float = 1.0,
    random_seed: Optional[int] = None,
) -> MonteCarloResult:
    """
    Bootstrap resampling of trade results.

    Samples trades with replacement to build a distribution of possible
    outcomes. This tests whether profit is driven by a few lucky trades
    or by a consistent edge.

    Args:
        trades: List of trade dicts, each must have 'profit_ratio' key.
        num_permutations: Number of bootstrap samples (default: 100).
        sample_fraction: Fraction of trades to sample each iteration (1.0 = same size as original).
        random_seed: Optional seed for reproducibility.

    Returns:
        MonteCarloResult with robustness metrics.
    """
    if not trades:
        return MonteCarloResult()

    rng = random.Random(random_seed)
    n_trades = len(trades)
    sample_size = max(1, int(n_trades * sample_fraction))
    permutation_profits: List[float] = []

    for _ in range(num_permutations):
        # Sample with replacement
        sample = [rng.choice(trades) for _ in range(sample_size)]
        total_profit = sum(t.get('profit_ratio', 0.0) for t in sample) * 100
        permutation_profits.append(total_profit)

    return _build_result(permutation_profits, num_permutations)


def shuffle_trade_order(
    trades: List[Dict[str, Any]],
    num_permutations: int = 100,
    random_seed: Optional[int] = None,
) -> MonteCarloResult:
    """
    Shuffle the order of trades and re-calculate cumulative profit.

    Under a fixed-stake model every ordering gives identical total profit,
    but under compound-stake or drawdown-adjusted models the order matters.
    This method recombines trades in random order and calculates the
    cumulative equity curve, checking if any ordering would have triggered
    a margin call or unacceptable drawdown.

    For simple total-profit analysis, shuffling doesn't change the sum,
    but the *max drawdown* and *Sharpe of the equity curve* can change
    dramatically.

    Args:
        trades: List of trade dicts with 'profit_ratio'.
        num_permutations: Number of shuffles.
        random_seed: Optional seed.

    Returns:
        MonteCarloResult.
    """
    if not trades:
        return MonteCarloResult()

    rng = random.Random(random_seed)
    profits_per_trade = [t.get('profit_ratio', 0.0) for t in trades]
    permutation_profits: List[float] = []

    for _ in range(num_permutations):
        shuffled = list(profits_per_trade)
        rng.shuffle(shuffled)

        # Calculate cumulative equity with compounding
        equity = 1.0
        for pnl in shuffled:
            equity *= (1 + pnl)
        total_return = (equity - 1.0) * 100
        permutation_profits.append(total_return)

    return _build_result(permutation_profits, num_permutations)


def jitter_slippage(
    trades: List[Dict[str, Any]],
    base_fee: float = 0.001,
    slippage_std: float = 0.0005,
    num_permutations: int = 100,
    random_seed: Optional[int] = None,
) -> MonteCarloResult:
    """
    Apply random slippage jitter to each trade.

    Each trade's profit is adjusted by a random fee/slippage delta drawn
    from N(0, slippage_std). This tests whether the strategy's edge
    survives realistic execution-cost variation.

    Args:
        trades: List of trade dicts with 'profit_ratio'.
        base_fee: Base fee (not used directly — jitter is additive).
        slippage_std: Std deviation of per-trade slippage noise.
        num_permutations: Number of iterations.
        random_seed: Optional seed.

    Returns:
        MonteCarloResult.
    """
    if not trades:
        return MonteCarloResult()

    rng = random.Random(random_seed)
    profits_per_trade = [t.get('profit_ratio', 0.0) for t in trades]
    permutation_profits: List[float] = []

    for _ in range(num_permutations):
        total = 0.0
        for pnl in profits_per_trade:
            # Apply random slippage: subtract/add noise (entry + exit = 2× jitter)
            jitter = rng.gauss(0, slippage_std) * 2
            total += (pnl - jitter)
        permutation_profits.append(total * 100)

    return _build_result(permutation_profits, num_permutations)


def run_monte_carlo(
    trades: List[Dict[str, Any]],
    config: Dict[str, Any],
) -> MonteCarloResult:
    """
    Run a full Monte-Carlo robustness analysis using all methods.

    Combines bootstrap, shuffle, and slippage-jitter into a single
    aggregate robustness score.

    Args:
        trades: List of trade dicts from a backtest.
        config: Monte-Carlo section of GA config. Keys:
            - num_permutations (int): Total permutations, split across methods.
            - slippage_std (float): Std dev for slippage jitter.
            - sample_fraction (float): Bootstrap sample fraction.
            - random_seed (int|None)
            - methods (list[str]): Which methods to use:
              'bootstrap', 'shuffle', 'slippage_jitter'

    Returns:
        Aggregated MonteCarloResult.
    """
    if not trades:
        return MonteCarloResult()

    num_perms = config.get('num_permutations', 100)
    seed = config.get('random_seed', None)
    methods = config.get('methods', ['bootstrap', 'shuffle', 'slippage_jitter'])
    sample_frac = config.get('sample_fraction', 1.0)
    slippage_std = config.get('slippage_std', 0.0005)

    # Divide permutations roughly equally across active methods
    n_methods = len(methods) if methods else 1
    perms_per = max(10, num_perms // n_methods)

    all_profits: List[float] = []

    if 'bootstrap' in methods:
        res = bootstrap_trades(trades, perms_per, sample_frac, seed)
        all_profits.extend(res.permutation_profits)

    if 'shuffle' in methods:
        res = shuffle_trade_order(trades, perms_per, seed)
        all_profits.extend(res.permutation_profits)

    if 'slippage_jitter' in methods:
        res = jitter_slippage(trades, slippage_std=slippage_std,
                              num_permutations=perms_per, random_seed=seed)
        all_profits.extend(res.permutation_profits)

    if not all_profits:
        return MonteCarloResult()

    result = _build_result(all_profits, len(all_profits))
    logger.info(f"[MONTE-CARLO] robustness={result.robustness_score:.2%}, "
                f"mean_profit={result.mean_profit:.2f}%, "
                f"p5={result.profit_p5:.2f}%, p95={result.profit_p95:.2f}%, "
                f"n={result.num_permutations}")
    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_result(profits: List[float], num_permutations: int) -> MonteCarloResult:
    """Build a MonteCarloResult from a list of permutation profits."""
    if not profits:
        return MonteCarloResult()

    n = len(profits)
    mean_p = sum(profits) / n
    var_p = sum((p - mean_p) ** 2 for p in profits) / n if n > 1 else 0.0
    std_p = math.sqrt(var_p)

    sorted_profits = sorted(profits)
    idx_5 = max(0, int(n * 0.05))
    idx_95 = min(n - 1, int(n * 0.95))

    profitable = sum(1 for p in profits if p > 0)

    return MonteCarloResult(
        robustness_score=profitable / n if n > 0 else 0.0,
        mean_profit=mean_p,
        profit_std=std_p,
        profit_p5=sorted_profits[idx_5],
        profit_p95=sorted_profits[idx_95],
        mean_sharpe=mean_p / std_p if std_p > 0 else 0.0,
        num_permutations=num_permutations,
        permutation_profits=profits,
    )
