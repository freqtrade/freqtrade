"""
Deflated Sharpe Ratio (DSR) Module

Implements the Deflated Sharpe Ratio from Bailey & López de Prado (2014):
"The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting,
and Non-Normality".

The DSR answers the question: "Given that we tried N strategies and observed
skewness γ₃ and kurtosis γ₄ in the returns, what is the probability that the
observed Sharpe Ratio is genuinely above zero rather than a statistical artifact?"

Key corrections:
1. **Multiple testing**: The more strategies we try (N_trials), the higher
   the expected maximum Sharpe by pure chance.
2. **Non-normality**: Fat tails (high kurtosis) and skew inflate apparent
   Sharpe ratios beyond what Gaussian assumptions predict.

Integration points:
- ``calculate_dsr()`` — standalone DSR p-value for a single strategy
- ``deflated_sharpe_penalty()`` — multiplicative fitness penalty [0, 1]
- Used inside ``FitnessEvaluator.calculate_fitness()`` as an optional
  anti-overfitting component (enabled by default)

Configuration (ga_config.yaml):
    deflated_sharpe:
        enabled: true          # Enable DSR penalty in fitness
        penalty_weight: 0.15   # How much DSR affects fitness (0 = off, 1 = full)
        min_trades: 20         # Minimum trades for reliable DSR
        significance: 0.05     # DSR threshold below which penalty kicks in
"""

import logging
import math
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def expected_max_sharpe(
    n_trials: int,
    variance_sr: float = 1.0,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """
    Compute the expected maximum Sharpe Ratio under the null hypothesis
    that all N strategies have zero true Sharpe, given non-normal returns.

    Uses the Euler-Mascheroni approximation for the expected maximum
    of N i.i.d. standard normal draws:

        E[max(Z_1, ..., Z_N)] ≈ (1 - γ_EM) * Φ⁻¹(1 - 1/N) + γ_EM * Φ⁻¹(1 - 1/(N*e))

    Then corrected for skew and kurtosis of the SR distribution.

    Args:
        n_trials: Number of strategies tried (N)
        variance_sr: Variance of SR estimator (≈ 1 + SR²/4 * (kurtosis-1) - SR * skewness)
                     For the null (SR=0), this simplifies to 1.
        skewness: Return skewness (γ₃)
        kurtosis: Return kurtosis (γ₄, excess = kurtosis - 3)

    Returns:
        Expected maximum SR under the null
    """
    from scipy.stats import norm

    if n_trials <= 1:
        return 0.0

    # Euler-Mascheroni constant
    gamma_em = 0.5772156649

    # Expected max of N i.i.d. standard normals
    try:
        z1 = norm.ppf(1 - 1.0 / n_trials)
        z2 = norm.ppf(1 - 1.0 / (n_trials * math.e))
    except (ValueError, FloatingPointError):
        z1 = norm.ppf(1 - 1e-10)
        z2 = z1
    
    e_max_z = (1 - gamma_em) * z1 + gamma_em * z2

    # Correct for non-normality (Bailey & López de Prado eq. 6)
    # SR* = E[max(Z)] * sqrt(V[SR])
    # where V[SR] ≈ 1 + (γ₄ - 1)/4 * SR² - γ₃ * SR  (under null SR=0 → V=1)
    e_max_sr = e_max_z * math.sqrt(max(variance_sr, 0.01))

    return e_max_sr


def calculate_dsr(
    observed_sharpe: float,
    n_trials: int,
    n_returns: int,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
    sr_benchmark: float = 0.0,
) -> float:
    """
    Calculate the Deflated Sharpe Ratio (probability that the observed SR
    is genuinely above the benchmark after accounting for selection bias
    and non-normal returns).

    DSR = Φ[ (SR_obs - SR*) / σ_SR ]

    where:
        SR* = expected max SR from N trials under null
        σ_SR = standard error of the SR estimator corrected for
               skewness and kurtosis

    Args:
        observed_sharpe: The observed annualized Sharpe Ratio
        n_trials: Number of strategies tested (selection bias adjustment)
        n_returns: Number of return observations (T)
        skewness: Return skewness (γ₃)
        kurtosis: Return kurtosis (γ₄, NOT excess — i.e., normal = 3.0)
        sr_benchmark: Benchmark SR to compare against (default 0.0)

    Returns:
        DSR p-value in [0, 1]. Higher = more confidence the SR is real.
        Values > 0.95 indicate strong evidence of genuine skill.
    """
    from scipy.stats import norm

    if n_returns < 10:
        return 0.0  # Not enough data for any meaningful inference

    if n_trials < 1:
        n_trials = 1

    # Excess kurtosis
    excess_kurtosis = kurtosis - 3.0

    # Standard error of the SR estimator (Lo 2002, corrected for non-normality)
    # Var(SR) ≈ [1 + (γ₄-1)/4 * SR² - γ₃ * SR] / T
    sr = observed_sharpe
    var_sr = (1.0 + (excess_kurtosis / 4.0) * sr * sr - skewness * sr) / n_returns
    var_sr = max(var_sr, 1e-10)  # Numerical guard
    se_sr = math.sqrt(var_sr)

    # Expected max SR under the null
    e_max_sr = expected_max_sharpe(
        n_trials=n_trials,
        variance_sr=1.0,  # Under H0: SR=0
        skewness=skewness,
        kurtosis=kurtosis,
    )

    # The benchmark is the maximum of (explicit benchmark, expected max from trials)
    effective_benchmark = max(sr_benchmark, e_max_sr)

    # DSR = P(SR > SR*) = Φ[(SR_obs - SR*) / σ_SR]
    if se_sr < 1e-12:
        return 1.0 if observed_sharpe > effective_benchmark else 0.0

    z_score = (observed_sharpe - effective_benchmark) / se_sr
    dsr = norm.cdf(z_score)

    return float(dsr)


def compute_return_statistics(
    trade_results: list,
) -> Dict[str, float]:
    """
    Compute return statistics needed for DSR from a list of trade results.

    Each trade result should have at least a 'profit_ratio' or 'profit_abs' field.

    Args:
        trade_results: List of trade dicts with profit information

    Returns:
        Dict with 'mean', 'std', 'skewness', 'kurtosis', 'n_returns',
        'sharpe_ratio' (annualized, assuming ~365 trades/year)
    """
    if not trade_results:
        return {
            'mean': 0.0, 'std': 0.0, 'skewness': 0.0,
            'kurtosis': 3.0, 'n_returns': 0, 'sharpe_ratio': 0.0,
        }

    # Extract returns
    returns = []
    for trade in trade_results:
        if isinstance(trade, dict):
            r = trade.get('profit_ratio', trade.get('profit_abs', 0))
        elif isinstance(trade, (int, float)):
            r = float(trade)
        else:
            continue
        if not math.isnan(r) and not math.isinf(r):
            returns.append(r)

    if len(returns) < 2:
        return {
            'mean': returns[0] if returns else 0.0,
            'std': 0.0, 'skewness': 0.0,
            'kurtosis': 3.0, 'n_returns': len(returns),
            'sharpe_ratio': 0.0,
        }

    arr = np.array(returns)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1))

    if std < 1e-12:
        return {
            'mean': mean, 'std': 0.0, 'skewness': 0.0,
            'kurtosis': 3.0, 'n_returns': len(returns),
            'sharpe_ratio': 0.0,
        }

    # Skewness and kurtosis (Fisher definition)
    from scipy.stats import skew, kurtosis as scipy_kurtosis
    skewness_val = float(skew(arr, bias=False))
    # scipy kurtosis returns EXCESS kurtosis by default; we want raw
    kurtosis_val = float(scipy_kurtosis(arr, bias=False, fisher=False))

    # Annualized Sharpe (rough: assume 1 trade ~ 1 day avg holding)
    sharpe = mean / std * math.sqrt(252) if std > 0 else 0.0

    return {
        'mean': mean,
        'std': std,
        'skewness': skewness_val,
        'kurtosis': kurtosis_val,
        'n_returns': len(returns),
        'sharpe_ratio': sharpe,
    }


def deflated_sharpe_penalty(
    observed_sharpe: float,
    n_trials: int,
    n_returns: int,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
    significance: float = 0.05,
    penalty_weight: float = 0.15,
) -> Tuple[float, Dict[str, float]]:
    """
    Compute a multiplicative fitness penalty based on the DSR.

    The penalty smoothly scales from 1.0 (no penalty) when DSR is high,
    to (1 - penalty_weight) when DSR is very low (i.e., the observed
    Sharpe is likely a statistical artifact).

    Penalty curve:
        - DSR > 0.95 → penalty = 1.0 (no reduction)
        - DSR = significance (e.g. 0.05) → penalty ≈ 1 - penalty_weight
        - DSR < significance → penalty = 1 - penalty_weight (floor)

    The transition is sigmoid-shaped for smooth gradient behavior.

    Args:
        observed_sharpe: Annualized Sharpe Ratio
        n_trials: Number of strategies tried so far in the GA run
        n_returns: Number of trade/return observations
        skewness: Return skewness
        kurtosis: Return kurtosis (raw, normal = 3.0)
        significance: DSR confidence threshold
        penalty_weight: Maximum penalty factor (0.15 = up to 15% fitness reduction)

    Returns:
        Tuple of (penalty_multiplier, info_dict)
        penalty_multiplier: float in [1 - penalty_weight, 1.0]
        info_dict: Contains DSR value, z-score, etc. for logging/metrics
    """
    if n_returns < 20 or n_trials < 2:
        # Not enough data for meaningful DSR — skip penalty
        return 1.0, {
            'dsr': float('nan'),
            'dsr_penalty': 1.0,
            'dsr_skipped': True,
            'skip_reason': 'insufficient_data',
        }

    dsr = calculate_dsr(
        observed_sharpe=observed_sharpe,
        n_trials=n_trials,
        n_returns=n_returns,
        skewness=skewness,
        kurtosis=kurtosis,
    )

    # Smooth penalty using a logistic function centered at the significance level
    # When DSR >> significance → multiplier ≈ 1.0
    # When DSR << significance → multiplier ≈ 1 - penalty_weight
    # Steepness controls how quickly the transition happens
    steepness = 10.0  # Sharpness of the sigmoid transition
    midpoint = significance + 0.2  # Center the transition slightly above significance
    
    # Logistic: f(x) = 1 / (1 + exp(-k*(x - x0)))
    try:
        logistic = 1.0 / (1.0 + math.exp(-steepness * (dsr - midpoint)))
    except OverflowError:
        logistic = 0.0 if dsr < midpoint else 1.0

    # Map from [0, 1] logistic to [1 - penalty_weight, 1.0] multiplier
    penalty_multiplier = (1.0 - penalty_weight) + penalty_weight * logistic

    info = {
        'dsr': dsr,
        'dsr_penalty': penalty_multiplier,
        'dsr_skipped': False,
        'dsr_n_trials': n_trials,
        'dsr_n_returns': n_returns,
        'dsr_skewness': skewness,
        'dsr_kurtosis': kurtosis,
        'dsr_observed_sharpe': observed_sharpe,
    }

    return penalty_multiplier, info


class DSRTracker:
    """
    Tracks the number of unique strategies evaluated across the GA run
    to provide an accurate n_trials count for DSR computation.

    Should be instantiated once per GA run and updated each generation.
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        dsr_config = self.config.get('deflated_sharpe', {})
        self.enabled = dsr_config.get('enabled', True)
        self.penalty_weight = dsr_config.get('penalty_weight', 0.15)
        self.min_trades = dsr_config.get('min_trades', 20)
        self.significance = dsr_config.get('significance', 0.05)
        self._total_evaluated = 0
        self._strategy_hashes: set = set()

    @property
    def n_trials(self) -> int:
        """Number of unique strategies evaluated so far."""
        return max(len(self._strategy_hashes), self._total_evaluated)

    def register_evaluation(self, strategy_hash: str = None):
        """Record that a strategy was evaluated."""
        self._total_evaluated += 1
        if strategy_hash:
            self._strategy_hashes.add(strategy_hash)

    def compute_penalty(
        self,
        observed_sharpe: float,
        n_returns: int,
        skewness: float = 0.0,
        kurtosis: float = 3.0,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute DSR penalty using the tracker's accumulated trial count.

        Args:
            observed_sharpe: Annualized Sharpe
            n_returns: Number of return observations
            skewness: Return skewness
            kurtosis: Return kurtosis (raw)

        Returns:
            (penalty_multiplier, info_dict)
        """
        if not self.enabled:
            return 1.0, {'dsr': float('nan'), 'dsr_skipped': True,
                         'skip_reason': 'disabled'}

        if n_returns < self.min_trades:
            return 1.0, {'dsr': float('nan'), 'dsr_skipped': True,
                         'skip_reason': f'trades({n_returns})<min({self.min_trades})'}

        return deflated_sharpe_penalty(
            observed_sharpe=observed_sharpe,
            n_trials=self.n_trials,
            n_returns=n_returns,
            skewness=skewness,
            kurtosis=kurtosis,
            significance=self.significance,
            penalty_weight=self.penalty_weight,
        )

    def reset(self):
        """Reset counters (e.g., for a new GA run)."""
        self._total_evaluated = 0
        self._strategy_hashes.clear()
