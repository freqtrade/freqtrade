"""
LLM Strategy Diagnostics

Diagnoses failure modes from strategy metrics to provide targeted
improvement objectives for LLM-guided mutation and objective-conditioned
prompts.
"""

import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

# Failure mode definitions with diagnostic strings and suggested fixes.
# Each entry: (check_fn, failure_label, guidance_string)

# Default thresholds — overridden by config when available.
_DEFAULT_THRESHOLDS = {
    'min_trades': 5,
    'max_drawdown': 0.20,
    'min_win_rate': 30.0,
    'min_sharpe': 0.5,
    'max_complexity': 8,       # indicators + conditions
    'wf_gap_threshold': 0.15,  # walk-forward fitness gap
}


def diagnose_failure_mode(
    metrics: Dict[str, Any],
    config: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """
    Inspect an individual's backtest metrics and return a targeted
    failure-mode description suitable for LLM prompts.

    The returned string tells the LLM *what* is wrong and *how* to fix it.
    Returns ``None`` if no clear failure mode is detected (strategy is
    reasonably healthy).

    Args:
        metrics: Dict of backtest metrics — expected keys include
            ``num_trades``, ``max_drawdown``, ``win_rate``, ``sharpe_ratio``,
            ``profit``, ``indicator_count``, ``condition_count``,
            ``wf_gap`` (optional).
        config: Optional GA config for threshold overrides.

    Returns:
        Failure-mode string or ``None``.
    """
    thresholds = dict(_DEFAULT_THRESHOLDS)
    if config:
        fitness_cfg = config.get('fitness', {})
        penalties = fitness_cfg.get('penalties', {})
        thresholds['min_trades'] = penalties.get('min_trades', thresholds['min_trades'])
        thresholds['max_drawdown'] = penalties.get('max_drawdown', thresholds['max_drawdown'])
        thresholds['min_win_rate'] = penalties.get('min_win_rate', thresholds['min_win_rate'])

    # Ordered by severity — return the *first* (most critical) failure.
    failures: List[str] = []

    # 1. Zero or too-few trades (most common LLM strategy failure)
    num_trades = metrics.get('num_trades', 0)
    if num_trades == 0:
        failures.append(
            "ZERO TRADES: The strategy never triggers. "
            "Loosen entry condition thresholds (e.g. RSI < 35 instead of < 20), "
            "use OR logic for alternative entries, or add operators like "
            "cross_above/cross_below that trigger on transitions."
        )
    elif num_trades < thresholds['min_trades']:
        failures.append(
            f"TOO FEW TRADES ({num_trades}): The entry conditions are too restrictive. "
            f"Target at least {thresholds['min_trades']} trades. "
            "Consider widening thresholds, removing one entry filter, "
            "or switching from AND to OR for some conditions."
        )

    # 2. Excessive drawdown
    max_dd = metrics.get('max_drawdown', 0)
    if max_dd > thresholds['max_drawdown']:
        failures.append(
            f"EXCESSIVE DRAWDOWN ({max_dd:.1%}): The strategy loses too much "
            f"during unfavorable periods (threshold: {thresholds['max_drawdown']:.0%}). "
            "Add a higher-timeframe trend filter (e.g. EMA or ADX on 1h), "
            "tighten the stoploss, add trailing stop, or add exit conditions "
            "that cut losing trades earlier."
        )

    # 3. Low win rate
    win_rate = metrics.get('win_rate', 0)
    if num_trades > 0 and win_rate < thresholds['min_win_rate']:
        failures.append(
            f"LOW WIN RATE ({win_rate:.0f}%): Most trades lose. "
            "Improve entry timing by using cross_above/cross_below operators "
            "instead of static thresholds, add volume confirmation (CMF > 0, VROC > 50), "
            "or add a trend-strength filter (ADX > 25)."
        )

    # 4. Poor risk-adjusted return
    sharpe = metrics.get('sharpe_ratio', 0)
    if num_trades > 0 and sharpe < thresholds['min_sharpe']:
        failures.append(
            f"POOR RISK-ADJUSTED RETURN (Sharpe={sharpe:.2f}): Returns don't "
            "compensate for the risk taken. Add exit conditions to lock in "
            "profits earlier, use trailing stops to protect gains, or tighten "
            "ROI targets (lower the '0' minute target to capture quick wins)."
        )

    # 5. Overfitting / walk-forward gap
    wf_gap = metrics.get('wf_gap', None)
    if wf_gap is not None and wf_gap > thresholds['wf_gap_threshold']:
        failures.append(
            f"OVERFITTING (walk-forward gap={wf_gap:.2f}): The strategy works "
            "in training but degrades out-of-sample. Simplify the strategy: "
            "remove 1–2 indicators, use longer lookback periods, prefer "
            "cross_above/cross_below operators over exact thresholds, and "
            "avoid exotic indicator combinations."
        )

    # 6. Excessive complexity
    indicator_count = metrics.get('indicator_count', 0)
    condition_count = metrics.get('condition_count', 0)
    complexity = indicator_count + condition_count
    if complexity > thresholds['max_complexity']:
        failures.append(
            f"EXCESSIVE COMPLEXITY ({indicator_count} indicators + "
            f"{condition_count} conditions = {complexity}): Overly complex "
            "strategies are prone to overfitting. Remove the least-important "
            "indicator or condition, merge similar conditions, or replace "
            "two correlated indicators with a single composite one."
        )

    # 7. Negative profit but with trades
    profit = metrics.get('profit', 0)
    if num_trades > 0 and profit < -5:
        failures.append(
            f"LARGE LOSS (profit={profit:.1f}%): The strategy loses money "
            "consistently. Check if entry/exit logic is inverted, tighten "
            "the stoploss (e.g. -8% instead of -15%), add a trend filter "
            "to avoid counter-trend entries, or improve exit timing."
        )

    if not failures:
        return None

    # Return the most critical failure (first in priority order)
    return failures[0]


def diagnose_all_failure_modes(
    metrics: Dict[str, Any],
    config: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """
    Return *all* applicable failure modes for a strategy, not just the
    most critical one. Useful for comprehensive LLM context.

    Args:
        metrics: Backtest metrics dict.
        config: Optional GA config.

    Returns:
        List of failure-mode strings (may be empty).
    """
    thresholds = dict(_DEFAULT_THRESHOLDS)
    if config:
        fitness_cfg = config.get('fitness', {})
        penalties = fitness_cfg.get('penalties', {})
        thresholds['min_trades'] = penalties.get('min_trades', thresholds['min_trades'])
        thresholds['max_drawdown'] = penalties.get('max_drawdown', thresholds['max_drawdown'])
        thresholds['min_win_rate'] = penalties.get('min_win_rate', thresholds['min_win_rate'])

    failures: List[str] = []
    num_trades = metrics.get('num_trades', 0)
    max_dd = metrics.get('max_drawdown', 0)
    win_rate = metrics.get('win_rate', 0)
    sharpe = metrics.get('sharpe_ratio', 0)
    profit = metrics.get('profit', 0)
    wf_gap = metrics.get('wf_gap', None)
    indicator_count = metrics.get('indicator_count', 0)
    condition_count = metrics.get('condition_count', 0)

    if num_trades == 0:
        failures.append("zero_trades")
    elif num_trades < thresholds['min_trades']:
        failures.append("too_few_trades")

    if max_dd > thresholds['max_drawdown']:
        failures.append("excessive_drawdown")

    if num_trades > 0 and win_rate < thresholds['min_win_rate']:
        failures.append("low_win_rate")

    if num_trades > 0 and sharpe < thresholds['min_sharpe']:
        failures.append("poor_sharpe")

    if wf_gap is not None and wf_gap > thresholds['wf_gap_threshold']:
        failures.append("overfitting")

    if (indicator_count + condition_count) > thresholds['max_complexity']:
        failures.append("excessive_complexity")

    if num_trades > 0 and profit < -5:
        failures.append("large_loss")

    return failures


def select_mutation_objective(
    metrics: Dict[str, Any],
    config: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Select the best mutation objective for a strategy based on its metrics.

    Returns one of: ``"increase_trades"``, ``"reduce_drawdown"``,
    ``"improve_entries"``, ``"improve_risk_adjusted"``, ``"simplify"``,
    ``"general_improvement"``.

    Args:
        metrics: Backtest metrics dict.
        config: Optional GA config.

    Returns:
        Objective string.
    """
    failure = diagnose_failure_mode(metrics, config)
    if failure is None:
        return "general_improvement"

    failure_lower = failure.lower()
    if "zero trades" in failure_lower or "too few trades" in failure_lower:
        return "increase_trades"
    elif "drawdown" in failure_lower:
        return "reduce_drawdown"
    elif "win rate" in failure_lower:
        return "improve_entries"
    elif "risk-adjusted" in failure_lower or "sharpe" in failure_lower:
        return "improve_risk_adjusted"
    elif "complexity" in failure_lower or "overfitting" in failure_lower:
        return "simplify"
    elif "large loss" in failure_lower:
        return "reduce_drawdown"
    else:
        return "general_improvement"
