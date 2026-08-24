# research/cost_stress.py
"""Fee-sensitivity stress test: re-evaluates a promotion gate candidate's
already-selected walk-forward parameters at progressively higher (worse)
transaction fee assumptions, to see whether its edge is a thin margin over
baseline costs or survives materially worse ones. Informational only --
never changes a gate's PASS/FAIL verdict.

This is NOT a slippage/market-impact model -- freqtrade's backtester has no
execution-price slippage simulation at all (orders fill at the exact
requested price). A flat fee-rate multiplier is a legitimate cost-sensitivity
/ margin-of-safety test, but it gets slippage's actual structure wrong (which
scales with order size/liquidity/volatility, not as a fixed percentage), so
it is named and reported as "fee sensitivity" throughout, never "slippage".
See docs/superpowers/specs/2026-08-24-fee-sensitivity-stress-test-design.md
for the full reasoning, including why n_trials=1 is correct here (no new
selection happens at any fee level, so no new multiple-testing penalty
applies -- the original gate's trial count already paid for that once).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from freqtrade.optimize.backtesting import Backtesting
from research.statistics import deflated_sharpe_ratio
from research.walkforward import WalkForwardRunner, WindowResult


DEFAULT_FEE_MULTIPLIERS = (1.0, 1.25, 1.5, 2.0)


def fee_sensitivity(
    config: dict,
    pairs: list[str],
    timeframe: str,
    datadir: Path,
    window_results: list[WindowResult],
    multipliers: tuple[float, ...] = DEFAULT_FEE_MULTIPLIERS,
    periods_per_year: int = 365,
) -> dict[float, dict]:
    """Re-evaluate every `WindowResult`'s already-selected `best_params` at
    each fee multiplier (`base_fee * multiplier`), aggregating exactly as
    `research.gate.run_promotion_gate` does: mean per-window test Sharpe fed
    into `deflated_sharpe_ratio` with `n_obs` = total concatenated
    `test_returns` and `n_trials=1`.

    `base_fee` is read once from a throwaway `Backtesting(config)` instance's
    resolved `.fee` -- the exact same fee the original gate run used, since
    `config` is passed through unchanged. The `1.0` multiplier therefore
    reproduces the original gate's own fee exactly (a control, not a stress
    level).

    Returns `{multiplier: {"mean_test_sharpe": float, "deflated_sharpe":
    float, "n_windows": int}}`, one entry per multiplier given.
    """
    if not multipliers or any(m <= 0 for m in multipliers):
        raise ValueError("multipliers must be non-empty and all values > 0")

    base_fee = Backtesting(config).fee
    runner = WalkForwardRunner(config, pairs, timeframe, datadir)

    report: dict[float, dict] = {}
    for multiplier in multipliers:
        fee = base_fee * multiplier
        results = [
            runner.evaluate_fixed_params(wr.window, wr.best_params, fee_override=fee)
            for wr in window_results
        ]
        all_test_returns = [r for res in results for r in res.test_returns]
        mean_test_sharpe = float(np.mean([res.test_sharpe for res in results]))
        deflated = deflated_sharpe_ratio(
            mean_test_sharpe,
            n_obs=len(all_test_returns),
            n_trials=1,
            periods_per_year=periods_per_year,
        )
        report[multiplier] = {
            "mean_test_sharpe": mean_test_sharpe,
            "deflated_sharpe": deflated,
            "n_windows": len(results),
        }
    return report
