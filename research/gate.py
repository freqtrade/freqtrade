# research/gate.py
"""Promotion gate: the final go/no-go check a strategy must pass before it is
trusted, built on top of walk-forward evaluation (`research.walkforward`) and
the statistical tests in `research.statistics`/`research.pbo`. Ties those
pieces together, logs the outcome to the candidate ledger (`research.ledger`),
and returns a single pass/fail verdict with the reasons for it.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from research.cost_stress import fee_sensitivity
from research.db import get_engine, get_session
from research.ledger import family_of, family_trial_count, log_candidate_result
from research.parameter_stability import parameter_stability
from research.pbo import probability_of_backtest_overfitting
from research.regime import classify_regimes, regime_report
from research.statistics import benjamini_hochberg, deflated_sharpe_ratio, permutation_test
from research.walkforward import WalkForwardRunner, generate_windows


@dataclass
class GateResult:
    """Verdict and supporting statistics from one `run_promotion_gate` call."""

    strategy_id: str
    passed: bool
    deflated_sharpe: float
    permutation_p: float
    pbo: float
    mean_test_sharpe: float
    n_trials: int
    reasons: list[str]
    fee_sensitivity: dict[float, dict] | None = None
    regime_breakdown: dict[str, dict] | None = None
    parameter_stability: float | None = None


def run_promotion_gate(
    config: dict,
    strategy_id: str,
    pairs: list[str],
    timeframe: str,
    datadir: Path,
    start: datetime,
    end: datetime,
    train_days: int,
    test_days: int,
    param_grid: list[dict],
    db_path: str = "user_data/research.sqlite",
    dsr_threshold: float = 0.95,
    fdr_q: float = 0.05,
    pbo_threshold: float = 0.5,
    periods_per_year: int = 365,
    fee_sensitivity_multipliers: tuple[float, ...] | None = None,
    include_regime_breakdown: bool = False,
    include_parameter_stability: bool = False,
) -> GateResult:
    """Run walk-forward evaluation for `strategy_id` and decide whether it is
    statistically fit to promote, logging the outcome to the candidate ledger.

    Requires at least 4 walk-forward windows (raises `ValueError` otherwise) --
    fewer windows leave the OOS statistics (DSR, permutation test, PBO) too
    thin to be meaningful, so it is better to fail loudly than to gate on noise.

    Trial count and ledger writes follow a strict count-then-write order:
    `family_trial_count` reads the ledger's accumulated trial history for this
    strategy's family BEFORE this run's own row is written, so the deflated
    Sharpe ratio never deflates against trials this run hasn't finished yet
    (which would be circular) and never omits trials prior runs already spent.
    The candidate row is logged regardless of pass/fail, so the ledger stays a
    complete trial history for future DSR calculations.

    A strategy passes only if it clears all three checks: deflated Sharpe >=
    `dsr_threshold`, the permutation-test p-value survives Benjamini-Hochberg
    FDR control at `fdr_q`, and PBO <= `pbo_threshold`.

    `include_regime_breakdown` classifies using only `pairs[0]` as the reference pair --
    multi-pair blending is out of scope for this feature.
    """
    windows = generate_windows(start, end, train_days, test_days)
    if len(windows) < 4:
        raise ValueError(
            f"Need at least 4 walk-forward windows for a meaningful gate, got "
            f"{len(windows)}. Widen start/end or shrink train_days/test_days."
        )

    runner = WalkForwardRunner(config, pairs, timeframe, datadir)
    results = runner.run(windows, param_grid)

    all_test_returns = [r for wr in results for r in wr.test_returns]
    n_obs = len(all_test_returns)
    mean_test_sharpe = float(np.mean([wr.test_sharpe for wr in results]))

    variant_keys = sorted({k for wr in results for k in wr.variant_returns})
    variant_matrix = np.array([[wr.variant_returns[key] for wr in results] for key in variant_keys])
    pbo_result = probability_of_backtest_overfitting(variant_matrix)

    engine = get_engine(db_path)
    session = get_session(engine)
    family = family_of(strategy_id)
    this_run_trials = len(param_grid) * len(windows)
    # count-then-write: read ledger history BEFORE writing this run's own row, so a
    # run never deflates its own significance test against trials it hasn't finished.
    n_trials = family_trial_count(session, family, declared=this_run_trials)

    deflated = deflated_sharpe_ratio(
        mean_test_sharpe, n_obs=n_obs, n_trials=n_trials, periods_per_year=periods_per_year
    )
    perm_p = permutation_test(mean_test_sharpe, np.array(all_test_returns))
    survived_bh = benjamini_hochberg([perm_p], q=fdr_q)[0]

    reasons: list[str] = []
    if deflated < dsr_threshold:
        reasons.append(f"deflated_sharpe {deflated:.3f} below threshold {dsr_threshold}")
    if not survived_bh:
        reasons.append(f"permutation p-value {perm_p:.3f} fails BH-FDR at q={fdr_q}")
    if pbo_result["pbo"] > pbo_threshold:
        reasons.append(f"PBO {pbo_result['pbo']:.3f} above threshold {pbo_threshold}")
    passed = not reasons

    fee_report = None
    if passed and fee_sensitivity_multipliers is not None:
        fee_report = fee_sensitivity(
            config,
            pairs,
            timeframe,
            datadir,
            results,
            multipliers=fee_sensitivity_multipliers,
            periods_per_year=periods_per_year,
        )

    log_candidate_result(
        session,
        strategy_id=strategy_id,
        params={"grid": param_grid},
        universe=",".join(pairs),
        timeframe=timeframe,
        discovery_start=start.isoformat(),
        discovery_end=end.isoformat(),
        n_trials_this_run=this_run_trials,
        is_sharpe=float(np.mean([wr.train_sharpe for wr in results])),
        oos_sharpe=mean_test_sharpe,
        deflated_sharpe=deflated,
        permutation_p=perm_p,
        pbo=pbo_result["pbo"],
        survived=passed,
    )
    session.commit()

    regime_breakdown = None
    if include_regime_breakdown:
        labels = classify_regimes(pairs[0], timeframe, datadir, windows)
        regime_breakdown = regime_report(results, labels)

    stability = None
    if include_parameter_stability:
        stability = parameter_stability(variant_matrix)

    return GateResult(
        strategy_id=strategy_id,
        passed=passed,
        deflated_sharpe=deflated,
        permutation_p=perm_p,
        pbo=pbo_result["pbo"],
        mean_test_sharpe=mean_test_sharpe,
        n_trials=n_trials,
        reasons=reasons,
        fee_sensitivity=fee_report,
        regime_breakdown=regime_breakdown,
        parameter_stability=stability,
    )
