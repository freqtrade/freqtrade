# research/scoring.py
"""Strategy scoring and reporting: synthesizes an already-computed GateResult (deflated
Sharpe, permutation p-value, PBO, and the optional fee-sensitivity/regime-breakdown
add-ons) into one continuous robustness_score() and one standardized strategy_report()
string. No new statistics are computed here -- every input is already produced by
research.gate.run_promotion_gate before either function in this module runs. See
docs/superpowers/specs/2026-08-24-strategy-scoring-and-report-design.md for the full
design, including why the score is only directly comparable between GateResults computed
with the same set of optional components present (a documented, deliberate tradeoff of
renormalizing over available evidence rather than penalizing runs that skip the
compute-costly optional fee-sensitivity/regime-breakdown analyses).
"""

from __future__ import annotations

from research.gate import GateResult


# ponytail: starting weights, not empirically derived -- adjust based on real usage once
# this runs against real strategies. Not a runtime parameter; see the spec for why.
WEIGHTS = {
    "deflated_sharpe": 0.35,
    "significance": 0.25,
    "pbo_inverse": 0.25,
    "cost_sensitivity": 0.10,
    "regime_consistency": 0.05,
}


def robustness_score(result: GateResult) -> float:
    """Weighted average of GateResult's already-computed statistics, renormalized over
    whichever components are actually present. Always in [0, 1].

    Always-present components: deflated_sharpe directly, 1 - permutation_p
    ("significance"), 1 - pbo ("pbo_inverse") -- all three are already probabilities in
    [0, 1] by construction (see research/statistics.py, research/pbo.py).

    "cost_sensitivity", only when result.fee_sensitivity is present: the fraction of the
    lowest-tested fee multiplier's deflated Sharpe that survives at the highest-tested
    multiplier, clipped to [0, 1]. 1.0 (fail open) if only one multiplier was tested --
    no stress signal to penalize. 0.0 (fail closed) if the baseline deflated Sharpe is
    non-positive -- the ratio would be meaningless.

    "regime_consistency", only when result.regime_breakdown is present: the fraction of
    regime buckets with a positive mean_test_sharpe -- does the edge show up broadly, or
    only in one favorable regime.

    Known limitation, deliberate: scores from GateResults with different optional
    components present are NOT directly comparable, since renormalization shifts the
    denominator. Read this as "how robust this candidate looks given the evidence
    gathered for it," not a universal ranking number.
    """
    components: dict[str, float] = {
        "deflated_sharpe": result.deflated_sharpe,
        "significance": 1.0 - result.permutation_p,
        "pbo_inverse": 1.0 - result.pbo,
    }

    if result.fee_sensitivity:
        baseline_mult = min(result.fee_sensitivity)
        stress_mult = max(result.fee_sensitivity)
        baseline_dsr = result.fee_sensitivity[baseline_mult]["deflated_sharpe"]
        stress_dsr = result.fee_sensitivity[stress_mult]["deflated_sharpe"]
        if baseline_mult == stress_mult:
            cost_sensitivity = 1.0
        elif baseline_dsr <= 0:
            cost_sensitivity = 0.0
        else:
            cost_sensitivity = max(0.0, min(1.0, stress_dsr / baseline_dsr))
        components["cost_sensitivity"] = cost_sensitivity

    if result.regime_breakdown:
        n_total = len(result.regime_breakdown)
        n_positive = sum(
            1 for stats in result.regime_breakdown.values() if stats["mean_test_sharpe"] > 0
        )
        components["regime_consistency"] = n_positive / n_total

    weighted_sum = sum(WEIGHTS[k] * v for k, v in components.items())
    weight_total = sum(WEIGHTS[k] for k in components)
    return weighted_sum / weight_total


def strategy_report(result: GateResult, pair: str | None = None) -> str:
    """Standardized human-readable report for one GateResult -- verdict, robustness
    score, the three core statistics, any reasons the gate failed, and the
    fee-sensitivity / regime-breakdown tables when present. Returns the text (does not
    print it).

    `pair` names which pair a present regime_breakdown was classified against (the
    caller's responsibility to supply, since GateResult itself doesn't carry it -- see
    the spec). When None, the regime-breakdown header omits the parenthetical pair name.
    """
    score = robustness_score(result)
    verdict = "PASS" if result.passed else "FAIL"
    lines = [
        f"{result.strategy_id}: {verdict}",
        f"  robustness score  {score:.3f}",
        f"  deflated_sharpe   {result.deflated_sharpe:.3f}",
        f"  permutation p     {result.permutation_p:.3f}",
        f"  PBO               {result.pbo:.3f}",
        f"  mean OOS sharpe   {result.mean_test_sharpe:.3f}",
        f"  trials (ledger)   {result.n_trials}",
    ]
    for reason in result.reasons:
        lines.append(f"  - {reason}")

    if result.fee_sensitivity:
        lines.append("  fee sensitivity (informational, not part of PASS/FAIL):")
        for mult, stats in result.fee_sensitivity.items():
            label = f"{mult:.2f}x fee" + (" (baseline)" if mult == 1.0 else "")
            lines.append(
                f"    {label:<22} mean OOS sharpe {stats['mean_test_sharpe']:>6.2f}"
                f"   deflated_sharpe (n_trials=1) {stats['deflated_sharpe']:.3f}"
            )

    if result.regime_breakdown:
        pair_part = f"{pair}, " if pair else ""
        lines.append(f"  regime breakdown ({pair_part}informational, not part of PASS/FAIL):")
        for label, stats in result.regime_breakdown.items():
            lines.append(
                f"    {label:<15} {stats['n_windows']:>2} windows"
                f"   {stats['n_trades']:>3} trades"
                f"   mean sharpe {stats['mean_test_sharpe']:>6.2f}"
                f"   total return {stats['total_return']:>8.4f}"
            )

    return "\n".join(lines)
