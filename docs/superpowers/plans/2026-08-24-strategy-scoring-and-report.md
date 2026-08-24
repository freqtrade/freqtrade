# Strategy Scoring & Report Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Synthesize an already-computed `GateResult` (deflated Sharpe, permutation
p-value, PBO, optional fee-sensitivity, optional regime-breakdown) into one continuous
`robustness_score()` and one standardized `strategy_report()` string, replacing
`research/cli.py`'s ad-hoc print block.

**Architecture:** One new module, `research/scoring.py`, with two pure functions —
`robustness_score` (a renormalized weighted average) and `strategy_report` (a formatted
text block, byte-identical in content to what `research/cli.py` currently prints, plus
one new score line). `research/cli.py` is reduced to one function call.

**Tech Stack:** Python, pytest with real `GateResult` construction (no mocking) — no new
backtesting, no new data load, no new dependency.

**Spec:** `docs/superpowers/specs/2026-08-24-strategy-scoring-and-report-design.md`

## Global Constraints

- `WEIGHTS = {"deflated_sharpe": 0.35, "significance": 0.25, "pbo_inverse": 0.25,
  "cost_sensitivity": 0.10, "regime_consistency": 0.05}` — module-level constants in
  `research/scoring.py`, not a runtime parameter. Do not add a way to override them at
  call time; this is a deliberate starting default (`ponytail:`-flagged), matching this
  package's existing pattern for `trend_threshold`/`DEFAULT_FEE_MULTIPLIERS`.
- `robustness_score(result: GateResult) -> float`: always-present components are
  `result.deflated_sharpe` directly, `1.0 - result.permutation_p`, and
  `1.0 - result.pbo`. `cost_sensitivity` is added only when `result.fee_sensitivity` is
  truthy; `regime_consistency` only when `result.regime_breakdown` is truthy. Final
  score is `sum(WEIGHTS[k] * v) / sum(WEIGHTS[k])` over whichever components are
  present — a renormalized weighted average, always in `[0, 1]`.
- `cost_sensitivity`: `baseline_mult = min(result.fee_sensitivity)`,
  `stress_mult = max(result.fee_sensitivity)`. If `baseline_mult == stress_mult`:
  `1.0`. Elif `baseline_dsr <= 0`: `0.0`. Else:
  `max(0.0, min(1.0, stress_dsr / baseline_dsr))` where `baseline_dsr`/`stress_dsr` are
  `result.fee_sensitivity[baseline_mult]["deflated_sharpe"]` /
  `result.fee_sensitivity[stress_mult]["deflated_sharpe"]`.
- `regime_consistency`: `n_positive / n_total` where `n_total = len(result.regime_breakdown)`
  and `n_positive` counts buckets with `stats["mean_test_sharpe"] > 0`.
- **Deliberate, documented limitation (do not "fix"):** the score is only directly
  comparable between `GateResult`s computed with the *same* set of optional components
  present, because renormalization shifts the denominator. This was surfaced by an
  lmchatbot design review and is intentional — see the spec's "Known limitation"
  paragraph. A reviewer flagging this as a bug should be pointed at that paragraph, not
  have it "fixed" into fixed-denominator scoring.
- `strategy_report(result: GateResult, pair: str | None = None) -> str` returns
  (does not print) text byte-identical in format to `research/cli.py`'s current print
  block, with one new `"  robustness score  {score:.3f}"` line inserted as the second
  line (right after the verdict line, before `deflated_sharpe`). When `pair` is `None`,
  the regime-breakdown header omits the parenthetical pair name entirely (no stray
  `"(, "`).
- No new CLI flags. No changes to `run_promotion_gate`'s signature, `GateResult`'s
  fields, `research/regime.py`, or `research/cost_stress.py` — this plan only adds
  `research/scoring.py` and reduces `research/cli.py`'s print block to one call.
- Existing `research/tests/test_cli.py` tests must all continue to pass **unchanged** —
  they are the regression proof that the `cli.py` refactor preserves exact output.

---

### Task 1: `research/scoring.py` — robustness score and report

**Files:**
- Create: `research/scoring.py`
- Test: `research/tests/test_scoring.py`

**Interfaces:**
- Consumes: `research.gate.GateResult` (existing dataclass — `strategy_id: str, passed:
  bool, deflated_sharpe: float, permutation_p: float, pbo: float, mean_test_sharpe:
  float, n_trials: int, reasons: list[str], fee_sensitivity: dict[float, dict] | None =
  None, regime_breakdown: dict[str, dict] | None = None`, see `research/gate.py:26-39`).
- Produces: `robustness_score(result: GateResult) -> float` and
  `strategy_report(result: GateResult, pair: str | None = None) -> str` — both consumed
  directly by Task 2.

- [ ] **Step 1: Write the failing tests for `robustness_score`**

Create `research/tests/test_scoring.py`:

```python
# research/tests/test_scoring.py
import pytest

from research.gate import GateResult
from research.scoring import WEIGHTS, robustness_score, strategy_report


def _core_result(deflated_sharpe=0.90, permutation_p=0.02, pbo=0.15, **kwargs):
    """A GateResult with only the three always-present statistics populated,
    unless overridden."""
    return GateResult(
        strategy_id="TestStrategy",
        passed=True,
        deflated_sharpe=deflated_sharpe,
        permutation_p=permutation_p,
        pbo=pbo,
        mean_test_sharpe=1.1,
        n_trials=10,
        reasons=[],
        **kwargs,
    )


def test_robustness_score_with_only_core_stats():
    result = _core_result()

    score = robustness_score(result)

    # Independently computed, not calling robustness_score recursively.
    weighted_sum = (
        WEIGHTS["deflated_sharpe"] * 0.90
        + WEIGHTS["significance"] * (1.0 - 0.02)
        + WEIGHTS["pbo_inverse"] * (1.0 - 0.15)
    )
    weight_total = WEIGHTS["deflated_sharpe"] + WEIGHTS["significance"] + WEIGHTS["pbo_inverse"]
    assert score == pytest.approx(weighted_sum / weight_total)
    assert score == pytest.approx(0.9088235294117647)


def test_robustness_score_includes_cost_sensitivity_when_fee_sensitivity_present():
    result = _core_result(
        fee_sensitivity={
            1.0: {"mean_test_sharpe": 0.8, "deflated_sharpe": 0.90, "n_windows": 5},
            1.5: {"mean_test_sharpe": 0.4, "deflated_sharpe": 0.60, "n_windows": 5},
        }
    )

    score = robustness_score(result)

    assert score == pytest.approx(0.8833333333333333)


def test_robustness_score_includes_regime_consistency_when_regime_breakdown_present():
    result = _core_result(
        regime_breakdown={
            "Bull/High": {"n_windows": 2, "n_trades": 10, "mean_test_sharpe": 0.5, "total_return": 0.01},
            "Bull/Low": {"n_windows": 1, "n_trades": 5, "mean_test_sharpe": 0.3, "total_return": 0.005},
            "Bear/Low": {"n_windows": 1, "n_trades": 4, "mean_test_sharpe": 0.1, "total_return": 0.002},
            "Bear/High": {"n_windows": 1, "n_trades": 3, "mean_test_sharpe": -0.2, "total_return": -0.004},
        }
    )

    score = robustness_score(result)

    assert score == pytest.approx(0.9)


def test_robustness_score_with_both_fee_sensitivity_and_regime_breakdown():
    result = _core_result(
        fee_sensitivity={
            1.0: {"mean_test_sharpe": 0.8, "deflated_sharpe": 0.90, "n_windows": 5},
            1.5: {"mean_test_sharpe": 0.4, "deflated_sharpe": 0.60, "n_windows": 5},
        },
        regime_breakdown={
            "Bull/High": {"n_windows": 2, "n_trades": 10, "mean_test_sharpe": 0.5, "total_return": 0.01},
            "Bull/Low": {"n_windows": 1, "n_trades": 5, "mean_test_sharpe": 0.3, "total_return": 0.005},
            "Bear/Low": {"n_windows": 1, "n_trades": 4, "mean_test_sharpe": 0.1, "total_return": 0.002},
            "Bear/High": {"n_windows": 1, "n_trades": 3, "mean_test_sharpe": -0.2, "total_return": -0.004},
        },
    )

    score = robustness_score(result)

    assert score == pytest.approx(0.8766666666666667)


def test_robustness_score_cost_sensitivity_single_multiplier_is_one():
    result = _core_result(
        fee_sensitivity={1.0: {"mean_test_sharpe": 0.5, "deflated_sharpe": 0.5, "n_windows": 5}}
    )

    score = robustness_score(result)

    assert score == pytest.approx(0.9184210526315789)


def test_robustness_score_cost_sensitivity_zero_baseline_is_zero():
    result = _core_result(
        fee_sensitivity={
            1.0: {"mean_test_sharpe": 0.0, "deflated_sharpe": 0.0, "n_windows": 5},
            2.0: {"mean_test_sharpe": 0.1, "deflated_sharpe": 0.3, "n_windows": 5},
        }
    )

    score = robustness_score(result)

    assert score == pytest.approx(0.8131578947368421)


def test_robustness_score_stays_in_unit_interval():
    worst = _core_result(deflated_sharpe=0.0, permutation_p=1.0, pbo=1.0)
    best = _core_result(deflated_sharpe=1.0, permutation_p=0.0, pbo=0.0)

    assert robustness_score(worst) == pytest.approx(0.0)
    assert robustness_score(best) == pytest.approx(1.0)


def test_robustness_score_renormalization_makes_scores_incomparable_across_optional_flags():
    """Documents a deliberate, spec-approved limitation (not a bug to fix): identical
    core statistics score differently once an optional component is added, because the
    weighted-average denominator shifts. See the spec's "Known limitation" paragraph."""
    without_regime = _core_result(deflated_sharpe=0.7, permutation_p=0.1, pbo=0.2)
    with_regime = _core_result(
        deflated_sharpe=0.7,
        permutation_p=0.1,
        pbo=0.2,
        regime_breakdown={
            "Bull/High": {"n_windows": 3, "n_trades": 20, "mean_test_sharpe": 0.6, "total_return": 0.02},
        },
    )

    assert robustness_score(without_regime) == pytest.approx(0.788235294117647)
    assert robustness_score(with_regime) == pytest.approx(0.8)
    assert robustness_score(without_regime) != pytest.approx(robustness_score(with_regime))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest research/tests/test_scoring.py -v -k "robustness_score"`
Expected: FAIL/ERROR with `ModuleNotFoundError: No module named 'research.scoring'`

- [ ] **Step 3: Implement `robustness_score`**

Create `research/scoring.py`:

```python
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest research/tests/test_scoring.py -v -k "robustness_score"`
Expected: PASS (8 tests)

- [ ] **Step 5: Write the failing tests for `strategy_report`**

Append to `research/tests/test_scoring.py`:

```python
def test_strategy_report_pass_case_shows_verdict_core_stats_and_score():
    result = _core_result(passed=True, deflated_sharpe=0.97, permutation_p=0.01, pbo=0.1)

    report = strategy_report(result)

    assert "TestStrategy: PASS" in report
    assert "robustness score" in report
    assert "deflated_sharpe   0.970" in report
    assert "permutation p     0.010" in report
    assert "PBO               0.100" in report
    assert "mean OOS sharpe   1.100" in report
    assert "trials (ledger)   10" in report


def test_strategy_report_fail_case_shows_reasons():
    result = GateResult(
        strategy_id="TestStrategy",
        passed=False,
        deflated_sharpe=0.4,
        permutation_p=0.3,
        pbo=0.7,
        mean_test_sharpe=0.1,
        n_trials=12,
        reasons=["deflated_sharpe 0.400 below threshold 0.95"],
    )

    report = strategy_report(result)

    assert "TestStrategy: FAIL" in report
    assert "deflated_sharpe 0.400 below threshold 0.95" in report


def test_strategy_report_includes_fee_sensitivity_table_when_present():
    result = _core_result(
        fee_sensitivity={
            1.0: {"mean_test_sharpe": 0.87, "deflated_sharpe": 0.91, "n_windows": 5},
            1.5: {"mean_test_sharpe": 0.33, "deflated_sharpe": 0.52, "n_windows": 5},
        }
    )

    report = strategy_report(result)

    assert "fee sensitivity" in report
    assert "baseline" in report
    assert "1.50x fee" in report
    assert "slippage" not in report.lower()


def test_strategy_report_includes_regime_breakdown_with_pair_name_when_given():
    result = _core_result(
        regime_breakdown={
            "Bull/High": {"n_windows": 2, "n_trades": 14, "mean_test_sharpe": 0.42, "total_return": 0.0012},
            "Bear/Low": {"n_windows": 1, "n_trades": 6, "mean_test_sharpe": -1.10, "total_return": -0.0034},
        }
    )

    report = strategy_report(result, pair="BTC/USDT")

    assert "regime breakdown" in report
    assert "BTC/USDT" in report
    assert "Bull/High" in report
    assert "Bear/Low" in report


def test_strategy_report_omits_pair_name_when_not_given():
    result = _core_result(
        regime_breakdown={
            "Bull/High": {"n_windows": 2, "n_trades": 14, "mean_test_sharpe": 0.42, "total_return": 0.0012},
        }
    )

    report = strategy_report(result, pair=None)

    assert "regime breakdown" in report
    assert "Bull/High" in report
    assert "(, " not in report
```

- [ ] **Step 6: Run the tests to verify they fail**

Run: `pytest research/tests/test_scoring.py -v -k "strategy_report"`
Expected: FAIL with `ImportError: cannot import name 'strategy_report' from 'research.scoring'`

- [ ] **Step 7: Implement `strategy_report`**

Append to `research/scoring.py`:

```python
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
```

- [ ] **Step 8: Run all tests in the file to verify they pass**

Run: `pytest research/tests/test_scoring.py -v`
Expected: PASS (13 tests)

- [ ] **Step 9: Lint and format**

Run: `ruff check research/scoring.py research/tests/test_scoring.py` and
`ruff format --check research/scoring.py research/tests/test_scoring.py`
Expected: no errors (fix with `ruff check --fix` / `ruff format` if needed, then re-run
the test suite from Step 8 to confirm nothing broke)

- [ ] **Step 10: Commit**

```bash
git add research/scoring.py research/tests/test_scoring.py
git commit -m "feat(research): add scoring.py -- robustness_score and strategy_report"
```

---

### Task 2: `research/cli.py` — replace print block with `strategy_report`

**Files:**
- Modify: `research/cli.py:74-103` (the `gate` command's print block)
- Test: `research/tests/test_cli.py` (no new tests — this task's job is to keep every
  existing test passing unchanged, proving the refactor preserves output byte-for-byte)

**Interfaces:**
- Consumes: `research.scoring.strategy_report` (Task 1).
- Produces: nothing further downstream — this is the final task in the plan.

- [ ] **Step 1: Run the existing tests to confirm the current baseline passes**

Run: `pytest research/tests/test_cli.py -v`
Expected: PASS (5 tests) — this is the pre-change baseline; Step 3 must not break any of
these.

- [ ] **Step 2: Add the import**

In `research/cli.py`, add to the existing import block (alongside
`from research.gate import run_promotion_gate`):

```python
from research.scoring import strategy_report
```

- [ ] **Step 3: Replace the print block**

In `research/cli.py`, replace this entire block (currently `research/cli.py:74-103`):

```python
        verdict = "PASS" if result.passed else "FAIL"
        print(f"{result.strategy_id}: {verdict}")
        print(f"  deflated_sharpe   {result.deflated_sharpe:.3f}")
        print(f"  permutation p     {result.permutation_p:.3f}")
        print(f"  PBO               {result.pbo:.3f}")
        print(f"  mean OOS sharpe   {result.mean_test_sharpe:.3f}")
        print(f"  trials (ledger)   {result.n_trials}")
        for reason in result.reasons:
            print(f"  - {reason}")
        if result.fee_sensitivity:
            print("  fee sensitivity (informational, not part of PASS/FAIL):")
            for mult, stats in result.fee_sensitivity.items():
                label = f"{mult:.2f}x fee" + (" (baseline)" if mult == 1.0 else "")
                print(
                    f"    {label:<22} mean OOS sharpe {stats['mean_test_sharpe']:>6.2f}"
                    f"   deflated_sharpe (n_trials=1) {stats['deflated_sharpe']:.3f}"
                )
        if result.regime_breakdown:
            print(
                f"  regime breakdown ({args.pairs.split(',')[0]}, informational, "
                "not part of PASS/FAIL):"
            )
            for label, stats in result.regime_breakdown.items():
                print(
                    f"    {label:<15} {stats['n_windows']:>2} windows"
                    f"   {stats['n_trades']:>3} trades"
                    f"   mean sharpe {stats['mean_test_sharpe']:>6.2f}"
                    f"   total return {stats['total_return']:>8.4f}"
                )
        return 0 if result.passed else 1
```

with:

```python
        print(strategy_report(result, pair=args.pairs.split(",")[0]))
        return 0 if result.passed else 1
```

- [ ] **Step 4: Run the existing tests to confirm nothing broke**

Run: `pytest research/tests/test_cli.py -v`
Expected: PASS (5 tests) — identical result to Step 1's baseline. If anything fails,
compare `strategy_report`'s output text against the exact format the removed print block
produced; the two must match byte-for-byte (this is why Task 1's `strategy_report` was
written to reproduce that exact format).

- [ ] **Step 5: Lint and format**

Run: `ruff check research/cli.py` and `ruff format --check research/cli.py`
Expected: no errors (fix and re-run Step 4 if needed)

- [ ] **Step 6: Run the full research suite**

Run: `pytest research/ -v`
Expected: PASS (every test in `research/tests/`, confirming Tasks 1-2 compose cleanly
with the rest of the package)

- [ ] **Step 7: Commit**

```bash
git add research/cli.py
git commit -m "refactor(research): cli.py prints strategy_report() instead of an inline print block"
```
