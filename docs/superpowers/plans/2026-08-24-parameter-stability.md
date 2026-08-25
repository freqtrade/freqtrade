# Parameter Stability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a "parameter stability" robustness check — does a candidate's edge hold across
a region of its parameter grid, not just one lucky combination — as a new, purely
informational `float` field on `GateResult`, folded into `robustness_score` and
`strategy_report`.

**Architecture:** One new pure function, `research/parameter_stability.py`'s
`parameter_stability(variant_matrix) -> float`, reusing the exact `variant_matrix`
`research/gate.py` already builds for PBO (zero new backtests). `research/gate.py`,
`research/scoring.py`, and `research/cli.py` are extended to wire an opt-in flag through to
it — no existing behavior changes when the flag is unused.

**Tech Stack:** Python, numpy, pytest (real `UNITTEST/BTC` fixture data for the
`run_promotion_gate` integration tests, plain array construction for the pure-function unit
tests — no mocking of `parameter_stability`'s own logic).

**Spec:** `docs/superpowers/specs/2026-08-24-parameter-stability-design.md`

## Global Constraints

- `parameter_stability(variant_matrix: np.ndarray) -> float` — fraction of rows (grid
  variants) whose mean across columns (windows) is `> 0`. Always in `[0, 1]`.
- `variant_matrix` is the **exact same array** `run_promotion_gate` already builds for
  `probability_of_backtest_overfitting` (`variant_keys`/`variant_matrix` at
  `research/gate.py:97-99`) — no new backtests, no new data collection.
- Single-variant grid (`variant_matrix.shape[0] == 1`): fails **open to `1.0`** — no region
  to be unstable across, so this is not evidence against the candidate. Same fail-open
  convention as `scoring.robustness_score`'s existing `cost_sensitivity` component.
- `variant_matrix.ndim != 2` or zero rows: raises `ValueError` — a caller-contract
  violation, not a data condition to paper over.
- `run_promotion_gate` gains `include_parameter_stability: bool = False` (default off —
  existing callers/tests unaffected) and `GateResult` gains
  `parameter_stability: float | None = None`.
- Computed **unconditionally on pass/fail** — mirrors the regime-breakdown precedent, not
  fee-sensitivity's `passed`-only gating (see spec's "What this is not"). Costs nothing
  extra either way since `variant_matrix` is already built.
- `scoring.WEIGHTS` gains `"parameter_stability": 0.05` — same order of magnitude as
  `regime_consistency`, still `ponytail:`-flagged as a starting value, not empirically
  derived (the existing disclaimer on `WEIGHTS` already covers this — do not add a second
  one).
- `robustness_score` uses `result.parameter_stability` directly as the component value
  (already a probability in `[0, 1]`, no `1 - x` transform — unlike `permutation_p`/`pbo`).
- `strategy_report` gets **one new single-float line**, same style as `deflated_sharpe`/
  `permutation p`/`PBO`/`mean OOS sharpe` — NOT a fraction breakdown (`GateResult` only
  stores the float, no `n_positive`/`n_total` counts are available at report time).
- No mocking of `parameter_stability`'s own logic in tests — plain `np.ndarray`
  construction, matching `research/tests/test_pbo.py`'s style for pure numeric functions.

---

### Task 1: `research/parameter_stability.py` — the pure stability function

**Files:**
- Create: `research/parameter_stability.py`
- Test: `research/tests/test_parameter_stability.py`

**Interfaces:**
- Consumes: nothing from this codebase — takes a plain `np.ndarray`.
- Produces: `parameter_stability(variant_matrix: np.ndarray) -> float`, consumed directly
  by Task 2.

- [ ] **Step 1: Write the failing tests**

Create `research/tests/test_parameter_stability.py`:

```python
# research/tests/test_parameter_stability.py
import numpy as np
import pytest

from research.parameter_stability import parameter_stability


def test_all_variants_profitable_returns_one():
    matrix = np.array([[0.01, 0.02, 0.01], [0.03, 0.01, 0.02]])
    assert parameter_stability(matrix) == pytest.approx(1.0)


def test_no_variants_profitable_returns_zero():
    matrix = np.array([[-0.01, -0.02, -0.01], [-0.03, -0.01, -0.02]])
    assert parameter_stability(matrix) == pytest.approx(0.0)


def test_mixed_variants_returns_exact_fraction():
    # row means: 0.01 (>0), -0.01 (<0), 0.04/3 (>0, cells straddle zero), -0.05/3 (<0)
    # -> 2 of 4 variants profitable
    matrix = np.array(
        [
            [0.01, 0.02, 0.00],
            [-0.01, -0.02, 0.00],
            [0.05, -0.03, 0.02],
            [-0.05, 0.01, -0.01],
        ]
    )
    assert parameter_stability(matrix) == pytest.approx(0.5)


def test_single_variant_grid_fails_open_to_one():
    matrix = np.array([[-0.5, -0.5, -0.5]])  # unprofitable, but no region to test
    assert parameter_stability(matrix) == pytest.approx(1.0)


def test_raises_on_non_2d_input():
    with pytest.raises(ValueError, match="2-D"):
        parameter_stability(np.array([0.01, 0.02]))


def test_raises_on_zero_rows():
    with pytest.raises(ValueError, match="2-D"):
        parameter_stability(np.zeros((0, 3)))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest research/tests/test_parameter_stability.py -v`
Expected: FAIL/ERROR with `ModuleNotFoundError: No module named 'research.parameter_stability'`

- [ ] **Step 3: Implement `parameter_stability`**

Create `research/parameter_stability.py`:

```python
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
            f"variant_matrix must be 2-D with at least one row, got shape "
            f"{variant_matrix.shape}"
        )
    if variant_matrix.shape[0] == 1:
        return 1.0

    row_means = variant_matrix.mean(axis=1)
    return float((row_means > 0).sum() / len(row_means))
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest research/tests/test_parameter_stability.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Lint and format**

Run: `ruff check research/parameter_stability.py research/tests/test_parameter_stability.py`
and `ruff format --check research/parameter_stability.py research/tests/test_parameter_stability.py`
Expected: no errors (fix with `ruff check --fix` / `ruff format` if needed, then re-run
Step 4 to confirm nothing broke)

- [ ] **Step 6: Commit**

```bash
git add research/parameter_stability.py research/tests/test_parameter_stability.py
git commit -m "feat(research): add parameter_stability.py -- in-sample grid-variant plateau check"
```

---

### Task 2: `research/gate.py` — wire parameter stability into the promotion gate

**Files:**
- Modify: `research/gate.py:17-23` (imports), `:26-39` (`GateResult`), `:42-170`
  (`run_promotion_gate`)
- Test: `research/tests/test_gate.py`

**Interfaces:**
- Consumes: `research.parameter_stability.parameter_stability` (Task 1). `variant_matrix`
  is already in scope inside `run_promotion_gate` (`research/gate.py:98`) — no duplicate
  computation.
- Produces: `GateResult.parameter_stability: float | None`, and
  `run_promotion_gate(..., include_parameter_stability: bool = False)` — both consumed by
  Task 3 and Task 4.

- [ ] **Step 1: Write the failing tests**

Append to `research/tests/test_gate.py`:

```python
def test_run_promotion_gate_attaches_parameter_stability_when_requested_and_passes(
    mocker, tmp_path
):
    conf = _conf()
    _patch(mocker)
    full_data = history.load_data(datadir=TESTDATADIR, timeframe="5m", pairs=["UNITTEST/BTC"])
    min_date, max_date = get_timerange(full_data)
    total_days = max(8, (max_date - min_date).days)
    train_days = max(1, total_days // 8)
    test_days = max(1, total_days // 16)

    # Permissive thresholds guarantee a real, unmocked PASS (see the fee-sensitivity pass
    # test above for the same pattern).
    result = run_promotion_gate(
        config=conf,
        strategy_id="StrategyTestV3",
        pairs=["UNITTEST/BTC"],
        timeframe="5m",
        datadir=TESTDATADIR,
        start=min_date,
        end=max_date,
        train_days=train_days,
        test_days=test_days,
        param_grid=[{"buy_rsi": 25}, {"buy_rsi": 35}],
        db_path=str(tmp_path / "research.sqlite"),
        dsr_threshold=0.0,
        fdr_q=1.0,
        pbo_threshold=1.0,
        include_parameter_stability=True,
    )

    assert result.passed is True
    assert result.parameter_stability is not None
    assert 0.0 <= result.parameter_stability <= 1.0


def test_run_promotion_gate_attaches_parameter_stability_when_requested_and_fails(
    mocker, tmp_path
):
    conf = _conf()
    _patch(mocker)
    full_data = history.load_data(datadir=TESTDATADIR, timeframe="5m", pairs=["UNITTEST/BTC"])
    min_date, max_date = get_timerange(full_data)
    total_days = max(8, (max_date - min_date).days)
    train_days = max(1, total_days // 8)
    test_days = max(1, total_days // 16)

    # Impossible dsr_threshold guarantees a real FAIL. Parameter stability must still be
    # attached -- this is the test that actually proves the pass/fail-independence was
    # implemented, not just described in the spec.
    result = run_promotion_gate(
        config=conf,
        strategy_id="StrategyTestV3",
        pairs=["UNITTEST/BTC"],
        timeframe="5m",
        datadir=TESTDATADIR,
        start=min_date,
        end=max_date,
        train_days=train_days,
        test_days=test_days,
        param_grid=[{"buy_rsi": 25}, {"buy_rsi": 35}],
        db_path=str(tmp_path / "research.sqlite"),
        dsr_threshold=1.1,
        include_parameter_stability=True,
    )

    assert result.passed is False
    assert result.parameter_stability is not None
    assert 0.0 <= result.parameter_stability <= 1.0


def test_run_promotion_gate_omits_parameter_stability_by_default(mocker, tmp_path):
    conf = _conf()
    _patch(mocker)
    full_data = history.load_data(datadir=TESTDATADIR, timeframe="5m", pairs=["UNITTEST/BTC"])
    min_date, max_date = get_timerange(full_data)
    total_days = max(8, (max_date - min_date).days)
    train_days = max(1, total_days // 8)
    test_days = max(1, total_days // 16)

    result = run_promotion_gate(
        config=conf,
        strategy_id="StrategyTestV3",
        pairs=["UNITTEST/BTC"],
        timeframe="5m",
        datadir=TESTDATADIR,
        start=min_date,
        end=max_date,
        train_days=train_days,
        test_days=test_days,
        param_grid=[{"buy_rsi": 25}, {"buy_rsi": 35}],
        db_path=str(tmp_path / "research.sqlite"),
    )

    assert result.parameter_stability is None
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest research/tests/test_gate.py -v -k parameter_stability`
Expected: FAIL with `TypeError: run_promotion_gate() got an unexpected keyword argument
'include_parameter_stability'`

- [ ] **Step 3: Wire `include_parameter_stability` into `run_promotion_gate` and `GateResult`**

In `research/gate.py`, add the import (alphabetically, between the existing `ledger` and
`pbo` imports):

```python
from research.ledger import family_of, family_trial_count, log_candidate_result
from research.parameter_stability import parameter_stability
from research.pbo import probability_of_backtest_overfitting
```

Add the new field to `GateResult` (after the existing `regime_breakdown` field):

```python
    fee_sensitivity: dict[float, dict] | None = None
    regime_breakdown: dict[str, dict] | None = None
    parameter_stability: float | None = None
```

Add the new parameter to `run_promotion_gate`'s signature (after
`include_regime_breakdown`):

```python
    fee_sensitivity_multipliers: tuple[float, ...] | None = None,
    include_regime_breakdown: bool = False,
    include_parameter_stability: bool = False,
) -> GateResult:
```

Inside `run_promotion_gate`, immediately after the existing `regime_breakdown` block (which
reads `windows`/`results`) and before the `return GateResult(...)` statement, add the
unconditional stability block:

```python
    stability = None
    if include_parameter_stability:
        stability = parameter_stability(variant_matrix)
```

Note this block does **not** check `passed` -- it costs nothing extra either way since
`variant_matrix` (line 98) is already built for PBO, unconditionally, earlier in this same
function.

Finally, thread it into the `GateResult(...)` construction (after the existing
`regime_breakdown=regime_breakdown,` line):

```python
        regime_breakdown=regime_breakdown,
        parameter_stability=stability,
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest research/tests/test_gate.py -v`
Expected: PASS (all tests in the file, including the 3 new ones and every pre-existing
one -- confirms no regression)

- [ ] **Step 5: Lint and format**

Run: `ruff check research/gate.py research/tests/test_gate.py` and
`ruff format --check research/gate.py research/tests/test_gate.py`
Expected: no errors (fix and re-run Step 4 if needed)

- [ ] **Step 6: Commit**

```bash
git add research/gate.py research/tests/test_gate.py
git commit -m "feat(research): wire parameter stability into run_promotion_gate (opt-in, runs on pass or fail)"
```

---

### Task 3: `research/scoring.py` — fold parameter stability into scoring and the report

**Files:**
- Modify: `research/scoring.py:21-27` (`WEIGHTS`), `:30-81` (`robustness_score`), `:84-128`
  (`strategy_report`)
- Test: `research/tests/test_scoring.py`

**Interfaces:**
- Consumes: `GateResult.parameter_stability` (Task 2).
- Produces: nothing further downstream -- `robustness_score`/`strategy_report` are the
  final consumers in this package; Task 4's CLI already calls `strategy_report` and needs
  no changes of its own to pick this up.

- [ ] **Step 1: Write the failing tests**

Append to `research/tests/test_scoring.py`:

```python
def test_robustness_score_includes_parameter_stability_when_present():
    result = _core_result(parameter_stability=0.75)

    score = robustness_score(result)

    weighted_sum = (
        WEIGHTS["deflated_sharpe"] * 0.90
        + WEIGHTS["significance"] * (1.0 - 0.02)
        + WEIGHTS["pbo_inverse"] * (1.0 - 0.15)
        + WEIGHTS["parameter_stability"] * 0.75
    )
    weight_total = (
        WEIGHTS["deflated_sharpe"]
        + WEIGHTS["significance"]
        + WEIGHTS["pbo_inverse"]
        + WEIGHTS["parameter_stability"]
    )
    assert score == pytest.approx(weighted_sum / weight_total)


def test_robustness_score_with_all_four_optional_components():
    result = _core_result(
        parameter_stability=0.75,
        fee_sensitivity={
            1.0: {"mean_test_sharpe": 0.8, "deflated_sharpe": 0.90, "n_windows": 5},
            1.5: {"mean_test_sharpe": 0.4, "deflated_sharpe": 0.60, "n_windows": 5},
        },
        regime_breakdown={
            "Bull/High": {
                "n_windows": 2,
                "n_trades": 10,
                "mean_test_sharpe": 0.5,
                "total_return": 0.01,
            },
            "Bear/Low": {
                "n_windows": 1,
                "n_trades": 4,
                "mean_test_sharpe": 0.1,
                "total_return": 0.002,
            },
        },
    )

    score = robustness_score(result)

    assert 0.0 <= score <= 1.0


def test_strategy_report_includes_parameter_stability_line_when_present():
    result = _core_result(parameter_stability=0.75)

    report = strategy_report(result)

    assert "parameter stability  0.750" in report


def test_strategy_report_omits_parameter_stability_line_when_absent():
    result = _core_result()

    report = strategy_report(result)

    assert "parameter stability" not in report
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest research/tests/test_scoring.py -v -k parameter_stability`
Expected: FAIL with `TypeError: GateResult.__init__() got an unexpected keyword argument
'parameter_stability'` (Task 2 already added the field to `GateResult` -- this failure
means Task 3's own changes to `scoring.py` haven't landed yet; if Task 2 wasn't actually
committed first, re-run Task 2 before continuing)

- [ ] **Step 3: Add the `parameter_stability` component to `WEIGHTS` and `robustness_score`**

In `research/scoring.py`, add to `WEIGHTS` (after the existing `regime_consistency` entry):

```python
WEIGHTS = {
    "deflated_sharpe": 0.35,
    "significance": 0.25,
    "pbo_inverse": 0.25,
    "cost_sensitivity": 0.10,
    "regime_consistency": 0.05,
    "parameter_stability": 0.05,
}
```

In `robustness_score`, after the existing `if result.regime_breakdown:` block and before
`weighted_sum = ...`, add:

```python
    if result.parameter_stability is not None:
        components["parameter_stability"] = result.parameter_stability
```

Also add one sentence to `robustness_score`'s docstring, alongside the existing
`"regime_consistency"` paragraph:

```python
    "parameter_stability", only when result.parameter_stability is not None: used directly
    -- already a probability in [0, 1] by construction (research.parameter_stability), no
    transform needed.
```

- [ ] **Step 4: Add the report line to `strategy_report`**

In `research/scoring.py`, inside `strategy_report`, after the existing `lines = [...]`
block and before the `for reason in result.reasons:` loop, add:

```python
    if result.parameter_stability is not None:
        lines.append(f"  parameter stability  {result.parameter_stability:.3f}")
    for reason in result.reasons:
        lines.append(f"  - {reason}")
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `pytest research/tests/test_scoring.py -v`
Expected: PASS (all tests in the file, including the 4 new ones and every pre-existing
one -- confirms no regression)

- [ ] **Step 6: Lint and format**

Run: `ruff check research/scoring.py research/tests/test_scoring.py` and
`ruff format --check research/scoring.py research/tests/test_scoring.py`
Expected: no errors (fix and re-run Step 5 if needed)

- [ ] **Step 7: Commit**

```bash
git add research/scoring.py research/tests/test_scoring.py
git commit -m "feat(research): fold parameter stability into robustness_score and strategy_report"
```

---

### Task 4: `research/cli.py` — `--parameter-stability` flag

**Files:**
- Modify: `research/cli.py:44-51` (arg parser), `:55-74` (dispatch)
- Test: `research/tests/test_cli.py`

**Interfaces:**
- Consumes: `GateResult.parameter_stability` (Task 2, printed automatically by
  `strategy_report` -- Task 3), `run_promotion_gate(..., include_parameter_stability=...)`
  (Task 2).
- Produces: nothing further downstream -- this is the final task in the plan.

- [ ] **Step 1: Write the failing test**

Append to `research/tests/test_cli.py`:

```python
def test_gate_command_threads_parameter_stability_flag_and_prints_report_line(
    mocker, capsys
):
    canned = GateResult(
        strategy_id="StrategyTestV3",
        passed=True,
        deflated_sharpe=0.97,
        permutation_p=0.01,
        pbo=0.1,
        mean_test_sharpe=1.2,
        n_trials=12,
        reasons=[],
        parameter_stability=0.75,
    )
    mock_gate = mocker.patch("research.cli.run_promotion_gate", return_value=canned)
    mocker.patch(
        "research.cli.Configuration.from_files", return_value={"datadir": "user_data/data"}
    )

    exit_code = main(
        [
            "gate",
            "--strategy",
            "StrategyTestV3",
            "--config",
            "config.json",
            "--pairs",
            "BTC/USDT",
            "--timeframe",
            "1h",
            "--start",
            "2024-01-01",
            "--end",
            "2024-06-01",
            "--train-days",
            "60",
            "--test-days",
            "20",
            "--param-grid",
            '[{"buy_rsi": 30}]',
            "--parameter-stability",
        ]
    )

    _, kwargs = mock_gate.call_args
    assert kwargs["include_parameter_stability"] is True

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "parameter stability  0.750" in captured.out
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest research/tests/test_cli.py -v -k parameter_stability`
Expected: FAIL with `error: unrecognized arguments: --parameter-stability` (an `argparse`
`SystemExit` -- Task 2/3's `GateResult.parameter_stability` field and report line already
exist by this point; the only missing piece here is the CLI flag itself)

- [ ] **Step 3: Add the `--parameter-stability` flag**

In `research/cli.py`, after the existing `--regime-breakdown` argument definition:

```python
    gate.add_argument(
        "--regime-breakdown",
        action="store_true",
        help=(
            "Also compute a regime (Trend x Volatility) breakdown of walk-forward "
            "results, regardless of pass/fail (informational)"
        ),
    )
    gate.add_argument(
        "--parameter-stability",
        action="store_true",
        help=(
            "Also compute the fraction of grid variants profitable in-sample across "
            "the walk-forward run, regardless of pass/fail (informational)"
        ),
    )
```

- [ ] **Step 4: Thread the flag into `run_promotion_gate`**

In the `run_promotion_gate(...)` call inside `main`, add the new keyword argument (after
`include_regime_breakdown=...`):

```python
            include_regime_breakdown=args.regime_breakdown,
            include_parameter_stability=args.parameter_stability,
        )
```

No new print logic is needed -- `strategy_report(result, ...)` (already called on the next
line) picks up `result.parameter_stability` automatically via Task 3's change.

- [ ] **Step 5: Run the test to verify it passes**

Run: `pytest research/tests/test_cli.py -v`
Expected: PASS (all tests in the file, including the new one and every pre-existing one)

- [ ] **Step 6: Lint and format**

Run: `ruff check research/cli.py research/tests/test_cli.py` and
`ruff format --check research/cli.py research/tests/test_cli.py`
Expected: no errors (fix and re-run Step 5 if needed)

- [ ] **Step 7: Run the full research test suite**

Run: `pytest research/ -v`
Expected: PASS (every test in `research/tests/`, confirming Tasks 1-4 compose cleanly)

- [ ] **Step 8: Commit**

```bash
git add research/cli.py research/tests/test_cli.py
git commit -m "feat(research): add --parameter-stability CLI flag"
```
