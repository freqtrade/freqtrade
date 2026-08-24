# Regime Breakdown Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Attribute a promotion-gate candidate's out-of-sample walk-forward performance to
the market conditions (Trend × Volatility) each window's test period actually occurred in,
as a new, purely informational field on `GateResult`.

**Architecture:** One new module, `research/regime.py`, with two pure functions:
`classify_regimes` (labels each `Window` from its own OHLCV test-period data) and
`regime_report` (groups an existing gate run's `WindowResult`s by those labels and
aggregates). `research/gate.py` and `research/cli.py` are extended to wire an opt-in flag
through to these two functions and print the result — no existing behavior changes when
the flag is unused.

**Tech Stack:** Python, pandas/numpy, freqtrade's `history.load_data`/`TimeRange` (already
used by `research/walkforward.py`), pytest with real fixture data (`UNITTEST/BTC`,
`tests/testdata/`) — no mocking of the classification/aggregation logic itself.

**Spec:** `docs/superpowers/specs/2026-08-24-regime-breakdown-design.md`

## Global Constraints

- Two axes only — Trend (Bull/Bear/Sideways) × Volatility (High/Low), 6 possible combined
  labels (`"{Trend}/{Volatility}"`). No Crash/Recovery, no unsupervised/statistical model,
  no new data source beyond the traded pair's own OHLCV.
- `classify_regimes(pair: str, timeframe: str, datadir: Path, windows: list[Window],
  trend_threshold: float = 0.05) -> list[str]` — one label per window, same order as
  `windows`. `trend_threshold` default `0.05` is a starting value, not empirically derived
  — mark with a `ponytail:` comment, don't remove or "improve" it during implementation.
- Trend label: `"Bull"` if `total_return > trend_threshold`, `"Bear"` if
  `total_return < -trend_threshold`, else `"Sideways"`.
- Volatility label: `"High"` if this window's `realized_vol` is strictly greater than the
  **median** `realized_vol` across every window in this same `classify_regimes` call,
  `"Low"` otherwise. Strict `>`, not `>=` — a tie with the median is `"Low"` by construction.
  This is a deliberate design choice, not something to "fix" during review.
- Degenerate window (test period has fewer than 2 candles): fail closed to
  `total_return=0.0, realized_vol=0.0` — no exception, no crash. `0.0` is inside
  `[-trend_threshold, trend_threshold]`, so this is `"Sideways"` on the trend axis by
  construction.
- `classify_regimes` raises `ValueError` on an empty `windows` list.
- `regime_report(window_results: list[WindowResult], labels: list[str]) -> dict[str, dict]`
  raises `ValueError` if `len(window_results) != len(labels)`.
- `regime_report`'s per-label aggregate is `{"n_windows": int, "n_trades": int,
  "mean_test_sharpe": float, "total_return": float}`. `total_return` is the plain
  **arithmetic sum** of every trade's fractional return across the group's windows — NOT
  geometrically compounded, NOT named `total_pnl`. Do not add a `NaN`-guard around
  `mean_test_sharpe` — `WindowResult.test_sharpe` is never `NaN` for a zero-trade window
  (`calculate_sharpe` returns the plain sentinel `0`, verified against
  `freqtrade/data/metrics.py`), so `np.mean` over any group is always well-defined.
- `run_promotion_gate` gains `include_regime_breakdown: bool = False` (default off —
  existing callers/tests unaffected) and `GateResult` gains
  `regime_breakdown: dict[str, dict] | None = None`.
- **Deliberate asymmetry with the fee-sensitivity precedent, do not "fix" to match it:**
  regime breakdown is computed whenever `include_regime_breakdown=True`, **regardless of
  `passed`** — unlike `fee_sensitivity_multipliers`, which only runs when the gate passes.
  A failed candidate's regime breakdown is diagnostic evidence, not wasted compute.
- Regime breakdown uses `pairs[0]` as the single reference pair for classification.
  Multi-pair blending is explicitly out of scope.
- No mocking of `classify_regimes`/`regime_report`'s own logic in tests — real
  `history.load_data` against real `tests/testdata/UNITTEST/BTC` fixture data, real
  `WindowResult`/`Window` dataclass construction, matching every existing file in
  `research/tests/`. `classify_regimes` never touches `Backtesting` or `Exchange` (it only
  calls `history.load_data`), so its tests need none of the `patch_exchange`/`_conf()`
  boilerplate the other `research/tests/` files use.

---

### Task 1: `research/regime.py` — classification and aggregation

**Files:**
- Create: `research/regime.py`
- Test: `research/tests/test_regime.py`

**Interfaces:**
- Consumes: `research.walkforward.Window`, `research.walkforward.WindowResult` (both
  existing dataclasses — see their definitions in `research/walkforward.py:19-49`),
  `freqtrade.configuration.TimeRange`, `freqtrade.data.history.load_data`.
- Produces: `classify_regimes(pair: str, timeframe: str, datadir: Path, windows:
  list[Window], trend_threshold: float = 0.05) -> list[str]` and
  `regime_report(window_results: list[WindowResult], labels: list[str]) -> dict[str,
  dict]` — both consumed directly by Task 2.

- [ ] **Step 1: Write the failing tests for `classify_regimes`**

Create `research/tests/test_regime.py`:

```python
# research/tests/test_regime.py
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from freqtrade.configuration import TimeRange
from freqtrade.data import history
from freqtrade.data.history import get_timerange
from research.regime import classify_regimes, regime_report
from research.walkforward import Window, WindowResult


TESTDATADIR = Path(__file__).resolve().parents[2] / "tests" / "testdata"
PAIR = "UNITTEST/BTC"
TIMEFRAME = "5m"


def _split_into_n_windows(min_date, max_date, n):
    """N equal-duration, contiguous windows spanning [min_date, max_date). Only
    test_start/test_end matter to classify_regimes, so train_start/train_end are set
    equal to test_start -- unused, but Window requires all four fields."""
    step = (max_date - min_date) / n
    windows = []
    for i in range(n):
        test_start = min_date + step * i
        test_end = min_date + step * (i + 1)
        windows.append(
            Window(
                train_start=test_start, train_end=test_start, test_start=test_start,
                test_end=test_end,
            )
        )
    return windows


def _real_return_and_vol(pair, timeframe, datadir, window):
    """Independently recompute a window's total_return/realized_vol via a fresh
    history.load_data call -- a separate execution path from classify_regimes' own
    internals, used to derive the expected label from real data rather than a
    hand-picked one."""
    timerange = TimeRange(
        "date", "date", int(window.test_start.timestamp()), int(window.test_end.timestamp())
    )
    data = history.load_data(datadir=datadir, timeframe=timeframe, pairs=[pair], timerange=timerange)
    close = data[pair]["close"]
    if len(close) < 2:
        return 0.0, 0.0
    return float(close.iloc[-1] / close.iloc[0] - 1), float(close.pct_change().dropna().std())


def test_classify_regimes_trend_axis_matches_threshold_on_real_data():
    full_data = history.load_data(datadir=TESTDATADIR, timeframe=TIMEFRAME, pairs=[PAIR])
    min_date, max_date = get_timerange(full_data)
    windows = _split_into_n_windows(min_date, max_date, n=5)
    trend_threshold = 0.05

    labels = classify_regimes(PAIR, TIMEFRAME, TESTDATADIR, windows, trend_threshold=trend_threshold)

    assert len(labels) == len(windows)
    for window, label in zip(windows, labels):
        total_return, _ = _real_return_and_vol(PAIR, TIMEFRAME, TESTDATADIR, window)
        if total_return > trend_threshold:
            expected_trend = "Bull"
        elif total_return < -trend_threshold:
            expected_trend = "Bear"
        else:
            expected_trend = "Sideways"
        assert label.split("/")[0] == expected_trend


def test_classify_regimes_volatility_axis_ranks_relative_to_median_on_real_data():
    full_data = history.load_data(datadir=TESTDATADIR, timeframe=TIMEFRAME, pairs=[PAIR])
    min_date, max_date = get_timerange(full_data)
    windows = _split_into_n_windows(min_date, max_date, n=5)

    labels = classify_regimes(PAIR, TIMEFRAME, TESTDATADIR, windows)

    vols = [_real_return_and_vol(PAIR, TIMEFRAME, TESTDATADIR, w)[1] for w in windows]
    median_vol = float(np.median(vols))
    for vol, label in zip(vols, labels):
        expected_volatility = "High" if vol > median_vol else "Low"
        assert label.split("/")[1] == expected_volatility


def test_classify_regimes_degenerate_window_is_sideways_and_does_not_raise():
    full_data = history.load_data(datadir=TESTDATADIR, timeframe=TIMEFRAME, pairs=[PAIR])
    min_date, _ = get_timerange(full_data)
    tiny_end = min_date + timedelta(seconds=1)  # far shorter than one 5m candle
    window = Window(
        train_start=min_date, train_end=min_date, test_start=min_date, test_end=tiny_end
    )

    labels = classify_regimes(PAIR, TIMEFRAME, TESTDATADIR, [window])

    # single-window input: median vol is that window's own (degenerate) vol of 0.0,
    # so 0.0 > 0.0 is False -- deterministically "Low", not just "does not crash".
    assert labels == ["Sideways/Low"]


def test_classify_regimes_raises_on_empty_windows():
    with pytest.raises(ValueError, match="windows"):
        classify_regimes(PAIR, TIMEFRAME, TESTDATADIR, [])
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest research/tests/test_regime.py -v`
Expected: FAIL/ERROR with `ModuleNotFoundError: No module named 'research.regime'` (the
`regime_report` tests added in Step 5 aren't written yet — only run the four tests above
for now, e.g. `pytest research/tests/test_regime.py -v -k "classify_regimes"`).

- [ ] **Step 3: Implement `classify_regimes`**

Create `research/regime.py`:

```python
# research/regime.py
"""Regime breakdown: labels each walk-forward window's out-of-sample test period by the
market conditions it actually occurred in (Trend x Volatility), and aggregates a gate's
window-level results by that label. Purely informational -- never changes
GateResult.passed. See docs/superpowers/specs/2026-08-24-regime-breakdown-design.md for
full reasoning, including why this classifier is NOT safe to reuse for a live/production
regime-switching signal: it ranks each window against the full-sample median of every
window in this run, including ones chronologically after it -- fine for a one-shot,
post-hoc report generated after the whole backtest already ran, not causal.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

from freqtrade.configuration import TimeRange
from freqtrade.data import history
from research.walkforward import Window, WindowResult


def classify_regimes(
    pair: str,
    timeframe: str,
    datadir: Path,
    windows: list[Window],
    # ponytail: starting default, not empirically derived -- adjust based on real usage
    # once this runs against real strategies.
    trend_threshold: float = 0.05,
) -> list[str]:
    """Label each window's test period "{Trend}/{Volatility}", e.g. "Bull/High".

    Trend: "Bull" if the test period's total return exceeds `trend_threshold`, "Bear" if
    below -`trend_threshold`, "Sideways" otherwise.

    Volatility: "High" if this window's realized volatility (std of per-candle pct
    change) is strictly above the median realized volatility across every window in
    `windows`, "Low" otherwise (a window sitting exactly on the median is "Low" by
    construction -- ">", not ">=").

    Self-referential to this run's own windows (no external volatility index needed) --
    honest about what it is: "more/less volatile than this backtest's other periods," not
    "objectively high/low."

    A window whose test period has fewer than 2 candles can't compute a return or
    volatility; it fails closed to total_return=0.0, realized_vol=0.0 (classified
    "Sideways" on the trend axis by construction, since 0.0 is inside
    [-trend_threshold, trend_threshold]).

    Returns one label per window, same order as `windows`.
    """
    if not windows:
        raise ValueError("windows must not be empty")

    returns: list[float] = []
    vols: list[float] = []
    for window in windows:
        timerange = TimeRange(
            "date", "date", int(window.test_start.timestamp()), int(window.test_end.timestamp())
        )
        data = history.load_data(
            datadir=datadir, timeframe=timeframe, pairs=[pair], timerange=timerange
        )
        close = data[pair]["close"]
        if len(close) < 2:
            returns.append(0.0)
            vols.append(0.0)
            continue
        returns.append(float(close.iloc[-1] / close.iloc[0] - 1))
        vols.append(float(close.pct_change().dropna().std()))

    median_vol = float(np.median(vols))

    labels = []
    for total_return, realized_vol in zip(returns, vols):
        if total_return > trend_threshold:
            trend = "Bull"
        elif total_return < -trend_threshold:
            trend = "Bear"
        else:
            trend = "Sideways"
        volatility = "High" if realized_vol > median_vol else "Low"
        labels.append(f"{trend}/{volatility}")
    return labels
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest research/tests/test_regime.py -v -k "classify_regimes"`
Expected: PASS (4 tests)

- [ ] **Step 5: Write the failing tests for `regime_report`**

Append to `research/tests/test_regime.py`:

```python
_DUMMY_WINDOW = Window(
    train_start=datetime(2020, 1, 1, tzinfo=UTC),
    train_end=datetime(2020, 1, 8, tzinfo=UTC),
    test_start=datetime(2020, 1, 8, tzinfo=UTC),
    test_end=datetime(2020, 1, 15, tzinfo=UTC),
)


def test_regime_report_aggregates_by_label():
    wr_a1 = WindowResult(
        window=_DUMMY_WINDOW, variant_returns={}, best_params={}, train_sharpe=0.0,
        test_sharpe=1.0, test_n_trades=3, test_returns=[0.01, 0.02, -0.01],
    )
    wr_a2 = WindowResult(
        window=_DUMMY_WINDOW, variant_returns={}, best_params={}, train_sharpe=0.0,
        test_sharpe=2.0, test_n_trades=2, test_returns=[0.03, 0.04],
    )
    wr_b1 = WindowResult(
        window=_DUMMY_WINDOW, variant_returns={}, best_params={}, train_sharpe=0.0,
        test_sharpe=-1.0, test_n_trades=5, test_returns=[-0.05, -0.02, 0.01, -0.03, -0.01],
    )

    report = regime_report([wr_a1, wr_a2, wr_b1], ["Bull/High", "Bull/High", "Bear/Low"])

    assert set(report) == {"Bull/High", "Bear/Low"}
    assert report["Bull/High"]["n_windows"] == 2
    assert report["Bull/High"]["n_trades"] == 5
    assert report["Bull/High"]["mean_test_sharpe"] == pytest.approx(1.5)
    assert report["Bull/High"]["total_return"] == pytest.approx(0.01 + 0.02 - 0.01 + 0.03 + 0.04)
    assert report["Bear/Low"]["n_windows"] == 1
    assert report["Bear/Low"]["n_trades"] == 5
    assert report["Bear/Low"]["mean_test_sharpe"] == pytest.approx(-1.0)
    assert report["Bear/Low"]["total_return"] == pytest.approx(-0.05 - 0.02 + 0.01 - 0.03 - 0.01)


def test_regime_report_raises_on_mismatched_lengths():
    wr = WindowResult(
        window=_DUMMY_WINDOW, variant_returns={}, best_params={}, train_sharpe=0.0,
        test_sharpe=1.0, test_n_trades=1, test_returns=[0.01],
    )
    with pytest.raises(ValueError, match="same length"):
        regime_report([wr], ["Bull/High", "Bear/Low"])
```

Add `from datetime import UTC, datetime, timedelta` is already at the top of the file from
Step 1 — no new import needed for `_DUMMY_WINDOW`.

- [ ] **Step 6: Run the tests to verify they fail**

Run: `pytest research/tests/test_regime.py -v -k "regime_report"`
Expected: FAIL with `ImportError: cannot import name 'regime_report' from 'research.regime'`

- [ ] **Step 7: Implement `regime_report`**

Append to `research/regime.py`:

```python
def regime_report(window_results: list[WindowResult], labels: list[str]) -> dict[str, dict]:
    """Group `window_results` by their parallel `labels` entry and aggregate each group.

    Raises ValueError if len(window_results) != len(labels) -- mismatched parallel lists
    is a caller-contract violation, not a data problem.

    Returns {label: {"n_windows": int, "n_trades": int, "mean_test_sharpe": float,
    "total_return": float}}. `total_return` is the plain arithmetic sum of every trade's
    fractional return across the group's windows -- NOT a geometrically compounded
    return, a rough same-units aggregate for comparing regime buckets against each
    other, not a claim about realized account growth.

    WindowResult.test_sharpe is never NaN for a zero-trade window (calculate_sharpe
    returns the plain sentinel 0), so np.mean over any group is always well-defined --
    no NaN-guard needed here.
    """
    if len(window_results) != len(labels):
        raise ValueError(
            f"window_results and labels must be the same length, got "
            f"{len(window_results)} and {len(labels)}"
        )

    grouped: dict[str, list[WindowResult]] = defaultdict(list)
    for wr, label in zip(window_results, labels):
        grouped[label].append(wr)

    report: dict[str, dict] = {}
    for label, group in grouped.items():
        report[label] = {
            "n_windows": len(group),
            "n_trades": sum(wr.test_n_trades for wr in group),
            "mean_test_sharpe": float(np.mean([wr.test_sharpe for wr in group])),
            "total_return": sum(r for wr in group for r in wr.test_returns),
        }
    return report
```

- [ ] **Step 8: Run all tests in the file to verify they pass**

Run: `pytest research/tests/test_regime.py -v`
Expected: PASS (6 tests)

- [ ] **Step 9: Lint and format**

Run: `ruff check research/regime.py research/tests/test_regime.py` and
`ruff format --check research/regime.py research/tests/test_regime.py`
Expected: no errors (fix with `ruff check --fix` / `ruff format` if needed, then re-run
the test suite from Step 8 to confirm nothing broke)

- [ ] **Step 10: Commit**

```bash
git add research/regime.py research/tests/test_regime.py
git commit -m "feat(research): add regime.py -- Trend x Volatility window classification"
```

---

### Task 2: `research/gate.py` — wire regime breakdown into the promotion gate

**Files:**
- Modify: `research/gate.py:1-22` (imports), `:25-37` (`GateResult`), `:40-158`
  (`run_promotion_gate`)
- Test: `research/tests/test_gate.py`

**Interfaces:**
- Consumes: `research.regime.classify_regimes`, `research.regime.regime_report` (Task 1).
  `windows` and `results` are already in scope inside `run_promotion_gate` from the
  existing walk-forward run (`research/gate.py:77` and `:85`) — no duplicate computation.
- Produces: `GateResult.regime_breakdown: dict[str, dict] | None`, and
  `run_promotion_gate(..., include_regime_breakdown: bool = False)` — both consumed by
  Task 3.

- [ ] **Step 1: Write the failing tests**

Append to `research/tests/test_gate.py`:

```python
def test_run_promotion_gate_attaches_regime_breakdown_when_requested_and_passes(mocker, tmp_path):
    conf = _conf()
    _patch(mocker)
    full_data = history.load_data(datadir=TESTDATADIR, timeframe="5m", pairs=["UNITTEST/BTC"])
    min_date, max_date = get_timerange(full_data)
    total_days = max(8, (max_date - min_date).days)
    train_days = max(1, total_days // 8)
    test_days = max(1, total_days // 16)

    # Permissive thresholds guarantee a real, unmocked PASS (see the fee-sensitivity
    # pass test above for the same pattern).
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
        include_regime_breakdown=True,
    )

    assert result.passed is True
    assert result.regime_breakdown is not None
    assert len(result.regime_breakdown) > 0


def test_run_promotion_gate_attaches_regime_breakdown_when_requested_and_fails(mocker, tmp_path):
    conf = _conf()
    _patch(mocker)
    full_data = history.load_data(datadir=TESTDATADIR, timeframe="5m", pairs=["UNITTEST/BTC"])
    min_date, max_date = get_timerange(full_data)
    total_days = max(8, (max_date - min_date).days)
    train_days = max(1, total_days // 8)
    test_days = max(1, total_days // 16)

    # Impossible dsr_threshold guarantees a real FAIL (see the fee-sensitivity skip test
    # above for the same pattern). Regime breakdown must still be attached -- this is the
    # test that actually proves the deliberate asymmetry with fee-sensitivity was
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
        include_regime_breakdown=True,
    )

    assert result.passed is False
    assert result.regime_breakdown is not None
    assert len(result.regime_breakdown) > 0


def test_run_promotion_gate_omits_regime_breakdown_by_default(mocker, tmp_path):
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

    assert result.regime_breakdown is None
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest research/tests/test_gate.py -v -k regime_breakdown`
Expected: FAIL with `TypeError: run_promotion_gate() got an unexpected keyword argument
'include_regime_breakdown'`

- [ ] **Step 3: Wire `include_regime_breakdown` into `run_promotion_gate` and `GateResult`**

In `research/gate.py`, add the import (alongside the existing `research.*` imports at the
top of the file):

```python
from research.regime import classify_regimes, regime_report
```

Add the new field to `GateResult` (after the existing `fee_sensitivity` field):

```python
    fee_sensitivity: dict[float, dict] | None = None
    regime_breakdown: dict[str, dict] | None = None
```

Add the new parameter to `run_promotion_gate`'s signature (after
`fee_sensitivity_multipliers`):

```python
    fee_sensitivity_multipliers: tuple[float, ...] | None = None,
    include_regime_breakdown: bool = False,
) -> GateResult:
```

Inside `run_promotion_gate`, after the existing `fee_report` block (which computes
`fee_report` conditionally on `passed`) and before the `log_candidate_result` call, add
the unconditional regime-breakdown block:

```python
    regime_breakdown = None
    if include_regime_breakdown:
        labels = classify_regimes(pairs[0], timeframe, datadir, windows)
        regime_breakdown = regime_report(results, labels)
```

Note this block does **not** check `passed` — that is the one deliberate asymmetry with
the `fee_report` block immediately above it (see Global Constraints). `windows` and
`results` are already in scope from earlier in this function (`generate_windows(...)` and
`runner.run(...)`).

Finally, thread it into the `GateResult(...)` construction at the end of the function
(after the existing `fee_sensitivity=fee_report` line):

```python
        fee_sensitivity=fee_report,
        regime_breakdown=regime_breakdown,
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest research/tests/test_gate.py -v`
Expected: PASS (all tests in the file, including the 3 new ones and every pre-existing
one — confirms no regression)

- [ ] **Step 5: Lint and format**

Run: `ruff check research/gate.py research/tests/test_gate.py` and
`ruff format --check research/gate.py research/tests/test_gate.py`
Expected: no errors (fix and re-run Step 4 if needed)

- [ ] **Step 6: Commit**

```bash
git add research/gate.py research/tests/test_gate.py
git commit -m "feat(research): wire regime breakdown into run_promotion_gate (opt-in, runs on pass or fail)"
```

---

### Task 3: `research/cli.py` — `--regime-breakdown` flag and table output

**Files:**
- Modify: `research/cli.py:22-43` (arg parser), `:44-83` (dispatch/printing)
- Test: `research/tests/test_cli.py`

**Interfaces:**
- Consumes: `GateResult.regime_breakdown` (Task 2), `run_promotion_gate(...,
  include_regime_breakdown=...)` (Task 2).
- Produces: nothing further downstream — this is the final task in the plan.

- [ ] **Step 1: Write the failing test**

Append to `research/tests/test_cli.py`:

```python
def test_gate_command_prints_regime_breakdown_table_when_present(mocker, capsys):
    canned = GateResult(
        strategy_id="StrategyTestV3",
        passed=False,
        deflated_sharpe=0.4,
        permutation_p=0.3,
        pbo=0.7,
        mean_test_sharpe=0.1,
        n_trials=12,
        reasons=["deflated_sharpe 0.400 below threshold 0.95"],
        regime_breakdown={
            "Bull/High": {
                "n_windows": 2, "n_trades": 14, "mean_test_sharpe": 0.42, "total_return": 0.0012,
            },
            "Bear/Low": {
                "n_windows": 1, "n_trades": 6, "mean_test_sharpe": -1.10, "total_return": -0.0034,
            },
        },
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
            "--regime-breakdown",
        ]
    )

    _, kwargs = mock_gate.call_args
    assert kwargs["include_regime_breakdown"] is True

    captured = capsys.readouterr()
    assert exit_code == 1  # this GateResult failed -- regime breakdown must print regardless
    assert "regime breakdown" in captured.out
    assert "Bull/High" in captured.out
    assert "Bear/Low" in captured.out
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest research/tests/test_cli.py -v -k regime_breakdown`
Expected: FAIL with `error: unrecognized arguments: --regime-breakdown` (an `argparse`
`SystemExit` — Task 2's `GateResult.regime_breakdown` field already exists by this point,
since Task 2 committed before this task started; the only missing piece here is the CLI
flag itself)

- [ ] **Step 3: Add the `--regime-breakdown` flag**

In `research/cli.py`, after the existing `--fee-sensitivity` argument definition:

```python
    gate.add_argument(
        "--fee-sensitivity",
        action="store_true",
        help="Also run a fee-sensitivity stress test if the gate passes (informational)",
    )
    gate.add_argument(
        "--regime-breakdown",
        action="store_true",
        help=(
            "Also compute a regime (Trend x Volatility) breakdown of walk-forward "
            "results, regardless of pass/fail (informational)"
        ),
    )
```

- [ ] **Step 4: Thread the flag into `run_promotion_gate` and print the table**

In the `run_promotion_gate(...)` call inside `main`, add the new keyword argument (after
`fee_sensitivity_multipliers=...`):

```python
            fee_sensitivity_multipliers=DEFAULT_FEE_MULTIPLIERS if args.fee_sensitivity else None,
            include_regime_breakdown=args.regime_breakdown,
        )
```

After the existing `if result.fee_sensitivity:` block, add:

```python
        if result.regime_breakdown:
            print("  regime breakdown (informational, not part of PASS/FAIL):")
            for label, stats in result.regime_breakdown.items():
                print(
                    f"    {label:<15} {stats['n_windows']:>2} windows"
                    f"   {stats['n_trades']:>3} trades"
                    f"   mean sharpe {stats['mean_test_sharpe']:>6.2f}"
                    f"   total return {stats['total_return']:>8.4f}"
                )
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `pytest research/tests/test_cli.py -v`
Expected: PASS (all tests in the file, including the new one and every pre-existing one)

- [ ] **Step 6: Lint and format**

Run: `ruff check research/cli.py research/tests/test_cli.py` and
`ruff format --check research/cli.py research/tests/test_cli.py`
Expected: no errors (fix and re-run Step 5 if needed)

- [ ] **Step 7: Run the full research test suite**

Run: `pytest research/ -v`
Expected: PASS (every test in `research/tests/`, confirming Tasks 1-3 compose cleanly)

- [ ] **Step 8: Commit**

```bash
git add research/cli.py research/tests/test_cli.py
git commit -m "feat(research): add --regime-breakdown CLI flag and table output"
```
