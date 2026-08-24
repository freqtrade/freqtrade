# Fee Sensitivity Stress Test Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an informational fee-sensitivity stress test to the `research/` promotion
gate — re-evaluates an already-passing candidate's already-selected walk-forward
parameters at progressively worse fee assumptions (1.0x/1.25x/1.5x/2.0x), reporting how
its edge degrades, without changing the PASS/FAIL verdict.

**Architecture:** Extract the test-period evaluation from `WalkForwardRunner.run_window`
into a new, independently-callable method (`evaluate_fixed_params`) that accepts an
optional fee override and never mutates the runner's config. A new `research/cost_stress.py`
module calls it once per fee multiplier for every already-computed `WindowResult`, and
`run_promotion_gate` calls that module (opt-in, only when the gate passed) using the
`results` list it already has in scope — no duplicate walk-forward run.

**Tech Stack:** Python, no new dependencies — same stack as the rest of `research/`.

**Spec:** `docs/superpowers/specs/2026-08-24-fee-sensitivity-stress-test-design.md`

## Global Constraints

- Python >=3.11, modern type hints.
- No new third-party dependencies.
- `freqtrade/` package is never modified by this plan.
- Tests live under `research/tests/`, run via `pytest research/tests -v` from the repo root.
- TDD every task: failing test first, watch it fail, minimal implementation, watch it pass.
- Lint before every commit: `ruff check research/ --fix && ruff format research/` (run from
  the repo root, with the project venv: `.venv/Scripts/python.exe -m ruff ...`).
- **Never named "slippage"** anywhere (module, function, CLI flag, report labels, comments)
  — this stress-tests fee sensitivity only; a real slippage/execution model is explicitly
  out of scope (spec's "What this is not").
- **`n_trials=1`** for every Deflated Sharpe Ratio computed inside this feature — no new
  selection happens at any fee level, so no new multiple-testing penalty applies (spec's
  "On `n_trials=1`" section has the full reasoning; don't re-derive it differently).
- **Never assert Sharpe is exactly monotonic** across fee multipliers in a test — it's an
  empirical tendency for a fixed trade set, not a mathematical guarantee (spec's Testing
  section). Use the invariants that are actually guaranteed instead (see Task 1/2 tests).
- `run_promotion_gate`'s existing callers and existing tests must keep working unchanged —
  the new parameter (`fee_sensitivity_multipliers`) defaults to `None` (off).

---

## Deviations from this plan (discovered during implementation)

The trade-set-invariance and P&L-monotonicity claims baked into Task 1's and Task 2's Step
1 test code blocks below were discovered false during implementation, for ROI-exit
strategies: freqtrade's `should_exit()` computes its ROI-threshold comparison from a
fee-adjusted profit ratio (`Trade.calc_profit_ratio`), so a higher fee can genuinely delay
or prevent an ROI exit and change which trades occur, not just their realized P&L — verified
against real freqtrade source (`freqtrade/strategy/interface.py`,
`freqtrade/persistence/trade_model.py`) and empirically on the real `UNITTEST/BTC` fixture.
See `docs/superpowers/specs/2026-08-24-fee-sensitivity-stress-test-design.md` (the
"Invariant this method must uphold" and "Post-implementation correction" sections) for the
full reasoning. As a result, `test_evaluate_fixed_params_fee_override_changes_pnl_but_not_trade_count`
(Task 1, Step 1 below) was renamed to `test_evaluate_fixed_params_fee_override_changes_results`
and its `cheap.test_n_trades == expensive.test_n_trades` assertion was removed (a P&L-
monotonicity assertion was also removed later, in the final-review fix wave, for the same
reason). `test_fee_sensitivity_net_pnl_is_non_increasing_as_fee_rises` (Task 2, Step 1
below) was dropped entirely rather than implemented, since its only other assertion
(`n_windows` matching across reports) is already covered by
`test_fee_sensitivity_reports_one_entry_per_multiplier`. The task bodies below are left as
originally written, for historical record; this section is the annotation that reconciles
them with what was actually built.

---

### Task 1: `WalkForwardRunner.evaluate_fixed_params` — extract and reuse for `run_window`

**Files:**
- Modify: `research/walkforward.py` (whole file shown below — small enough to replace wholesale)
- Test: `research/tests/test_walkforward.py` (add tests, keep the 2 existing ones unchanged)

**Interfaces:**
- Consumes: nothing new — same freqtrade APIs `run_window` already uses (`Backtesting`,
  `history.load_data`, `trim_dataframes`, `calculate_sharpe`, `TimeRange`).
- Produces: `WalkForwardRunner.evaluate_fixed_params(window: Window, params: dict, fee_override: float | None = None) -> WindowResult`
  — Task 2's `research/cost_stress.py` calls this directly. Returns a `WindowResult` with
  `variant_returns={}` (explicit empty dict — no grid was searched).

The existing `Window`, `WindowResult`, `variant_key`, `generate_windows`,
`WalkForwardRunner.__init__`, and `WalkForwardRunner.run` are **unchanged** — only
`run_window`'s body changes (to call the new method for its final phase) and the new
method is added.

- [ ] **Step 1: Write the failing tests**

Add these three tests to `research/tests/test_walkforward.py`, below the two existing tests
(keep the existing `TESTDATADIR`, `EXMS`, `_conf` at the top of the file unchanged):

```python
def test_evaluate_fixed_params_matches_run_window_for_the_winning_variant(mocker):
    conf = _conf()
    patch_exchange(mocker)
    mocker.patch(f"{EXMS}.get_min_pair_stake_amount", return_value=0.00001)
    mocker.patch(f"{EXMS}.get_max_pair_stake_amount", return_value=float("inf"))
    mocker.patch(f"{EXMS}.get_pair_base_currency", lambda _, x: x.split("/")[0])

    full_data = history.load_data(datadir=TESTDATADIR, timeframe="5m", pairs=["UNITTEST/BTC"])
    min_date, max_date = get_timerange(full_data)
    train_days = max(1, int((max_date - min_date).days * 0.7))
    train_end = min_date + timedelta(days=train_days)
    window = Window(
        train_start=min_date, train_end=train_end, test_start=train_end, test_end=max_date
    )

    runner = WalkForwardRunner(conf, pairs=["UNITTEST/BTC"], timeframe="5m", datadir=TESTDATADIR)
    param_grid = [{"buy_rsi": 25}, {"buy_rsi": 35}]
    run_window_result = runner.run_window(window, param_grid)

    direct_result = runner.evaluate_fixed_params(window, run_window_result.best_params)

    assert direct_result.test_sharpe == run_window_result.test_sharpe
    assert direct_result.test_returns == run_window_result.test_returns
    assert direct_result.test_n_trades == run_window_result.test_n_trades
    assert direct_result.variant_returns == {}

    # fee_override=None and fee_override=<the config's own base fee> must be equivalent --
    # they resolve to the same fee, just via a different code path.
    from freqtrade.optimize.backtesting import Backtesting

    base_fee = Backtesting(conf).fee
    override_result = runner.evaluate_fixed_params(
        window, run_window_result.best_params, fee_override=base_fee
    )
    assert override_result.test_sharpe == run_window_result.test_sharpe
    assert override_result.test_returns == run_window_result.test_returns


def test_evaluate_fixed_params_fee_override_changes_pnl_but_not_trade_count(mocker):
    conf = _conf()
    patch_exchange(mocker)
    mocker.patch(f"{EXMS}.get_min_pair_stake_amount", return_value=0.00001)
    mocker.patch(f"{EXMS}.get_max_pair_stake_amount", return_value=float("inf"))
    mocker.patch(f"{EXMS}.get_pair_base_currency", lambda _, x: x.split("/")[0])

    full_data = history.load_data(datadir=TESTDATADIR, timeframe="5m", pairs=["UNITTEST/BTC"])
    min_date, max_date = get_timerange(full_data)
    train_days = max(1, int((max_date - min_date).days * 0.7))
    train_end = min_date + timedelta(days=train_days)
    window = Window(
        train_start=min_date, train_end=train_end, test_start=train_end, test_end=max_date
    )

    runner = WalkForwardRunner(conf, pairs=["UNITTEST/BTC"], timeframe="5m", datadir=TESTDATADIR)
    params = {"buy_rsi": 25}

    config_before = dict(runner.config)
    cheap = runner.evaluate_fixed_params(window, params, fee_override=0.0)
    expensive = runner.evaluate_fixed_params(window, params, fee_override=0.05)

    # Same window, same params -> same trades. Only realized P&L differs (higher fee ->
    # less or equal total P&L on the same trade set; never asserted as exactly monotonic
    # Sharpe -- see the design spec's Testing section for why that's not guaranteed).
    assert cheap.test_n_trades == expensive.test_n_trades
    assert sum(cheap.test_returns) >= sum(expensive.test_returns)

    # self.config must never be mutated by a fee_override call.
    assert runner.config == config_before


def test_run_window_still_works_after_the_refactor(mocker):
    """Regression coverage: run_window's own pre-existing test already covers this,
    but this direct check makes the refactor's intent explicit -- run_window's final
    phase now delegates to evaluate_fixed_params rather than duplicating it."""
    conf = _conf()
    patch_exchange(mocker)
    mocker.patch(f"{EXMS}.get_min_pair_stake_amount", return_value=0.00001)
    mocker.patch(f"{EXMS}.get_max_pair_stake_amount", return_value=float("inf"))
    mocker.patch(f"{EXMS}.get_pair_base_currency", lambda _, x: x.split("/")[0])

    full_data = history.load_data(datadir=TESTDATADIR, timeframe="5m", pairs=["UNITTEST/BTC"])
    min_date, max_date = get_timerange(full_data)
    train_days = max(1, int((max_date - min_date).days * 0.7))
    train_end = min_date + timedelta(days=train_days)
    window = Window(
        train_start=min_date, train_end=train_end, test_start=train_end, test_end=max_date
    )

    runner = WalkForwardRunner(conf, pairs=["UNITTEST/BTC"], timeframe="5m", datadir=TESTDATADIR)
    result = runner.run_window(window, [{"buy_rsi": 25}, {"buy_rsi": 35}])

    assert result.best_params in [{"buy_rsi": 25}, {"buy_rsi": 35}]
    assert set(result.variant_returns) == {variant_key({"buy_rsi": 25}), variant_key({"buy_rsi": 35})}
    assert isinstance(result.train_sharpe, float)
    assert isinstance(result.test_sharpe, float)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest research/tests/test_walkforward.py -v`
Expected: the two new `evaluate_fixed_params` tests FAIL with
`AttributeError: 'WalkForwardRunner' object has no attribute 'evaluate_fixed_params'`.
The third new test (`test_run_window_still_works_after_the_refactor`) currently PASSES
already (it exercises only existing behavior) — that's expected and fine; it becomes a
real regression check once Step 3 changes `run_window`'s implementation.

- [ ] **Step 3: Implement `evaluate_fixed_params` and refactor `run_window` to use it**

Replace `research/walkforward.py` in full with:

```python
# research/walkforward.py
from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from freqtrade.configuration import TimeRange
from freqtrade.data import history
from freqtrade.data.converter import trim_dataframes
from freqtrade.data.metrics import calculate_sharpe
from freqtrade.optimize.backtesting import Backtesting


@dataclass
class Window:
    """One walk-forward train/test period. `test_start` always equals
    `train_end`; see `generate_windows` for how consecutive windows relate."""

    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime


@dataclass
class WindowResult:
    """Outcome of running one `Window` through `WalkForwardRunner.run_window`:
    the train-period Sharpe of every param variant (`variant_returns`, keyed by
    `variant_key`, feeding `research.pbo`), the train-selected best params, and
    the resulting out-of-sample (test-period) performance.

    `evaluate_fixed_params` also returns a `WindowResult`, but a deliberately
    partial one: no grid was searched, so `variant_returns` is always `{}` on
    that path. `research.cost_stress` is the only intended consumer of that
    partial form -- don't assume `variant_returns` reflects a real grid search
    unless the `WindowResult` came from `run_window`."""

    window: Window
    variant_returns: dict[str, float]
    best_params: dict
    train_sharpe: float
    test_sharpe: float
    test_n_trades: int
    test_returns: list[float]


def variant_key(params: dict) -> str:
    """Canonical, order-independent string key for a param-variant dict, used
    to identify the same variant across windows (e.g. in `variant_returns`)."""
    return json.dumps(params, sort_keys=True)


def generate_windows(
    start: datetime, end: datetime, train_days: int, test_days: int
) -> list[Window]:
    """Rolling, contiguous windows: each window's test period starts exactly
    where its train period ends, and test periods are gapless across windows
    (the next window's test period starts exactly where the previous
    window's test period ended). Cursor advances by test_days each step, so
    every day in [start, end) is covered by at most one window's OOS test
    period, maximizing out-of-sample statistical power for Task 5's
    DSR/permutation-test/PBO evaluation."""
    windows: list[Window] = []
    cursor = start
    while True:
        train_end = cursor + timedelta(days=train_days)
        test_end = train_end + timedelta(days=test_days)
        if test_end > end:
            break
        windows.append(Window(cursor, train_end, train_end, test_end))
        cursor = cursor + timedelta(days=test_days)
    return windows


class WalkForwardRunner:
    """Runs a strategy through freqtrade's `Backtesting` engine across a series
    of walk-forward windows, selecting the best param variant on each window's
    train period and evaluating it on that window's held-out test period."""

    def __init__(self, config: dict, pairs: list[str], timeframe: str, datadir: Path):
        self.config = config
        self.pairs = pairs
        self.timeframe = timeframe
        self.datadir = datadir

    def evaluate_fixed_params(
        self, window: Window, params: dict, fee_override: float | None = None
    ) -> WindowResult:
        """Backtest a single, already-chosen `params` on `window` -- no grid
        search. Used directly by `research.cost_stress.fee_sensitivity` to
        re-evaluate an already-selected candidate at a different fee level,
        and by `run_window` (below) for its own final test-phase step -- both
        paths share this one implementation rather than drifting apart.

        `fee_override`, when given, is used ONLY for this call's `Backtesting`
        instance -- `self.config` is never mutated, so a stress-test fee can
        never leak into other calls sharing this `WalkForwardRunner`.

        Returns a `WindowResult` with `variant_returns={}` (no grid was
        searched) and `train_sharpe` reflecting only `params`' own
        train-period Sharpe (not a selection among alternatives).
        """
        cfg = {**self.config, "fee": fee_override} if fee_override is not None else self.config
        backtesting = Backtesting(cfg)
        backtesting._set_strategy(backtesting.strategylist[0])

        timerange = TimeRange(
            "date", "date", int(window.train_start.timestamp()), int(window.test_end.timestamp())
        )
        data = history.load_data(
            datadir=self.datadir,
            timeframe=self.timeframe,
            pairs=self.pairs,
            timerange=timerange,
            startup_candles=backtesting.required_startup,
        )

        for name, value in params.items():
            getattr(backtesting.strategy, name).value = value
        processed = backtesting.strategy.advise_all_indicators(data)
        processed = trim_dataframes(processed, timerange, backtesting.required_startup)

        train_result = backtesting.backtest(
            processed=deepcopy(processed),
            start_date=window.train_start,
            end_date=window.train_end,
        )
        train_trades = train_result["results"]
        train_sharpe = calculate_sharpe(
            train_trades, window.train_start, window.train_end, self.config["dry_run_wallet"]
        )

        test_result = backtesting.backtest(
            processed=deepcopy(processed),
            start_date=window.test_start,
            end_date=window.test_end,
        )
        test_trades = test_result["results"]
        test_returns = (test_trades["profit_abs"] / self.config["dry_run_wallet"]).tolist()
        test_sharpe = calculate_sharpe(
            test_trades, window.test_start, window.test_end, self.config["dry_run_wallet"]
        )

        return WindowResult(
            window=window,
            variant_returns={},
            best_params=params,
            train_sharpe=train_sharpe,
            test_sharpe=test_sharpe,
            test_n_trades=len(test_trades),
            test_returns=test_returns,
        )

    def run_window(self, window: Window, param_grid: list[dict]) -> WindowResult:
        """Backtest every variant in `param_grid` on `window`'s train period,
        pick the highest-train-Sharpe variant, then evaluate ONLY that variant
        on the test period via `evaluate_fixed_params`.

        This ordering is the load-bearing invariant: parameter selection is
        strictly train-only. Test-period data is never touched until after
        `best_params` is fixed, so no parameter choice can be informed by data
        it will later be scored against -- the test-period Sharpe this returns
        is a genuine out-of-sample estimate, not a look-ahead-contaminated one.

        Indicators are (re)computed per param variant over the full
        [train_start, test_end] span before either backtest call (freqtrade's
        own convention), which is safe for backward-looking, row-local
        indicators but can leak test-period information into train-period
        selection for non-causal or DataFrame-wide-normalized indicators --
        see the inline note below.
        """
        backtesting = Backtesting(self.config)
        backtesting._set_strategy(backtesting.strategylist[0])

        timerange = TimeRange(
            "date", "date", int(window.train_start.timestamp()), int(window.test_end.timestamp())
        )
        data = history.load_data(
            datadir=self.datadir,
            timeframe=self.timeframe,
            pairs=self.pairs,
            timerange=timerange,
            startup_candles=backtesting.required_startup,
        )

        if not param_grid:
            raise ValueError("param_grid must not be empty")

        variant_returns: dict[str, float] = {}
        best_sharpe = -np.inf
        best_params: dict | None = None

        for params in param_grid:
            for name, value in params.items():
                getattr(backtesting.strategy, name).value = value

            # ponytail: indicators are (re)computed per param variant over
            # [train_start, test_end], per freqtrade's own convention (§3 of the
            # architecture doc). Safe for backward-looking, row-local indicators
            # (this strategy's RSI); two families of indicator would leak
            # test-period information into the train-period parameter selection
            # done below even though they're "backward-looking" in the naive
            # sense: (1) a non-causal indicator (centered rolling, .shift(-n)),
            # and (2) any indicator normalized against a DataFrame-wide
            # statistic (global z-score, min/max scaling, global percentile
            # rank) computed over the whole [train_start, test_end] span, since
            # that statistic itself is contaminated by test-period rows. Run
            # freqtrade's lookahead-analysis (and audit for global
            # normalization) on any strategy before trusting this runner's
            # results for it.
            processed = backtesting.strategy.advise_all_indicators(data)
            processed = trim_dataframes(processed, timerange, backtesting.required_startup)

            train_result = backtesting.backtest(
                processed=deepcopy(processed),
                start_date=window.train_start,
                end_date=window.train_end,
            )
            train_trades = train_result["results"]
            sharpe = calculate_sharpe(
                train_trades, window.train_start, window.train_end, self.config["dry_run_wallet"]
            )
            key = variant_key(params)
            variant_returns[key] = (
                float((train_trades["profit_abs"] / self.config["dry_run_wallet"]).mean())
                if len(train_trades)
                else 0.0
            )

            if sharpe > best_sharpe:
                best_sharpe, best_params = sharpe, params

        if best_params is None:
            raise RuntimeError(
                "no param variant produced a result"
            )  # unreachable: param_grid non-empty

        # ponytail: this recomputes data + indicators for the winning variant a
        # second time (evaluate_fixed_params does its own history.load_data +
        # advise_all_indicators call) rather than reusing the grid loop's
        # already-computed `processed` dataframe for that variant. Trades a
        # small, deterministic amount of duplicate work for one shared,
        # single-tested code path between run_window and evaluate_fixed_params
        # (see the fee-sensitivity design doc) -- revisit only if this becomes
        # a measured bottleneck.
        result = self.evaluate_fixed_params(window, best_params)
        result.variant_returns = variant_returns
        result.train_sharpe = best_sharpe
        return result

    def run(self, windows: list[Window], param_grid: list[dict]) -> list[WindowResult]:
        """Run `run_window` over every window in sequence and collect results."""
        return [self.run_window(w, param_grid) for w in windows]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest research/tests/test_walkforward.py -v`
Expected: PASS (5 tests: 2 pre-existing + 3 new).

- [ ] **Step 5: Run the full research suite to check for regressions**

Run: `.venv/Scripts/python.exe -m pytest research/tests -v`
Expected: PASS (29 tests: 26 pre-existing across the whole `research/tests` suite + the
3 new tests added to `test_walkforward.py` in this task).

- [ ] **Step 6: Get a second opinion via lmchatbot**

```bash
curl -s localhost:3000/ 2>/dev/null || (node C:/dev/lmchatbot/server.js &)
cat > /tmp/lmchatbot_fee_task1.json <<'EOF'
{"provider":"gemini","prompt":"I refactored a walk-forward backtest runner's run_window method: it used to inline its final test-period-only backtest call, and now delegates that to a new sibling method evaluate_fixed_params(window, params, fee_override=None) which builds a config dict with an overridden 'fee' key (via dict unpacking: {**self.config, 'fee': fee_override} if fee_override is not None else self.config) and constructs a fresh freqtrade Backtesting(cfg) instance from it, never mutating self.config. run_window now calls self.evaluate_fixed_params(window, best_params) for its final phase and overwrites the returned WindowResult's variant_returns/train_sharpe fields with its own grid-search results before returning. Does this refactor look correct and safe -- any risk in overwriting dataclass fields on an object returned from a method the class also uses internally, and does swapping a 'fee' key into a shallow-copied config dict actually work for how freqtrade's Backtesting class reads its fee, or could a shallow copy miss nested config structure freqtrade also needs?"}
EOF
curl -s localhost:3000/ask -d @/tmp/lmchatbot_fee_task1.json
rm /tmp/lmchatbot_fee_task1.json
```

Read the reply. If it surfaces a real gap (e.g. `Backtesting.__init__` mutating the config
dict it's given in a way a shallow copy wouldn't isolate, or a cleaner way to avoid
overwriting dataclass fields after construction), fix it before continuing; otherwise
proceed.

- [ ] **Step 7: Lint and commit**

```bash
.venv/Scripts/python.exe -m ruff check research/ --fix
.venv/Scripts/python.exe -m ruff format research/
git add research/walkforward.py research/tests/test_walkforward.py
git commit -m "feat(research): extract WalkForwardRunner.evaluate_fixed_params with fee override"
```

---

### Task 2: `research/cost_stress.py` + wire into `run_promotion_gate` and the CLI

**Files:**
- Create: `research/cost_stress.py`
- Modify: `research/gate.py`
- Modify: `research/cli.py`
- Test: `research/tests/test_cost_stress.py` (new)
- Test: `research/tests/test_gate.py` (add tests, keep existing 2 unchanged)
- Test: `research/tests/test_cli.py` (add a test, keep existing 3 unchanged)

**Interfaces:**
- Consumes: `WalkForwardRunner.evaluate_fixed_params` and `WindowResult` (Task 1);
  `research.statistics.deflated_sharpe_ratio` (existing); `freqtrade.optimize.backtesting.Backtesting`
  (existing, for reading `.fee`).
- Produces: `research.cost_stress.fee_sensitivity(config, pairs, timeframe, datadir, window_results, multipliers=(1.0, 1.25, 1.5, 2.0), periods_per_year=365) -> dict[float, dict]`
  — each value is `{"mean_test_sharpe": float, "deflated_sharpe": float, "n_windows": int}`.
  `research.gate.GateResult` gains `fee_sensitivity: dict[float, dict] | None = None`.
  `research.gate.run_promotion_gate` gains `fee_sensitivity_multipliers: tuple[float, ...] | None = None`.

- [ ] **Step 1: Write the failing tests**

Create `research/tests/test_cost_stress.py`:

```python
from pathlib import Path

import numpy as np
import pytest

from freqtrade.data import history
from freqtrade.data.history import get_timerange
from freqtrade.enums import RunMode
from freqtrade.optimize.backtesting import Backtesting
from research.cost_stress import fee_sensitivity
from research.statistics import deflated_sharpe_ratio
from research.walkforward import WalkForwardRunner, generate_windows
from tests.conftest import get_default_conf, patch_exchange


TESTDATADIR = Path(__file__).resolve().parents[2] / "tests" / "testdata"
EXMS = "freqtrade.exchange.exchange.Exchange"


def _conf():
    conf = get_default_conf(TESTDATADIR)
    conf["runmode"] = RunMode.BACKTEST
    conf["max_open_trades"] = 10
    conf["use_exit_signal"] = False
    return conf


def _patch(mocker):
    patch_exchange(mocker)
    mocker.patch(f"{EXMS}.get_min_pair_stake_amount", return_value=0.00001)
    mocker.patch(f"{EXMS}.get_max_pair_stake_amount", return_value=float("inf"))
    mocker.patch(f"{EXMS}.get_pair_base_currency", lambda _, x: x.split("/")[0])


def _window_results(conf):
    full_data = history.load_data(datadir=TESTDATADIR, timeframe="5m", pairs=["UNITTEST/BTC"])
    min_date, max_date = get_timerange(full_data)
    total_days = max(8, (max_date - min_date).days)
    train_days = max(1, total_days // 8)
    test_days = max(1, total_days // 16)
    windows = generate_windows(min_date, max_date, train_days, test_days)
    runner = WalkForwardRunner(conf, pairs=["UNITTEST/BTC"], timeframe="5m", datadir=TESTDATADIR)
    return runner, runner.run(windows, [{"buy_rsi": 25}, {"buy_rsi": 35}])


def test_fee_sensitivity_reports_one_entry_per_multiplier(mocker):
    conf = _conf()
    _patch(mocker)
    _runner, results = _window_results(conf)

    report = fee_sensitivity(
        conf, pairs=["UNITTEST/BTC"], timeframe="5m", datadir=TESTDATADIR,
        window_results=results, multipliers=(1.0, 1.5),
    )

    assert set(report) == {1.0, 1.5}
    for stats in report.values():
        assert isinstance(stats["mean_test_sharpe"], float)
        assert 0.0 <= stats["deflated_sharpe"] <= 1.0
        assert stats["n_windows"] == len(results)


def test_fee_sensitivity_baseline_matches_direct_recomputation(mocker):
    conf = _conf()
    _patch(mocker)
    runner, results = _window_results(conf)

    report = fee_sensitivity(
        conf, pairs=["UNITTEST/BTC"], timeframe="5m", datadir=TESTDATADIR,
        window_results=results, multipliers=(1.0,),
    )

    base_fee = Backtesting(conf).fee
    direct = [
        runner.evaluate_fixed_params(wr.window, wr.best_params, fee_override=base_fee)
        for wr in results
    ]
    expected_mean_sharpe = float(np.mean([r.test_sharpe for r in direct]))
    expected_n_obs = sum(len(r.test_returns) for r in direct)
    expected_deflated = deflated_sharpe_ratio(
        expected_mean_sharpe, n_obs=expected_n_obs, n_trials=1, periods_per_year=365
    )

    assert report[1.0]["mean_test_sharpe"] == pytest.approx(expected_mean_sharpe)
    assert report[1.0]["deflated_sharpe"] == pytest.approx(expected_deflated)


def test_fee_sensitivity_net_pnl_is_non_increasing_as_fee_rises(mocker):
    conf = _conf()
    _patch(mocker)
    runner, results = _window_results(conf)

    report_1 = fee_sensitivity(
        conf, pairs=["UNITTEST/BTC"], timeframe="5m", datadir=TESTDATADIR,
        window_results=results, multipliers=(1.0,),
    )
    report_2 = fee_sensitivity(
        conf, pairs=["UNITTEST/BTC"], timeframe="5m", datadir=TESTDATADIR,
        window_results=results, multipliers=(2.0,),
    )

    base_fee = Backtesting(conf).fee
    pnl_1x = sum(
        sum(runner.evaluate_fixed_params(wr.window, wr.best_params, fee_override=base_fee * 1.0).test_returns)
        for wr in results
    )
    pnl_2x = sum(
        sum(runner.evaluate_fixed_params(wr.window, wr.best_params, fee_override=base_fee * 2.0).test_returns)
        for wr in results
    )
    assert pnl_1x >= pnl_2x
    assert report_1[1.0]["n_windows"] == report_2[2.0]["n_windows"]


def test_fee_sensitivity_raises_on_empty_or_non_positive_multipliers(mocker):
    conf = _conf()
    _patch(mocker)
    _runner, results = _window_results(conf)

    with pytest.raises(ValueError, match="multipliers"):
        fee_sensitivity(
            conf, pairs=["UNITTEST/BTC"], timeframe="5m", datadir=TESTDATADIR,
            window_results=results, multipliers=(),
        )
    with pytest.raises(ValueError, match="multipliers"):
        fee_sensitivity(
            conf, pairs=["UNITTEST/BTC"], timeframe="5m", datadir=TESTDATADIR,
            window_results=results, multipliers=(0.0,),
        )
```

Add to `research/tests/test_gate.py` (after the two existing tests, keep the existing
`TESTDATADIR`/`EXMS`/`_conf`/`_patch` helpers unchanged):

```python
def test_run_promotion_gate_attaches_fee_sensitivity_when_it_passes(mocker, tmp_path):
    conf = _conf()
    _patch(mocker)
    full_data = history.load_data(datadir=TESTDATADIR, timeframe="5m", pairs=["UNITTEST/BTC"])
    min_date, max_date = get_timerange(full_data)
    total_days = max(8, (max_date - min_date).days)
    train_days = max(1, total_days // 8)
    test_days = max(1, total_days // 16)

    # Permissive thresholds guarantee a real, unmocked PASS on the small fixture dataset
    # (any run with at least one trade clears them) without mocking the statistics --
    # deterministic real-execution test, not a forced/mocked pass.
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
        fee_sensitivity_multipliers=(1.0, 1.5),
    )

    assert result.passed is True
    assert result.fee_sensitivity is not None
    assert set(result.fee_sensitivity) == {1.0, 1.5}


def test_run_promotion_gate_skips_fee_sensitivity_when_it_fails(mocker, tmp_path):
    conf = _conf()
    _patch(mocker)
    full_data = history.load_data(datadir=TESTDATADIR, timeframe="5m", pairs=["UNITTEST/BTC"])
    min_date, max_date = get_timerange(full_data)
    total_days = max(8, (max_date - min_date).days)
    train_days = max(1, total_days // 8)
    test_days = max(1, total_days // 16)

    # Impossible dsr_threshold guarantees a real FAIL, regardless of the actual result.
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
        fee_sensitivity_multipliers=(1.0, 1.5),
    )

    assert result.passed is False
    assert result.fee_sensitivity is None
```

Add to `research/tests/test_cli.py` (after the existing tests, keep the existing imports
and both existing tests unchanged):

```python
def test_gate_command_prints_fee_sensitivity_table_when_present(mocker, capsys):
    canned = GateResult(
        strategy_id="StrategyTestV3",
        passed=True,
        deflated_sharpe=0.97,
        permutation_p=0.01,
        pbo=0.1,
        mean_test_sharpe=1.2,
        n_trials=12,
        reasons=[],
        fee_sensitivity={
            1.0: {"mean_test_sharpe": 0.87, "deflated_sharpe": 0.91, "n_windows": 5},
            1.5: {"mean_test_sharpe": 0.33, "deflated_sharpe": 0.52, "n_windows": 5},
        },
    )
    mocker.patch("research.cli.run_promotion_gate", return_value=canned)
    mocker.patch(
        "research.cli.Configuration.from_files", return_value={"datadir": "user_data/data"}
    )

    exit_code = main(
        [
            "gate",
            "--strategy", "StrategyTestV3",
            "--config", "config.json",
            "--pairs", "BTC/USDT",
            "--timeframe", "1h",
            "--start", "2024-01-01",
            "--end", "2024-06-01",
            "--train-days", "60",
            "--test-days", "20",
            "--param-grid", '[{"buy_rsi": 30}]',
            "--fee-sensitivity",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "fee sensitivity" in captured.out
    assert "baseline" in captured.out
    assert "1.50x fee" in captured.out
    assert "slippage" not in captured.out.lower()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest research/tests/test_cost_stress.py research/tests/test_gate.py research/tests/test_cli.py -v`
Expected: `test_cost_stress.py`'s tests FAIL with
`ModuleNotFoundError: No module named 'research.cost_stress'`. The two new `test_gate.py`
tests FAIL with `TypeError: run_promotion_gate() got an unexpected keyword argument
'fee_sensitivity_multipliers'`. The new `test_cli.py` test FAILS with `TypeError:
GateResult.__init__() got an unexpected keyword argument 'fee_sensitivity'`.

- [ ] **Step 3: Implement `research/cost_stress.py`**

```python
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


def fee_sensitivity(
    config: dict,
    pairs: list[str],
    timeframe: str,
    datadir: Path,
    window_results: list[WindowResult],
    multipliers: tuple[float, ...] = (1.0, 1.25, 1.5, 2.0),
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
```

- [ ] **Step 4: Extend `research/gate.py`**

In `research/gate.py`:

1. Add an import: `from research.cost_stress import fee_sensitivity`
2. Add one field to `GateResult`, after `reasons: list[str]`:

```python
    fee_sensitivity: dict[float, dict] | None = None
```

3. Add one parameter to `run_promotion_gate`'s signature, after `periods_per_year: int = 365,`:

```python
    fee_sensitivity_multipliers: tuple[float, ...] | None = None,
```

4. After the `passed = not reasons` line and before the `log_candidate_result(...)` call, add:

```python
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
```

5. Add `fee_sensitivity=fee_report,` to the final `return GateResult(...)` call, after `reasons=reasons,`.

- [ ] **Step 5: Extend `research/cli.py`**

In `research/cli.py`, add one CLI argument after the `--db-path` line:

```python
    gate.add_argument(
        "--fee-sensitivity",
        action="store_true",
        help="Also run a fee-sensitivity stress test if the gate passes (informational)",
    )
```

Change the `run_promotion_gate(...)` call to add one more keyword argument, after `db_path=args.db_path,`:

```python
            fee_sensitivity_multipliers=(1.0, 1.25, 1.5, 2.0) if args.fee_sensitivity else None,
```

After the existing `for reason in result.reasons:` loop and its `print`, and before the
`return 0 if result.passed else 1` line, add:

```python
        if result.fee_sensitivity:
            print("  fee sensitivity (informational, not part of PASS/FAIL):")
            for i, (mult, stats) in enumerate(result.fee_sensitivity.items()):
                label = f"{mult:.2f}x fee" + (" (baseline)" if i == 0 else "")
                print(
                    f"    {label:<22} mean OOS sharpe {stats['mean_test_sharpe']:>6.2f}"
                    f"   deflated_sharpe {stats['deflated_sharpe']:.3f}"
                )
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest research/tests -v`
Expected: PASS (36 tests: 29 from after Task 1 + 4 new in `test_cost_stress.py` + 2 new in
`test_gate.py` + 1 new in `test_cli.py`).

- [ ] **Step 7: Get a second opinion via lmchatbot**

```bash
curl -s localhost:3000/ 2>/dev/null || (node C:/dev/lmchatbot/server.js &)
cat > /tmp/lmchatbot_fee_task2.json <<'EOF'
{"provider":"chatgpt","verify":"gemini","prompt":"Reviewing a fee-sensitivity stress test wired into a trading-strategy promotion gate. research/cost_stress.py::fee_sensitivity(config, pairs, timeframe, datadir, window_results, multipliers=(1.0,1.25,1.5,2.0), periods_per_year=365) reads a base fee once from a throwaway freqtrade Backtesting(config).fee, then for each multiplier re-evaluates every already-selected WindowResult.best_params via WalkForwardRunner.evaluate_fixed_params(window, best_params, fee_override=base_fee*multiplier), aggregates mean per-window test_sharpe and feeds it into deflated_sharpe_ratio(mean_sharpe, n_obs=<total concatenated test_returns count>, n_trials=1, periods_per_year=365). This is called from research/gate.py's run_promotion_gate ONLY when the gate already passed (using n_trials=1 deliberately, since no new parameter search/selection happens at any fee level -- the original gate's own n_trials already paid for the one selection event that chose these params). It's opt-in via a --fee-sensitivity CLI flag, informational only, never changes the PASS/FAIL verdict, and is explicitly never called 'slippage' anywhere (freqtrade's backtester has no real slippage/execution-price model, only a configurable fee). Any correctness issues, real design gaps, or reasoning errors in this? Is the n_trials=1 reasoning actually sound?"}
EOF
curl -s localhost:3000/ask -d @/tmp/lmchatbot_fee_task2.json
rm /tmp/lmchatbot_fee_task2.json
```

Read the reply, including both `draft` and `reply` (verifier) fields. If it surfaces a
real, concrete gap, fix it and rerun Step 6; otherwise proceed.

- [ ] **Step 8: Lint, run the full suite once more, and commit**

```bash
.venv/Scripts/python.exe -m ruff check research/ --fix
.venv/Scripts/python.exe -m ruff format research/
.venv/Scripts/python.exe -m pytest research/tests -v
git add research/cost_stress.py research/gate.py research/cli.py \
        research/tests/test_cost_stress.py research/tests/test_gate.py research/tests/test_cli.py
git commit -m "feat(research): add fee-sensitivity stress test, opt-in via --fee-sensitivity"
```

---

## Self-Review Notes

- **Spec coverage:** every component in the spec (`research/cost_stress.py`, the new
  `WalkForwardRunner.evaluate_fixed_params` method with the `run_window` refactor to reuse
  it, `GateResult.fee_sensitivity`, `run_promotion_gate`'s new parameter, the CLI flag and
  table) maps to a concrete step above. The spec's "What this is not" constraints (no
  slippage naming, no hard gate, no re-search, `n_trials=1`) are all encoded as Global
  Constraints and enforced in the actual code (not just prose) — e.g. the CLI test asserts
  `"slippage" not in captured.out.lower()`.
- **No placeholders:** every step has complete, real code — no `# TODO`, no "similar to
  above." The lmchatbot steps are genuine second-opinion checks with concrete prompts, not
  filler.
- **Type/signature consistency:** `evaluate_fixed_params`'s signature
  (`window, params, fee_override=None`) is defined once in Task 1 and called identically
  in Task 2's `cost_stress.py` and every test. `fee_sensitivity`'s signature and return
  shape (`dict[float, dict]` with `mean_test_sharpe`/`deflated_sharpe`/`n_windows` keys) is
  defined once and consumed identically by `gate.py` and the CLI's table-printing code.
  `GateResult.fee_sensitivity` and `run_promotion_gate`'s `fee_sensitivity_multipliers`
  parameter names match between the dataclass, the function signature, and every call site
  across both tasks' tests.
