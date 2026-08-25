# Parameter Stability — Design

## Context

Third sub-project decomposed from the research-MVP gap analysis of
`CRYPTO_STRATEGY_DISCOVERY_PROPOSAL.md` §12 (robustness testing) — after fee-sensitivity
(`docs/superpowers/specs/2026-08-24-fee-sensitivity-stress-test-design.md`, shipped) and
regime breakdown (`docs/superpowers/specs/2026-08-24-regime-breakdown-design.md`, shipped).
§12 asks: "does profitability exist across a region of parameter space (not just one lucky
value)?"

`research/walkforward.py`'s `WalkForwardRunner.run_window` already grid-searches
`param_grid` on every window's TRAIN period, picks the highest-train-Sharpe variant, and
backtests only that winner on the held-out TEST period. `research/gate.py` already builds a
`variant_matrix` (shape `n_variants x n_windows`, each cell a variant's mean train-period
fractional return) from every window's `variant_returns`, purely to feed
`probability_of_backtest_overfitting`. This sub-project reuses that same matrix — already
computed, zero new backtests — to answer a different question than PBO does.

## What this is not

**Not a re-backtest of every grid variant's OOS performance.** The alternative design (run
every variant's TEST-period backtest too, not just the window's winner, and measure how many
are OOS-profitable) was considered and rejected: it costs `len(param_grid)x` more
backtesting, AND it lets parameter-stability evaluation peek at test-period data for variants
that were never selected as a window's winner — reintroducing a form of the exact
data-snooping problem PBO already exists to catch. Cross-checked with an external model
(lmchatbot/Gemini) before locking this in: confirmed this in-sample-only approach is the
standard technique, commonly called **Parameter Plateau Analysis** in quant research — robust
strategies occupy a broad, flat region of train-period profitability; overfit strategies
occupy a single sharp spike. Consensus favors evaluating plateau shape on train data
specifically so the OOS test period stays a single-pass validation gate, undiluted by
per-variant peeking.

**Not a local-neighborhood/plateau-radius metric.** The same external review flagged that a
global "fraction of all grid variants profitable" can mislead when a grid spans wildly
different regimes (e.g. fast-scalping params mixed with slow-trend params in one grid) — a
tighter version would measure sign-consistency only in the local neighborhood around the
winning variant. Not built here: it requires a distance metric over arbitrary param dicts
(numeric, categorical, or mixed), which is real added complexity for a component that's
informational, not gate-deciding. Global fraction-positive is the MVP metric; the
neighborhood-vs-global tradeoff is documented as a known limitation in code, not silently
assumed away — a caller with a deliberately heterogeneous grid should read a modest
`parameter_stability` score with that caveat in mind.

**Not folded into the promotion decision.** Like fee-sensitivity and regime breakdown, this
never changes `GateResult.passed` — informational evidence for `robustness_score`/
`strategy_report`, not a new gate criterion.

**Not gated on `passed`.** Fee-sensitivity only runs for an already-passing candidate (a
validated edge's cost margin is what's interesting). Parameter stability follows regime
breakdown's precedent instead: it describes whether the *training* signal was a plateau or a
spike, which is diagnostic evidence for a **failed** candidate too (and it costs nothing extra
either way, since `variant_matrix` is already built unconditionally for PBO). Runs whenever
`include_parameter_stability=True`, regardless of `passed`.

## Architecture

```
run_promotion_gate() [existing, research/gate.py]
  │
  ├─ ... existing walk-forward / variant_matrix / PBO logic [unchanged]
  │
  └─ if include_parameter_stability:                (regardless of passed/failed)
         stability = parameter_stability(variant_matrix)
         attached to GateResult.parameter_stability

research/parameter_stability.py  [new]
  parameter_stability(variant_matrix) -> float
    -- fraction of grid variants (rows) whose mean train-period return across
       windows (columns) is positive. Fail-open to 1.0 for a single-variant
       grid (no region to test, nothing to penalize) -- same fail-open
       convention as scoring.py's cost_sensitivity.
```

## Components

**`research/parameter_stability.py`** (new file) — one pure function:

`parameter_stability(variant_matrix: np.ndarray) -> float`

`variant_matrix` is the exact `n_variants x n_windows` array `run_promotion_gate` already
builds for `probability_of_backtest_overfitting` — each cell is one param variant's mean
train-period fractional return on one window (from `WindowResult.variant_returns`, itself
built by `WalkForwardRunner.run_window`). No new data collection, no new backtests.

Computes `row_means = variant_matrix.mean(axis=1)` (each variant's mean train-period return
across all windows), then returns `float((row_means > 0).sum() / len(row_means))` — the
fraction of the grid that is, on average, profitable in-sample. Always in `[0, 1]`.

Fail-open: if `variant_matrix.shape[0] == 1` (a single-variant grid — the caller didn't
actually search a region), returns `1.0` — there's no region to be unstable across, so this
isn't evidence against the candidate, matching `scoring.robustness_score`'s existing
"1.0 fail-open when only one [multiplier/variant] was tested" convention for
`cost_sensitivity`.

Raises `ValueError` if `variant_matrix.ndim != 2` or `variant_matrix.shape[0] == 0` — a
caller-contract violation (mirrors `regime_report`'s `ValueError` on a caller-side shape
mismatch), not a data condition this function should silently paper over.

**`research/gate.py`** (existing file, extended) — `run_promotion_gate` gains one new
optional parameter, `include_parameter_stability: bool = False` (default off — existing
callers/tests unaffected). Computed **unconditionally on pass/fail** from the `variant_matrix`
already in scope (built earlier in the function for PBO — no duplicate work, no new
backtests). Attached to a new `GateResult.parameter_stability: float | None = None` field.

**`research/scoring.py`** (existing file, extended) — `WEIGHTS` gains
`"parameter_stability": 0.05` (same order of magnitude as `regime_consistency`; still
`ponytail:`-flagged as a starting value, not empirically derived — the existing disclaimer on
`WEIGHTS` already covers this). `robustness_score` adds a `parameter_stability` component
straight from `result.parameter_stability` when not `None` — already a probability in
`[0, 1]` by construction, no transform needed (unlike `permutation_p`/`pbo`, which need
`1 - x`). `strategy_report` gets one new informational line, alongside the existing
`deflated_sharpe`/`permutation p`/`PBO`/`mean OOS sharpe`/`trials` block — a single float,
same style as those existing lines (not a fraction breakdown: `GateResult.parameter_stability`
is designed as one float, like `deflated_sharpe`/`permutation_p`/`pbo`, not a dict of counts
like `regime_breakdown`, so there is no `n_positive`/`n_total` available to print at report
time):

```
  parameter stability  0.750
```

**`research/cli.py`** (existing file, extended) — `gate` subcommand gains
`--parameter-stability` (flag, default off), threading `include_parameter_stability=True`
through to `run_promotion_gate`. No new print logic needed beyond what `strategy_report`
already produces (the CLI already prints `strategy_report(result)`).

## Data flow

1. `research gate --strategy X ... --parameter-stability` → `cli.main()`
2. `run_promotion_gate(..., include_parameter_stability=True)`
3. Existing walk-forward / `variant_matrix` / PBO logic runs unchanged; `variant_matrix`
   already in scope
4. Regardless of `passed`: `stability = parameter_stability(variant_matrix)`
5. Attached to `GateResult.parameter_stability`
6. `scoring.robustness_score`/`strategy_report` (called by the CLI, as today) pick it up
   automatically once present — no CLI-specific formatting code required

## Error handling

- `parameter_stability` raises `ValueError` on a malformed `variant_matrix` (wrong ndim, zero
  rows) — a caller-contract violation, fail loudly rather than return a misleading number.
- Single-variant grid: NOT an error — fails open to `1.0`, documented above.
- `run_promotion_gate` passes its own `variant_matrix` through unchanged; no new failure mode
  is introduced at the gate level beyond what already exists for PBO on that same array.

## Testing

- `parameter_stability` (`research/tests/test_parameter_stability.py`, new): pure-function
  unit tests, no real backtesting needed (construct `np.ndarray` directly, same style as
  `research/tests/test_pbo.py`):
  1. All variants profitable (every row's mean > 0) → `1.0`.
  2. No variants profitable → `0.0`.
  3. Mixed (e.g. 2 of 4 rows have positive mean, 2 negative — including a row whose
     individual cells straddle zero but whose row mean is unambiguously signed) → exact
     expected fraction.
  4. Single-variant grid (`shape[0] == 1`, any values) → fails open to `1.0`.
  5. Malformed input (`ndim != 2`, or zero rows) → `ValueError`.
- `run_promotion_gate` (`research/tests/test_gate.py`, extended): mirrors the existing
  `test_run_promotion_gate_attaches_regime_breakdown_when_requested_and_{passes,fails}` /
  `test_run_promotion_gate_omits_regime_breakdown_by_default` trio — three new real
  end-to-end cases (real `UNITTEST/BTC` fixture data, `_patch(mocker)`, no mocking of
  `parameter_stability`'s own logic): populated when `include_parameter_stability=True` and
  the gate passes, populated the same way when it fails (proves the pass/fail-independence
  asymmetry is actually implemented), and `None` by default when the flag is omitted.
- `robustness_score`/`strategy_report` (`research/tests/test_scoring.py`, extended): mirrors
  the existing `test_robustness_score_includes_regime_consistency_when_regime_breakdown_present`
  / `test_robustness_score_with_both_...` / `test_strategy_report_includes_regime_breakdown_...`
  pattern — one test confirming `parameter_stability` folds into the weighted average with an
  independently-hand-computed expected score, one confirming it composes correctly alongside
  the other optional components already present in `_core_result`, one confirming
  `strategy_report` includes the new line when present and omits it when absent.

## Open items resolved during brainstorming

- Data source: **reuse the existing `variant_matrix`** (train-period only, zero extra
  backtests) — considered and rejected re-backtesting every variant's OOS performance, both
  for cost and for reintroducing per-variant OOS peeking (user-approved design fork,
  cross-checked with lmchatbot/Gemini: confirmed as the standard "Parameter Plateau Analysis"
  technique and quant-research consensus).
- Metric: **global fraction of grid variants with positive mean train-period return**, not a
  local-neighborhood/plateau-radius metric — deferred as real added complexity for an
  informational-only component; documented as a known limitation for heterogeneous grids
  (see "What this is not" above; flagged by the same external review).
- Gating: **runs regardless of PASS/FAIL** (mirrors regime breakdown's precedent, not
  fee-sensitivity's `passed`-only gating) — free to compute either way, and diagnostic for a
  failed candidate too.
- Fail-open behavior for a single-variant grid: **`1.0`**, matching `cost_sensitivity`'s
  existing fail-open convention in `scoring.py`.
