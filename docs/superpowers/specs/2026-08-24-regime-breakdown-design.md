# Regime Breakdown — Design

## Context

Second sub-project decomposed from `CRYPTO_STRATEGY_DISCOVERY_PROPOSAL.md` (§5, market
regimes) — the first was the fee-sensitivity stress test
(`docs/superpowers/specs/2026-08-24-fee-sensitivity-stress-test-design.md`, shipped).
`research/` (ledger, DSR/BH-FDR, PBO, walk-forward runner, promotion gate, fee-sensitivity
stress test) already ships and has been proven end-to-end against a real strategy
(`EmaTrendFollow`, correctly rejected on BTC/USDT).

This sub-project attributes a candidate's out-of-sample performance to the market
conditions each walk-forward window's test period actually occurred in. Unlike
fee-sensitivity (only meaningful for a candidate that already passed), regime breakdown is
valuable for a **failed** candidate too — it can distinguish "bad everywhere" from "only
bad in one kind of market," which is real diagnostic information the plain PASS/FAIL
verdict throws away.

## What this is not

**Not the proposal's full 7-category taxonomy.** §5 asks for Bull, Bear, Sideways, High
volatility, Low volatility, Crash, Recovery. This spec ships **two independent axes**
instead — Trend (Bull/Bear/Sideways) × Volatility (High/Low) — dropping Crash and Recovery
from the MVP. Trend and volatility are genuinely orthogonal (the proposal's own example
table lists them as separate rows, not a single enum), so reporting them as two axes is
more informative than collapsing them into one label, not less. Crash/Recovery need a
drawdown-magnitude threshold that's harder to calibrate defensibly with only price data
and no drawdown-history baseline — deferred rather than shipped as an arbitrary guess.

**Not a statistical/unsupervised regime model.** A Hidden Markov Model or Gaussian mixture
on returns was considered and rejected: more "principled"-sounding, but it's a black box a
non-expert can't audit, which cuts directly against this whole project's transparency
goal (§14 of the proposal: "hide complexity, not evidence"). Simple, explicit,
threshold-based rules — reproducible and auditable by inspection — are what the rest of
`research/` already does (DSR, BH-FDR, PBO are all closed-form, not fit models).

**Not richer signals (funding rate, BTC dominance).** Classified purely from the traded
pair's own OHLCV — already downloaded for backtesting, no new data source, no new fetch
code. A real VIX-style external volatility index doesn't have a crypto equivalent readily
available; deferred rather than half-built.

**Not a causal/live-safe classifier (verified via lmchatbot cross-check).** The volatility
axis ranks each window against the **median across every window in the run**, including
windows chronologically after the one being classified. That's fine for this spec's stated
use — a one-shot, post-hoc historical report generated after the whole backtest already
ran, so it can't leak into the trading decisions it's describing (the classifier never
touches signal generation or fill logic). It is **not** safe to reuse this exact function
for a live/production regime-switching signal, which would need a trailing, point-in-time
baseline instead (e.g. a rolling historical volatility window) rather than a full-sample
median. If a live regime signal is ever wanted, that's a different function, not a new
call site for this one.

**Not per-trade attribution.** Each walk-forward window's whole test period gets ONE
regime label, not each individual trade. Finer-grained per-trade tagging would require
exposing per-trade entry/exit timestamps from `WalkForwardRunner` — a real change to
code that's already shipped and reviewed twice (Task 1/2 of the fee-sensitivity plan). Per
the proposal's own example output (a single PASS/FAIL/WEAK per regime row, not per trade),
window-level granularity is what's actually being asked for.

**Not gated on the gate's PASS/FAIL verdict.** Fee-sensitivity only runs for a passing
candidate (an already-validated edge's cost margin is what's interesting). Regime
breakdown deliberately runs regardless of pass/fail — a failed candidate's regime
breakdown is diagnostic evidence for the next research iteration, not wasted compute.

**Not folded into the promotion decision.** Like fee-sensitivity, this is informational —
it never changes `GateResult.passed`. The proposal's §5 closing line ("This should
influence robustness scoring") is future work (a robustness-scoring sub-project doesn't
exist yet); this spec produces the raw evidence a future scorer could consume, not the
scorer itself.

**Not Deflated-Sharpe-scored per regime bucket.** Per-regime sample sizes (often 1-3
windows) are too thin for DSR to say anything meaningful — reporting a DSR per bucket
would manufacture false precision. Each bucket reports raw `mean_test_sharpe`, `total_return`,
`n_trades`, `n_windows` — real numbers a human can weigh, explicitly not a statistical
verdict.

## Architecture

```
run_promotion_gate() [existing, research/gate.py]
  │
  ├─ windows = generate_windows(...)              [existing]
  ├─ results = runner.run(windows, param_grid)    [existing — list[WindowResult]]
  ├─ ... existing DSR / BH-FDR / PBO / ledger logic [unchanged]
  │
  └─ if include_regime_breakdown:                  (regardless of passed/failed)
         labels = classify_regimes(pairs[0], timeframe, datadir, windows)
         regime = regime_report(results, labels)
         attached to GateResult.regime_breakdown

research/regime.py  [new]
  classify_regimes(pair, timeframe, datadir, windows, trend_threshold=0.05) -> list[str]
    -- two-pass: (1) load each window's own OHLCV test-period close series (no
       Backtesting instance needed -- just history.load_data), compute total
       return + realized volatility per window; (2) find the median realized
       volatility ACROSS all windows in this run, classify each window's vol
       as High/Low relative to that median (self-referential baseline, no
       external index needed), combine with trend into e.g. "Bull/High".
       One label per window, same order as `windows`.

  regime_report(window_results, labels) -> dict[str, dict]
    -- groups window_results by label (parallel list, not a dict key --
       Window is an unfrozen dataclass, not hashable), aggregates exactly
       like gate.py/cost_stress.py already do (concatenate test_returns,
       mean per-window test_sharpe) -- consistent with the established
       pattern, no new aggregation convention invented.
```

## Components

**`research/regime.py`** (new file) — two functions:

`classify_regimes(pair: str, timeframe: str, datadir: Path, windows: list[Window], trend_threshold: float = 0.05) -> list[str]`

For each window, builds a `freqtrade.configuration.TimeRange` from that window's own
`test_start`/`test_end` fields (the same `TimeRange("date", "date", int(ts_start.timestamp()),
int(ts_end.timestamp()))` construction `research/walkforward.py` already uses), and loads
that `[test_start, test_end)` close-price series via one `freqtrade.data.history.load_data`
call per window (no `Backtesting` instance — this needs only price data, not indicators or
a strategy; no `startup_candles` needed either, since there's no indicator warmup to
satisfy). One load per window, not a single bulk load re-sliced — simpler, and the number
of windows in a typical run (single digits to low tens) makes the repeated I/O cheap;
revisit only if this becomes a measured bottleneck. From the loaded close series, computes:
- `total_return = close.iloc[-1] / close.iloc[0] - 1`
- `realized_vol = close.pct_change().dropna().std()`

(Windows within one walk-forward run always share the same `train_days`/`test_days` —
`generate_windows` takes single scalar values applied to every window it produces — so
every window's test period is the same duration. The fixed `trend_threshold` is therefore
being compared against a consistent basis within a run; comparing thresholds across runs
with different `test_days` is a separate concern this spec doesn't address.)

Trend label: `"Bull"` if `total_return > trend_threshold`, `"Bear"` if
`total_return < -trend_threshold`, else `"Sideways"`. `trend_threshold` defaults to `0.05`
(5% total return over the window) — a starting default, not empirically derived; adjust
based on real usage once this runs against real strategies (`ponytail:`-flagged in code).

Volatility label: computed in a second pass once every window's `realized_vol` is known —
`"High"` if this window's vol is strictly above the **median** vol across all windows in
this run, `"Low"` otherwise (a window sitting exactly on the median, or in a tied group
with the median, is `"Low"` by construction — deliberate, not an unresolved edge case:
`>` not `>=`, so ties resolve toward `"Low"`). Self-referential to the run's own data (no
external volatility index needed), which is honest about what it is: "more/less volatile
than this backtest's other periods," not "objectively high/low." See "Not a causal/live-safe
classifier" above for why this full-sample median is fine for this report but not for a
live signal.

Final label is `f"{trend}/{volatility}"`, e.g. `"Bull/High"`, `"Bear/Low"`,
`"Sideways/High"` — 6 possible combinations. Returns one label per window, in the same
order as `windows`.

Degenerate input: a window whose test period has fewer than 2 candles (can't compute a
return or volatility) is fail-closed to `total_return=0.0, realized_vol=0.0` —
classified `"Sideways"` on the trend axis by construction (`0.0` is inside the
`[-threshold, threshold]` band) and whatever the vol median comparison yields on the
volatility axis (deterministic, not a crash).

`regime_report(window_results: list[WindowResult], labels: list[str]) -> dict[str, dict]`

Raises `ValueError` if `len(window_results) != len(labels)` (mismatched parallel lists — a
caller error, fail loudly). Groups `window_results` by their parallel `labels` entry, and
for each distinct label reports:
- `n_windows`: how many windows got this label
- `n_trades`: total trade count across those windows (`sum(wr.test_n_trades)`)
- `mean_test_sharpe`: mean of `wr.test_sharpe` across those windows — same aggregation
  style already used in `gate.py`/`cost_stress.py`, not a new convention
- `total_return` (named precisely per lmchatbot review — NOT `total_pnl`, which would
  imply compounded equity/dollar P&L; this is neither): the plain sum of every trade's
  fractional return across those windows (`sum(r for wr in group for r in wr.test_returns)`,
  where each `r` is already `profit_abs / dry_run_wallet`, matching how `test_returns` is
  populated everywhere else in `research/`). An arithmetic sum of fractional returns, not
  a geometrically compounded return — fine as a rough same-units aggregate for comparing
  regime buckets against each other, not a claim about realized account growth.

`WindowResult.test_sharpe` is never `NaN` for a zero-trade window — verified against
`freqtrade/data/metrics.py`'s `calculate_sharpe`, which returns the plain sentinel `0` when
`len(trades) == 0` (not `NaN`, not a different sentinel). `np.mean` over a group containing
such a window is therefore always well-defined, no explicit `NaN`-guard needed in
`regime_report`.

**`research/gate.py`** (existing file, extended) — `run_promotion_gate` gains one new
optional parameter, `include_regime_breakdown: bool = False` (default off — existing
callers and existing tests are unaffected). Computed **unconditionally on pass/fail** (the
one deliberate asymmetry with fee-sensitivity, documented above) using `pairs[0]` as the
reference pair, `windows` and `results` already in scope from the existing walk-forward
run — no duplicate work. Attached to a new `GateResult.regime_breakdown: dict[str, dict] |
None = None` field. Multi-pair regime blending (if `pairs` has more than one entry) is out
of scope — `pairs[0]` is used as the single reference asset, documented as a scope limit,
not silently wrong behavior.

**`research/cli.py`** (existing file, extended) — `gate` subcommand gains
`--regime-breakdown` (flag, default off). When set, threads
`include_regime_breakdown=True` through to `run_promotion_gate`, and if
`result.regime_breakdown` is present (always, when the flag was passed — unlike
fee-sensitivity, this isn't conditional on `passed`), prints a table:

```
  regime breakdown (informational, not part of PASS/FAIL):
    Bull/High       2 windows   14 trades   mean sharpe  0.42   total return  0.0012
    Bear/Low        1 windows    6 trades   mean sharpe -1.10   total return -0.0034
    Sideways/Low     3 windows   19 trades   mean sharpe -0.05   total return -0.0002
```

## Data flow

1. `research gate --strategy X ... --regime-breakdown` → `cli.main()`
2. `run_promotion_gate(..., include_regime_breakdown=True)`
3. Existing gate logic runs unchanged; `windows`, `results` already in scope
4. Regardless of `passed`: `labels = classify_regimes(pairs[0], timeframe, datadir, windows)`
5. Inside `classify_regimes`: one `history.load_data` call per window (price-only, no
   backtest) to get total return + realized vol; second pass computes the vol median
   across all windows and finalizes each label
6. `regime = regime_report(results, labels)` — groups and aggregates
7. Attached to `GateResult.regime_breakdown`; CLI prints the table if present

## Error handling

- `classify_regimes` raises `ValueError` on an empty `windows` list — nothing to classify,
  fail loudly rather than return an empty/misleading result (mirrors
  `run_promotion_gate`'s own `< 4 windows` `ValueError` pattern).
- `regime_report` raises `ValueError` on mismatched list lengths between `window_results`
  and `labels` — a caller-contract violation, not a data problem.
- A degenerate (near-empty) window's price data fails closed to a defined, deterministic
  label rather than raising or crashing (see "Degenerate input" above) — a single thin
  window shouldn't abort the whole report.

## Testing

- `classify_regimes`: real execution (real `history.load_data`, real `UNITTEST/BTC`
  fixture data, matching the existing `research/tests/` pattern):
  1. **Trend boundary test**: construct windows over known-different price regions of the
     real fixture (discovered programmatically via the fixture's actual price series, not
     hand-picked indices — same approach `research/tests/test_walkforward.py` already
     uses for its own fixtures) and confirm the trend axis assigns `"Bull"` when total
     return exceeds `trend_threshold`, `"Bear"` when below `-trend_threshold`, `"Sideways"`
     otherwise — derived from the real computed `total_return`, not hand-calculated.
  2. **Volatility relative-ranking test**: construct several windows with genuinely
     different realized volatility (e.g. a flat-price window vs. a sharply oscillating
     one) and confirm the higher-vol window(s) get `"High"` and the lower-vol window(s)
     get `"Low"`, relative to the set's own median — not asserting an absolute threshold,
     since the design deliberately doesn't have one.
  3. **Degenerate window test**: a window whose test period has 0 or 1 candles gets a
     deterministic `"Sideways"` trend label, no exception raised.
  4. **Empty windows list**: raises `ValueError`.
- `regime_report`: no real backtesting needed — construct `WindowResult` objects directly
  (plain dataclass construction, real `research.walkforward.WindowResult`/`Window`, not
  mocks) with known `test_sharpe`/`test_n_trades`/`test_returns` values and known parallel
  labels, assert the grouped aggregates (`n_windows`, `n_trades`, `mean_test_sharpe`,
  `total_return`) match hand-computed expected values for a small, fully-specified case.
  Mismatched-length `ValueError` test.
- `run_promotion_gate` / `cli.py`: extend the existing real end-to-end gate tests — one
  assertion path confirming `GateResult.regime_breakdown` is populated when
  `include_regime_breakdown=True` **and the gate fails** (the one deliberate asymmetry
  with fee-sensitivity — this is the test that actually proves the asymmetry was
  implemented, not just described), and populated the same way when it passes.
  `test_cli.py` gets one more case: `--regime-breakdown` flag present, `run_promotion_gate`
  mocked to return a `GateResult` with `regime_breakdown` populated, assert the table
  appears in stdout.

## Open items resolved during brainstorming

- Granularity: **per-window**, not per-trade (user decision — avoids touching already-shipped
  `WalkForwardRunner`/`WindowResult` structures again).
- Signal source: **BTC/USDT's own OHLCV only** (user decision — no new data source/fetch code).
- Category scope: **two axes (Trend × Volatility), not the proposal's 7-category taxonomy**
  — Crash/Recovery deferred (design decision, presented and approved in chat).
- Gating: **runs regardless of PASS/FAIL** (design decision, presented and approved in chat)
  — the one deliberate asymmetry with the fee-sensitivity precedent, called out explicitly
  everywhere it matters (spec, and to be mirrored in the eventual plan's Global Constraints)
  so it doesn't read as an inconsistency with the established pattern.

**Spec review round (lmchatbot, Gemini draft cross-verified by ChatGPT).** The draft raised
eight points; the verify pass corrected several as overstated (raw per-candle `.std()`
doesn't need annualizing to be a valid volatility measure; a fixed ±5% threshold is fine
across equal-duration windows, which every window in one run already is by construction;
"statistically unsound below 6 windows" isn't a real threshold, just "noisier with fewer
windows"; equal-median handling is a deliberate, already-specified `>`-not-`>=` choice, not
an unresolved ambiguity). Four real findings survived the cross-check and were applied
above: (1) `total_pnl` was a misleading name for a plain sum of fractional trade returns —
renamed `total_return` with an explicit non-compounding disclaimer; (2) the
`Window`→`TimeRange` translation and the one-load-per-window approach needed to be spelled
out, not left implicit; (3) the median-relative volatility classifier's non-causal,
research-only nature needed its own explicit "not" scope item, not just an implied
limitation; (4) zero-trade `test_sharpe` NaN-safety was unspecified — resolved definitively
by citing `calculate_sharpe`'s actual `0`-not-`NaN` sentinel behavior (verified against real
freqtrade source, not assumed).
