# Fee Sensitivity Stress Test — Design

## Context

First sub-project decomposed from `CRYPTO_STRATEGY_DISCOVERY_PROPOSAL.md` (§9, cost
stress testing). The proposal's own MVP list groups this with regime analysis,
robustness scoring, and reporting — genuinely independent subsystems, decomposed
per the brainstorming process rather than specced together.

`research/` (ledger, DSR/BH-FDR, PBO, walk-forward runner, promotion gate + CLI) already
ships and has been proven end-to-end against a real strategy (`EmaTrendFollow`, correctly
rejected on BTC/USDT). This sub-project adds one more piece of evidence to a candidate
that already passed the main statistical gate: does its edge survive materially worse
transaction costs?

## What this is not

**Not a slippage/market-impact model.** freqtrade's backtester has no execution-price
slippage simulation at all — orders fill at the exact requested price as long as it's
within the candle's high/low range, only a configurable percentage fee is modeled
(verified: `FREQTRADE_RESEARCH_ARCHITECTURE.md` §3, `freqtrade/optimize/backtesting.py`).
A flat fee-rate multiplier is a legitimate cost-sensitivity / margin-of-safety test, but
it gets slippage's actual structure wrong, not just approximately right — slippage scales
with order size, book depth, and volatility, not as a fixed percentage of notional
(cross-checked via lmchatbot, ChatGPT+Gemini, both converged on this distinction
independently). This sub-project is scoped to **fee sensitivity only**, named
accordingly everywhere (module, function, CLI flag, report labels) — never "slippage." A
real execution-shortfall model (moving the simulated fill price based on volatility/
spread/size) is explicitly deferred as its own future sub-project.

**Not a hard gate criterion.** Runs only on a candidate that has already PASSed the main
statistical gate (DSR + BH-FDR + PBO). Informational — it doesn't change the PASS/FAIL
verdict, it adds evidence for a human to weigh. (If experience later shows this should
become a hard gate, that's a follow-up decision made with real data in hand, not upfront.)

**Not a re-run of the parameter search.** The main gate has already selected the best
parameters for each walk-forward window (`WindowResult.best_params`). This sub-project
re-evaluates those *already-chosen* parameters at worse fee levels — it does not
re-optimize at each fee level (that would be a different, much more expensive experiment,
and would reopen the multiple-testing question this whole system exists to guard against).

**On `n_trials=1` for the fee-stress DSR (reviewed and deliberately kept, not an
oversight):** it's tempting to think the fee-stressed Sharpe should inherit the original
gate's `n_trials`, since the strategy being stress-tested was still chosen via a search.
That reasoning conflates two different questions. DSR's trial-count corrects for
selection bias *at the moment a maximum is chosen from N candidates* — that correction
was already paid once, at the original gate's PASS decision, using the ledger's real
`family_trial_count`. Fee-stressing doesn't select anything: it applies one deterministic
transformation (a higher fee) to one already-fixed, already-selected trade sequence, once
per multiplier, with no comparison across candidates and no new maximum being taken.
There is no new selection event to correct for, so no new multiplicity penalty accrues —
using the original `n_trials` here would double-count the same correction rather than
apply a second, independent one. `n_trials=1` here answers a narrower, different
question than the original gate's DSR: "is this specific, already-fixed, already-selected
return series distinguishable from noise at this cost level" — not "was the search that
found it valid" (that was already settled).

## Architecture

```
run_promotion_gate() [existing, research/gate.py]
  │
  ├─ results = runner.run(windows, param_grid)   [existing — list[WindowResult]]
  ├─ ... existing DSR / BH-FDR / PBO / ledger logic [unchanged]
  │
  └─ if passed and fee_sensitivity_multipliers given:
         fee_report = cost_stress.fee_sensitivity(
             config, pairs, timeframe, datadir, results, fee_sensitivity_multipliers
         )
         attached to GateResult.fee_sensitivity

cost_stress.fee_sensitivity()  [new, research/cost_stress.py]
  │  for each multiplier (default 1.0, 1.25, 1.5, 2.0):
  │    for each WindowResult already in hand:
  │      runner.evaluate_fixed_params(window, best_params, fee_override=base_fee*multiplier)
  │    aggregate exactly as run_promotion_gate does (mean Sharpe, concatenated
│    n_obs) -> deflated_sharpe_ratio(..., n_trials=1) at this fee level
  └─ returns {multiplier: {mean_test_sharpe, deflated_sharpe, n_windows}}

WalkForwardRunner.evaluate_fixed_params()  [new method, research/walkforward.py]
  -- extracted from run_window's existing "test phase" (load data, compute indicators
     for the given fixed params, trim, single backtest call, compute Sharpe/returns) --
  now reusable with an optional fee override, without re-running the grid search.
  run_window() itself is unchanged; this is a new method alongside it, not a rewrite.
```

## Components

**`research/cost_stress.py`** (new file) — one function:

`fee_sensitivity(config, pairs, timeframe, datadir, window_results, multipliers=(1.0, 1.25, 1.5, 2.0), periods_per_year=365) -> dict[float, dict]`

(`periods_per_year` passed straight through from `run_promotion_gate`'s own parameter of
the same name and default — never a second, independently-specified value.)

Reads the config's effective base fee once (constructing one throwaway `Backtesting`
instance and reading its resolved `.fee`, mirroring how freqtrade itself resolves fee —
explicit `config["fee"]` if set, else the exchange's own taker/maker fee). This is the
exact same fee value the original `run_promotion_gate` call used, since `config` is
passed through unchanged — the `1.0` multiplier is therefore an exact-reproduction
control, not a stress level; report labels call it "baseline," not "1.0x stress."

For each multiplier, re-evaluates every `WindowResult`'s already-selected `best_params` on
that window's test period at `base_fee * multiplier`, via the new
`WalkForwardRunner.evaluate_fixed_params`. **Aggregation method (exactly mirrors
`run_promotion_gate`'s own existing aggregation — same functions, same shape, so the
`1.0`/baseline result reproduces the original gate's `mean_test_sharpe` exactly; its
`deflated_sharpe` is NOT directly comparable to the gate's own — this module always calls
`deflated_sharpe_ratio` with `n_trials=1` by design, while the gate's own headline
`deflated_sharpe` uses the ledger's real `n_trials`, and DSR is monotone non-increasing in
`n_trials` — see "On `n_trials=1`" above):** concatenate `test_returns` across all windows for `n_obs`; average
each window's `test_sharpe` for the point Sharpe estimate; feed that point estimate into
`deflated_sharpe_ratio(mean_sharpe, n_obs=n_obs, n_trials=1, periods_per_year=...)` (same
`periods_per_year` as the original gate call). This is not a new aggregation convention —
it is `research/gate.py`'s existing `mean_test_sharpe`/`n_obs`/`deflated_sharpe_ratio`
call, reused verbatim at each fee level.

**`WalkForwardRunner.evaluate_fixed_params(window, params, fee_override=None) -> WindowResult`**
(new method on the existing class in `research/walkforward.py`) — the single-params,
single-window evaluation that `run_window`'s final phase already does once it has settled
on `best_params`, extracted into its own method so it's callable directly without a grid
search. It runs BOTH a train-period and a test-period backtest, mirroring `run_window`'s
own per-variant structure so the two share one implementation rather than drifting apart —
but only the test-period result (`test_sharpe`, `test_n_trades`, `test_returns`) is
meaningful to a caller evaluating an already-fixed parameter set. The train-period result
(`train_sharpe`) exists only because `run_window`'s own final phase needs this method to
look identical to its grid-search loop's per-variant calls; every current caller
(`run_window`'s final phase, `cost_stress.fee_sensitivity`) either overwrites or ignores
it. `fee_override`, when given, builds a config with that fee
for just this call (never mutates `self.config` — avoids leaking a stress-test fee across
other calls sharing the same `WalkForwardRunner` instance). Returns a `WindowResult` with
`variant_returns={}` (an explicit empty dict, not `None` — no grid was searched, so there
are no variants to report; only `best_params`, `test_sharpe`, `test_n_trades`,
`test_returns` are meaningful). This is a deliberately **partial** `WindowResult` —
`cost_stress.py` is documented as its only intended consumer; any future caller that reads
`variant_returns` off a `WindowResult` must not assume it came from a full grid search
unless it came from `run_window`. `run_window` itself is refactored to call this new
method for its own final test-phase step, so the two paths share one implementation
rather than drifting — this also means `run_window`'s own existing tests double as
regression coverage for `evaluate_fixed_params`'s core behavior, not just the new tests
below.

**Invariant this method must uphold (revised after implementation — the original claim
below was wrong, kept struck through so the correction has context):**
`fee_override` changes only the fee rate charged in the backtest's own P&L accounting. It
must never affect ~~indicator computation or signal generation — the same `params` on the
same `window` must produce the exact same trade entries/exits (same count, same
timestamps) at every fee level; only each trade's realized `profit_abs` changes~~ *indicator
computation* — that part still holds (fee never touches `populate_indicators`/
`populate_entry_trend`). But **trade count and timing are not invariant to fee for
ROI-exit strategies**, discovered during Task 1's implementation and independently
verified against real freqtrade source and real fixture data (see the "Deviations from
this plan" section near the top of
`docs/superpowers/plans/2026-08-24-fee-sensitivity-stress-test.md` for the durable
record): `should_exit()` (`freqtrade/strategy/interface.py:1440-1464`) computes
`current_profit = trade.calc_profit_ratio(...)`, which is fee-adjusted
(`freqtrade/persistence/trade_model.py:1206-1234`), and compares it against `minimal_roi`
thresholds — so a higher fee can genuinely delay or prevent an ROI exit, changing *which*
trades occur, not just their realized P&L. Empirically confirmed on the real
`UNITTEST/BTC` fixture: 6 trades at `fee=0.0` vs. 2 trades at `fee=0.05`.

This means the fee-sensitivity comparison is **not** a clean like-for-like comparison of
identical trades at different costs — it's closer to "how does this strategy's realized
performance change under worse fee assumptions, including any resulting shift in exit
timing." That's still meaningful evidence (it's still asking "does raising costs hurt this
candidate"), just a less surgically isolated question than originally specified. Two
concrete consequences: (1) the trade-set-identity test below is now a documented
non-invariant, not a guarantee; (2) net P&L is **not** guaranteed monotonic across fee
levels either — the same mechanism that breaks trade-count invariance also breaks the P&L
monotonicity claim previously in this section, confirmed empirically on the real fixture
at a wide fee sweep (0.0 → 0.1): total P&L was *higher* at `fee=0.1` than at `fee=0.05`,
because fewer/different trades occurred. Neither is asserted as a hard invariant in the
test suite below.

**`research/gate.py`** (existing file, extended) — `run_promotion_gate` gains one new
optional parameter, `fee_sensitivity_multipliers: tuple[float, ...] | None = None`
(default `None` — existing callers and existing tests are unaffected). When set and the
gate PASSed, calls `cost_stress.fee_sensitivity` using the `results` list already
computed in this call (no duplicate walk-forward run) and attaches it to a new
`GateResult.fee_sensitivity: dict | None = None` field.

**`research/cli.py`** (existing file, extended) — `gate` subcommand gains
`--fee-sensitivity` (flag, default off — opt-in, since it multiplies the already-completed
walk-forward evaluation's backtest count by `len(multipliers)`, roughly ~4x the work for
the default 4 multipliers, and is only useful once a candidate has passed). When set,
threads `fee_sensitivity_multipliers=(1.0, 1.25, 1.5, 2.0)` through to
`run_promotion_gate`, and if `result.fee_sensitivity` is present, prints a small table:

```
  fee sensitivity (informational, not part of PASS/FAIL):
    1.00x fee (baseline)   mean OOS sharpe  0.87   deflated_sharpe 0.91
    1.25x fee               mean OOS sharpe  0.61   deflated_sharpe 0.74
    1.50x fee               mean OOS sharpe  0.33   deflated_sharpe 0.52
    2.00x fee               mean OOS sharpe -0.12   deflated_sharpe 0.08
```

## Data flow

1. `research gate --strategy X ... --fee-sensitivity` → `cli.main()`
2. `run_promotion_gate(..., fee_sensitivity_multipliers=(1.0,1.25,1.5,2.0))`
3. Existing gate logic runs unchanged; `results: list[WindowResult]` already in scope
4. If `passed`: `cost_stress.fee_sensitivity(config, pairs, timeframe, datadir, results, multipliers)`
5. Inside: one throwaway `Backtesting(config)` to resolve `base_fee`; then per multiplier,
   per window, `runner.evaluate_fixed_params(window, window_result.best_params, fee_override)`
6. Aggregated dict attached to `GateResult.fee_sensitivity`; CLI prints the table

## Error handling

- If `passed` is `False`, fee sensitivity is skipped entirely regardless of the flag
  (nothing to stress-test — matches the "informational report on passing candidates"
  decision). `GateResult.fee_sensitivity` stays `None`; the CLI prints nothing extra.
- `multipliers` must be non-empty and all values `> 0`; `fee_sensitivity` raises
  `ValueError` on an empty tuple or a non-positive multiplier (mirrors the existing
  `run_promotion_gate`'s `< 4 windows` `ValueError` pattern — fail loudly on a
  caller-supplied parameter that can't produce a meaningful result, don't silently
  return an empty report).
- `evaluate_fixed_params` reuses `run_window`'s existing data-loading/backtest error
  behavior unchanged (no new error handling needed there — same freqtrade calls, same
  failure modes already exercised by the existing `run_window` tests).

## Testing

**Deliberately not tested as a hard invariant: Sharpe monotonically decreasing as fee
increases.** It's true for a fixed trade set *if* each trade's dollar fee were exactly
constant, but real per-trade fees scale with that trade's notional value, which varies
slightly across trades (fill price, position size) — so it's a strong empirical tendency,
not a mathematical guarantee, and asserting it as exact would risk a flaky test on a
technically-correct-but-non-monotonic result. Testing the invariants that *are* actually
guaranteed instead:

- `WalkForwardRunner.evaluate_fixed_params` — three real-execution tests (real
  `Backtesting`, real fixture data, matching the existing `test_walkforward.py` pattern):
  1. **Equivalence chain:** for the winning variant of an existing walk-forward window,
     `run_window`'s own test-phase result, `evaluate_fixed_params(window, best_params,
     fee_override=None)`, and `evaluate_fixed_params(window, best_params,
     fee_override=base_fee)` all produce identical `test_sharpe` and `test_returns` —
     proving the extraction preserved behavior and that `fee_override=None` and
     `fee_override=base_fee` are truly equivalent.
  2. **~~Trade-set invariance~~ Fee-override produces a real, measurable result change**
     (revised — see "Invariant this method must uphold" above): calling
     `evaluate_fixed_params` on the same window/params at two different fee levels does
     **not** reliably produce the same `test_n_trades` for ROI-exit strategies — confirmed
     false on real data, not merely untested. The test instead asserts fee_override
     produces a measurable difference in the result (Sharpe or trade count) and that P&L
     moves in the expected direction on average, without asserting trade-count equality or
     strict cross-level monotonicity as a hard invariant.
  3. **Config isolation:** `self.config` is byte-for-byte unchanged after a call with
     `fee_override` set (dict equality before/after) — proving no leakage across calls
     sharing the same `WalkForwardRunner` instance.
- `cost_stress.fee_sensitivity` — real end-to-end test reusing the existing
  `test_walkforward.py` / `test_gate.py` fixture pattern (`StrategyTestV3`, real
  `UNITTEST/BTC` data): run the existing walk-forward fixture to get real `WindowResult`s,
  call `fee_sensitivity` with a small multiplier tuple, assert: keys match the multipliers
  given; the `1.0` (baseline) entry's `mean_test_sharpe`/`deflated_sharpe` exactly match
  independently recomputing via `evaluate_fixed_params` + the same aggregation by hand in
  the test. **Cross-fee-level P&L/Sharpe monotonicity is deliberately not asserted as a
  hard invariant** (revised from the original spec, which called this "a real guarantee" —
  it isn't; see "Invariant this method must uphold" above, confirmed false by direct
  empirical measurement at a wide fee sweep on real data). Also a `ValueError` test for
  empty/non-positive multipliers.
- `run_promotion_gate` / `cli.py`: extend the existing real end-to-end gate test with one
  more assertion path — call with `fee_sensitivity_multipliers` set, assert
  `GateResult.fee_sensitivity` is populated when the gate passes; a second test (can reuse
  the existing "too few windows" or a canned `FAIL` scenario) confirms it stays `None`
  when the gate fails, even if multipliers were requested. `test_cli.py` gets one more
  case: `--fee-sensitivity` flag present, `run_promotion_gate` mocked to return a
  `GateResult` with `fee_sensitivity` populated, assert the table appears in stdout.

## Open items resolved during brainstorming

- Gate vs. report: **informational report**, not a hard PASS/FAIL criterion (user decision).
- Slippage: **out of scope**, named "fee sensitivity" throughout, never "slippage" (user
  decision after lmchatbot cross-check).
- Multiplier levels: **1.0, 1.25, 1.5, 2.0** (dropped the proposal's 3.0x — lmchatbot's
  recommended set, "optionally 2.0x" taken as included since it's cheap and informative).

**Spec review round (lmchatbot, ChatGPT draft cross-verified by Gemini — the two
disagreed with each other on two points, resolved by direct technical judgment rather
than picking a side by default):**
- `n_trials=1` for the fee-stress DSR: ChatGPT's draft flagged this as wrong (should
  inherit the original gate's `n_trials`); Gemini's verify pass corrected it back —
  confirmed correct as originally specified, reasoning strengthened in "What this is not"
  above so the question doesn't reopen for a future reader.
- Monotonic-Sharpe test invariant: ChatGPT's draft flagged it as too strong a claim to
  assert as exact; Gemini's verify pass defended it as exact. Neither fully right —
  approximately true for a fixed stake amount, not a strict guarantee. Replaced with
  invariants that are actually guaranteed (trade-set identity, net-P&L monotonicity,
  three-way equivalence chain) rather than assuming Sharpe monotonicity.
- Both providers agreed the DSR aggregation method was underspecified (resolved: reuse
  `run_promotion_gate`'s existing aggregation verbatim) and that the partial `WindowResult`
  contract from `evaluate_fixed_params` needed to be explicit (resolved above).

**Post-implementation correction (Task 1, not a review-round finding — discovered by the
implementer, independently verified by the task reviewer against real freqtrade source and
real fixture data):** this spec's two strongest "these ARE real guarantees, unlike Sharpe"
claims — trade-set invariance under `fee_override`, and net-P&L monotonicity across fee
levels — are both false for ROI-exit strategies. The lmchatbot review round above debated
whether *Sharpe* monotonicity was safe to assert and correctly concluded no; it did not
catch that the *replacement* invariants this spec substituted (trade-count identity, P&L
monotonicity) rest on the same "same trades, only cost differs" assumption, which turned
out to be the actually-wrong premise. Root cause and empirical confirmation are in
"Invariant this method must uphold" above. Lesson for future specs in this codebase: a
confident-sounding "unlike X, this one IS guaranteed" claim still needs the same
skepticism as the thing it's being contrasted with — being right that one invariant is
shaky doesn't make the alternative you propose automatically solid.
