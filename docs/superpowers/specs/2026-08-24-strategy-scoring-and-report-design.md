# Strategy Scoring & Report — Design

## Context

Third sub-project decomposed from `CRYPTO_STRATEGY_DISCOVERY_PROPOSAL.md` (§14-15, strategy
scoring and strategy report). The first two were the fee-sensitivity stress test and regime
breakdown (both shipped). `research/` now produces, per gate run: a PASS/FAIL verdict with
three thresholded statistics (deflated Sharpe, permutation p-value, PBO), and two optional
informational dimensions (fee-sensitivity, regime breakdown). This sub-project adds nothing
new to *compute* — it synthesizes what `GateResult` already carries into (a) one continuous
robustness score for ranking/comparing candidates, and (b) one standardized human-readable
report, replacing the ad-hoc print statements currently duplicated inline in `research/cli.py`.

## What this is not

**Not a new statistical test.** Every input to the score is already computed by
`run_promotion_gate` before this sub-project's code ever runs. No new backtesting, no new
walk-forward evaluation, no new data load.

**Not the proposal's full component list.** §14 lists OOS performance, OOS consistency,
drawdown, Sharpe/Sortino, deflated Sharpe, parameter stability, time stability, asset
stability, regime stability, cost sensitivity, trade count, Monte Carlo risk, profit
concentration, IS/OOS degradation. Only five of these are ever actually available on a
`GateResult` today: deflated Sharpe, the permutation test (the proposal's "Monte Carlo
risk" — `permutation_test` already **is** a sign-flip Monte Carlo test, not a separate
thing to build), PBO, cost sensitivity (from `fee_sensitivity`, when requested), and regime
stability (from `regime_breakdown`, when requested). Drawdown, Sortino, parameter/time/asset
stability, profit concentration, and IS/OOS degradation would all require new fields on
`WindowResult` or new equity-curve reconstruction that no current code produces — deferred
rather than half-built from data that doesn't exist yet. `passed` itself is not a
component — the score is deliberately not a re-derivation of pass/fail (see below).

**Not a replacement for the PASS/FAIL verdict.** `GateResult.passed` and `reasons` are
untouched — the score is a supplementary, continuous signal (particularly useful for
comparing several FAILED candidates to see which is closest, or ranking several PASSED
ones), not a second gate. A strategy that fails the gate can still have a computed score;
the report shows both, never lets one substitute for the other.

**Not a new CLI flag.** The report is not conditional — it's what `research gate` already
prints, reorganized into one reusable function instead of ten scattered `print()` calls.
`--fee-sensitivity` and `--regime-breakdown` keep gating whether those *sections* of the
report have content, exactly as today.

**Not configurable weights.** The score's five component weights are fixed module-level
constants, not a runtime parameter — matches this package's established pattern for
similar starting defaults (`trend_threshold` in `research/regime.py`,
`DEFAULT_FEE_MULTIPLIERS` in `research/cost_stress.py`): a `ponytail:`-flagged starting
point, adjustable in code once this runs against real strategies, not prematurely
generalized into a config object nobody has asked to tune yet.

## Architecture

```
research/scoring.py  [new]
  robustness_score(result: GateResult) -> float
    -- weighted average of whichever components GateResult actually has populated,
       renormalized over the weights actually used. Always in [0, 1].

  strategy_report(result: GateResult, pair: str | None = None) -> str
    -- the exact text research/cli.py currently prints, unchanged in content, now built
       as one string instead of ad-hoc print() calls, plus one new line embedding
       robustness_score(result).

research/cli.py  [modified]
  gate command's print block replaced with print(strategy_report(result, pair=...))
  -- no new flags, no behavior change to what gets shown or the exit code.
```

## Components

**`research/scoring.py`** (new file):

```python
WEIGHTS = {
    "deflated_sharpe": 0.35,
    "significance": 0.25,       # 1 - permutation_p
    "pbo_inverse": 0.25,        # 1 - pbo
    "cost_sensitivity": 0.10,   # only when result.fee_sensitivity is present
    "regime_consistency": 0.05, # only when result.regime_breakdown is present
}
```

`robustness_score(result: GateResult) -> float`

Every one of `deflated_sharpe`, `permutation_p`, and `pbo` is already a probability in
`[0, 1]` by construction (`deflated_sharpe_ratio` returns `norm.cdf(...)`;
`permutation_test` returns a fraction of matching permutations; `probability_of_backtest_
overfitting` returns a probability) — no rescaling needed, just `1 - x` for the two where
lower is better (`permutation_p`, `pbo`).

Always-present components:
- `"deflated_sharpe"`: `result.deflated_sharpe` directly.
- `"significance"`: `1.0 - result.permutation_p`.
- `"pbo_inverse"`: `1.0 - result.pbo`.

`"cost_sensitivity"`, only when `result.fee_sensitivity` is truthy: let
`baseline_mult = min(result.fee_sensitivity)`, `stress_mult = max(result.fee_sensitivity)`
(the lowest and highest multiplier actually tested — not hardcoded to `1.0`, since a caller
could in principle pass `fee_sensitivity` computed with custom multipliers that don't
include `1.0`). Let `baseline_dsr = result.fee_sensitivity[baseline_mult]["deflated_sharpe"]`,
`stress_dsr = result.fee_sensitivity[stress_mult]["deflated_sharpe"]`.
- If `baseline_mult == stress_mult` (only one multiplier was tested — no stress signal
  either way): component = `1.0` (fail open, not closed — there's genuinely no
  information to penalize on).
- Elif `baseline_dsr <= 0`: component = `0.0` (fail closed — dividing by a non-positive
  baseline is meaningless, and an already-zero-or-negative baseline deflated Sharpe is
  the worst case regardless of how stress compares to it).
- Else: component = `max(0.0, min(1.0, stress_dsr / baseline_dsr))` — how much of the
  baseline's statistical edge survives at the worst tested fee level, clipped to `[0, 1]`
  (a `stress_dsr` that's *higher* than baseline, e.g. from a genuinely noisy small sample,
  should not push the component above `1.0` and give an unearned bonus).

`"regime_consistency"`, only when `result.regime_breakdown` is truthy:
`n_positive = sum(1 for stats in result.regime_breakdown.values() if stats["mean_test_sharpe"] > 0)`,
`n_total = len(result.regime_breakdown)`, component = `n_positive / n_total` — the fraction
of regime buckets where the strategy was net-positive on average, not just one favorable
bucket carrying the whole result.

Final score: `sum(WEIGHTS[k] * v for k, v in components.items()) / sum(WEIGHTS[k] for k in
components)` — a weighted average renormalized over whichever components are actually
present. Always in `[0, 1]`: every component value is itself in `[0, 1]`, and a weighted
average of values in `[0, 1]` stays in `[0, 1]`. The denominator is never zero — the three
always-present components alone sum to `0.85` of weight.

**Known limitation, deliberate (verified via lmchatbot cross-check): scores are only
directly comparable between `GateResult`s computed with the *same* set of optional
components present.** Renormalizing over available weight means an identical underlying
performance profile scores differently depending on whether `fee_sensitivity`/
`regime_breakdown` were requested for that run — e.g. adding a strong
`regime_consistency` value shifts the denominator from `0.85` to `0.90`, changing the
result even though the three core statistics didn't move. This is a real property of
the design, not an oversight: the alternative (a fixed `1.0` denominator that treats a
missing optional component as `0`) would instead punish every ordinary run that never
requested the optional, compute-costly fee-sensitivity/regime-breakdown analyses,
capping their maximum achievable score below what a more expensively-evaluated
candidate could reach for reasons having nothing to do with actual strategy quality —
worse than the chosen tradeoff, since both `fee_sensitivity` and `regime_breakdown` stay
opt-in specifically to avoid forcing extra backtesting nobody asked for (see "Not a new
CLI flag" above). Read `robustness_score` as "how robust this candidate looks *given the
evidence gathered for it*," not a universal ranking number — a future batch-ranking tool
built on top of this (see §16's non-expert workflow) should compare only
same-configuration runs, or surface which optional components fed into each score
alongside the number. `strategy_report`'s printed sections already make this visible:
which of the fee-sensitivity/regime-breakdown blocks appear tells a reader which
components the accompanying score used.

`strategy_report(result: GateResult, pair: str | None = None) -> str`

Builds and returns (does not print) the exact text `research/cli.py`'s `gate` command
prints today, verbatim in content and format, plus one new line for the robustness score.
Reusing the current format exactly means every existing `research/tests/test_cli.py`
assertion on printed substrings (`"PASS"`, `"FAIL"`, `"fee sensitivity"`, `"baseline"`,
`"1.50x fee"`, `"regime breakdown"`, `"Bull/High"`, `"Bear/Low"`, `"slippage" not in
...lower()`) continues to pass unchanged once `cli.py` is switched to print this
function's return value — this sub-project is a pure refactor-plus-one-line-addition at
the CLI layer, not a format change.

```
{strategy_id}: {PASS|FAIL}
  robustness score  {score:.3f}
  deflated_sharpe   {deflated_sharpe:.3f}
  permutation p     {permutation_p:.3f}
  PBO               {pbo:.3f}
  mean OOS sharpe   {mean_test_sharpe:.3f}
  trials (ledger)   {n_trials}
  - {reason}                                    [one line per entry in reasons]
  fee sensitivity (informational, not part of PASS/FAIL):     [only if fee_sensitivity truthy]
    {mult:.2f}x fee{" (baseline)" if mult == 1.0 else ""}   mean OOS sharpe {..:>6.2f}   deflated_sharpe (n_trials=1) {..:.3f}
  regime breakdown ({pair}, informational, not part of PASS/FAIL):   [only if regime_breakdown truthy; omit "({pair}, " -> "(" when pair is None]
    {label:<15} {n_windows:>2} windows   {n_trades:>3} trades   mean sharpe {mean_test_sharpe:>6.2f}   total return {total_return:>8.4f}
```

The `robustness score` line's placement (second line, right after the verdict) is
deliberate: it is the one number meant to be read alongside PASS/FAIL, not buried after
the individual statistics that already justify the verdict on their own.

`pair` is optional because `strategy_report` operates on a `GateResult` alone — it has no
way to know which pair was classified for a `regime_breakdown` unless the caller tells it
(the CLI knows, from `args.pairs`; a caller scoring several already-computed `GateResult`s
programmatically, e.g. for a ranking table, may not always have it handy). When `pair` is
`None`, the regime-breakdown header line simply omits the parenthetical pair name rather
than raising or guessing.

**`research/cli.py`** (existing file, modified): the `gate` command's block of ~12
individual `print()` calls (verdict through the regime-breakdown table) is replaced with:

```python
from research.scoring import strategy_report
...
print(strategy_report(result, pair=args.pairs.split(",")[0]))
```

No new argparse flags. `return 0 if result.passed else 1` is unchanged and stays after
this line, in the same position.

## Data flow

1. `run_promotion_gate(...)` (unchanged) produces a `GateResult`.
2. `research gate` CLI command calls `strategy_report(result, pair=args.pairs.split(",")[0])`.
3. Inside `strategy_report`: calls `robustness_score(result)` once, embeds it, then builds
   the rest of the text from `result`'s existing fields exactly as `cli.py` does today.
4. `print()`s the returned string; exit code logic is unchanged.

A caller other than the CLI (e.g. a future batch-ranking script iterating over several
`GateResult`s from the ledger) can call `robustness_score` directly without going through
`strategy_report` at all — the two functions are independently useful, not coupled.

## Error handling

- `robustness_score` never raises on a well-formed `GateResult` — every branch above is
  total (covers `fee_sensitivity`/`regime_breakdown` present-and-absent, and the two
  cost-sensitivity edge cases explicitly). A malformed `GateResult` (e.g.
  `permutation_p` outside `[0, 1]`) is a caller-contract violation from `run_promotion_
  gate` itself, which is not this sub-project's concern to re-validate — `GateResult`'s
  fields are already trusted throughout the rest of `research/`.
- `strategy_report` never raises — it only formats fields already present on a
  well-formed `GateResult`, the same fields `cli.py` already formats today without
  guards.

## Testing

- `research/tests/test_scoring.py` (new): real `GateResult` construction (no mocking),
  matching this package's established house style.
  1. Only the three always-present components (no `fee_sensitivity`/`regime_breakdown`):
     assert the result equals an independently-computed weighted average using the same
     `WEIGHTS` constants (imported, not hardcoded decimal literals in the test) but a
     fresh, separately-written arithmetic expression — not calling `robustness_score`
     recursively.
  2. With `fee_sensitivity` added (two multipliers, deflated Sharpe genuinely lower at
     the higher one): assert the result equals an independently-computed weighted
     average over all four weights actually used (`deflated_sharpe`, `significance`,
     `pbo_inverse`, `cost_sensitivity`), renormalized over their sum (`0.95`) — an exact
     value, not a directional claim, computed the same way as case 1.
  3. With `regime_breakdown` added (a mix of positive- and negative-mean-Sharpe buckets):
     assert the result equals an independently-computed weighted average over the four
     weights actually used (`deflated_sharpe`, `significance`, `pbo_inverse`,
     `regime_consistency`), renormalized over their sum (`0.90`) — same exact-value
     approach.
  4. Both `fee_sensitivity` and `regime_breakdown` present together: assert the result
     equals an independently-computed weighted average over all five weights,
     renormalized over the full `1.0` denominator.
  5. Cost-sensitivity edge case: `fee_sensitivity` with a single multiplier key →
     component is exactly `1.0`.
  6. Cost-sensitivity edge case: baseline (`min` multiplier) `deflated_sharpe` is `0.0` →
     component is exactly `0.0`.
  7. Sanity bound: construct several varied `GateResult`s (including a deliberately bad
     one — `deflated_sharpe=0.0, permutation_p=1.0, pbo=1.0`) and assert every score is
     in `[0.0, 1.0]`.
  8. Documents the known renormalization limitation as real, intentional behavior (not a
     regression to "fix" later): two `GateResult`s with identical `deflated_sharpe`/
     `permutation_p`/`pbo`, one with a `regime_breakdown` added whose buckets are all
     positive (`regime_consistency = 1.0`), assert the two scores are NOT equal —
     confirms the documented tradeoff is real and the test suite would catch anyone
     "fixing" it into fixed-denominator scoring without an explicit spec change.
- `research/tests/test_scoring.py` also covers `strategy_report`:
  9. PASS case, no fee/regime: assert `"PASS"`, the formatted `deflated_sharpe`/
     `permutation p`/`PBO`/`mean OOS sharpe`/`trials (ledger)` lines, and a
     `"robustness score"` line with a numeric value, all appear in the returned string.
  10. FAIL case with `reasons`: assert `"FAIL"` and every reason string appear.
  11. `fee_sensitivity` present: assert `"fee sensitivity"`, `"baseline"`, a
      `"1.50x fee"`-style label, and that `"slippage"` never appears (case-insensitive) —
      the same three assertions `test_cli.py`'s existing fee-sensitivity test makes today,
      now made directly against `strategy_report`'s return value.
  12. `regime_breakdown` present with `pair="BTC/USDT"`: assert `"regime breakdown"`,
      `"BTC/USDT"`, and both regime labels appear.
  13. `regime_breakdown` present with `pair=None`: assert `"regime breakdown"` and the
      regime labels still appear, and the string does not contain a stray `"(, "` or
      similar malformed parenthetical from a missing pair name.
- `research/tests/test_cli.py` (existing file): **no test changes required** — every
  existing assertion (verdict text, exit codes, fee-sensitivity substrings,
  regime-breakdown substrings and the `include_regime_breakdown` kwarg check) continues
  to pass once `cli.py`'s print block is replaced with `print(strategy_report(result,
  pair=...))`, because `strategy_report`'s format is byte-identical to the text those
  tests already check for. Re-running the existing suite after the `cli.py` change is
  itself the regression proof that this refactor didn't alter any user-visible output.
