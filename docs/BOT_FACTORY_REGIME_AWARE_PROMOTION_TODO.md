# Bot Factory Regime-Aware Promotion Gate TODO

Status: proposed follow-up for the Regime-Aware Strategy Portfolio and Shadow
Strategy Observation direction.

Created: 2026-05-19 JST.
Updated: 2026-05-21 JST after multi-regime local selector simulation.

Product direction: `docs/BOT_FACTORY_PRODUCT_VISION_TODO.md` is the current
product-level North Star. This document is a local selector-eligibility follow-up
under the Strategy Evidence and Suitability / Matching contracts.

## Goal

Prevent the existing long-window promotion gate and future parallel dry-run /
shadow observation flow from interfering with each other.

The system must not promote a strategy only because aggregate dry-run PnL looks
good over a long period. It must first prove where the strategy is useful,
where it is unsafe, and where `no_trade` is the correct output.

In this document, promotion means local selector eligibility only. It is
necessary but not sufficient for Phase 3 paper readiness and never bypasses
paper readiness, paper run planning, runtime validation, drift reporting, or an
explicit user request.

The intended local evidence unit is:

```text
strategy_version
+ signal_version
+ activation_regime_scope
+ risk_policy_version
+ regime_classifier_version
+ cost_model_id
```

Not:

```text
strategy_version only
```

## Non-Goals / Safety Boundaries

- Do not start paper trading, dry-run trading, live trading, exchange order
  endpoints, API-key usage, leverage, or shorting as part of this TODO.
- Do not let shadow observation start, stop, or switch a running bot in the
  current scope.
- Do not let recent dry-run strength override failed historical, walk-forward,
  cost robustness, readiness, runtime validation, or drift evidence.
- Do not globally promote a strategy from evidence concentrated in one market
  regime, one pair, one short calendar window, or one parameter-only retry
  family.
- Do not treat `no_trade` as success only because losses were avoided. Always
  measure opportunity cost against predeclared baselines, never hindsight-best
  strategy choice.

## Core Design Invariant

Promotion gate, shadow observation, and regime fitness must remain separate
layers.

```text
local backtest / walk-forward / local shadow replay
  -> observation ledger
  -> regime-stratified fitness scorecard
  -> local selector eligibility decision
  -> Phase 3 readiness input only
  -> future selector / risk governor input
```

Each layer must consume and emit explicit artifacts. The promotion gate must
not consume raw aggregate PnL directly.

## Decision Model

Use explicit local eligibility outcomes:

- `GLOBAL_SELECTOR_ELIGIBLE`
  - Allowed only when the candidate has sufficient cost-adjusted,
    walk-forward, multi-pair, and multi-regime evidence and no excluded regime
    requires blocking behavior.
- `REGIME_SCOPED_SELECTOR_ELIGIBLE`
  - Allowed when a candidate is useful only under specific regimes and has
    explicit blocked or `no_trade` behavior outside them.
- `SHADOW_ONLY`
  - Candidate remains observable, but cannot be selected.
- `NO_TRADE_POLICY`
  - The best decision for a regime is to stay out of market, with opportunity
    cost recorded.
- `QUARANTINE`
  - Candidate was previously useful but recent evidence drifted beyond
    tolerance.
- `REJECT`
  - Candidate lacks sufficient edge, robustness, or safety evidence.
- `INSUFFICIENT_EVIDENCE`
  - Regime evidence is under-sampled or data quality is not adequate.

Do not use `GLOBAL_PROMOTE` or `REGIME_SCOPED_PROMOTE` for current artifacts;
those names are too easy to confuse with Phase 3 paper/live promotion.

## Required Artifacts

### 1. Observation Ledger

Add a local-only artifact schema that records every comparable observation.

Current-scope `source_type` values:

```text
backtest
walk_forward
local_shadow_replay
```

Future values such as `future_paper`, `future_dry_run`,
`future_paper_observation`, and `future_dry_run_observation` must be rejected
by validators until later phase documentation explicitly permits the exact
paper/dry-run observation path.

Required fields:

```text
observation_id
created_at
source_type
strategy_id
strategy_version
candidate_id
signal_version
risk_policy_version
pair
timeframe
window_start
window_end
market_regime
regime_classifier_version
baseline_id
cost_model_id
normal_cost_bps
stress_cost_bps
trade_count
exposure_ratio
gross_return
net_return_normal_cost
net_return_stress_cost
max_drawdown
downside_deviation
win_rate
profit_factor
no_trade_reason
no_trade_opportunity_cost
data_quality_flags
reason_codes
```

Implemented foundation:

- [x] Add `observation_ledger` schema constants and validation in
  `freqtrade_ext/bot_factory/regime_promotion.py`.
- [x] Reject future paper/dry-run source types in the current scope.
- [x] Add local JSON ledger assembly with safety-scope flags showing no
  process control.

Remaining TODO:

- [ ] Add line-oriented JSONL output so multiple candidates can be compared
  without rewriting whole files.
- [ ] Add markdown summary rendering for human review.
- [ ] Ensure all future CLI artifacts contain no secrets, API keys, exchange
  credentials, or live order identifiers.

### 2. Market Regime Classifier

Add deterministic local-only regime labeling.

Initial coarse labels:

```text
trend_up
trend_down
range
high_volatility
low_volatility
liquidity_stress
post_spike_reversion
mixed
unknown
```

Keep action decisions out of labels. For example, `trend_down` is a regime;
`avoid_long` belongs in `no_trade_conditions`.

TODO:

- [ ] Define regime feature extraction from local candles.
- [ ] Include realized volatility, trend strength, range efficiency,
  volume/liquidity proxy, candle spread proxy, and data-quality flags.
- [ ] Keep taxonomy small and predeclared to reduce overfitting.
- [ ] Persist `regime_classifier_version`.
- [ ] Emit `unknown` instead of forcing a label when evidence is weak or data
  quality is poor.
- [ ] Add tests for deterministic labeling.
- [ ] Add tests that small candle perturbations do not cause excessive label
  churn.

### 3. Regime Fitness Scorecard

Add a scorecard that evaluates strategy usefulness per regime before local
selector eligibility.

Required dimensions:

```text
strategy_version
activation_regime_scope
market_regime
sample_days
window_count
trade_count
exposure_ratio
expectancy
profit_factor
win_rate
max_drawdown
downside_deviation
net_pnl_normal_cost
net_pnl_stress_cost
baseline_delta_normal_cost
baseline_delta_stress_cost
incumbent_delta
no_trade_opportunity_cost
confidence_interval
lower_confidence_bound
walk_forward_pass_rate
pair_concentration
calendar_concentration
data_quality_pass
decision
reason_codes
```

Implemented foundation:

- [x] Add regime scorecard aggregation by `strategy_version x market_regime`
  in `freqtrade_ext/bot_factory/regime_promotion.py`.
- [x] Require positive candidate return and baseline delta after normal and
  stress costs.
- [x] Require minimum sample days, windows, trades, pass rate, confidence
  lower bound, pair concentration, calendar concentration, and drawdown limits.
- [x] Mark under-sampled regimes as `INSUFFICIENT_EVIDENCE`, not pass/fail.
- [x] Persist reason codes for each regime decision.

Remaining TODO:

- [ ] Compare against incumbent approved strategy when present.
- [ ] Add explicit `no_trade` baseline generation and opportunity-cost
  fixtures.
- [ ] Add confidence interval calculation beyond supplied lower bound or the
  conservative fallback.

### 4. Promotion Gate Interference Guard

Protect the existing promotion gate from consuming misleading aggregate dry-run
evidence.

Implemented foundation:

- [x] Add invariant: scorecards record
  `raw_aggregate_pnl_promotion_allowed=false`.
- [x] Add invariant: scorecards record
  `phase3_readiness_required_after_scorecard=true`.
- [x] Add invariant: version fields segment evidence and are included in the
  scorecard evidence unit.
- [x] Add tests that aggregate positive PnL cannot bypass bad regime-sliced
  evidence.
- [x] Add tests that a strategy with positive edge only in `range` receives
  `REGIME_SCOPED_SELECTOR_ELIGIBLE`, not `GLOBAL_SELECTOR_ELIGIBLE`.
- [x] Add tests that a strategy with large `high_volatility` drawdown cannot be
  globally eligible even if total PnL is positive.

Remaining TODO:

- [ ] Wire scorecard decisions into any future paper-readiness submission path.
- [ ] Reject raw aggregate PnL-only promotion requests at the future CLI/API
  boundary.
- [ ] Add tests for almost-always-`no_trade` candidates with opportunity-cost
  analysis.

### 5. Shadow Observation / Parallel Dry-Run Compatibility

Prepare future multi-dry-run observation without allowing it to interfere with
current release gates.

Implemented foundation:

- [x] Define current accepted source `local_shadow_replay`.
- [x] Reject future paper/dry-run source types in current-scope validation.

Remaining TODO:

- [ ] Require future paper/dry-run observations to use the same observation
  ledger schema after a later phase explicitly permits them.
- [ ] Require every observed strategy to maintain separate readiness, runtime
  validation, drift report, and stop reasons.
- [ ] Do not allow one passing strategy to bless a strategy family.
- [ ] Do not allow recent observed performance to override failed research
  gates.
- [ ] Add leaderboard logic that separates long-term evidence, current-regime
  evidence, recent observation evidence, and data-quality confidence.
- [ ] Add cooldown and hysteresis rules to prevent frequent selector churn.

### 6. Regime-Scoped Strategy Contract

Extend candidate metadata with a local eligibility contract.

Required fields:

```text
strategy_version
signal_version
risk_policy_version
regime_classifier_version
cost_model_id
intended_regimes
excluded_regimes
activation_conditions
no_trade_conditions
regime_shift_stop_conditions
required_features
minimum_evidence
maximum_drawdown_by_regime
cost_sensitivity_limits
cooldown_after_regime_change
allowed_pairs
allowed_timeframes
```

Implemented foundation:

- [x] Add `RegimeStrategyContract` and validation in
  `freqtrade_ext/bot_factory/regime_promotion.py`.
- [x] Validate intended/excluded regimes against the predeclared taxonomy.
- [x] Validate that excluded regimes require explicit `no_trade_conditions`.
- [x] Validate that regime-shift stop conditions, allowed pairs, and allowed
  timeframes are explicit.
- [x] Add local-only logic specs for strong uptrend, long-only downtrend
  defensive rebound, and range mean reversion candidates.
- [x] Add assumed-runtime selector simulation proving that multiple candidates
  choose the candidate matching the current regime, and that same-regime
  candidates are ranked by stress-cost robustness before normal-cost PnL.

Remaining TODO:

- [ ] Extend proposal/generated metadata with this contract.
- [ ] Validate that missing required features force `no_trade` or
  `INSUFFICIENT_EVIDENCE`.
- [ ] Validate that contracts cannot include unsupported pair/timeframe/cost
  contexts.

## Example Expected Decisions

### Case A: Trend Strategy Wins Only In Uptrend

Expected decision:

```text
REGIME_SCOPED_SELECTOR_ELIGIBLE for trend_up
no activation outside trend_up
```

### Case B: Mean-Reversion Strategy Wins Only In Range

Expected decision:

```text
REGIME_SCOPED_SELECTOR_ELIGIBLE for range
blocked in trend_up and high_volatility
```

### Case C: Strategy Has Positive Aggregate PnL But High-Volatility Crash Risk

Expected decision:

```text
SHADOW_ONLY or REJECT
never GLOBAL_SELECTOR_ELIGIBLE
```

### Case D: No-Trade Avoids Losses During Volatility

Expected decision:

```text
NO_TRADE_POLICY only for high_volatility or unknown/liquidity_stress
opportunity cost recorded
```

### Case E: Dry-Run Period Is Regime-Biased

Expected decision:

```text
INSUFFICIENT_EVIDENCE for global eligibility
possible trend_up scoped observation only
```

## Implementation Phases

### Phase 0: Documentation and Schema

- [x] Add this TODO file under
  `docs/BOT_FACTORY_REGIME_AWARE_PROMOTION_TODO.md`.
- [x] Link it from `docs/BOT_FACTORY_MVP_TODO.md`.
- [x] Add schema drafts for observation ledger, scorecard, and strategy
  contract.
- [ ] Add expected decision examples as external fixtures.

### Phase 1: Local-Only Regime Fitness

- [ ] Implement deterministic regime labeling from local candle artifacts.
- [ ] Implement local observation ledger output from existing backtest /
  walk-forward artifacts.
- [x] Implement regime fitness scorecard aggregation foundation.
- [x] Add focused tests for representative expected decisions.
- [x] Confirm no paper/dry-run/live process is started.

### Phase 2: Promotion Gate Integration

- [ ] Update future readiness inputs to accept scorecard decisions.
- [ ] Reject raw aggregate PnL-only promotion requests.
- [ ] Add `REGIME_SCOPED_SELECTOR_ELIGIBLE` and `NO_TRADE_POLICY` outcomes to
  future promotion/readiness review artifacts.
- [ ] Add reason-code-first Markdown reports to future CLIs.

### Phase 3: Local Shadow Replay

- [ ] Replay multiple candidates through the same observation ledger schema.
- [ ] Build candidate leaderboard separated by regime.
- [ ] Compare selector decisions against always-on, best-single,
  equal-rotation, and no-trade baselines.
- [ ] Add cooldown/hysteresis simulations.
- [ ] Keep all outputs local artifacts.

### Phase 4: Future Paper / Dry-Run Observation Design Only

- [ ] Draft paper/dry-run observation requirements.
- [ ] Draft monitoring, stop, cleanup, and drift-report requirements.
- [ ] Require explicit human approval before any actual paper/dry-run command
  exists.
- [ ] Keep this phase blocked until repository phase documentation permits the
  exact command.

## Acceptance Criteria

This TODO is complete only when:

- [ ] A candidate cannot be globally eligible from aggregate PnL alone.
- [ ] Eligibility decisions are made from regime-stratified scorecards.
- [ ] `REGIME_SCOPED_SELECTOR_ELIGIBLE` is supported and tested.
- [ ] `NO_TRADE_POLICY` is supported and evaluated with opportunity cost.
- [ ] Dry-run/shadow evidence is recorded as observations, not direct
  promotion authority.
- [ ] Version changes segment evidence.
- [ ] Under-sampled regimes produce `INSUFFICIENT_EVIDENCE`.
- [ ] Positive PnL with unsafe volatility behavior cannot globally pass.
- [ ] All implementation remains local-only until a later explicitly approved
  paper/dry-run phase.
- [ ] Markdown reports explain decisions with reason codes that a reviewer can
  audit.
