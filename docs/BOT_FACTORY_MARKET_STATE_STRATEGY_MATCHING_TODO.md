# Bot Factory Market State / Strategy Matching TODO

Status: P0 local market-state artifact, state-conditioned scorecard,
strategy suitability matrix, offline selector/no-trade matching, and strict
paper-readiness validation foundation implemented.

Created: 2026-05-27 JST.

Updated: 2026-05-28 JST.

Scope: local-only market-state diagnosis, state-conditioned strategy evaluation, and safe strategy/no-trade matching.

This TODO extends, but does not replace:

- `docs/BOT_FACTORY_REGIME_AWARE_PROMOTION_TODO.md`
- `docs/BOT_FACTORY_ARCHITECTURE_RISK_TODO.md`
- `docs/BOT_FACTORY_GATE_GLOSSARY.md`
- `docs/BOT_FACTORY_PHASE3_PAPER_DESIGN.md`
- `freqtrade_ext/bot_factory/market_regime.py`
- `freqtrade_ext/bot_factory/regime_promotion.py`
- `freqtrade_ext/bot_factory/evidence_pipeline.py`
- `freqtrade_ext/bot_factory/candidate_identity.py`
- `freqtrade_ext/bot_factory/candidate_ranking.py`
- `freqtrade_ext/bot_factory/feature_quality.py`

## Goal

Move Bot Factory from simple regime labels such as `trend_up` / `range` toward an auditable system that can answer:

```text
1. What market state are we in, as of the latest trusted local data?
2. Which strategies have evidence for this state and horizon profile?
3. Which strategies are unsafe, under-sampled, stale, or out-of-scope here?
4. Is `no_trade` the correct output because the state is uncertain, out-of-distribution, or unsupported?
```

The target architecture must separate:

```text
market state estimation
  != strategy edge discovery
  != state-conditioned strategy evaluation
  != selector/no_trade matching
  != paper readiness
  != paper/live execution
```

## Non-Goals / Safety Boundaries

- Do not start `freqtrade trade`.
- Do not start paper trading, dry-run trading, canary live, live trading, or any bot process.
- Do not use API keys, secrets, private environment values, exchange order endpoints, account state, or order placement.
- Do not introduce leverage above `1.0`.
- Do not introduce shorting.
- Do not treat current market-state reports as paper/live approval.
- Do not let ML output directly select, start, stop, or switch a bot.
- Do not let recent observation strength override historical, walk-forward, cost, readiness, runtime validation, or drift gates.
- Do not define market-state clusters after looking at strategy profitability for the same holdout period.
- Do not use hindsight-best strategy labels as a selector training target.

## Existing Foundation To Preserve

Current Bot Factory already has important safety and lineage foundations:

- `StrategyCandidateIdentity` segments evidence by strategy, signal, risk policy, regime classifier, and cost model versions.
- `REGIME_SCOPED_SELECTOR_ELIGIBLE` means local selector simulation eligibility only, not paper/live promotion.
- Regime scorecards record that raw aggregate PnL cannot directly authorize promotion.
- Shadow observation currently accepts only local sources such as `backtest`, `walk_forward`, and `local_shadow_replay`.
- Phase 3 paper readiness is a no-startup local evidence check and cannot start paper/dry-run/live processes.
- Existing deterministic regime labels are useful as a first baseline, but they are not enough to represent all market states.

This TODO should build on those invariants instead of bypassing them.

## Core Design

Introduce a new intermediate layer:

```text
local OHLCV / safe local public artifacts
  -> multi-horizon market state artifact
  -> state-conditioned strategy scorecards
  -> strategy suitability matrix
  -> selector/no_trade matching report
  -> future Phase 3 readiness input only
```

The matching unit must be:

```text
strategy_version
+ signal_version
+ risk_policy_version
+ cost_model_id
+ state_encoder_version
+ horizon_profile_id
+ activation_state_scope
```

Not:

```text
strategy_version only
```

## Terminology

### Market State

A numeric and categorical description of the market as of a timestamp and horizon.

Examples:

- `trend_up`
- `trend_down`
- `range`
- `high_volatility`
- `low_volatility`
- `liquidity_stress`
- `post_spike_reversion`
- `mixed`
- `transition`
- `out_of_distribution`
- `unknown`

### Horizon Profile

A multi-timeframe view of state.

Example:

```text
5m: noisy_breakout
1h: trend_up
4h: high_volatility_trend_up
1d: late_trend_or_overextended
```

### Strategy Suitability

Evidence that a strategy is useful, unsafe, or unsupported in a specific market state and horizon profile.

### Matching Policy

A selector layer that chooses one of:

- `select_strategy`
- `watch_only`
- `shadow_only`
- `quarantine`
- `retire`
- `no_trade`

### Abstention

An explicit decision not to trade when evidence is weak, state confidence is low, state is out-of-distribution, or no strategy has sufficient state-conditioned evidence.

## P0: Multi-Horizon Market State Artifact

### Goal

Replace single-label regime decisions with a multi-horizon state artifact that records labels, numeric features, confidence, uncertainty, and data quality.

### Future File

```text
docs/BOT_FACTORY_MARKET_STATE_STRATEGY_MATCHING_TODO.md
docs/bot_factory/market_state_schema.md
data/market_state/<run_id>/market_state_snapshot.json
data/market_state/<run_id>/market_state_windows.jsonl
data/market_state/<run_id>/market_state_report.md
```

### Required Schema Fields

```text
factory
schema_version
run_id
generated_at
data_asof
latest_local_candle_at
git_commit
source_data_paths
source_data_hashes
pair
pair_group
base_timeframe
horizons
state_encoder_version
regime_classifier_version
feature_version
cost_model_id
data_quality_summary
feature_quality_summary
state_confidence
uncertainty
unknown_reason
out_of_distribution_score
safety_scope
```

Each horizon row should include:

```text
horizon
lookback_window
label
confidence
uncertainty
state_vector
feature_cutoff_timestamp
label_cutoff_timestamp
data_quality_flags
reason_codes
```

### Initial Horizons

Use only horizons for which local data exists.

```text
5m
15m
1h
4h
1d
1w
```

Suggested grouping:

```text
micro: 5m / 15m
intraday: 1h / 4h
swing: 1d / 1w
```

### Required Features

Local-only features:

- rolling return
- realized volatility
- volatility z-score
- trend slope
- moving-average distance
- range efficiency
- drawdown from local high
- high-low range percentage
- candle gap proxy
- volume/liquidity proxy from local candles
- turnover/cost pressure from local artifacts
- missing candle flags
- data freshness flags

Optional only when already present as local public artifacts:

- mark-price context
- funding-rate context
- open-interest context
- liquidation context
- order-book/depth context

Explicitly disallowed in current scope:

- live order book
- live spread endpoint
- private account data
- exchange order/fill endpoint
- exchange wallet/balance state
- API keys, secrets, or private env values

### TODO

- [x] Define `market_state_snapshot_v1` schema in
  `docs/bot_factory/market_state_schema.md`.
- [x] Define `market_state_window_v1` JSONL schema in
  `docs/bot_factory/market_state_schema.md`.
- [x] Extend the deterministic classifier output into multi-horizon artifacts.
- [x] Add `state_vector` fields independent from human-readable labels.
- [x] Add `confidence`, `uncertainty`, `unknown_reason`, and `out_of_distribution_score`.
- [x] Add `data_asof`, `latest_local_candle_at`, and `latest_local_candle_close_at`; never call a stale local snapshot "current" without the as-of timestamp.
- [x] Add anti-leakage fields:
  - `feature_cutoff_timestamp`
  - `label_cutoff_timestamp`
  - `decision_window_start`
  - `decision_window_end`
  - `future_data_used=false`
- [x] Add tests where low-confidence or conflicting horizons emit `mixed`, `transition`, `out_of_distribution`, or `unknown`.
- [x] Add tests where stale latest local candles force `unknown` or `stale_data_no_trade`.

Implemented on 2026-05-28 JST:

- Added multi-horizon snapshot construction in
  `freqtrade_ext/bot_factory/market_regime.py`, reusing the existing
  deterministic local OHLCV classifier and writing `market_state_snapshot_v1`
  plus `market_state_window_v1` rows.
- Added local-only state vectors, as-of timestamps, anti-leakage cutoffs,
  confidence/uncertainty, OOD score proxy, stale-data unknown fallback, horizon
  conflict detection, `horizon_profile_id`, and no-trade defaults for uncertain
  local state.
- Review follow-up on 2026-05-28 JST: staleness now ages from candle close
  time, `data_asof` points to latest local candle close, and resampled
  higher-timeframe rows are dropped until their candle close time is reached.
- Added `scripts/bot_factory_build_market_state.py` to write
  `market_state_snapshot.json`, `market_state_windows.jsonl`,
  `market_state_report.md`, `current_market_state.json`, and
  `current_market_state_report.md` from local OHLCV only.
- Added focused tests for multi-horizon artifact writing, conflicting horizons
  defaulting to `mixed` plus `no_trade_default=true`, and stale local candles
  forcing `unknown` plus no-trade.
- Remaining P0 limitations: OOD scoring is a deterministic confidence proxy,
  not a learned analog-distance model; optional market-structure contexts are
  not yet joined; historical as-of selector replay remains a future increment.

## P0: Current Market State Report

### Goal

Add a local-only report for "what the latest trusted local data says now."

This report must be explicit that "current" means:

```text
current as of latest local artifact timestamp
```

Not:

```text
live exchange state
```

### Future Files

```text
data/market_state/current/<run_id>/current_market_state.json
data/market_state/current/<run_id>/current_market_state_report.md
```

### TODO

- [x] Build a current-state reporter that consumes local candle artifacts only.
- [x] Report `data_asof` and stale-data status at the top of the Markdown.
- [x] Include each horizon's label, confidence, uncertainty, and top reason codes.
- [x] Include a compact "horizon conflict" section.
- [ ] Include `no_trade_default=true` when:
  - data is stale;
  - state is unknown;
  - horizon signals conflict;
  - state is out-of-distribution;
  - feature quality fails;
  - no strategy has state-conditioned evidence.
- [x] Include a "not allowed" section confirming no paper/dry-run/live process was started.
- [ ] Add tests for stale data, conflicting horizons, and out-of-distribution state.

Implemented on 2026-05-28 JST:

- Added `current_market_state_v1` construction and Markdown rendering in
  `freqtrade_ext/bot_factory/market_regime.py`.
- The current-state report explicitly states that "current" means current as of
  the latest local data timestamp, includes horizon labels/confidence/reasons,
  and records that no paper, dry-run, live, bot startup, order placement, or
  process-control path was started.
- Added stale-data and conflicting-horizon tests. Dedicated
  out-of-distribution analog-distance tests remain open until the OOD model is
  more than a deterministic confidence proxy.
- Review follow-up on 2026-05-29 JST: `current_market_state_v1` now preserves
  the source snapshot `cost_model_id`, so `current_market_state.json` can be
  passed directly into selector matching without losing the market-identity
  boundary required by strategy suitability rows.

## P0: State-Conditioned Strategy Evaluation

### Goal

Evaluate strategies by market state and horizon profile, not only by global backtest metrics.

Existing scorecards already aggregate by `strategy_version x market_regime`. Extend this to:

```text
strategy_version
+ signal_version
+ risk_policy_version
+ state_id
+ horizon_profile_id
+ pair
+ timeframe
+ cost_model_id
```

### Future Files

```text
docs/bot_factory/state_conditioned_scorecard_schema.md
data/state_scorecards/<candidate_id>/<run_id>/state_conditioned_scorecard.json
data/state_scorecards/<candidate_id>/<run_id>/state_conditioned_scorecard_report.md
data/state_scorecards/<candidate_id>/<run_id>/strategy_state_suitability_matrix.json
```

### Required Metrics

Per strategy / state / horizon profile:

```text
sample_days
independent_window_count
non_overlapping_window_count
trade_count
exposure_ratio
average_holding_time
gross_return
net_return_normal_cost
net_return_stress_cost
expectancy
profit_factor
win_rate
max_drawdown
downside_deviation
turnover
cost_burden
no_trade_delta
no_trade_opportunity_cost
hold_delta
incumbent_delta
lower_confidence_bound
pair_concentration
calendar_concentration
state_sample_count
state_cluster_stability
data_quality_pass
decision
reason_codes
```

### Decisions

```text
STATE_SELECTOR_ELIGIBLE
STATE_SHADOW_ONLY
STATE_INSUFFICIENT_EVIDENCE
STATE_UNSAFE
STATE_NO_TRADE_POLICY
STATE_DIAGNOSTIC_ONLY
```

`STATE_SELECTOR_ELIGIBLE` must not mean paper readiness or live approval.

### TODO

- [x] Extend observation ledger rows with `state_id`, `horizon_profile_id`, and `state_encoder_version`.
- [x] Build state-conditioned scorecards from checked backtest and walk-forward artifacts.
- [x] Require actual checked strategy evidence for selector eligibility.
- [x] Keep proxy replays and relaxed-threshold demos as `STATE_DIAGNOSTIC_ONLY`.
- [ ] Require baseline deltas against:
  - `no_trade`
  - hold baseline
  - incumbent strategy when present
  - style-specific baseline
- [x] Split baseline deltas by `baseline_id`; never sum hold and no-trade baselines into one aggregate.
- [ ] Add hard vetoes:
  - insufficient windows
  - insufficient trades
  - negative stress-cost edge
  - lower confidence bound not positive
  - drawdown beyond state contract
  - pair concentration too high
  - calendar concentration too high
  - data quality failure
  - state coverage too narrow
- [ ] Add tests for:
  - trend strategy eligible only in uptrend horizon profile;
  - range strategy eligible only in range horizon profile;
  - strategy with positive global PnL but unsafe high-volatility state returns `STATE_SHADOW_ONLY` or `STATE_UNSAFE`;
  - strategy that underperforms hold in a bull trend is not selector-eligible without risk-reduction rationale;
  - almost-always-`no_trade` policy records opportunity cost.

Implemented on 2026-05-28 JST:

- Added `freqtrade_ext/bot_factory/state_conditioning.py` with
  `state_conditioned_scorecard_v1` construction from an existing deterministic
  `regime_fitness_scorecard_v1` plus `market_state_snapshot_v1`.
- Added `scripts/bot_factory_build_state_scorecard.py` to write
  `data/state_scorecards/<candidate_id>/<run_id>/state_conditioned_scorecard.json`
  and `state_conditioned_scorecard_report.md` from local JSON artifacts only.
- The builder preserves canonical candidate identity, maps regime scorecard rows
  onto `state_id` / `horizon_profile_id`, splits no-trade and hold baseline
  deltas into separate `baseline_id` rows, and records explicit
  `selector_candidate_creation_allowed` / `paper_readiness_input_allowed`
  flags.
- Strict selector eligibility now requires a deterministic source regime
  scorecard, checked candidate identity, no proxy evidence, no relaxed
  thresholds, selector-eligible state rows, and walk-forward evidence when the
  caller requires it. Missing walk-forward evidence produces
  `diagnostic_only=true`.
- Review follow-up on 2026-05-28 JST: source regime scorecard top-level
  `decision` must pass the historical selector gate before
  `selector_candidate_creation_allowed` or `paper_readiness_input_allowed` can
  be true.
- Review follow-up on 2026-05-29 JST: `--allow-missing-walk-forward` only
  allows artifact creation. Missing walk-forward evidence still records
  `walk_forward_gate_passed=false`, keeps the artifact diagnostic-only, and
  cannot set selector or paper-readiness input flags.
- Review follow-up on 2026-05-29 JST: derived `1w` horizons now use
  Monday-anchored weekly resampling to match Freqtrade candle boundaries.
- Review follow-up on 2026-05-29 JST: state-conditioned selector eligibility
  now requires complete source observation state scope and `future_data_used=false`;
  snapshot label matching is retained only as diagnostic/report context.
- Review follow-up on 2026-05-29 JST: Phase 3 paper readiness now joins
  market-state scorecard candidate identity to the readiness strategy,
  historical metrics, walk-forward metrics, embedded strategy source identity
  when present, and supplied suitability-matrix selector rows.
- Review follow-up on 2026-05-29 JST: suitability-matrix strategy rows now
  retain the full candidate identity unit needed for paper-readiness identity
  joins.
- Added native optional observation-ledger state fields. When any state field is
  present, validation now requires complete `state_id`, `horizon_profile_id`,
  `state_encoder_version`, `state_window_id`, feature/label cutoff timestamps,
  decision-window start/end, plus `future_data_used=false`.
- Added state-scorecard hard veto coverage for single-window selector rows,
  missing trades, negative stress-cost edge, non-positive lower confidence
  bound, pair/calendar concentration, and data quality failures.
- Added tests covering observation state-field validation, trend-only and
  range-only selector scope through the suitability/matching layer, and
  no-trade opportunity-cost accounting.
- Remaining limitations: incumbent/style baselines are not yet implemented;
  drawdown contract and underperform-hold bull-trend rationale still need
  deeper strategy-family-specific checks.
- Follow-up from PR #11 review on 2026-05-29 JST: multi-window source
  observations should be grouped by state scope instead of requiring all
  rows in a regime to share the same `state_window_id` and cutoff timestamps.
  Preserve `state_window_ids[]`, `decision_windows[]`,
  `feature_cutoff_range`, and `label_cutoff_range`, while still requiring
  `future_data_used=false` for every source row.

## P0: Diagnostic vs Selector-Eligible Boundary

### Goal

Prevent research demos, proxy replays, or relaxed-threshold artifacts from becoming selector inputs.

### Required Fields

```text
evidence_eligibility
proxy_evidence
diagnostic_only
relaxed_thresholds_used
actual_strategy_backtest_required
historical_gate_passed
walk_forward_gate_passed
selector_candidate_creation_allowed
paper_readiness_input_allowed
```

### TODO

- [x] Add `evidence_eligibility = diagnostic_only | selector_eligible_candidate`.
- [x] Require `selector_candidate_creation_allowed=false` for:
  - proxy close-to-close replays;
  - manually assembled scorecards;
  - relaxed calendar concentration;
  - single-window demos;
  - scorecards without checked strategy identity;
  - scorecards without walk-forward evidence.
- [x] Require `paper_readiness_input_allowed=false` unless the full strict scorecard schema passes.
- [x] Update `selection_candidate_from_scorecard()` to reject diagnostic-only artifacts.
- [x] Update paper readiness scorecard validation to reject minimal JSONs that only contain top-level flags.
- [x] Add tests that diagnostic replays cannot become selector candidates.

Implemented on 2026-05-28 JST:

- Added state-conditioned scorecard eligibility flags:
  `evidence_eligibility`, `diagnostic_only`, `proxy_evidence`,
  `relaxed_thresholds_used`, `actual_strategy_backtest_required`,
  `historical_gate_passed`, `walk_forward_gate_passed`,
  `selector_candidate_creation_allowed`, and `paper_readiness_input_allowed`.
- Added selector validation for state-conditioned scorecards and hardened
  `selection_candidate_from_scorecard()` so diagnostic-only, proxy,
  relaxed-threshold, or selector-disallowed scorecards cannot become selector
  candidates.
- Added focused tests proving strict state-conditioned evidence can validate for
  selector input, missing walk-forward evidence remains diagnostic-only, and a
  diagnostic scorecard cannot become a selector candidate.
- Added full state-conditioned scorecard validation to Phase 3 paper readiness
  optional inputs. Minimal JSONs with only top-level flags are rejected because
  selector rows, identity, baselines, and no-startup safety scope must pass.
- Remaining limitations: old regime scorecard validation is preserved for
  backward compatibility; future paper readiness still needs full generated
  strategy identity joins across state scorecard, suitability matrix, config,
  backtest, and walk-forward artifacts.

## P1: State Discovery / Clustering Research

### Goal

Handle market states that are not cleanly described by rule labels such as `trend_up` or `range`.

Start with diagnostic-only unsupervised or semi-supervised state discovery. Do not use clusters directly for live or paper selection.

### Future Files

```text
data/state_discovery/<run_id>/state_clusters.json
data/state_discovery/<run_id>/state_cluster_report.md
data/state_discovery/<run_id>/representative_windows.jsonl
```

### TODO

- [ ] Build offline state discovery using only local historical windows.
- [ ] Cluster windows using predeclared features:
  - returns
  - realized volatility
  - range efficiency
  - drawdown
  - volume/liquidity proxy
  - cost pressure
  - data quality flags
- [ ] Persist:
  - `cluster_id`
  - cluster centroid / summary
  - representative windows
  - nearest historical analogs
  - cluster stability
  - out-of-distribution threshold
  - feature version
  - clustering version
- [ ] Compare discovered clusters against deterministic labels.
- [ ] Mark clusters as `diagnostic_only` until reviewed.
- [ ] Add temporal stability checks.
- [ ] Add tests that tiny feature perturbations do not completely reshuffle clusters.
- [ ] Add tests that clusters with too few analog windows produce `INSUFFICIENT_EVIDENCE`.

## P1: Strategy Suitability Matrix

### Goal

Create an auditable matrix that answers:

```text
Which strategy can be used in which state and horizon profile?
Which strategy should be blocked?
Which states have no supported strategy?
```

### Future Files

```text
data/strategy_suitability/<run_id>/strategy_state_suitability_matrix.json
data/strategy_suitability/<run_id>/strategy_state_suitability_report.md
```

### Matrix Shape

```text
rows:
  strategy_identity_unit
  state_id
  horizon_profile_id
  pair_group
  pair
  timeframe
  cost_model_id

columns:
  decision
  evidence_quality
  expected_utility_after_cost
  risk_adjusted_score
  uncertainty
  no_trade_delta
  hold_delta
  incumbent_delta
  blockers
  reason_codes
```

### TODO

- [x] Generate matrix from state-conditioned scorecards only.
- [x] Include one row for `no_trade` as a first-class policy.
- [x] Mark missing states as `NO_SUPPORTED_STRATEGY`.
- [x] Mark states with weak data as `UNKNOWN_NO_TRADE`.
- [x] Mark states outside historical analog coverage as `OUT_OF_DISTRIBUTION_NO_TRADE`.
- [x] Add matrix diff report between candidate versions.
- [x] Add tests proving a strategy cannot inherit another strategy's state evidence.

Implemented on 2026-05-28 JST:

- Added `freqtrade_ext/bot_factory/strategy_suitability.py` with
  `strategy_suitability_matrix_v1` construction from one or more strict
  `state_conditioned_scorecard_v1` artifacts.
- Added `scripts/bot_factory_build_strategy_suitability.py` to write
  `data/strategy_suitability/<run_id>/strategy_state_suitability_matrix.json`
  and `strategy_suitability_report.md` from local JSON artifacts only.
- Matrix rows preserve the state-scoped strategy identity unit, state ID,
  horizon profile, cost model, stress-cost utility, uncertainty, baseline
  deltas, blockers, and reason codes.
- `no_trade` is represented as a first-class row for every known state scope.
  Missing states produce explicit no-trade rows, weak/diagnostic rows cannot
  become selector inputs, and identity mismatches produce
  `IDENTITY_MISMATCH` rather than inheriting another strategy's evidence.
- Added matrix validation and matrix-diff payload support, plus regression
  tests proving a range strategy cannot inherit trend-state evidence and a
  tampered candidate identity is not selector-eligible.
- Review follow-up on 2026-05-29 JST: selector validation now requires the
  matrix top-level `safety_scope` to prove local-artifacts-only,
  historical-evaluation-only use and no trading, order placement, process
  control, secrets, leverage above one, shorting, or promotion authority.

## P1: Selector / Matching Policy

### Goal

Use the current multi-horizon market state and the strategy suitability matrix to choose a safe local decision.

### Selector Inputs

```text
current_market_state
strategy_state_suitability_matrix
candidate_identity
feature_quality_report
cost_model
selector_state
cooldown/hysteresis config
no_trade_policy
```

### Selector Outputs

```text
select_strategy
no_trade
shadow_only
watch
quarantine
retire
```

### Required Decision Metadata

```text
decision_id
generated_at
data_asof
selected_action
selected_strategy_id
selected_candidate_id
selected_state_id
selected_horizon_profile_id
no_trade_reason
selector_version
state_encoder_version
evidence_unit
confidence
uncertainty
reason_codes
rejected_alternatives
safety_scope
```

### TODO

- [x] Implement offline selector matching over local artifacts only.
- [x] Default to `no_trade` when:
  - state confidence is below threshold;
  - state is out-of-distribution;
  - state evidence is under-sampled;
  - required features are missing or stale;
  - cost model is stale;
  - all candidate state scorecards are weak;
  - strategy identity mismatches;
  - cooldown/hysteresis blocks switching.
- [ ] Use predeclared scoring weights per state/horizon group.
- [x] Include `no_trade` and incumbent in every comparison.
- [x] Log rejected alternatives and reason codes.
- [x] Add tests for:
  - clear uptrend selects only a trend-eligible strategy;
  - clear range selects only a range-eligible strategy;
  - multi-horizon conflict returns `no_trade`;
  - out-of-distribution state returns `no_trade`;
  - stale data returns `no_trade`;
  - two same-state candidates are ranked by stress-cost utility, not raw PnL;
  - cooldown prevents churn;
  - hysteresis keeps previous strategy unless improvement margin is material.

Implemented on 2026-05-28 JST:

- Added `freqtrade_ext/bot_factory/selector_matching.py` with
  `selector_matching_decision_v1` and local-only matching over
  `current_market_state_v1` or `market_state_snapshot_v1` plus
  `strategy_suitability_matrix_v1`.
- Added `scripts/bot_factory_match_strategy_to_market_state.py` to emit
  `selector_matching_decision.json`, `selector_matching_report.md`, and
  `no_trade_scorecard.json`.
- Matching defaults to `no_trade` for stale local data, low confidence, OOD,
  mixed/transition/unknown states, horizon conflict, feature-quality failure,
  stale cost-model flags, matrix validation failure, no eligible state row,
  identity mismatch, and cooldown-blocked switching.
- Selector comparisons include no-trade policy rows and incumbent rows; selected
  strategies are ranked by stress-cost utility / post-cost utility / lower
  confidence bound rather than raw PnL.
- Added tests for clear trend-up, clear range, multi-horizon conflict, OOD,
  stale data, stress-utility ranking, cooldown, and hysteresis behavior.
- Review follow-up on 2026-05-29 JST: selector row matching now requires the
  strategy suitability row to match the current market identity (`pair`,
  `base_timeframe`, and `cost_model_id`) in addition to `state_id` and
  `horizon_profile_id`. Label-derived state IDs cannot cross pair, timeframe,
  or cost-model evidence boundaries.

Remaining limitation: scoring weights are deterministic and explicit in code,
but not yet externalized as a versioned per-state/horizon weight artifact.

## P1: No-Trade Evaluation

### Goal

Treat `no_trade` as an explicit policy with both benefits and costs.

### Required Metrics

```text
avoided_drawdown
avoided_negative_expectancy
opportunity_cost_vs_hold
opportunity_cost_vs_incumbent
opportunity_cost_vs_best_selector_eligible_strategy
uncertainty_reduction_value
state_confidence
reason_codes
```

### TODO

- [x] Build `no_trade_scorecard.json`.
- [x] Evaluate no-trade by state and horizon profile.
- [x] Separate:
  - loss avoidance;
  - opportunity cost;
  - uncertainty / OOD safety value.
- [x] Do not reward no-trade merely for avoiding losses in hindsight.
- [x] Add tests:
  - no-trade is good in high-volatility crash state;
  - no-trade is costly in clear trend-up state;
  - no-trade is acceptable in unknown/OOD state even if opportunity cost exists.

Implemented on 2026-05-28 JST:

- `build_no_trade_scorecard()` emits `no_trade_scorecard_v1` with avoided
  drawdown, opportunity cost versus hold/incumbent/best selector-eligible
  strategy, uncertainty-reduction value, confidence, assessment, and reason
  codes.
- The no-trade scorecard records opportunity cost in supported clear trends and
  explicitly includes `no_hindsight_profit_credit`, so avoiding losses is not
  treated as a standalone reward.
- Added tests for costly clear trend-up no-trade, acceptable unknown/OOD-style
  no-trade, and high-volatility no-trade safety value.

## P1: ML-Assisted State And Suitability Research

### Goal

Add ML only as an offline diagnostic and scoring aid, not as direct execution control.

Recommended first ML use cases:

```text
1. market state embedding
2. state clustering / analog search
3. out-of-distribution detection
4. strategy suitability scoring
5. abstention / no_trade probability
```

Avoid initially:

```text
direct buy/sell signal generation
direct bot switching
direct paper/live execution
reinforcement learning over order placement
```

### TODO

- [ ] Define `state_encoder_model_v1` artifact schema.
- [ ] Define `strategy_suitability_model_v1` artifact schema.
- [ ] Define training dataset rows:
  - state vector
  - horizon profile
  - strategy identity
  - cost model
  - state-conditioned outcome
  - no-trade baseline
  - hold baseline
  - future-data leakage guard
- [ ] Use time-based train/validation/holdout splits.
- [ ] Use pair-held-out or pair-group-held-out tests where data permits.
- [ ] Use purged/embargoed split logic where labels overlap future windows.
- [ ] Optimize risk-adjusted utility after costs, not raw return.
- [ ] Calibrate uncertainty.
- [ ] Require model explanations:
  - nearest historical analog windows;
  - feature contribution summary;
  - reason code mapping;
  - OOD score.
- [ ] Require abstention:
  - high uncertainty -> no_trade;
  - low analog count -> no_trade;
  - state drift -> no_trade or quarantine.
- [ ] Keep ML outputs as `diagnostic_only` until offline replay beats deterministic baselines out-of-sample.
- [ ] Add tests that high ML score cannot bypass missing scorecard, failed walk-forward, stale data, or identity mismatch.

## P1: Backtest And Walk-Forward Evaluation By State

### Goal

Make backtest evaluation answer "where does this strategy work?" rather than "did it work globally?"

### TODO

- [ ] Add state-sliced backtest report sections:
  - by state label
  - by state cluster
  - by horizon profile
  - by pair
  - by timeframe
  - by cost regime
- [ ] Add walk-forward state coverage table.
- [ ] Add per-state pass/fail/insufficient decisions.
- [ ] Add style-aware gates per strategy family:
  - micro / short-horizon
  - intraday trend
  - swing trend
  - range mean reversion
  - defensive no-trade
  - ML-assisted state suitability
- [ ] Add state-balanced holdout evaluation when possible.
- [ ] Reject strategy eligibility when positive evidence is concentrated in one easy or hand-picked state.
- [ ] Add "state missingness" report showing which states have no evidence.
- [ ] Add tests that global positive backtest cannot hide a state-specific crash.

## P2: Current-State To Strategy-Matching Replay

### Goal

Simulate historical "as-of" decision points to prove the matcher behaves without using future data.

### Future Files

```text
data/selector_replays/<run_id>/market_state_decisions.jsonl
data/selector_replays/<run_id>/selector_replay_report.md
```

### TODO

- [ ] For each historical decision timestamp, build a market-state snapshot using only data available before that timestamp.
- [ ] Join only strategies whose state-conditioned evidence was available before that timestamp.
- [ ] Emit selector decision:
  - selected strategy
  - no_trade
  - shadow_only
  - rejected alternatives
  - reason codes
- [ ] Compare replay against baselines:
  - always no_trade
  - always hold
  - best single eligible strategy
  - equal rotation
  - incumbent selector
- [ ] Report:
  - net return after cost
  - drawdown
  - exposure
  - turnover
  - missed opportunity
  - no_trade quality
  - selector churn
- [ ] Add tests that intentionally leaked future state labels are rejected.

## P2: Paper Readiness Integration

### Goal

Feed strict state-conditioned scorecards into Phase 3 paper readiness without granting startup authority.

### TODO

- [x] Add optional `--market-state-scorecard` and `--strategy-suitability-matrix` inputs to future paper readiness.
- [x] Require full schema validation, not only top-level flags.
- [x] Require candidate identity to match generated strategy, backtest, walk-forward, scorecard, and suitability matrix.
- [x] Require `paper_readiness_input_allowed=true` only for strict evidence, not diagnostic evidence.
- [ ] Reject scorecards with:
  - `diagnostic_only=true`
  - `proxy_evidence=true`
  - `relaxed_thresholds_used=true`
  - missing walk-forward evidence
  - missing state coverage
  - missing no-trade baseline
- [x] Preserve existing Phase 3 no-startup semantics.
- [x] Add tests that a market-state scorecard cannot start paper/dry-run/live processes.

Implemented on 2026-05-28 JST:

- Extended `PaperReadinessInputs` and
  `scripts/bot_factory_check_paper_readiness.py` with optional
  `--market-state-scorecard`, `--requires-market-state-scorecard`,
  `--strategy-suitability-matrix`, and
  `--requires-strategy-suitability-matrix` inputs.
- Paper readiness now validates state-conditioned scorecards and suitability
  matrices with full schema checks, rejects diagnostic/minimal top-level-only
  scorecard JSON, and preserves the existing no-startup/no-process safety
  scope.
- Review follow-up on 2026-05-28 JST: a supplied strategy suitability matrix
  must contain a selector-eligible row matching the readiness target strategy
  class or strategy id.
- Review follow-up on 2026-05-29 JST: market-state scorecard identity is
  compared against the readiness strategy, historical metrics, walk-forward
  metrics, embedded strategy source identity when present, and supplied
  strategy suitability matrix selector rows.

Remaining limitation: generated strategy metadata and config-level identity
joins may need a later consolidated readiness identity proof if those
artifacts are supplied separately.

## P2: Future Shadow Observation Compatibility

### Goal

Ensure future paper/dry-run observations are only additional evidence and cannot override state-conditioned gates.

### TODO

- [ ] Future paper/dry-run observation rows must use the same observation ledger schema.
- [ ] Future observation must include state snapshot ID and horizon profile ID.
- [ ] Recent observation evidence must be separated from:
  - historical evidence
  - walk-forward evidence
  - training evidence
  - readiness evidence
  - runtime validation
  - drift evidence
- [ ] Recent observation may influence ranking only when the strategy already has strict state-conditioned evidence.
- [ ] Add drift detection:
  - state distribution drift
  - feature distribution drift
  - cost/turnover drift
  - drawdown envelope breach
  - selector churn increase
- [ ] Add quarantine rules when live-like observations contradict historical state evidence.
- [ ] Keep current phase local-only until later explicit approval.

## P2: Reporting And Review UX

### Goal

Make it possible to review from one report:

```text
Current local market state
-> supported strategies
-> unsupported states
-> selected action
-> no_trade rationale
-> evidence lineage
```

### Future Report Sections

```text
Summary
Data As-Of / Freshness
Multi-Horizon State
Horizon Conflicts
OOD / Unknown Reasons
Supported Strategy Matrix
No-Trade Evaluation
Selected Action
Rejected Alternatives
Evidence Lineage
Baseline Comparisons
Drift / Feature Quality
Safety Boundary Confirmation
Next Required Gate
```

### TODO

- [x] Add `current_market_state_report.md`.
- [x] Add `strategy_suitability_report.md`.
- [x] Add `selector_matching_report.md`.
- [x] Include stable reason codes and remediation hints.
- [x] Include "why not trade?" section.
- [x] Include "what would need to become true before selection?" section.
- [x] Include "this does not permit paper/live" section.

## P3: Research Questions

Track these as explicit research questions rather than hidden tuning.

- [ ] Which horizon combinations are stable enough to define strategy activation scopes?
- [ ] Which states are common enough to support strategy development?
- [ ] Which states have no supported strategy and should default to no_trade?
- [ ] Does deterministic labeling explain enough of strategy performance, or is state embedding needed?
- [ ] Does clustering find reproducible states that are not just calendar artifacts?
- [ ] Can a strategy suitability model beat deterministic selector rules out-of-sample?
- [ ] How much opportunity cost is acceptable for no_trade in uncertain states?
- [ ] How should BTC/ETH/global market state influence pair-specific strategy selection?
- [ ] How should state drift quarantine existing selector-eligible strategies?

## Implementation Order

Recommended safe order:

1. Add this TODO and schema docs only.
2. Add multi-horizon market-state schema and tests.
3. Add local current-state report from latest available local data.
4. Extend observation ledger with `state_id` and `horizon_profile_id`.
5. Add state-conditioned scorecard from checked backtest/walk-forward artifacts.
6. Add diagnostic-vs-selector-eligible boundary flags.
7. Add strategy suitability matrix.
8. Add no-trade scorecard.
9. Add offline selector matching report.
10. Add historical as-of selector replay.
11. Add diagnostic-only state clustering.
12. Add diagnostic-only ML suitability model.
13. Integrate strict state-conditioned evidence into Phase 3 readiness.
14. Only after separate approval, design future paper/dry-run observation input.

## Acceptance Criteria

This design is ready to implement only when:

- [x] Market state artifacts are multi-horizon, timestamped, and local-only.
- [x] Stale local data cannot be reported as live/current without an as-of warning.
- [ ] Strategy evaluation is state-conditioned and includes no-trade / hold / incumbent baselines.
- [x] Diagnostic replay cannot become selector eligibility.
- [ ] ML output cannot bypass strict scorecards or paper readiness.
- [x] Unknown, mixed, transition, OOD, and stale states default to no_trade.
- [x] Time horizons are explicit; 5m trend and multi-month trend are not treated as the same state.
- [ ] State definitions are frozen before evaluating holdout strategy performance.
- [x] Selector decisions include rejected alternatives and reason codes.
- [x] All scorecards preserve candidate identity and evidence version lineage.
- [x] Phase 3 readiness remains no-startup and cannot start paper/dry-run/live.
- [x] Reports clearly state what the artifact permits and what it does not permit.

## Suggested First Safe Increment

### Increment 0: Documentation And Boundary Schema

Scope:

- Add this TODO.
- Add `market_state_snapshot_v1` draft schema.
- Add `state_conditioned_scorecard_v1` draft schema.
- Add `strategy_suitability_matrix_v1` draft schema.
- Add diagnostic-vs-selector-eligible boundary fields.
- Add example JSON snippets only.

Non-scope:

- no strategy generation
- no backtest execution
- no Freqtrade trade
- no paper/dry-run/live/canary
- no exchange API
- no secrets
- no leverage > 1.0
- no shorting
- no ML training execution

Expected output:

```text
docs/BOT_FACTORY_MARKET_STATE_STRATEGY_MATCHING_TODO.md
docs/bot_factory/market_state_schema.md
docs/bot_factory/state_conditioned_scorecard_schema.md
docs/bot_factory/strategy_suitability_matrix_schema.md
```

Implemented on 2026-05-28 JST:

- Added `docs/bot_factory/market_state_schema.md` with draft
  `market_state_snapshot_v1`, `market_state_window_v1`, current-state report
  boundary fields, anti-leakage timestamps, state vectors, stale/unknown/OOD
  no-trade defaults, safety scope, and example JSON snippets.
- Added `docs/bot_factory/state_conditioned_scorecard_schema.md` with draft
  `state_conditioned_scorecard_v1`, observation-ledger extension fields,
  strict evidence-unit fields, baseline separation by `baseline_id`,
  diagnostic-vs-selector eligibility flags, hard veto reason codes, safety
  scope, and example JSON snippets.
- Added `docs/bot_factory/strategy_suitability_matrix_schema.md` with draft
  `strategy_suitability_matrix_v1`, first-class `no_trade` rows, missing-state
  rows, selector/non-authority boundaries, matrix diff fields, safety scope, and
  example JSON snippets.
- This increment is documentation and schema boundary work only. It does not
  implement market-state artifact generation, state-conditioned scorecard
  construction, strategy matching, paper readiness integration, paper trading,
  dry-run trading, live trading, exchange order placement, process control,
  shorting, or leverage above `1.0`.

## Open Questions

- Which horizons are available reliably in local artifacts: 5m, 15m, 1h, 4h, 1d, 1w?
- Should BTC/ETH global state be a separate market-state context used by all pair-level selectors?
- Should `state_id` be deterministic-label based, cluster based, or both?
- What is the minimum analog-window count for a state to become selector-eligible?
- Should `no_trade` have separate thresholds for opportunity cost in trend states versus unknown states?
- Which strategy families should exist first: trend, range, defensive rebound, volatility breakout, or ML-assisted?
- How strict should paper readiness be for regime-scoped strategies that intentionally trade only rare states?
- What is the first incumbent baseline once there is no production incumbent yet?
