# Bot Factory State-Conditioned Scorecard Schema

Status: draft schema for Increment 0.

Implementation status: state-conditioned scorecard construction and selector
validation are implemented for local checked artifacts. Multi-window state
scope aggregation is implemented as of 2026-05-30 JST. This file does not run
backtests, generate strategies, start paper trading, start dry-run trading,
start live trading, or authorize selector use by itself.

## Purpose

State-conditioned scorecards answer where a checked strategy has evidence and
where it is unsupported, unsafe, diagnostic-only, or explicitly a no-trade
policy. They extend the existing regime scorecard concept from a single
`market_regime` dimension to a market-state and horizon-profile evidence unit.

## Artifacts

Future producers should write:

```text
data/state_scorecards/<candidate_id>/<run_id>/state_conditioned_scorecard.json
data/state_scorecards/<candidate_id>/<run_id>/state_conditioned_scorecard_report.md
data/state_scorecards/<candidate_id>/<run_id>/strategy_state_suitability_matrix.json
```

## Evidence Unit

The scorecard evidence unit is:

```text
strategy_version
+ signal_version
+ risk_policy_version
+ state_id
+ horizon_profile_id
+ pair
+ timeframe
+ cost_model_id
+ state_encoder_version
```

`strategy_version` alone is never sufficient. The scorecard must also embed or
reference the canonical `StrategyCandidateIdentity` so evidence cannot be reused
for a different strategy, signal, risk policy, classifier, or cost model.

## Observation Ledger Extension

Future observation rows that feed this scorecard should add these fields to the
existing local observation ledger contract:

| Field | Type | Notes |
| --- | --- | --- |
| `state_id` | string | Market-state id assigned before evaluating strategy results. |
| `horizon_profile_id` | string | Multi-horizon profile id. |
| `state_encoder_version` | string | Market-state encoder version. |
| `market_state_snapshot_id` | string | Source snapshot run id or artifact id. |
| `market_state_window_id` | string or null | Source window row id when applicable. |
| `feature_cutoff_timestamp` | timestamp | Must not exceed the decision window end. |
| `label_cutoff_timestamp` | timestamp | Must not exceed the decision window end. |
| `future_data_used` | boolean | Must be `false`. |

When multiple source observations share the same state scope, producers may
aggregate them into one state-conditioned evidence row. Aggregation groups by
strategy identity unit, `state_id`, `horizon_profile_id`,
`state_encoder_version`, `cost_model_id`, `pair_group`, and `timeframe`, while
preserving `state_window_ids[]`, `decision_windows[]`,
`feature_cutoff_range`, `label_cutoff_range`, and
`source_observation_count`.

Observation sources remain limited to checked local evidence paths permitted by
the active phase documentation. Future paper or dry-run observations remain
invalid until later phase docs explicitly permit that exact source type.

## `state_conditioned_scorecard_v1`

Required top-level fields:

| Field | Type | Notes |
| --- | --- | --- |
| `factory` | string | Must be `bot_factory`. |
| `schema_version` | string | Must be `state_conditioned_scorecard_v1`. |
| `run_id` | string | Stable local scorecard run id. |
| `generated_at` | timestamp | Artifact creation time. |
| `candidate_id` | string | Candidate id under evaluation. |
| `candidate_identity` | object | Canonical strategy identity. |
| `candidate_identity_schema_version` | string | Expected `strategy_candidate_identity_v1`. |
| `source_artifacts` | object | Checked backtest, walk-forward, trade, and market-state paths. |
| `source_artifact_hashes` | object | Local content hashes. |
| `state_encoder_version` | string | Version used by the market-state artifact. |
| `horizon_profile_ids` | array[string] | Profiles represented in the scorecard. |
| `cost_model_id` | string | Cost model used for normal and stress costs. |
| `evidence_eligibility` | string | `diagnostic_only` or `selector_eligible_candidate`. |
| `diagnostic_only` | boolean | True when the artifact cannot become a selector candidate. |
| `proxy_evidence` | boolean | True for proxy replay or non-strategy evidence. |
| `relaxed_thresholds_used` | boolean | True when strict thresholds were relaxed. |
| `actual_strategy_backtest_required` | boolean | True for selector eligibility. |
| `historical_gate_passed` | boolean | Historical checked evidence gate result. |
| `walk_forward_gate_passed` | boolean | Walk-forward checked evidence gate result. |
| `selector_candidate_creation_allowed` | boolean | False unless strict evidence passes. |
| `paper_readiness_input_allowed` | boolean | False unless the full strict schema passes. |
| `rows` | array[object] | State-conditioned scorecard rows. |
| `baseline_comparisons` | array[object] | Separate baseline deltas by `baseline_id`. |
| `summary_decision` | string | Aggregate review decision. |
| `blockers` | array[string] | Hard blockers. |
| `reason_codes` | array[string] | Stable audit reasons. |
| `safety_scope` | object | Required no-startup, no-live, no-secret flags. |

## Eligibility Boundary Fields

`evidence_eligibility` values:

```text
diagnostic_only
selector_eligible_candidate
```

`selector_candidate_creation_allowed` must be `false` when any of these are
true:

- `diagnostic_only=true`;
- `proxy_evidence=true`;
- `relaxed_thresholds_used=true`;
- checked strategy identity is missing or mismatched;
- checked historical strategy evidence is missing;
- checked walk-forward evidence is missing;
- the scorecard was manually assembled;
- the scorecard uses a single-window demo;
- state coverage, trade count, or independent window count is insufficient.

`paper_readiness_input_allowed` must be `false` unless
`selector_candidate_creation_allowed=true` and the full strict scorecard schema
passes. Even then, the scorecard is only a future Phase 3 readiness input. It
does not authorize startup.

## Scorecard Row

Each row evaluates one strategy evidence unit in one market state and horizon
profile.

Required fields:

| Field | Type | Notes |
| --- | --- | --- |
| `strategy_version` | string | Strategy version from candidate identity. |
| `signal_version` | string | Signal version from candidate identity. |
| `risk_policy_version` | string | Risk policy version from candidate identity. |
| `state_id` | string | Market-state id. |
| `horizon_profile_id` | string | Multi-horizon profile id. |
| `state_encoder_version` | string | Market-state encoder version. |
| `pair` | string | Pair under evaluation. |
| `pair_group` | string | Pair group used for state-scope aggregation; current implementation defaults to the pair when no broader group is supplied. |
| `timeframe` | string | Strategy timeframe. |
| `cost_model_id` | string | Cost model id. |
| `state_window_ids` | array[string] | Source state windows represented by this row. |
| `decision_windows` | array[object] | Source decision windows with `start` and `end`. |
| `feature_cutoff_range` | object | Earliest and latest feature cutoff represented by the row. |
| `label_cutoff_range` | object | Earliest and latest label cutoff represented by the row. |
| `source_observation_count` | integer | Count of source observations represented by this row. |
| `sample_days` | number | Total covered days. |
| `independent_window_count` | integer | Independent evaluation windows. |
| `non_overlapping_window_count` | integer | Non-overlapping windows. |
| `trade_count` | integer | Checked strategy trades. |
| `exposure_ratio` | number | Fraction of time in market. |
| `average_holding_time` | string or number | Duration or minutes. |
| `gross_return` | number | Gross return. |
| `net_return_normal_cost` | number | Net return after normal cost. |
| `net_return_stress_cost` | number | Net return after stress cost. |
| `expectancy` | number | Expectancy after costs. |
| `profit_factor` | number or null | Profit factor. |
| `win_rate` | number | Win rate. |
| `max_drawdown` | number | Maximum drawdown. |
| `downside_deviation` | number | Downside deviation. |
| `turnover` | number | Turnover. |
| `cost_burden` | number | Cost impact. |
| `no_trade_delta` | number | Delta versus no-trade baseline. |
| `hold_delta` | number | Delta versus hold baseline. |
| `incumbent_delta` | number or null | Delta versus incumbent when present. |
| `lower_confidence_bound` | number | Conservative edge estimate. |
| `pair_concentration` | number | Concentration score. |
| `calendar_concentration` | number | Calendar concentration score. |
| `state_sample_count` | integer | State analog count. |
| `state_cluster_stability` | number or null | Cluster stability when available. |
| `data_quality_pass` | boolean | Local data-quality result. |
| `feature_quality_pass` | boolean | Feature-quality result. |
| `decision` | string | State decision enum. |
| `blockers` | array[string] | Hard vetoes. |
| `reason_codes` | array[string] | Stable audit reasons. |

## Decisions

Allowed row decisions:

```text
STATE_SELECTOR_ELIGIBLE
STATE_SHADOW_ONLY
STATE_INSUFFICIENT_EVIDENCE
STATE_UNSAFE
STATE_NO_TRADE_POLICY
STATE_DIAGNOSTIC_ONLY
```

`STATE_SELECTOR_ELIGIBLE` means a future local selector simulation may consider
the candidate for the matching state and horizon profile. It does not mean paper
readiness, dry-run approval, live approval, process startup, or order placement.

## Baseline Comparisons

Baseline deltas must be split by `baseline_id`. Producers must not sum hold and
no-trade baselines into one aggregate.

Required baseline row fields:

| Field | Type | Notes |
| --- | --- | --- |
| `baseline_id` | string | Examples: `no_trade`, `hold`, `incumbent:<candidate_id>`, `style:<id>`. |
| `state_id` | string | Same state id as scorecard row. |
| `horizon_profile_id` | string | Same horizon profile id. |
| `pair` | string | Pair. |
| `timeframe` | string | Timeframe. |
| `net_return_delta` | number | Candidate net return minus baseline. |
| `drawdown_delta` | number | Candidate drawdown minus baseline. |
| `exposure_delta` | number | Candidate exposure minus baseline. |
| `opportunity_cost` | number or null | Required for no-trade evaluation. |
| `reason_codes` | array[string] | Stable audit reasons. |

## Hard Vetoes

Any strict selector-eligible scorecard must fail closed when these blockers are
present:

```text
insufficient_independent_windows
insufficient_non_overlapping_windows
insufficient_trades
negative_stress_cost_edge
non_positive_lower_confidence_bound
drawdown_beyond_state_contract
pair_concentration_too_high
calendar_concentration_too_high
data_quality_failed
feature_quality_failed
state_coverage_too_narrow
missing_no_trade_baseline
missing_hold_baseline
identity_mismatch
missing_walk_forward_evidence
```

## Safety Scope

Required safety flags:

```json
{
  "local_artifacts_source_of_truth": true,
  "historical_evaluation_only": true,
  "freqtrade_trade_started": false,
  "paper_trading_started": false,
  "dry_run_trading_started": false,
  "live_trading_started": false,
  "exchange_order_placement": false,
  "uses_api_keys_or_secrets": false,
  "metadata_contains_secrets": false,
  "process_control": false,
  "leverage_above_one": false,
  "shorting": false,
  "promotion_authorized_by_this_artifact": false,
  "phase3_readiness_required_after_scorecard": true
}
```

## Example

```json
{
  "factory": "bot_factory",
  "schema_version": "state_conditioned_scorecard_v1",
  "run_id": "20260528T001000Z_candidate_state_scorecard",
  "generated_at": "2026-05-28T00:10:00+09:00",
  "candidate_id": "strong-uptrend-historical-ohlcv-candidate",
  "candidate_identity_schema_version": "strategy_candidate_identity_v1",
  "candidate_identity": {
    "candidate_id": "strong-uptrend-historical-ohlcv-candidate",
    "strategy_version": "DonchianTrendBullStrategy:v1",
    "signal_version": "strong_uptrend_momentum_v1",
    "risk_policy_version": "long_only_static_risk_v1",
    "regime_classifier_version": "deterministic_regime_classifier_v1",
    "cost_model_id": "calibrated_cost_model_v1"
  },
  "source_artifacts": {
    "backtest_metrics": "data/backtests/DonchianTrendBullStrategy/example/metrics.json",
    "walk_forward_metrics": "data/walk_forward/DonchianTrendBullStrategy/example/walk_forward_metrics.json",
    "market_state_snapshot": "data/market_state/example/market_state_snapshot.json"
  },
  "source_artifact_hashes": {
    "backtest_metrics": "sha256:example"
  },
  "state_encoder_version": "deterministic_market_state_encoder_v1",
  "horizon_profile_ids": [
    "deterministic_market_state_encoder_v1:micro=mixed:intraday=trend_up:swing=unknown"
  ],
  "cost_model_id": "calibrated_cost_model_v1",
  "evidence_eligibility": "selector_eligible_candidate",
  "diagnostic_only": false,
  "proxy_evidence": false,
  "relaxed_thresholds_used": false,
  "actual_strategy_backtest_required": true,
  "historical_gate_passed": true,
  "walk_forward_gate_passed": true,
  "selector_candidate_creation_allowed": true,
  "paper_readiness_input_allowed": false,
  "rows": [
    {
      "strategy_version": "DonchianTrendBullStrategy:v1",
      "signal_version": "strong_uptrend_momentum_v1",
      "risk_policy_version": "long_only_static_risk_v1",
      "state_id": "deterministic_market_state_encoder_v1:1h:trend_up:medium:ohlcv_state_features_v1",
      "horizon_profile_id": "deterministic_market_state_encoder_v1:micro=mixed:intraday=trend_up:swing=unknown",
      "pair": "BTC/USDT:USDT",
      "timeframe": "5m",
      "cost_model_id": "calibrated_cost_model_v1",
      "sample_days": 120.0,
      "independent_window_count": 8,
      "non_overlapping_window_count": 6,
      "trade_count": 42,
      "exposure_ratio": 0.18,
      "average_holding_time": "PT1H20M",
      "gross_return": 0.12,
      "net_return_normal_cost": 0.09,
      "net_return_stress_cost": 0.05,
      "expectancy": 0.0012,
      "profit_factor": 1.35,
      "win_rate": 0.55,
      "max_drawdown": -0.04,
      "downside_deviation": 0.012,
      "turnover": 1.8,
      "cost_burden": 0.03,
      "no_trade_delta": 0.05,
      "hold_delta": 0.01,
      "incumbent_delta": null,
      "lower_confidence_bound": 0.01,
      "pair_concentration": 1.0,
      "calendar_concentration": 0.35,
      "state_sample_count": 180,
      "state_cluster_stability": null,
      "data_quality_pass": true,
      "feature_quality_pass": true,
      "decision": "STATE_SELECTOR_ELIGIBLE",
      "blockers": [],
      "reason_codes": ["positive_stress_cost_edge", "beats_hold", "walk_forward_gate_passed"]
    }
  ],
  "baseline_comparisons": [
    {
      "baseline_id": "no_trade",
      "state_id": "deterministic_market_state_encoder_v1:1h:trend_up:medium:ohlcv_state_features_v1",
      "horizon_profile_id": "deterministic_market_state_encoder_v1:micro=mixed:intraday=trend_up:swing=unknown",
      "pair": "BTC/USDT:USDT",
      "timeframe": "5m",
      "net_return_delta": 0.05,
      "drawdown_delta": -0.04,
      "exposure_delta": 0.18,
      "opportunity_cost": null,
      "reason_codes": ["candidate_beats_no_trade_after_cost"]
    },
    {
      "baseline_id": "hold",
      "state_id": "deterministic_market_state_encoder_v1:1h:trend_up:medium:ohlcv_state_features_v1",
      "horizon_profile_id": "deterministic_market_state_encoder_v1:micro=mixed:intraday=trend_up:swing=unknown",
      "pair": "BTC/USDT:USDT",
      "timeframe": "5m",
      "net_return_delta": 0.01,
      "drawdown_delta": 0.02,
      "exposure_delta": -0.82,
      "opportunity_cost": null,
      "reason_codes": ["candidate_beats_hold_after_cost"]
    }
  ],
  "summary_decision": "STATE_SELECTOR_ELIGIBLE",
  "blockers": [],
  "reason_codes": ["strict_checked_evidence"],
  "safety_scope": {
    "local_artifacts_source_of_truth": true,
    "historical_evaluation_only": true,
    "freqtrade_trade_started": false,
    "paper_trading_started": false,
    "dry_run_trading_started": false,
    "live_trading_started": false,
    "exchange_order_placement": false,
    "uses_api_keys_or_secrets": false,
    "metadata_contains_secrets": false,
    "process_control": false,
    "leverage_above_one": false,
    "shorting": false,
    "promotion_authorized_by_this_artifact": false,
    "phase3_readiness_required_after_scorecard": true
  }
}
```
