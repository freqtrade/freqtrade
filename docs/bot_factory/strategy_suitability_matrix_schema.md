# Bot Factory Strategy Suitability Matrix Schema

Status: draft schema for Increment 0.

Implementation status: documentation contract only. This file defines the
artifact shape for future strategy-state suitability and offline matching
review. It does not implement matching, start a bot, switch a bot, run paper or
dry-run trading, run live trading, or authorize order placement.

## Purpose

The strategy suitability matrix answers:

- which strategy evidence units are supported in each state and horizon profile;
- which strategies are blocked, diagnostic-only, or unsafe;
- which states have no supported strategy and must default to no-trade;
- why a future selector should reject alternatives.

Rows are built from strict state-conditioned scorecards only. Diagnostic
scorecards may be included for review visibility, but they cannot create
selector candidates.

## Artifacts

Future producers should write:

```text
data/strategy_suitability/<run_id>/strategy_state_suitability_matrix.json
data/strategy_suitability/<run_id>/strategy_state_suitability_report.md
```

## Matrix Unit

Each strategy row is keyed by:

```text
strategy_identity_unit
+ state_id
+ horizon_profile_id
+ pair_group
+ timeframe
+ cost_model_id
+ state_encoder_version
```

The `strategy_identity_unit` must include:

```text
candidate_id
strategy_version
signal_version
risk_policy_version
cost_model_id
state_encoder_version
horizon_profile_id
activation_state_scope
```

No row may inherit evidence from another candidate id, strategy version, signal
version, risk policy version, state encoder version, horizon profile, activation
scope, or cost model.

## `strategy_suitability_matrix_v1`

Required top-level fields:

| Field | Type | Notes |
| --- | --- | --- |
| `factory` | string | Must be `bot_factory`. |
| `schema_version` | string | Must be `strategy_suitability_matrix_v1`. |
| `run_id` | string | Stable local matrix run id. |
| `generated_at` | timestamp | Artifact creation time. |
| `source_scorecards` | array[object] | State-conditioned scorecard paths and hashes. |
| `source_market_state_schema_version` | string | Expected `market_state_snapshot_v1`. |
| `source_scorecard_schema_version` | string | Expected `state_conditioned_scorecard_v1`. |
| `state_encoder_version` | string | Market-state encoder version. |
| `cost_model_id` | string | Cost model id. |
| `pair_groups` | array[string] | Pair groups covered by the matrix. |
| `timeframes` | array[string] | Timeframes covered by the matrix. |
| `horizon_profile_ids` | array[string] | Horizon profiles covered by the matrix. |
| `rows` | array[object] | Suitability rows, including a first-class no-trade row. |
| `missing_state_rows` | array[object] | States with no supported strategy. |
| `matrix_summary` | object | Counts by decision and blocker. |
| `diff_against_matrix` | object or null | Optional previous matrix comparison. |
| `reason_codes` | array[string] | Stable audit reasons. |
| `safety_scope` | object | Required no-startup, no-live, no-secret flags. |

## Suitability Row

Required row fields:

| Field | Type | Notes |
| --- | --- | --- |
| `row_id` | string | Stable row identifier. |
| `row_type` | string | `strategy`, `no_trade`, or `missing_state`. |
| `strategy_identity_unit` | object or null | Required for strategy rows. |
| `state_id` | string | Market-state id. |
| `horizon_profile_id` | string | Horizon profile id. |
| `pair_group` | string | Pair group. |
| `timeframe` | string | Timeframe. |
| `cost_model_id` | string | Cost model id. |
| `activation_state_scope` | array[string] | States where this row may activate. |
| `decision` | string | Matrix decision enum. |
| `matching_action` | string | Future matching action. |
| `evidence_quality` | string | `strict`, `weak`, `diagnostic_only`, `missing`, or `failed`. |
| `expected_utility_after_cost` | number or null | Conservative utility after costs. |
| `risk_adjusted_score` | number or null | Predeclared scoring output. |
| `uncertainty` | number | Uncertainty in `[0.0, 1.0]`. |
| `state_confidence_min` | number | Minimum state confidence required. |
| `no_trade_delta` | number or null | Delta versus no-trade baseline. |
| `hold_delta` | number or null | Delta versus hold baseline. |
| `incumbent_delta` | number or null | Delta versus incumbent when present. |
| `lower_confidence_bound` | number or null | Conservative edge estimate. |
| `stress_cost_utility` | number or null | Stress-cost utility. |
| `blockers` | array[string] | Hard blockers. |
| `reason_codes` | array[string] | Stable audit reasons. |
| `source_scorecard_row_ids` | array[string] | Lineage to scorecard rows. |
| `selector_candidate_creation_allowed` | boolean | False for diagnostic, missing, or unsafe rows. |
| `paper_readiness_input_allowed` | boolean | False unless strict downstream schema passes. |

## Decisions

Allowed matrix decisions:

```text
STATE_SELECTOR_ELIGIBLE
STATE_SHADOW_ONLY
STATE_INSUFFICIENT_EVIDENCE
STATE_UNSAFE
STATE_NO_TRADE_POLICY
STATE_DIAGNOSTIC_ONLY
NO_SUPPORTED_STRATEGY
UNKNOWN_NO_TRADE
OUT_OF_DISTRIBUTION_NO_TRADE
```

Allowed `matching_action` values:

```text
select_strategy
watch_only
shadow_only
quarantine
retire
no_trade
```

`select_strategy` remains a local offline selector decision only. It does not
authorize Phase 3 paper readiness, paper startup, dry-run startup, live trading,
process control, or exchange order placement.

## No-Trade Row

Every matrix must include at least one first-class no-trade row per covered
state or missing state.

Required no-trade row behavior:

- `row_type=no_trade`;
- `matching_action=no_trade`;
- `strategy_identity_unit=null`;
- `selector_candidate_creation_allowed=false`;
- `paper_readiness_input_allowed=false`;
- reason codes distinguish uncertainty safety value from hindsight loss
  avoidance.

Recommended no-trade reason codes:

```text
stale_local_data_no_trade
unknown_state_no_trade
horizon_conflict_no_trade
out_of_distribution_no_trade
no_supported_strategy
insufficient_state_evidence
feature_quality_failed_no_trade
cooldown_or_hysteresis_no_trade
```

## Missing State Rows

States with no strict supported strategy should emit `missing_state` rows rather
than disappearing from the matrix.

Required fields:

| Field | Type | Notes |
| --- | --- | --- |
| `state_id` | string | Missing or unsupported state. |
| `horizon_profile_id` | string | Profile with missing support. |
| `decision` | string | `NO_SUPPORTED_STRATEGY`, `UNKNOWN_NO_TRADE`, or `OUT_OF_DISTRIBUTION_NO_TRADE`. |
| `matching_action` | string | Must be `no_trade`. |
| `evidence_gap` | string | Human-readable gap summary. |
| `minimum_required_evidence` | object | Predeclared minimum evidence. |
| `reason_codes` | array[string] | Stable audit reasons. |

## Matrix Diff

When comparing versions, `diff_against_matrix` should include:

| Field | Type | Notes |
| --- | --- | --- |
| `previous_matrix_path` | string | Local path to prior matrix. |
| `previous_matrix_hash` | string | Content hash. |
| `added_rows` | integer | Added row count. |
| `removed_rows` | integer | Removed row count. |
| `changed_decisions` | integer | Rows whose decision changed. |
| `changed_blockers` | integer | Rows whose blockers changed. |
| `requires_reviewer_attention` | boolean | True for eligibility or no-trade changes. |

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
  "strategy_switch_started": false,
  "leverage_above_one": false,
  "shorting": false,
  "paper_or_live_authorized_by_this_artifact": false
}
```

## Example

```json
{
  "factory": "bot_factory",
  "schema_version": "strategy_suitability_matrix_v1",
  "run_id": "20260528T002000Z_strategy_suitability",
  "generated_at": "2026-05-28T00:20:00+09:00",
  "source_scorecards": [
    {
      "path": "data/state_scorecards/strong-uptrend-historical-ohlcv-candidate/example/state_conditioned_scorecard.json",
      "sha256": "example"
    }
  ],
  "source_market_state_schema_version": "market_state_snapshot_v1",
  "source_scorecard_schema_version": "state_conditioned_scorecard_v1",
  "state_encoder_version": "deterministic_market_state_encoder_v1",
  "cost_model_id": "calibrated_cost_model_v1",
  "pair_groups": ["btc_major"],
  "timeframes": ["5m"],
  "horizon_profile_ids": [
    "deterministic_market_state_encoder_v1:micro=mixed:intraday=trend_up:swing=unknown"
  ],
  "rows": [
    {
      "row_id": "strategy:strong-uptrend-historical-ohlcv-candidate:trend_up",
      "row_type": "strategy",
      "strategy_identity_unit": {
        "candidate_id": "strong-uptrend-historical-ohlcv-candidate",
        "strategy_version": "DonchianTrendBullStrategy:v1",
        "signal_version": "strong_uptrend_momentum_v1",
        "risk_policy_version": "long_only_static_risk_v1",
        "cost_model_id": "calibrated_cost_model_v1",
        "state_encoder_version": "deterministic_market_state_encoder_v1",
        "horizon_profile_id": "deterministic_market_state_encoder_v1:micro=mixed:intraday=trend_up:swing=unknown",
        "activation_state_scope": ["trend_up"]
      },
      "state_id": "deterministic_market_state_encoder_v1:1h:trend_up:medium:ohlcv_state_features_v1",
      "horizon_profile_id": "deterministic_market_state_encoder_v1:micro=mixed:intraday=trend_up:swing=unknown",
      "pair_group": "btc_major",
      "timeframe": "5m",
      "cost_model_id": "calibrated_cost_model_v1",
      "activation_state_scope": ["trend_up"],
      "decision": "STATE_SELECTOR_ELIGIBLE",
      "matching_action": "select_strategy",
      "evidence_quality": "strict",
      "expected_utility_after_cost": 0.05,
      "risk_adjusted_score": 0.72,
      "uncertainty": 0.28,
      "state_confidence_min": 0.65,
      "no_trade_delta": 0.05,
      "hold_delta": 0.01,
      "incumbent_delta": null,
      "lower_confidence_bound": 0.01,
      "stress_cost_utility": 0.04,
      "blockers": [],
      "reason_codes": ["strict_state_scorecard_passed", "positive_stress_cost_utility"],
      "source_scorecard_row_ids": ["scorecard-row:trend_up:btc_major:5m"],
      "selector_candidate_creation_allowed": true,
      "paper_readiness_input_allowed": false
    },
    {
      "row_id": "no_trade:trend_up",
      "row_type": "no_trade",
      "strategy_identity_unit": null,
      "state_id": "deterministic_market_state_encoder_v1:1h:trend_up:medium:ohlcv_state_features_v1",
      "horizon_profile_id": "deterministic_market_state_encoder_v1:micro=mixed:intraday=trend_up:swing=unknown",
      "pair_group": "btc_major",
      "timeframe": "5m",
      "cost_model_id": "calibrated_cost_model_v1",
      "activation_state_scope": ["trend_up"],
      "decision": "STATE_NO_TRADE_POLICY",
      "matching_action": "no_trade",
      "evidence_quality": "strict",
      "expected_utility_after_cost": 0.0,
      "risk_adjusted_score": 0.0,
      "uncertainty": 0.28,
      "state_confidence_min": 0.0,
      "no_trade_delta": 0.0,
      "hold_delta": -0.04,
      "incumbent_delta": null,
      "lower_confidence_bound": 0.0,
      "stress_cost_utility": 0.0,
      "blockers": [],
      "reason_codes": ["first_class_no_trade_baseline", "opportunity_cost_recorded"],
      "source_scorecard_row_ids": ["scorecard-row:trend_up:btc_major:5m"],
      "selector_candidate_creation_allowed": false,
      "paper_readiness_input_allowed": false
    }
  ],
  "missing_state_rows": [
    {
      "state_id": "deterministic_market_state_encoder_v1:4h:out_of_distribution:low:ohlcv_state_features_v1",
      "horizon_profile_id": "deterministic_market_state_encoder_v1:micro=mixed:intraday=out_of_distribution:swing=unknown",
      "decision": "OUT_OF_DISTRIBUTION_NO_TRADE",
      "matching_action": "no_trade",
      "evidence_gap": "No strict checked strategy scorecard covers this OOD state.",
      "minimum_required_evidence": {
        "independent_window_count": 6,
        "trade_count": 20,
        "walk_forward_gate_passed": true
      },
      "reason_codes": ["out_of_distribution_state", "no_supported_strategy"]
    }
  ],
  "matrix_summary": {
    "selector_eligible_rows": 1,
    "no_trade_rows": 1,
    "missing_state_rows": 1,
    "diagnostic_only_rows": 0
  },
  "diff_against_matrix": null,
  "reason_codes": ["matrix_built_from_state_conditioned_scorecards"],
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
    "strategy_switch_started": false,
    "leverage_above_one": false,
    "shorting": false,
    "paper_or_live_authorized_by_this_artifact": false
  }
}
```
