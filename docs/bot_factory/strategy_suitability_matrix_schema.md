# Bot Factory Strategy Suitability Matrix Schema

Status: implemented local artifact schema.

Implementation status: implemented by
`freqtrade_ext/bot_factory/strategy_suitability.py` and
`scripts/bot_factory_build_strategy_suitability.py`. This artifact supports
offline selector simulation only. It does not start a bot, switch a bot, run
paper or dry-run trading, run live trading, or authorize order placement.

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

The implemented producer writes:

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
| `factory` | string | Must be `strategy_suitability_matrix`. |
| `schema_version` | string | Must be `strategy_suitability_matrix_v1`. |
| `run_id` | string | Stable local matrix run id. |
| `generated_at` | timestamp | Artifact creation time. |
| `source_artifacts` | object | State-conditioned scorecard paths and optional source paths. |
| `source_market_state_schema_version` | string | Expected `market_state_snapshot_v1`. |
| `source_state_scorecard_count` | integer | Number of source scorecards. |
| `scorecard_validations` | array[object] | Full validation results for source scorecards. |
| `selector_row_count` | integer | Count of selector-eligible strategy rows. |
| `state_count` | integer | Count of state/horizon scopes represented. |
| `rows` | array[object] | Suitability rows, including a first-class no-trade row. |
| `summary` | object | Counts by decision and row type. |
| `reason_codes` | array[string] | Stable audit reasons. |
| `safety_scope` | object | Required no-startup, no-live, no-secret flags. |

## Suitability Row

Required row fields:

| Field | Type | Notes |
| --- | --- | --- |
| `row_type` | string | `strategy`, `no_trade`, or `missing_state`. |
| `strategy_identity_unit` | object or null | Required for strategy rows. |
| `candidate_id` | string | Candidate id or policy id. |
| `strategy_id` | string | Strategy id or policy id. |
| `state_id` | string | Market-state id. |
| `state_label` | string | Deterministic state label where available. |
| `horizon_profile_id` | string | Horizon profile id. |
| `pair_group` | string | Pair group. |
| `pair` | string | Pair where evidence applies. |
| `timeframe` | string | Timeframe. |
| `cost_model_id` | string | Cost model id. |
| `state_encoder_version` | string | Market-state encoder version. |
| `decision` | string | Matrix decision enum. |
| `matching_action` | string | Future matching action. |
| `selector_eligible` | boolean | True only for checked selector rows. |
| `evidence_quality` | string | `checked`, `weak`, `policy`, `missing`, or diagnostic. |
| `expected_utility_after_cost` | number or null | Conservative utility after costs. |
| `risk_adjusted_score` | number or null | Predeclared scoring output. |
| `stress_cost_utility` | number or null | Stress-cost utility used for ranking. |
| `rank_score` | number or null | Deterministic local rank score. |
| `uncertainty` | number | Uncertainty in `[0.0, 1.0]`. |
| `no_trade_delta` | number or null | Delta versus no-trade baseline. |
| `hold_delta` | number or null | Delta versus hold baseline. |
| `incumbent_delta` | number or null | Delta versus incumbent when present. |
| `lower_confidence_bound` | number or null | Conservative edge estimate. |
| `identity_mismatch` | boolean | True when row identity differs from source candidate identity. |
| `blockers` | array[string] | Hard blockers. |
| `reason_codes` | array[string] | Stable audit reasons. |

## Decisions

Allowed matrix decisions:

```text
SELECTOR_ELIGIBLE
SHADOW_ONLY
DIAGNOSTIC_ONLY
IDENTITY_MISMATCH
UNSAFE_NO_TRADE
NO_TRADE_POLICY
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
- `strategy_identity_unit={"policy_id": "no_trade"}`;
- `selector_eligible=false`;
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
  "factory": "strategy_suitability_matrix",
  "schema_version": "strategy_suitability_matrix_v1",
  "run_id": "20260528T002000Z_strategy_suitability",
  "generated_at": "2026-05-28T00:20:00+09:00",
  "source_artifacts": {
    "state_scorecard_1": "data/state_scorecards/trend/example/state_conditioned_scorecard.json"
  },
  "source_state_scorecard_count": 1,
  "source_market_state_schema_version": "market_state_snapshot_v1",
  "scorecard_validations": [
    {
      "candidate_id": "trend-candidate",
      "run_id": "trend_state_scorecard",
      "ok": true,
      "reason_codes": ["state_conditioned_scorecard_selector_valid"]
    }
  ],
  "selector_row_count": 1,
  "state_count": 2,
  "rows": [
    {
      "row_type": "strategy",
      "strategy_identity_unit": {
        "candidate_id": "trend-candidate",
        "strategy_id": "trend-strategy",
        "strategy_version": "strategy_v1",
        "signal_version": "signal_v1",
        "risk_policy_version": "risk_v1",
        "cost_model_id": "cost_model_v1",
        "allowed_pairs": ["BTC/USDT:USDT"],
        "allowed_timeframes": ["5m"]
      },
      "candidate_id": "trend-candidate",
      "strategy_id": "trend-strategy",
      "state_id": "deterministic_market_state_encoder_v1:5m:trend_up:high:ohlcv_state_features_v1",
      "state_label": "trend_up",
      "horizon_profile_id": "deterministic_market_state_encoder_v1:micro=trend_up:intraday=missing:swing=missing",
      "pair_group": "btc_major",
      "pair": "BTC/USDT:USDT",
      "timeframe": "5m",
      "cost_model_id": "cost_model_v1",
      "state_encoder_version": "deterministic_market_state_encoder_v1",
      "decision": "SELECTOR_ELIGIBLE",
      "matching_action": "select_strategy",
      "selector_eligible": true,
      "evidence_quality": "checked",
      "expected_utility_after_cost": 3.2,
      "risk_adjusted_score": 0.2,
      "stress_cost_utility": 3.5,
      "rank_score": 3.5,
      "uncertainty": 0.1,
      "no_trade_delta": 5.0,
      "hold_delta": 3.0,
      "incumbent_delta": null,
      "lower_confidence_bound": 0.4,
      "identity_mismatch": false,
      "blockers": [],
      "reason_codes": ["selector_eligible_checked_state_evidence"]
    },
    {
      "row_type": "no_trade",
      "strategy_identity_unit": {"policy_id": "no_trade"},
      "candidate_id": "no_trade",
      "strategy_id": "no_trade",
      "state_id": "deterministic_market_state_encoder_v1:5m:trend_up:high:ohlcv_state_features_v1",
      "state_label": "trend_up",
      "horizon_profile_id": "deterministic_market_state_encoder_v1:micro=trend_up:intraday=missing:swing=missing",
      "pair_group": "btc_major",
      "pair": "BTC/USDT:USDT",
      "timeframe": "5m",
      "cost_model_id": "cost_model_v1",
      "state_encoder_version": "deterministic_market_state_encoder_v1",
      "decision": "NO_TRADE_POLICY",
      "matching_action": "no_trade",
      "selector_eligible": false,
      "evidence_quality": "policy",
      "expected_utility_after_cost": 0.0,
      "risk_adjusted_score": 0.0,
      "stress_cost_utility": 0.0,
      "rank_score": 0.0,
      "uncertainty": 0.0,
      "blockers": [],
      "reason_codes": ["first_class_no_trade_policy"]
    },
    {
      "row_type": "missing_state",
      "candidate_id": "missing_state",
      "strategy_id": "missing_state",
      "state_id": "deterministic_market_state_encoder_v1:15m:out_of_distribution:low:ohlcv_state_features_v1",
      "state_label": "out_of_distribution",
      "horizon_profile_id": "deterministic_market_state_encoder_v1:micro=trend_up:intraday=missing:swing=missing",
      "decision": "OUT_OF_DISTRIBUTION_NO_TRADE",
      "matching_action": "no_trade",
      "selector_eligible": false,
      "evidence_quality": "missing",
      "blockers": ["no_selector_eligible_strategy_for_state"],
      "reason_codes": ["out_of_distribution_no_trade"]
    }
  ],
  "summary": {
    "selector_eligible_rows": 1,
    "no_trade_rows": 1,
    "missing_state_rows": 1,
    "diagnostic_rows": 0
  },
  "reason_codes": ["strategy_suitability_matrix_built_from_state_scorecards"],
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
    "promotion_authorized_by_this_artifact": false
  }
}
```
