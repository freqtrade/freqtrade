# Bot Factory Selector Matching Schema

Status: implemented local artifact schema.

Implementation status: implemented by
`freqtrade_ext/bot_factory/selector_matching.py` and
`scripts/bot_factory_match_strategy_to_market_state.py`. This is an offline
selector-simulation artifact only. It does not start paper, dry-run, live
trading, `freqtrade trade`, process control, or exchange order placement.

## Purpose

Selector matching joins:

```text
current_market_state_v1 or market_state_snapshot_v1
+ strategy_suitability_matrix_v1
+ optional selector_state
```

and emits one local decision:

```text
select_strategy
no_trade
```

Future states such as `watch`, `quarantine`, and `retire` can be added without
changing the no-startup boundary.

## `selector_matching_decision_v1`

Required top-level fields:

| Field | Type | Notes |
| --- | --- | --- |
| `factory` | string | Must be `selector_matching`. |
| `schema_version` | string | Must be `selector_matching_decision_v1`. |
| `decision_id` | string | Stable local decision id. |
| `generated_at` | timestamp | Artifact creation time. |
| `data_asof` | timestamp or null | Local data as-of timestamp. |
| `selected_action` | string | `select_strategy` or `no_trade`. |
| `selected_strategy_id` | string | Strategy id, or `no_trade`. |
| `selected_candidate_id` | string or null | Candidate id when selected. |
| `selected_state_id` | string or null | Selected market-state id. |
| `selected_horizon_profile_id` | string or null | Selected horizon profile. |
| `no_trade_reason` | string or null | Primary no-trade reason. |
| `selector_version` | string | Deterministic selector version. |
| `state_encoder_version` | string | Market-state encoder version. |
| `evidence_unit` | object or null | Selected strategy identity unit. |
| `confidence` | number or null | Current state confidence. |
| `uncertainty` | number or null | Current state uncertainty. |
| `reason_codes` | array[string] | Stable audit reasons. |
| `comparison_set` | array[object] | No-trade, incumbent, and candidate comparisons. |
| `rejected_alternatives` | array[object] | Candidate rows rejected by the selector. |
| `selector_state` | object | Input selector state. |
| `next_selector_state` | object | Deterministic next selector state. |
| `source_artifacts` | object | Local source paths. |
| `safety_scope` | object | Required no-startup, no-live, no-secret flags. |

## No-Trade Defaults

The selector must output `no_trade` for:

- stale local data;
- low state confidence;
- unknown, mixed, transition, or out-of-distribution state;
- horizon conflict;
- feature-quality failure;
- stale cost-model flags;
- invalid suitability matrix;
- no selector-eligible row for the current state/horizon profile;
- strategy identity mismatch;
- cooldown-blocked switching.

## Ranking

Selector-eligible strategy rows are ranked by deterministic stress-aware
metrics:

```text
stress_cost_utility
expected_utility_after_cost
lower_confidence_bound
lower max_drawdown
```

Raw PnL alone is not a selector ranking key.

## `no_trade_scorecard_v1`

Required top-level fields:

| Field | Type | Notes |
| --- | --- | --- |
| `factory` | string | Must be `no_trade_scorecard`. |
| `schema_version` | string | Must be `no_trade_scorecard_v1`. |
| `run_id` | string | Stable local scorecard id. |
| `generated_at` | timestamp | Artifact creation time. |
| `selector_decision_id` | string or null | Linked selector decision id. |
| `rows` | array[object] | Per-state no-trade evaluation rows. |
| `reason_codes` | array[string] | Stable audit reasons. |
| `safety_scope` | object | Required no-startup, no-live, no-secret flags. |

No-trade rows separate:

- avoided drawdown;
- avoided negative expectancy;
- opportunity cost versus hold, incumbent, and best selector-eligible strategy;
- uncertainty reduction value.

No-trade scorecards include `no_hindsight_profit_credit` so loss avoidance is
not treated as a standalone hindsight reward.

## Safety Scope

Required safety flags:

```json
{
  "local_artifacts_source_of_truth": true,
  "historical_evaluation_only": true,
  "selector_simulation_only": true,
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
```
