# Bot Factory Historical Selector Replay Schema

Status: implemented local artifact contract for Product Vision Phase 2.

Implementation status: `historical_selector_replay_v1` is implemented by
`freqtrade_ext/bot_factory/selector_replay.py` and the local CLI
`scripts/bot_factory_run_selector_replay.py`.

This artifact is historical/as-of selector simulation only. It does not start
`freqtrade trade`, paper trading, dry-run trading, live trading, process
control, exchange order placement, leverage, shorting, or API-key usage.

## Purpose

Historical selector replay answers whether the state-aware
`select_strategy`/`no_trade` policy would have made defensible decisions through
time using only local information available at each decision timestamp.

## Inputs

- `market_state_snapshot_v1` or compatible current-state snapshots with
  `data_asof`.
- `strategy_suitability_matrix_v1` artifacts with `generated_at`.
- Optional realized-return rows keyed by `data_asof`, with candidate ids and
  `hold` returns.

The replay sorts snapshots by `data_asof` and uses only suitability matrices
whose `generated_at` is less than or equal to that decision timestamp.

## Future Leakage Guards

The replay is invalid when:

- a market-state horizon row has `future_data_used=true`;
- a market-state `feature_cutoff_timestamp` or `label_cutoff_timestamp` is
  after the decision `data_asof`;
- a suitability row has `future_data_used=true`;
- a suitability row has `evidence_available_at` after the matrix
  `generated_at`.

Suitability matrices generated after a decision timestamp are not used for that
decision. The decision falls back to `no_trade` with
`no_strategy_evidence_available_asof` when no eligible as-of matrix exists.

## Outputs

```text
data/selector_replay/<run_id>/selector_replay.json
data/selector_replay/<run_id>/selector_decisions.jsonl
data/selector_replay/<run_id>/selector_replay_report.md
```

## `historical_selector_replay_v1`

Required top-level fields:

| Field | Type | Notes |
| --- | --- | --- |
| `factory` | string | `historical_selector_replay`. |
| `schema_version` | string | `historical_selector_replay_v1`. |
| `run_id` | string | Stable local replay id. |
| `generated_at` | timestamp | Artifact generation time. |
| `selector_version` | string | Selector replay version. |
| `status` | string | `completed` or `invalid`. |
| `input_validation` | object | Future-leakage and evidence-availability checks. |
| `decision_count` | integer | Count of replay decisions. |
| `decisions` | array[object] | Same content as `selector_decisions.jsonl`. |
| `metrics_summary` | object | Selector-level metrics. |
| `baseline_comparisons` | array[object] | Baseline rows. |
| `future_evidence_rejected_count` | integer | Matrices ignored because they were generated after a decision timestamp. |
| `reason_codes` | array[string] | Stable audit reasons. |
| `source_artifacts` | object | Input path references. |
| `safety_scope` | object | No-startup safety flags. |

## Decision Row

Each decision row records:

- `decision_at`;
- selected action and candidate;
- selected state and horizon profile;
- no-trade reason when applicable;
- eligible candidate ids and rejected alternatives;
- selector realized return;
- hold return;
- best eligible return;
- missed opportunity;
- no-trade loss avoidance;
- `future_data_used=false`.

## Baselines

Replay reports include:

- `always_no_trade`;
- `always_hold`;
- `best_single_eligible_strategy`;
- `equal_rotation`;
- `incumbent:<candidate_id>` or `incumbent:none`.

## Metrics

The current implementation reports:

- selector net return under normal and stress turnover costs;
- max drawdown;
- downside deviation;
- exposure ratio;
- turnover / selector churn;
- missed opportunity;
- no-trade loss avoidance;
- no-trade count;
- unsupported-state rate;
- future-leakage check result;
- identity-scope check result.

These metrics are local replay metrics only. They do not authorize paper,
dry-run, live trading, or process startup.
