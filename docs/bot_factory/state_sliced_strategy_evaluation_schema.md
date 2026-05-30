# Bot Factory State-Sliced Strategy Evaluation Schema

Status: implemented local artifact contract for Product Vision Phase 3.

Implementation status: `state_sliced_strategy_evaluation_v1` is implemented by
`freqtrade_ext/bot_factory/state_sliced_reporting.py` and
`scripts/bot_factory_build_state_sliced_report.py`.

This artifact is local historical/state-conditioned evaluation only. It does
not start `freqtrade trade`, paper trading, dry-run trading, live trading,
process control, exchange order placement, leverage, shorting, or API-key
usage.

## Purpose

State-sliced strategy evaluation makes backtest and walk-forward evidence answer
where a strategy works, where it is unsupported, and where a globally positive
result hides state-specific failure.

## Inputs

- `state_conditioned_scorecard_v1`;
- optional historical metrics JSON;
- optional walk-forward metrics JSON;
- optional expected state ids;
- optional incumbent and style-specific baseline maps by `state_id`.

## Outputs

```text
data/state_sliced_evaluations/<candidate_id>/<run_id>/state_sliced_evaluation.json
data/state_sliced_evaluations/<candidate_id>/<run_id>/state_sliced_evaluation_report.md
```

## `state_sliced_strategy_evaluation_v1`

Required top-level fields:

| Field | Type | Notes |
| --- | --- | --- |
| `factory` | string | `state_sliced_strategy_evaluation`. |
| `schema_version` | string | `state_sliced_strategy_evaluation_v1`. |
| `run_id` | string | Stable local report id. |
| `generated_at` | timestamp | Artifact generation time. |
| `candidate_id` | string | Candidate under review. |
| `candidate_identity` | object | Strategy candidate identity. |
| `candidate_style` | string | Style profile used for state gates. |
| `state_coverage` | object | Covered, unsupported, missing states and ratios. |
| `backtest_state_slices` | array[object] | Per-state backtest evidence. |
| `walk_forward_state_slices` | array[object] | Per-state walk-forward evidence. |
| `baseline_deltas_by_state` | array[object] | No-trade, hold, incumbent, and style-specific deltas. |
| `style_specific_state_gates` | array[object] | Per-state style gates. |
| `state_specific_crashes` | array[object] | Crashes hidden by positive global results. |
| `summary_decision` | string | `STATE_SLICED_PASS`, `STATE_SLICED_REVIEW`, or `STATE_SLICED_FAIL`. |
| `reason_codes` | array[string] | Stable audit reasons. |
| `safety_scope` | object | No-startup safety flags. |

## Gate Semantics

`STATE_SLICED_FAIL` is emitted when a positive global result hides a
state-specific crash such as negative stress-cost edge or drawdown beyond the
state threshold.

`STATE_SLICED_REVIEW` is emitted when states are missing, unsupported, or fail a
style-specific gate.

`STATE_SLICED_PASS` means the state-sliced evidence is internally consistent for
local review only. It does not authorize paper, dry-run, live trading, process
startup, or promotion.
