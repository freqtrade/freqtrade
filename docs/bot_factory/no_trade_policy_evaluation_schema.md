# No-Trade Policy Evaluation Schema

Status: implemented for local historical selector replay evaluation.

Updated: 2026-05-30 JST.

## Purpose

`no_trade_policy_evaluation_v1` treats `no_trade` as a measurable historical
policy decision. It consumes selector replay decisions and reports whether
abstention avoided loss, created acceptable opportunity cost, or was overused.

This artifact is local-only and diagnostic/governance evidence. It cannot start
paper trading, dry-run trading, live trading, `freqtrade trade`, exchange order
placement, process control, promotion, leverage, or shorting.

## Artifact Paths

Default writer output:

```text
data/no_trade_evaluations/<run_id>/no_trade_policy_evaluation.json
data/no_trade_evaluations/<run_id>/no_trade_policy_evaluation_report.md
```

CLI:

```powershell
.\.venv\Scripts\python.exe scripts\bot_factory_evaluate_no_trade_policy.py --selector-replay-json data\selector_replay\<run_id>\selector_replay.json
```

## Input

Required:

- `selector_replay.json` produced by `historical_selector_replay_v1`.

Optional:

- `opportunity_cost_thresholds_json`, a JSON object keyed by state type:
  - `uncertain_or_ood`;
  - `unsupported`;
  - `cooldown`;
  - `supported`.

Default thresholds:

```json
{
  "uncertain_or_ood": 0.03,
  "unsupported": 0.02,
  "cooldown": 0.015,
  "supported": 0.01
}
```

## Required Fields

Top-level fields:

```text
factory
schema_version
run_id
generated_at
source_selector_replay_run_id
opportunity_cost_thresholds
no_trade_decisions[]
state_no_trade_quality[]
summary
summary_decision
reason_codes[]
source_artifacts
safety_scope
```

Each `no_trade_decisions[]` row records:

```text
decision_at
state_id
state_type
no_trade_reason
avoided_drawdown
opportunity_cost_vs_hold
opportunity_cost_vs_best
uncertainty_ood_safety_value
opportunity_cost_threshold
assessment
reason_codes[]
```

Each `state_no_trade_quality[]` row aggregates by `(state_type, state_id)` and
records:

```text
state_type
state_id
no_trade_count
avoided_drawdown
opportunity_cost_vs_hold
opportunity_cost_vs_best
uncertainty_ood_safety_value
opportunity_cost_threshold
assessment
```

## Assessment Semantics

Valid `assessment` values:

- `good`: no-trade avoided drawdown while opportunity cost stayed within the
  configured state-type threshold.
- `acceptable`: no-trade did not clearly avoid loss but did not exceed the
  opportunity-cost threshold.
- `costly`: no-trade exceeded the threshold without avoided drawdown.
- `overused`: no-trade exceeded twice the threshold without avoided drawdown.

Summary decisions:

```text
NO_TRADE_GOOD
NO_TRADE_ACCEPTABLE
NO_TRADE_COSTLY
NO_TRADE_OVERUSED
```

`NO_TRADE_OVERUSED` takes precedence over `NO_TRADE_COSTLY`, which takes
precedence over `NO_TRADE_GOOD`.

## Safety Scope

The artifact must include:

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
