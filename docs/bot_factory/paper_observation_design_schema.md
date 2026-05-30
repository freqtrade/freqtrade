# Paper Observation Design Schema

Status: implemented as local design and validation artifacts.

Updated: 2026-05-30 JST.

## Purpose

`paper_observation_design_v1` defines how a future paper or dry-run observation
would be recorded without allowing that observation to start a process, override
historical evidence, bypass readiness, or directly promote a strategy.

This artifact is not a paper startup path. It is a local schema and governance
artifact.

## Artifact Paths

Default writer output:

```text
data/paper_observation_design/<run_id>/paper_observation_design.json
data/paper_observation_design/<run_id>/paper_observation_design_report.md
data/paper_observation_design/<run_id>/paper_observation_schema.json
data/paper_observation_design/<run_id>/paper_drift_report_schema.json
```

CLI:

```powershell
.\.venv\Scripts\python.exe scripts\bot_factory_build_paper_observation_design.py
```

## Observation Ledger Compatibility

Future paper/dry-run observations must use the same
`regime_observation_ledger_v1` base schema as local backtest and walk-forward
observations.

Future observation rows additionally require:

```text
state_snapshot_id
state_id
horizon_profile_id
state_encoder_version
state_window_id
feature_cutoff_timestamp
label_cutoff_timestamp
decision_window_start
decision_window_end
future_data_used=false
```

The artifact records that future source types remain blocked by default unless
the caller explicitly validates with future sources allowed.

## Evidence Separation

Recent paper-like observation evidence remains separate from:

- historical evidence;
- walk-forward evidence;
- training evidence;
- readiness evidence;
- runtime validation;
- drift evidence.

Recent observations may influence ranking only after strict state-conditioned
evidence already exists. They cannot override failed historical, walk-forward,
readiness, runtime, or drift gates.

## Drift Report

The design emits a `paper_observation_drift_report_v1` schema and local drift
decision with these checks:

- state distribution drift;
- feature distribution drift;
- cost/turnover drift;
- drawdown envelope breach;
- selector churn increase.

Drift reports do not perform process control and do not authorize promotion.

## Quarantine And Retirement

Quarantine triggers include:

- invalid future observation schema;
- failed drift report;
- live-like observations contradicting historical state evidence.

Retirement review triggers include:

- persistent quarantine beyond the configured threshold;
- stale evidence;
- retired state identity.

Quarantine and retirement reports are review artifacts only.

## Startup Boundary

The artifact always records:

```json
{
  "requires_explicit_future_approval": true,
  "startup_eligible_by_this_artifact": false,
  "paper_trading_started": false,
  "dry_run_trading_started": false,
  "live_trading_started": false,
  "freqtrade_trade_started": false,
  "exchange_order_placement": false,
  "process_control": false,
  "promotion_authorized_by_this_artifact": false
}
```

Paper observation is additional evidence only, not direct promotion.
