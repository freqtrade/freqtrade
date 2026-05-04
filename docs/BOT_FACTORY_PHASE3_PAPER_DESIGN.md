# Bot Factory Phase 3 Paper Readiness Design

This document covers the first Phase 3 increment: a no-startup readiness layer
for future paper trading.

It does not authorize `freqtrade trade`, paper startup, dry-run startup, live
trading, canary live trading, exchange order placement, API keys, leverage above
`1.0`, or shorting.

## Goal

The readiness layer answers one question from local evidence only:

```text
Is this strategy candidate eligible for a tightly scoped future paper run?
```

The answer is written as `pass`, `fail`, or `blocked`.

- `pass`: all local evidence, gates, static checks, long-only checks, config
  safety checks, and reviewer-note requirements pass.
- `fail`: the check completed, but candidate quality gates or review
  requirements do not pass.
- `blocked`: required evidence is missing or a safety/config/static issue makes
  paper readiness unsafe to evaluate.

## Scope

Allowed in this increment:

- Read historical Phase 2 FreqAI artifacts.
- Read walk-forward artifacts.
- Read training factory artifacts.
- Run or consume static safety checks.
- Inspect strategy source for long-only constraints.
- Inspect a proposed dry-run config without writing credential values.
- Write local JSON and Markdown readiness artifacts.

Not allowed in this increment:

- Starting any bot process.
- Running `freqtrade trade`.
- Paper, dry-run, canary, or live startup.
- API keys or secrets.
- Exchange order endpoints.
- Shorting or leverage above `1.0`.
- Promotion based on failed Phase 2 gates.

## CLI

```powershell
.\.venv\Scripts\python.exe scripts\bot_factory_check_paper_readiness.py `
  --config user_data\config_freqai_phase2_safe.json `
  --strategy LongOnlyFreqAIStrategy `
  --historical-dir data\freqai\LongOnlyFreqAIStrategy\phase2_safe_20250105_20250107 `
  --walk-forward-dir data\walk_forward\LongOnlyFreqAIStrategy\phase2_walk_forward_20250105_20250109 `
  --training-dir data\freqai_training\LongOnlyFreqAIStrategy\phase2_training_20250105_20250107 `
  --run-id phase3_readiness_20260503 `
  --reviewer-note "Phase 3 no-startup paper readiness check only; do not start paper trading."
```

This command reads local files and writes reports. It does not call Freqtrade.

## Required Evidence

The checker requires these local files:

- historical backtest `metrics.json`
- historical backtest `report.md`
- historical backtest `freqai_metadata.json`
- historical backtest `trades.csv`
- walk-forward `walk_forward_metrics.json`
- walk-forward `walk_forward_report.md`
- walk-forward child window `metrics.json`, `trades.csv`, and
  `freqai_metadata.json` for each recorded window
- training factory `training_manifest.json`
- training factory `training_report.md`
- training factory `freqai_backtest` child `metrics.json`, `trades.csv`, and
  `freqai_metadata.json`

Failed historical, walk-forward, or training recommendations produce `fail` and
block paper readiness. Missing top-level or child files produce `blocked`.
Historical, walk-forward child, and training child trade exports must contain no
short trades and no leverage above `1.0`.

## Config Safety

The proposed config must be dry-run only and sanitized:

- `dry_run=true`
- explicit strategy and timeframe
- explicit positive `max_open_trades`
- `max_open_trades <= 3`
- explicit positive numeric `stake_amount <= 1000`
- explicit positive numeric `dry_run_wallet <= 10000`
- `stake_amount <= dry_run_wallet`
- explicit non-empty `exchange.pair_whitelist`
- `force_entry_enable` absent or `false`
- `initial_state=stopped`
- explicit boolean `cancel_open_orders_on_exit`
- API server disabled
- no non-empty API keys, secrets, passwords, UIDs, tokens, or credential-like
  values
- no private environment variable references
- no leverage above `1.0`
- no private or order endpoint overrides

The generated `config_safety.json` contains sanitized metadata only. It records
credential key paths when unsafe values are present, but never writes the values.
It also records the accepted simulation limits used by the policy check.

## Long-Only Strategy Safety

The checker parses the strategy source and requires:

- `can_short = False`
- no `enter_short` or `exit_short` signal references
- no `leverage()` hook, or only statically capped returns at `1.0`
- historical exported trades contain no shorts and no leverage above `1.0`
- walk-forward child exported trades contain no shorts and no leverage above
  `1.0`
- training child exported trades contain no shorts and no leverage above `1.0`

The existing static checker is also run or consumed, and any static safety error
blocks readiness.

## Artifacts

Readiness artifacts are written under:

```text
data/paper_readiness/<strategy>/<run_id>/
```

Files:

- `paper_readiness.json`
- `paper_readiness_report.md`
- `candidate_artifacts.json`
- `config_safety.json`
- `command.txt`

Local JSON, CSV, and Markdown artifacts remain the source of truth. MLflow is
not involved in this readiness layer.

## Paper Run Planning Gate

The second Phase 3 increment adds a no-startup planner for a future paper run:

```powershell
.\.venv\Scripts\python.exe scripts\bot_factory_plan_paper_run.py `
  --readiness-json data\paper_readiness\LongOnlyFreqAIStrategy\phase3_readiness_20260503\paper_readiness.json `
  --strategy LongOnlyFreqAIStrategy `
  --run-id phase3_paper_plan_20260503 `
  --reviewer-note "Phase 3 paper run planning only; do not start paper trading."
```

This command reads an existing readiness JSON and writes plan artifacts only. It
does not call Freqtrade, start `freqtrade trade`, start paper trading, start
dry-run trading, or manage any bot process.

The planner can become `ready` only when all of these conditions are true:

- readiness JSON is a Phase 3 `paper_readiness` report
- readiness JSON is `pass`
- readiness JSON has no blockers or failures
- readiness JSON records no-startup, no-live, no-order-placement safety scope
- readiness metadata is sanitized
- readiness JSON records no API keys/secrets, no shorting, no leverage above
  `1.0`, and local artifacts as the source of truth
- the referenced dry-run config still exists locally
- the referenced config and strategy paths resolve inside the repository
  workspace
- `--confirm-paper` is explicitly supplied
- at least one `--reviewer-note` is supplied

Even when all gates pass, the planner records
`startup_authorized_by_this_command=false`; a separate explicit user request is
still required before any future start command is run.

Plan artifacts are written under:

```text
data/paper/<strategy>/<run_id>/
```

Files:

- `paper_run_plan.json`
- `paper_run_checklist.md`
- `stop_cleanup.md`
- `command.txt`

For the current `LongOnlyFreqAIStrategy` evidence, the planner returns
`status=blocked` because readiness remains `fail`; without a user-supplied
`--confirm-paper`, the explicit confirmation gate is also blocked. The generated
blocked plan contains no startup command preview.

## Paper Startup Preflight Gate

The third Phase 3 increment adds a no-startup startup preflight for a future
paper run:

```powershell
.\.venv\Scripts\python.exe scripts\bot_factory_prepare_paper_start.py `
  --plan-json data\paper\LongOnlyFreqAIStrategy\phase3_paper_plan_20260503\paper_run_plan.json `
  --strategy LongOnlyFreqAIStrategy `
  --run-id phase3_paper_startup_preflight_20260503 `
  --reviewer-note "Phase 3 paper startup preflight only; do not start paper trading."
```

This command reads an existing `paper_run_plan.json` and writes startup
preflight artifacts only. It does not call Freqtrade, start `freqtrade trade`,
start paper trading, start dry-run trading, or manage a bot process.

The startup preflight can become `ready` only when all of these conditions are
true:

- paper run plan is a Phase 3 `paper_run_plan`
- paper run plan is `ready`
- paper run plan has no blockers
- paper run plan future startup eligibility is true
- paper run plan still requires a separate explicit user request
- paper run plan still requires stop and cleanup review first
- paper run plan did not authorize startup by itself
- paper run plan includes a `freqtrade trade` command preview
- command preview uses a `freqtrade` executable and the `trade` subcommand
- command preview includes exactly one `--config`, `--strategy`, and
  `--strategy-path`
- command preview config and strategy path match the plan metadata, resolve
  inside the repository workspace, and exist locally
- command preview strategy matches the startup candidate
- the plan's no-startup, no-live, no-order-placement, no-secret, long-only
  safety scope is intact and records local artifacts as the source of truth
- `--confirm-paper-start` is explicitly supplied
- `--requested-start-command` exactly matches the plan command preview
- at least one `--reviewer-note` is supplied
- the referenced stop/cleanup and paper run checklist artifacts resolve inside
  the repository workspace and exist locally

Even when this preflight is `ready`, it records
`startup_authorized_by_this_command=false` and `startup_executed=false`. A later
explicit execution step would still be required before any process could start.

Startup preflight artifacts are written under:

```text
data/paper/<strategy>/<run_id>/
```

Files:

- `paper_startup_preflight.json`
- `paper_startup_preflight_report.md`
- `process_metadata_template.json`
- `status_snapshot_template.json`
- `start_command_preview.txt`
- `command.txt`

For the current `LongOnlyFreqAIStrategy` evidence, startup preflight returns
`status=blocked` because the existing paper run plan is still blocked, readiness
is still `fail`, no startup command preview exists, and no explicit
`--confirm-paper-start` or exact requested command was supplied. The generated
`start_command_preview.txt` is empty.

## Paper Monitoring/Status Schema Gate

The fourth Phase 3 increment adds a no-startup, no-process-control monitoring
schema planner:

```powershell
.\.venv\Scripts\python.exe scripts\bot_factory_plan_paper_monitoring.py `
  --startup-preflight-json data\paper\LongOnlyFreqAIStrategy\phase3_paper_startup_preflight_20260503\paper_startup_preflight.json `
  --strategy LongOnlyFreqAIStrategy `
  --run-id phase3_paper_monitoring_plan_20260503 `
  --reviewer-note "Phase 3 paper monitoring schema planning only; do not start, stop, poll, or manage paper trading."
```

This command reads an existing `paper_startup_preflight.json` and writes local
schema artifacts only. It does not call Freqtrade, start `freqtrade trade`,
start paper trading, stop a process, poll a process, or manage a bot process.

The monitoring plan can become `ready` only when all of these conditions are
true:

- startup preflight is a Phase 3 `paper_startup_preflight`
- startup preflight strategy matches the monitoring candidate
- startup preflight is `ready`
- startup preflight has no blockers
- startup preflight startup eligibility is true
- startup preflight records no startup execution, no process start, and no
  startup authorization by the preflight command
- startup preflight still requires a separate execution step
- process metadata and status snapshot template paths resolve inside the
  repository workspace and exist locally
- stdout, stderr, and paper metrics paths resolve inside the repository
  workspace
- startup preflight no-startup, no-live, no-order-placement, no-secret,
  long-only, local-artifact safety scope is intact
- status snapshot template records no startup execution
- at least one `--reviewer-note` is supplied

Even when the monitoring plan is `ready`, it records:

- `monitoring_started=false`
- `status_polling_started=false`
- `process_control=false`
- `process_stop_started=false`

Monitoring schema artifacts are written under:

```text
data/paper/<strategy>/<run_id>/
```

Files:

- `paper_monitoring_plan.json`
- `paper_monitoring_report.md`
- `status_snapshot_schema.json`
- `paper_metrics_schema.json`
- `process_metadata_schema.json`
- `command.txt`

The status snapshot schema defines local status values such as `not_started`,
`starting`, `running`, `stopping`, `stopped`, and `failed`, plus required safety
flags for no live trading and no exchange order placement. The paper metrics
schema is limited to local paper artifact summaries, trade counts, profit/risk
fields, and safety scope. The process metadata schema requires local stdout,
stderr, status snapshot, and paper metrics paths.

For the current `LongOnlyFreqAIStrategy` evidence, monitoring planning returns
`status=blocked` because the existing startup preflight is still blocked, still
has blockers, and startup eligibility is false. It still writes schema artifacts
for future review.

## Paper Stop/Cleanup Planning Gate

The fifth Phase 3 increment adds a no-process-control stop and cleanup planner:

```powershell
.\.venv\Scripts\python.exe scripts\bot_factory_plan_paper_stop_cleanup.py `
  --monitoring-plan-json data\paper\LongOnlyFreqAIStrategy\phase3_paper_monitoring_plan_20260503\paper_monitoring_plan.json `
  --strategy LongOnlyFreqAIStrategy `
  --run-id phase3_paper_stop_cleanup_plan_20260503 `
  --reviewer-note "Phase 3 paper stop/cleanup planning only; do not start, stop, poll, terminate, or manage paper trading."
```

This command reads an existing `paper_monitoring_plan.json` and writes local
future stop request and cleanup review artifacts only. It does not call
Freqtrade, start `freqtrade trade`, start paper trading, stop a process, poll a
process, terminate a process, or manage a bot process.

The stop/cleanup plan can become `ready` only when all of these conditions are
true:

- monitoring plan is a Phase 3 `paper_monitoring_plan`
- monitoring plan strategy matches the stop/cleanup candidate
- monitoring plan is `ready`
- monitoring plan has no blockers
- monitoring eligibility is true
- monitoring plan records no monitoring start, no status polling, no process
  control, and no process stop
- monitoring plan requires process metadata, status snapshot paths, stdout and
  stderr logs, and paper metrics
- process metadata and status snapshot template paths resolve inside the
  repository workspace and exist locally
- stdout, stderr, and paper metrics paths resolve inside the repository
  workspace
- monitoring schemas include stop-relevant status, metrics, process identity,
  and local log fields
- monitoring plan no-startup, no-live, no-order-placement, no-secret,
  long-only, local-artifact safety scope is intact
- at least one `--reviewer-note` is supplied

Even when the stop/cleanup plan is `ready`, it records:

- `stop_executed=false`
- `cleanup_executed=false`
- `process_control=false`
- `process_stop_started=false`
- `status_polling_started=false`
- `stop_authorized_by_this_command=false`
- `cleanup_authorized_by_this_command=false`

Stop/cleanup planning artifacts are written under:

```text
data/paper/<strategy>/<run_id>/
```

Files:

- `paper_stop_cleanup_plan.json`
- `paper_stop_cleanup_report.md`
- `stop_request_schema.json`
- `cleanup_checklist.md`
- `command.txt`

For the current `LongOnlyFreqAIStrategy` evidence, stop/cleanup planning
returns `status=blocked` because the existing monitoring plan is still blocked,
still has blockers, and monitoring eligibility is false. It still writes local
stop request schema and cleanup checklist artifacts for future review.

## Paper Start Execution Request Gate

The sixth Phase 3 increment adds a no-startup, no-process-control execution
request gate:

```powershell
.\.venv\Scripts\python.exe scripts\bot_factory_request_paper_start.py `
  --readiness-json data\paper_readiness\LongOnlyFreqAIStrategy\phase3_readiness_20260503\paper_readiness.json `
  --plan-json data\paper\LongOnlyFreqAIStrategy\phase3_paper_plan_20260503\paper_run_plan.json `
  --startup-preflight-json data\paper\LongOnlyFreqAIStrategy\phase3_paper_startup_preflight_20260503\paper_startup_preflight.json `
  --monitoring-plan-json data\paper\LongOnlyFreqAIStrategy\phase3_paper_monitoring_plan_20260503\paper_monitoring_plan.json `
  --stop-cleanup-plan-json data\paper\LongOnlyFreqAIStrategy\phase3_paper_stop_cleanup_plan_20260503\paper_stop_cleanup_plan.json `
  --strategy LongOnlyFreqAIStrategy `
  --run-id phase3_paper_execution_request_20260503 `
  --reviewer-note "Phase 3 paper execution request planning only; do not start, stop, poll, terminate, clean up, or manage paper trading."
```

This command reads the existing readiness, plan, startup preflight, monitoring,
and stop/cleanup artifacts and writes local request artifacts only. It does not
call Freqtrade, start `freqtrade trade`, start paper trading, stop a process,
poll a process, terminate a process, clean up artifacts, or manage a bot
process.

The execution request can become `ready` only when all of these conditions are
true:

- readiness is a Phase 3 `paper_readiness` report for the same strategy
- readiness is `pass` and has no blockers or failures
- paper run plan is a Phase 3 `paper_run_plan` for the same strategy
- paper run plan is `ready`, has no blockers, and future startup eligibility is
  true
- startup preflight is a Phase 3 `paper_startup_preflight` for the same
  strategy
- startup preflight is `ready`, has no blockers, startup eligibility is true,
  and records no startup execution
- monitoring plan is a Phase 3 `paper_monitoring_plan` for the same strategy
- monitoring plan is `ready`, has no blockers, monitoring eligibility is true,
  and records no process control
- stop/cleanup plan is a Phase 3 `paper_stop_cleanup_plan` for the same
  strategy
- stop/cleanup plan is `ready`, has no blockers, stop/cleanup eligibility is
  true, preserves operator-review guardrails, and records no process control
- the artifact chain paths match exactly: plan to readiness, preflight to plan,
  monitoring to preflight, and stop/cleanup to monitoring
- startup preflight, monitoring, and stop/cleanup runtime paths agree and
  resolve inside the repository workspace
- process metadata and status snapshot templates exist locally
- stdout, stderr, and paper metrics paths resolve inside the repository
  workspace
- paper run plan and startup preflight command previews match exactly
- all upstream safety scopes remain no-startup or no-process-control,
  no-live, no-order-placement, no-secret, long-only, and local-artifact based
- `--confirm-paper-execution` is explicitly supplied
- `--requested-start-command` exactly matches the startup preflight command
  preview
- at least one `--reviewer-note` is supplied

Even when the request is `ready`, it records:

- `startup_executed=false`
- `process_started=false`
- `process_control=false`
- `status_polling_started=false`
- `process_stop_started=false`
- `cleanup_executed=false`
- `startup_authorized_by_this_command=false`
- `requires_separate_process_executor=true`

Execution request artifacts are written under:

```text
data/paper/<strategy>/<run_id>/
```

Files:

- `paper_execution_request.json`
- `paper_execution_request_report.md`
- `execution_manifest_template.json`
- `start_command_request.txt`
- `command.txt`

For the current `LongOnlyFreqAIStrategy` evidence, the execution request returns
`status=blocked` because readiness remains `fail`, the upstream
plan/preflight/monitoring/stop-cleanup artifacts remain `blocked`, no startup
command preview exists while readiness is failed, and no explicit
`--confirm-paper-execution` or exact requested start command was supplied. It
still writes request artifacts for review.

## Paper Process Executor Planning Gate

The seventh Phase 3 increment adds a no-startup, no-process-control process
executor planning gate:

```powershell
.\.venv\Scripts\python.exe scripts\bot_factory_plan_paper_executor.py `
  --execution-request-json data\paper\LongOnlyFreqAIStrategy\phase3_paper_execution_request_20260503\paper_execution_request.json `
  --strategy LongOnlyFreqAIStrategy `
  --run-id phase3_paper_executor_plan_20260503 `
  --reviewer-note "Phase 3 paper process executor planning only; do not start, stop, poll, terminate, clean up, or manage paper trading."
```

This command reads an existing `paper_execution_request.json` and writes a
local executor manifest draft plus operator checklist only. It does not call
Freqtrade, start `freqtrade trade`, start paper trading, stop a process, poll a
process, terminate a process, clean up artifacts, or manage a bot process.

The process executor plan can become `ready` only when all of these conditions
are true:

- execution request is a Phase 3 `paper_execution_request` for the same
  strategy
- execution request is `ready`, has no blockers, and execution eligibility is
  true
- execution request still requires a separate process executor
- execution request and its manifest record no startup execution, no process
  start, no process control, no status polling, no process stop, and no cleanup
- execution request command preview exists, uses `freqtrade trade`, includes
  exactly one `--config`, `--strategy`, and `--strategy-path`, and targets the
  same strategy
- execution request expected/requested command strings, manifest command, and
  newly supplied `--requested-start-command` all match exactly
- process metadata, status snapshot, stdout, stderr, paper metrics, execution
  manifest template, and start command request paths resolve inside the
  repository workspace; required template/request files exist locally
- execution request safety scope remains no-startup, no-process-control,
  no-live, no-order-placement, no-secret, long-only, and local-artifact based
- `--confirm-process-executor-plan` is explicitly supplied
- at least one `--reviewer-note` is supplied

Even when the process executor plan is `ready`, it records:

- `startup_executed=false`
- `process_started=false`
- `process_control=false`
- `status_polling_started=false`
- `process_stop_started=false`
- `cleanup_executed=false`
- `start_authorized_by_this_command=false`
- `requires_explicit_user_start_after_plan=true`

Process executor planning artifacts are written under:

```text
data/paper/<strategy>/<run_id>/
```

Files:

- `paper_process_executor_plan.json`
- `paper_process_executor_report.md`
- `process_executor_manifest.json`
- `operator_start_checklist.md`
- `start_command_review.txt`
- `command.txt`

For the current `LongOnlyFreqAIStrategy` evidence, process executor planning
returns `status=blocked` because the execution request remains `blocked`, has
blockers, is not eligible, no startup command preview exists while readiness is
failed, no exact requested start command was supplied, and
`--confirm-process-executor-plan` was not supplied. It still verifies local
runtime paths, no-startup/no-process-control safety scope, and writes local
executor planning artifacts for review.

## Paper Runtime Artifact Validation Gate

The eighth Phase 3 increment adds a no-process-control runtime artifact
validator:

```powershell
.\.venv\Scripts\python.exe scripts\bot_factory_validate_paper_runtime.py `
  --process-executor-plan-json data\paper\LongOnlyFreqAIStrategy\phase3_paper_executor_plan_20260503\paper_process_executor_plan.json `
  --process-metadata-json data\paper\LongOnlyFreqAIStrategy\phase3_paper_startup_preflight_20260503\process_metadata_template.json `
  --status-snapshot-json data\paper\LongOnlyFreqAIStrategy\phase3_paper_startup_preflight_20260503\status_snapshot_template.json `
  --stdout-log data\paper\LongOnlyFreqAIStrategy\phase3_paper_startup_preflight_20260503\logs\stdout.log `
  --stderr-log data\paper\LongOnlyFreqAIStrategy\phase3_paper_startup_preflight_20260503\logs\stderr.log `
  --paper-metrics-json data\paper\LongOnlyFreqAIStrategy\phase3_paper_startup_preflight_20260503\paper_metrics.json `
  --strategy LongOnlyFreqAIStrategy `
  --run-id phase3_paper_runtime_validation_20260504 `
  --reviewer-note "Phase 3 paper runtime artifact validation only; do not start, stop, poll, terminate, clean up, or manage paper trading."
```

This command reads a process executor plan and supplied local runtime artifact
paths only. It does not call Freqtrade, start `freqtrade trade`, start paper
trading, stop a process, poll a process, terminate a process, clean up
artifacts, or manage a bot process. Missing runtime artifacts are reported as
blocked checks and still produce validation artifacts.

Runtime validation can pass only when all of these conditions are true:

- process executor plan is a Phase 3 `paper_process_executor_plan` for the same
  strategy
- process executor plan is `ready`, has no blockers, is eligible, preserves
  no-plan-side startup/process-control flags, and still requires a separate
  explicit user start after planning
- supplied process metadata, status snapshot, stdout log, stderr log, and paper
  metrics paths resolve inside the repository workspace and exist locally
- supplied runtime paths match the process executor plan and executor manifest
- process metadata, status snapshot, and paper metrics include the required
  schema fields
- process metadata, status snapshot, and paper metrics agree on strategy and on
  the process executor plan run ID
- process metadata command matches the reviewed executor plan command
- status snapshot and paper metrics status values match and use known local
  status values
- paper metrics use `source=local_paper_artifacts`, contain internally
  consistent trade counts, and keep local artifacts as source of truth
- runtime metadata contains no non-empty API keys, secrets, tokens, UIDs,
  passwords, or private environment references
- runtime artifacts record no live/canary trading, no exchange order placement,
  no leverage above `1.0`, no shorting, and no process-control/poll/stop/cleanup
  execution by the validation path
- at least one `--reviewer-note` is supplied

Even when validation passes, it records:

- `bot_startup_performed_by_validator=false`
- `polling_performed_by_validator=false`
- `stop_performed_by_validator=false`
- `process_control=false`
- `status_polling_started=false`
- `process_stop_started=false`
- `cleanup_executed=false`

Runtime validation artifacts are written under:

```text
data/paper/<strategy>/<run_id>/
```

Files:

- `paper_runtime_validation.json`
- `paper_runtime_validation_report.md`
- `runtime_artifacts_manifest.json`
- `command.txt`

For the current `LongOnlyFreqAIStrategy` evidence, runtime validation returns
`status=blocked` because the process executor plan is still blocked, no reviewed
command preview exists, stdout/stderr logs and paper metrics do not exist, and
the existing template metadata/status snapshot run IDs belong to the blocked
startup preflight rather than a ready process executor plan. It still verifies
available local template paths and records that no process control was performed
by the validator.

## Paper/Backtest Drift Report

The ninth Phase 3 increment adds a no-process-control drift report that compares
future local paper metrics against prior historical, walk-forward, and training
evidence:

```powershell
.\.venv\Scripts\python.exe scripts\bot_factory_report_paper_drift.py `
  --historical-metrics-json data\freqai\LongOnlyFreqAIStrategy\phase2_safe_20250105_20250107\metrics.json `
  --walk-forward-metrics-json data\walk_forward\LongOnlyFreqAIStrategy\phase2_walk_forward_20250105_20250109\walk_forward_metrics.json `
  --training-manifest-json data\freqai_training\LongOnlyFreqAIStrategy\phase2_training_20250105_20250107\training_manifest.json `
  --paper-runtime-validation-json data\paper\LongOnlyFreqAIStrategy\phase3_paper_runtime_validation_20260504\paper_runtime_validation.json `
  --strategy LongOnlyFreqAIStrategy `
  --run-id phase3_paper_drift_report_20260504 `
  --reviewer-note "Phase 3 paper/backtest drift reporting only; do not start, stop, poll, terminate, clean up, promote, or manage paper trading."
```

This command reads local JSON artifacts only. It does not call Freqtrade, start
`freqtrade trade`, start paper trading, stop a process, poll a process,
terminate a process, clean up artifacts, promote a strategy, or manage a bot
process.

The drift report can pass only when all of these conditions are true:

- historical metrics, walk-forward metrics, training manifest, paper runtime
  validation, and paper metrics paths resolve inside the repository workspace
  and exist locally
- historical metrics, walk-forward metrics, training manifest, runtime
  validation, and paper metrics all reference the same strategy
- walk-forward metrics are completed Phase 2 evidence
- training manifest is a completed Phase 2 FreqAI training artifact
- paper runtime validation is a passing Phase 3 `paper_runtime_validation`
  artifact
- the paper metrics path supplied to the drift reporter matches the exact
  `paper_runtime_validation.input_paths.paper_metrics` artifact consumed by the
  runtime validator
- paper metrics use `source=local_paper_artifacts` and match the runtime
  validation process executor plan run ID
- paper metrics include numeric total return, max drawdown, and trade count
- walk-forward and training recommendations are `pass`
- paper return and drawdown drift remain inside configured thresholds
- runtime validation and paper metrics safety scopes remain sanitized,
  long-only, no-live, no-order-placement, no-leverage-above-`1.0`, no-shorting,
  no-process-control, and local-artifact based
- historical metrics, walk-forward metrics, training manifest, runtime
  validation, and paper metrics contain no non-empty credential-like metadata or
  private environment references
- at least one `--reviewer-note` is supplied

Even when the drift report passes, it records:

- `paper_promotion_eligible=false`
- `promotion_authorized_by_this_command=false`
- `process_control=false`
- `status_polling_started=false`
- `process_stop_started=false`
- `cleanup_executed=false`

Drift artifacts are written under:

```text
data/paper/<strategy>/<run_id>/
```

Files:

- `paper_drift_report.json`
- `paper_drift_report.md`
- `drift_metrics.json`
- `command.txt`

For the current `LongOnlyFreqAIStrategy` evidence, drift reporting returns
`status=blocked` because the runtime validation artifact is still blocked and
the referenced paper metrics file does not exist. It also records failed prior
walk-forward and training recommendations, so the report cannot support
promotion review. It still writes local drift report artifacts and records that
no process control or promotion was performed.

## Current Limitation

The verified `LongOnlyFreqAIStrategy` Phase 2 artifacts are pipeline
verification artifacts. Their recent historical, walk-forward, and training
gates fail, so the readiness checker must return `fail` for that candidate.
The hardened evidence and config checks pass for the current local artifacts,
so the current `fail` status is driven by those Phase 2 quality gates rather
than missing child evidence or unsafe config.

A future human-approved infrastructure-only smoke test is a separate path and
is not implemented here. No bot startup, monitoring loop, running paper process,
status polling implementation, process stop implementation, cleanup executor,
process-control executor, or paper/live promotion path is implemented by the
readiness checker, planner, startup preflight, monitoring schema planner,
stop/cleanup planner, execution request gate, process executor planning gate,
runtime artifact validator, or paper/backtest drift reporter.
