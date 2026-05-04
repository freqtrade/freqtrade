# Bot Factory Phase 3 Next Agent Prompt

Use the following prompt for the next coding agent.

````markdown
Continue Bot Factory Phase 3 from the current no-startup/no-process-control state.

First command, required:

```powershell
git status --short --untracked-files=all
```

Read these files before making changes:

- `AGENTS.md`
- `docs/BOT_FACTORY_MVP_TODO.md`
- `docs/BOT_FACTORY_PHASE3_NEXT_AGENT_PROMPT.md`
- `docs/BOT_FACTORY_PHASE3_AGENT_INSTRUCTIONS.md`
- `docs/BOT_FACTORY_PHASE3_PAPER_DESIGN.md`
- `docs/BOT_FACTORY_PHASE2_RUNBOOK.md`
- `docs/BOT_FACTORY_PHASE2_AGENT_INSTRUCTIONS.md`

Current branch:

- `develop`

Current worktree context:

- This handoff may include uncommitted Phase 3 no-startup/no-process-control changes:
  - `docs/BOT_FACTORY_MVP_TODO.md`
  - `docs/BOT_FACTORY_PHASE3_NEXT_AGENT_PROMPT.md`
  - `docs/BOT_FACTORY_PHASE3_PAPER_DESIGN.md`
  - `freqtrade_ext/bot_factory/paper_plan.py`
  - `freqtrade_ext/bot_factory/paper_startup.py`
  - `freqtrade_ext/bot_factory/paper_monitoring.py`
  - `freqtrade_ext/bot_factory/paper_stop_cleanup.py`
  - `freqtrade_ext/bot_factory/paper_execution.py`
  - `freqtrade_ext/bot_factory/paper_executor.py`
  - `freqtrade_ext/bot_factory/paper_runtime.py`
  - `freqtrade_ext/bot_factory/paper_drift.py`
  - `scripts/bot_factory_plan_paper_run.py`
  - `scripts/bot_factory_prepare_paper_start.py`
  - `scripts/bot_factory_plan_paper_monitoring.py`
  - `scripts/bot_factory_plan_paper_stop_cleanup.py`
  - `scripts/bot_factory_request_paper_start.py`
  - `scripts/bot_factory_plan_paper_executor.py`
  - `scripts/bot_factory_validate_paper_runtime.py`
  - `scripts/bot_factory_report_paper_drift.py`
  - `tests/test_bot_factory.py`
  - `data/paper/LongOnlyFreqAIStrategy/phase3_paper_plan_20260503/`
  - `data/paper/LongOnlyFreqAIStrategy/phase3_paper_startup_preflight_20260503/`
  - `data/paper/LongOnlyFreqAIStrategy/phase3_paper_monitoring_plan_20260503/`
  - `data/paper/LongOnlyFreqAIStrategy/phase3_paper_stop_cleanup_plan_20260503/`
  - `data/paper/LongOnlyFreqAIStrategy/phase3_paper_execution_request_20260503/`
  - `data/paper/LongOnlyFreqAIStrategy/phase3_paper_executor_plan_20260503/`
  - `data/paper/LongOnlyFreqAIStrategy/phase3_paper_runtime_validation_20260504/`
  - `data/paper/LongOnlyFreqAIStrategy/phase3_paper_drift_report_20260504/`
- Existing readiness artifacts may exist under:
  - `data/paper_readiness/LongOnlyFreqAIStrategy/phase3_readiness_20260503/`
- Known Windows ACL warnings may appear in `git status`:
  - `.codex_tmp/pytest-of-yoro4/`
  - `bot_factory_pytest_tmp/`
  - `codex_tmp/pytest/`
- Do not remove ACL-blocked directories unless normal file access is available
  and the resolved path is safely inside the repository.

Handoff priority for the next agent:

- Treat the no-startup/no-process-control Phase 3 paper scaffolding as complete
  unless a concrete bug or missing validation gap is found.
- Do not implement a real paper process starter, monitor, stopper, cleanup
  executor, or promotion path from the current artifacts. The current
  `LongOnlyFreqAIStrategy` chain is blocked because readiness is `fail`,
  runtime validation is `blocked`, paper metrics are missing, and walk-forward
  plus training recommendations are still `fail`.
- The most useful safe next step is to improve or replace the strategy
  candidate using Phase 2-safe historical work only: dependency checks, static
  checks, OHLCV quality checks, FreqAI validation, historical backtesting,
  walk-forward evaluation, training factory orchestration, and local reports.
- If the user explicitly asks to continue Phase 3 code instead, restrict work
  to hardening existing local-artifact gates or their schemas/docs. Do not
  start, stop, poll, terminate, clean up, promote, or manage a bot process.
- Add `docs/BOT_FACTORY_PHASE3_RUNBOOK.md` only after an actual paper path has
  been explicitly requested, preflight-approved, implemented, verified, and
  documented.

Phase 3 remaining work:

- The unfinished Phase 3 item is `Paper trading deployment`. The current code
  implements local-artifact safety gates only; it does not yet implement a real
  paper process starter, running paper process monitor, status poller, stop
  executor, cleanup executor, or promotion path.
- The first blocker is strategy evidence, not process code. The current
  `LongOnlyFreqAIStrategy` artifacts are pipeline verification artifacts and
  still fail historical, walk-forward, and training recommendations. The next
  useful step is to improve or replace the strategy candidate using Phase
  2-safe historical work only until a candidate can produce passing historical,
  walk-forward, and training artifacts.
- After a stronger candidate exists, rerun the Phase 3 local-artifact chain:
  paper readiness must be `pass`, and paper run plan, startup preflight,
  monitoring plan, stop/cleanup plan, execution request, and process executor
  plan must all be `ready` with matching artifact paths and reviewer notes.
- Only after every upstream gate is ready and the user explicitly requests the
  exact reviewed paper start command should a later agent consider a real
  process executor. That executor must be minimal, dry-run only, no-secret,
  long-only, no leverage above `1.0`, no shorting, and must write sanitized
  process metadata, stdout/stderr logs, status snapshot, and paper metrics.
- After a real paper process path exists, remaining Phase 3 implementation is:
  local status polling/monitoring, local paper metrics capture, explicit
  stop/cleanup execution with artifact preservation, paper/backtest drift
  reporting from the generated paper metrics, and only then a Phase 3 runbook.
- Do not mark `Paper trading deployment` complete until an explicitly
  requested, preflight-approved paper path has been implemented, verified, and
  documented. Do not create `docs/BOT_FACTORY_PHASE3_RUNBOOK.md` before that
  verified path exists.

Completed Phase 3 work:

- No-startup paper readiness preflight is implemented:
  - `freqtrade_ext/bot_factory/paper.py`
  - `scripts/bot_factory_check_paper_readiness.py`
  - docs in `docs/BOT_FACTORY_PHASE3_PAPER_DESIGN.md`
  - focused tests in `tests/test_bot_factory.py`
- Readiness preflight is hardened:
  - requires historical metrics/report/metadata/trades
  - requires walk-forward metrics/report and child window
    `metrics.json`, `trades.csv`, `freqai_metadata.json`
  - requires training manifest/report and training `freqai_backtest` child
    `metrics.json`, `trades.csv`, `freqai_metadata.json`
  - verifies historical, walk-forward child, and training child trade exports
    contain no shorts and no leverage above `1.0`
  - validates dry-run config safety, including `dry_run=true`,
    `initial_state=stopped`, `force_entry_enable=false`, explicit
    `cancel_open_orders_on_exit`, no credentials, no private env references,
    no order endpoint overrides, conservative `max_open_trades`,
    `stake_amount`, and `dry_run_wallet`
- No-startup paper run planner is implemented:
  - `freqtrade_ext/bot_factory/paper_plan.py`
  - `scripts/bot_factory_plan_paper_run.py`
  - consumes `paper_readiness.json`
  - requires a Phase 3 `paper_readiness` source
  - blocks unless readiness is `pass`
  - blocks if readiness has blockers or failures
  - verifies readiness safety scope is no-startup/no-live/no-order-placement
  - verifies readiness scope is no-secret, long-only, no leverage above `1.0`,
    and keeps local artifacts as the source of truth
  - requires the referenced dry-run config file to exist
  - requires config and strategy paths to resolve inside the repository
    workspace
  - requires `--confirm-paper` and at least one reviewer note before a plan can
    become `ready`
  - writes `paper_run_plan.json`, `paper_run_checklist.md`,
    `stop_cleanup.md`, and `command.txt`
  - records `startup_authorized_by_this_command=false`
  - never starts `freqtrade trade` or any bot process
- No-startup paper startup preflight is implemented:
  - `freqtrade_ext/bot_factory/paper_startup.py`
  - `scripts/bot_factory_prepare_paper_start.py`
  - consumes `paper_run_plan.json`
  - requires a Phase 3 `paper_run_plan` source
  - blocks unless the paper run plan is `ready`
  - blocks unless the plan has no blockers and future startup eligibility is
    true
  - verifies the plan still requires a separate explicit user request, stop
    and cleanup review, no plan-side startup authorization, and no-startup /
    no-live / no-order-placement / no-secret / long-only safety scope
  - verifies the plan keeps local artifacts as the source of truth
  - verifies the command preview uses `freqtrade trade`, has exactly one
    `--config`, `--strategy`, and `--strategy-path`, matches the plan metadata,
    resolves to local workspace paths, and targets the startup candidate
  - verifies referenced stop/cleanup and paper run checklist artifacts resolve
    inside the workspace and exist locally
  - requires `--confirm-paper-start`, exact `--requested-start-command`, and
    at least one reviewer note before startup preflight can become `ready`
  - writes `paper_startup_preflight.json`,
    `paper_startup_preflight_report.md`, `process_metadata_template.json`,
    `status_snapshot_template.json`, `start_command_preview.txt`, and
    `command.txt`
  - records `startup_executed=false` and
    `startup_authorized_by_this_command=false`
  - never starts `freqtrade trade` or any bot process
- No-startup paper monitoring/status schema planner is implemented:
  - `freqtrade_ext/bot_factory/paper_monitoring.py`
  - `scripts/bot_factory_plan_paper_monitoring.py`
  - consumes `paper_startup_preflight.json`
  - requires a Phase 3 `paper_startup_preflight` source
  - blocks unless startup preflight is `ready`, has no blockers, and startup
    eligibility is true
  - verifies startup preflight records no startup execution, no process start,
    no startup authorization by the preflight command, and a separate execution
    requirement
  - verifies process metadata and status snapshot templates resolve inside the
    workspace and exist locally
  - verifies stdout, stderr, and paper metrics paths resolve inside the
    workspace
  - verifies no-startup / no-live / no-order-placement / no-secret /
    long-only / local-artifact safety scope
  - requires at least one reviewer note before monitoring schemas can become
    `ready`
  - writes `paper_monitoring_plan.json`, `paper_monitoring_report.md`,
    `status_snapshot_schema.json`, `paper_metrics_schema.json`,
    `process_metadata_schema.json`, and `command.txt`
  - records `monitoring_started=false`, `status_polling_started=false`,
    `process_control=false`, and `process_stop_started=false`
  - never starts, stops, polls, or manages `freqtrade trade` or any bot process
- No-process-control paper stop/cleanup planner is implemented:
  - `freqtrade_ext/bot_factory/paper_stop_cleanup.py`
  - `scripts/bot_factory_plan_paper_stop_cleanup.py`
  - consumes `paper_monitoring_plan.json`
  - requires a Phase 3 `paper_monitoring_plan` source
  - blocks unless monitoring plan is `ready`, has no blockers, and monitoring
    eligibility is true
  - verifies the monitoring plan records no monitoring start, no status polling,
    no process control, and no process stop
  - verifies process metadata and status snapshot template paths resolve inside
    the workspace and exist locally
  - verifies stdout, stderr, and paper metrics paths resolve inside the
    workspace
  - verifies monitoring schemas include stop-relevant status, metrics, process
    identity, and local log fields
  - verifies no-startup / no-live / no-order-placement / no-secret /
    long-only / local-artifact safety scope
  - requires at least one reviewer note before stop/cleanup planning can become
    `ready`
  - writes `paper_stop_cleanup_plan.json`, `paper_stop_cleanup_report.md`,
    `stop_request_schema.json`, `cleanup_checklist.md`, and `command.txt`
  - records `stop_executed=false`, `cleanup_executed=false`,
    `process_control=false`, `process_stop_started=false`,
    `status_polling_started=false`, `stop_authorized_by_this_command=false`,
    and `cleanup_authorized_by_this_command=false`
  - never starts, stops, polls, terminates, or manages `freqtrade trade` or any
    bot process
- No-startup/no-process-control paper start execution request gate is implemented:
  - `freqtrade_ext/bot_factory/paper_execution.py`
  - `scripts/bot_factory_request_paper_start.py`
  - consumes `paper_readiness.json`, `paper_run_plan.json`,
    `paper_startup_preflight.json`, `paper_monitoring_plan.json`, and
    `paper_stop_cleanup_plan.json`
  - requires matching Phase 3 sources for the same strategy
  - blocks unless readiness is `pass`
  - blocks unless the paper run plan, startup preflight, monitoring plan, and
    stop/cleanup plan are all `ready`, have no blockers, and have eligible
    future-start/monitoring/stop-cleanup flags
  - verifies artifact-chain path consistency from readiness through
    stop/cleanup
  - verifies startup preflight, monitoring, and stop/cleanup runtime paths
    match and resolve inside the workspace
  - verifies process metadata and status snapshot templates exist locally
  - verifies the plan and startup preflight command previews match exactly
  - verifies no-startup / no-process-control / no-live / no-order-placement /
    no-secret / long-only / local-artifact safety scope
  - requires `--confirm-paper-execution`, exact `--requested-start-command`,
    and at least one reviewer note before the request can become `ready`
  - writes `paper_execution_request.json`,
    `paper_execution_request_report.md`, `execution_manifest_template.json`,
    `start_command_request.txt`, and `command.txt`
  - records `startup_executed=false`, `process_started=false`,
    `process_control=false`, `status_polling_started=false`,
    `process_stop_started=false`, `cleanup_executed=false`, and
    `startup_authorized_by_this_command=false`
  - never starts, stops, polls, terminates, cleans up, or manages
    `freqtrade trade` or any bot process
- No-startup/no-process-control paper process executor planning gate is
  implemented:
  - `freqtrade_ext/bot_factory/paper_executor.py`
  - `scripts/bot_factory_plan_paper_executor.py`
  - consumes `paper_execution_request.json`
  - requires a Phase 3 `paper_execution_request` source for the same strategy
  - blocks unless the execution request is `ready`, has no blockers, and
    execution request eligibility is true
  - verifies the execution request still requires a separate process executor
  - verifies the execution request and manifest record no startup execution, no
    process start, no process control, no status polling, no process stop, and
    no cleanup
  - verifies the command preview exists, uses `freqtrade trade`, includes one
    `--config`, `--strategy`, and `--strategy-path`, targets the same strategy,
    and exactly matches the execution request expected/requested command,
    execution manifest command, and newly supplied `--requested-start-command`
  - verifies process metadata, status snapshot, stdout, stderr, paper metrics,
    execution manifest template, and start command request paths resolve inside
    the workspace; required template/request files exist locally
  - verifies no-startup / no-process-control / no-live / no-order-placement /
    no-secret / long-only / local-artifact safety scope
  - requires `--confirm-process-executor-plan` and at least one reviewer note
    before the plan can become `ready`
  - writes `paper_process_executor_plan.json`,
    `paper_process_executor_report.md`, `process_executor_manifest.json`,
    `operator_start_checklist.md`, `start_command_review.txt`, and
    `command.txt`
  - records `startup_executed=false`, `process_started=false`,
    `process_control=false`, `status_polling_started=false`,
    `process_stop_started=false`, `cleanup_executed=false`,
    `start_authorized_by_this_command=false`, and
    `requires_explicit_user_start_after_plan=true`
  - never starts, stops, polls, terminates, cleans up, or manages
    `freqtrade trade` or any bot process
- No-process-control paper runtime artifact validation gate is implemented:
  - `freqtrade_ext/bot_factory/paper_runtime.py`
  - `scripts/bot_factory_validate_paper_runtime.py`
  - consumes `paper_process_executor_plan.json`, process metadata JSON, status
    snapshot JSON, stdout/stderr log paths, and paper metrics JSON
  - blocks unless the process executor plan is `ready`, has no blockers, is
    eligible, preserves no plan-side startup/process-control flags, and still
    requires a separate explicit user start after planning
  - verifies supplied runtime paths resolve inside the workspace, exist locally,
    and match the executor plan plus executor manifest
  - verifies required runtime schema fields, known local status values,
    consistent paper trade counts, matching strategy/run IDs, and command
    consistency
  - verifies runtime metadata contains no non-empty credential values or private
    environment references
  - verifies no live/canary trading, no exchange order placement, no leverage
    above `1.0`, no shorting, no process control, no status polling, no process
    stop, and no cleanup execution is recorded by the validation path
  - requires at least one reviewer note before validation can pass
  - writes `paper_runtime_validation.json`,
    `paper_runtime_validation_report.md`, `runtime_artifacts_manifest.json`,
    and `command.txt`
  - records `bot_startup_performed_by_validator=false`,
    `polling_performed_by_validator=false`, `stop_performed_by_validator=false`,
    `process_control=false`, `status_polling_started=false`,
    `process_stop_started=false`, and `cleanup_executed=false`
  - never starts, stops, polls, terminates, cleans up, or manages
    `freqtrade trade` or any bot process
- No-process-control paper/backtest drift reporting layer is implemented:
  - `freqtrade_ext/bot_factory/paper_drift.py`
  - `scripts/bot_factory_report_paper_drift.py`
  - consumes historical `metrics.json`, `walk_forward_metrics.json`,
    `training_manifest.json`, `paper_runtime_validation.json`, and local
    `paper_metrics.json`
  - blocks unless required input paths resolve inside the workspace and exist
    locally, runtime validation is a passing Phase 3
    `paper_runtime_validation`, the paper metrics path matches the exact
    `paper_runtime_validation.input_paths.paper_metrics` artifact consumed by
    runtime validation, paper metrics use local paper artifacts, and
    strategy/run IDs match
  - records failed prior walk-forward/training recommendations and large
    return/drawdown drift as `fail`, not as promotion approval
  - verifies no live/canary trading, no exchange order placement, no leverage
    above `1.0`, no shorting, no credential/private environment metadata, and
    no process-control/poll/stop/cleanup scope in the consumed artifacts;
    credential/private environment scanning covers historical metrics,
    walk-forward metrics, training manifest, runtime validation, and paper
    metrics
  - requires at least one reviewer note before the report can pass
  - writes `paper_drift_report.json`, `paper_drift_report.md`,
    `drift_metrics.json`, and `command.txt`
  - records `paper_promotion_eligible=false`,
    `promotion_authorized_by_this_command=false`, `process_control=false`,
    `status_polling_started=false`, `process_stop_started=false`, and
    `cleanup_executed=false`
  - never starts, stops, polls, terminates, cleans up, promotes, or manages
    `freqtrade trade` or any bot process

Verified current candidate state:

- Current `LongOnlyFreqAIStrategy` Phase 2 artifacts are pipeline verification
  artifacts, not a profitable strategy approval.
- The latest readiness run returned `readiness=fail`, as expected, because
  historical, walk-forward, and training gates still recommend `fail`.
- The no-startup planner run returned `status=blocked`, as expected, because
  readiness is still `fail` and no user-supplied `--confirm-paper`
  acknowledgement was provided.
- The blocked plan wrote no startup command preview.
- The no-startup startup preflight run returned `status=blocked`, as expected,
  because the plan is still blocked, no command preview exists, and no
  `--confirm-paper-start` or exact requested start command was supplied.
- The blocked startup preflight wrote an empty `start_command_preview.txt`.
- The no-startup monitoring/status schema planner returned `status=blocked`, as
  expected, because the upstream startup preflight is still blocked, still has
  blockers, and startup eligibility is false. It still wrote local schema
  artifacts for review.
- The no-process-control stop/cleanup planner returned `status=blocked`, as
  expected, because the upstream monitoring plan is still blocked, still has
  blockers, and monitoring eligibility is false. It still wrote local stop
  request schema and cleanup checklist artifacts for review.
- The no-startup/no-process-control paper start execution request gate returned
  `status=blocked`, as expected, because readiness is still `fail`, upstream
  plan/preflight/monitoring/stop-cleanup artifacts are still blocked, no startup
  command preview exists while readiness remains failed, and no
  `--confirm-paper-execution` or exact requested start command was supplied. It
  still wrote local execution request artifacts for review.
- The no-startup/no-process-control paper process executor planning gate
  returned `status=blocked`, as expected, because the execution request is still
  blocked, has blockers, is not eligible, no startup command preview exists
  while readiness remains failed, and no `--confirm-process-executor-plan` or
  exact requested start command was supplied. It still wrote local executor
  planning artifacts for review.
- The no-process-control paper runtime artifact validator returned
  `status=blocked`, as expected, because the process executor plan is still
  blocked, has blockers, is not eligible, no reviewed command preview exists,
  stdout/stderr logs and paper metrics do not exist, and the available template
  runtime files belong to the blocked startup preflight rather than a ready
  process executor plan. It still wrote local validation artifacts for review
  and did not start, stop, poll, terminate, clean up, or manage any process.
- The no-process-control paper/backtest drift reporter returned
  `status=blocked`, as expected, because the runtime validation artifact is
  still blocked, the referenced paper metrics file does not exist, and the
  current walk-forward and training recommendations are still `fail`. The
  latest hardening also verifies that drift reporting uses the same paper
  metrics path consumed by runtime validation. It still wrote local drift
  artifacts for review and did not start, stop, poll, terminate, clean up,
  promote, or manage any process.

Latest verification commands and results:

```powershell
.\.venv\Scripts\python.exe -m py_compile `
  freqtrade_ext\bot_factory\paper.py `
  freqtrade_ext\bot_factory\paper_plan.py `
  freqtrade_ext\bot_factory\paper_startup.py `
  freqtrade_ext\bot_factory\paper_monitoring.py `
  freqtrade_ext\bot_factory\paper_stop_cleanup.py `
  freqtrade_ext\bot_factory\paper_execution.py `
  freqtrade_ext\bot_factory\paper_executor.py `
  freqtrade_ext\bot_factory\paper_runtime.py `
  freqtrade_ext\bot_factory\paper_drift.py `
  scripts\bot_factory_check_paper_readiness.py `
  scripts\bot_factory_plan_paper_run.py `
  scripts\bot_factory_prepare_paper_start.py `
  scripts\bot_factory_plan_paper_monitoring.py `
  scripts\bot_factory_plan_paper_stop_cleanup.py `
  scripts\bot_factory_request_paper_start.py `
  scripts\bot_factory_plan_paper_executor.py `
  scripts\bot_factory_validate_paper_runtime.py `
  scripts\bot_factory_report_paper_drift.py `
  tests\test_bot_factory.py
```

Result: passed.

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_bot_factory.py
```

Result: sandbox temp/cache ACLs blocked the sandboxed run at
`C:\Users\yoro4\AppData\Local\Temp\pytest-of-yoro4`, producing 64 fixture setup
errors before test bodies ran. The same focused command was re-run with normal
filesystem temp/cache permissions and passed after the drift-report increment:
`64 passed`.

```powershell
.\.venv\Scripts\python.exe scripts\bot_factory_static_check.py user_data\strategies
```

Result: `ok=true`, 7 files checked, no errors. Existing review warnings remain
in `5mV1.py` and `FreqAICustomStrategy.py`. Report written to
`registry/strategies/checks/20260504T055512Z_static_check.json`.

```powershell
.\.venv\Scripts\python.exe scripts\bot_factory_plan_paper_run.py `
  --readiness-json data\paper_readiness\LongOnlyFreqAIStrategy\phase3_readiness_20260503\paper_readiness.json `
  --strategy LongOnlyFreqAIStrategy `
  --run-id phase3_paper_plan_20260503 `
  --reviewer-note "Phase 3 paper run planning hardening check only; do not start paper trading."
```

Result: completed without bot startup and returned `status=blocked`. Artifacts
were written under
`data/paper/LongOnlyFreqAIStrategy/phase3_paper_plan_20260503/`. The newly
added source, safety-scope, local-artifact, and workspace-path planner checks
passed for the current readiness artifact.

```powershell
.\.venv\Scripts\python.exe scripts\bot_factory_prepare_paper_start.py `
  --plan-json data\paper\LongOnlyFreqAIStrategy\phase3_paper_plan_20260503\paper_run_plan.json `
  --strategy LongOnlyFreqAIStrategy `
  --run-id phase3_paper_startup_preflight_20260503 `
  --reviewer-note "Phase 3 paper startup preflight hardening check only; do not start paper trading."
```

Result: completed without bot startup and returned `status=blocked`, as
expected. Artifacts were written under
`data/paper/LongOnlyFreqAIStrategy/phase3_paper_startup_preflight_20260503/`.
The generated `start_command_preview.txt` is empty. The newly added plan-source,
workspace-artifact, and local-artifact checks passed; command-preview integrity
checks are blocked because the plan correctly has no command preview while
readiness remains `fail`.

```powershell
.\.venv\Scripts\python.exe scripts\bot_factory_plan_paper_monitoring.py `
  --startup-preflight-json data\paper\LongOnlyFreqAIStrategy\phase3_paper_startup_preflight_20260503\paper_startup_preflight.json `
  --strategy LongOnlyFreqAIStrategy `
  --run-id phase3_paper_monitoring_plan_20260503 `
  --reviewer-note "Phase 3 paper monitoring schema planning only; do not start, stop, poll, or manage paper trading."
```

Result: completed without bot startup, process polling, process stop, or
process management and returned `status=blocked`, as expected. Artifacts were
written under
`data/paper/LongOnlyFreqAIStrategy/phase3_paper_monitoring_plan_20260503/`.
The planner blocks because the upstream startup preflight is still `blocked`,
still has blockers, and startup eligibility is false. Local template/log/metric
path checks and safety-scope checks passed.

```powershell
.\.venv\Scripts\python.exe scripts\bot_factory_plan_paper_stop_cleanup.py `
  --monitoring-plan-json data\paper\LongOnlyFreqAIStrategy\phase3_paper_monitoring_plan_20260503\paper_monitoring_plan.json `
  --strategy LongOnlyFreqAIStrategy `
  --run-id phase3_paper_stop_cleanup_plan_20260503 `
  --reviewer-note "Phase 3 paper stop/cleanup planning only; do not start, stop, poll, terminate, or manage paper trading."
```

Result: completed without bot startup, process polling, process stop,
termination, cleanup execution, or process management and returned
`status=blocked`, as expected. Artifacts were written under
`data/paper/LongOnlyFreqAIStrategy/phase3_paper_stop_cleanup_plan_20260503/`.
The planner blocks because the upstream monitoring plan is still `blocked`,
still has blockers, and monitoring eligibility is false. Local process metadata,
status snapshot, stdout, stderr, paper metrics path, schema, and safety-scope
checks passed.

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

Result: completed without bot startup, process polling, process stop,
termination, cleanup execution, or process management and returned
`status=blocked`, as expected. Artifacts were written under
`data/paper/LongOnlyFreqAIStrategy/phase3_paper_execution_request_20260503/`.
The gate blocks because readiness is still `fail`, upstream
plan/preflight/monitoring/stop-cleanup artifacts are still `blocked`, no
startup command preview exists while readiness remains failed, and no
`--confirm-paper-execution` or exact `--requested-start-command` was supplied.

```powershell
.\.venv\Scripts\python.exe scripts\bot_factory_plan_paper_executor.py `
  --execution-request-json data\paper\LongOnlyFreqAIStrategy\phase3_paper_execution_request_20260503\paper_execution_request.json `
  --strategy LongOnlyFreqAIStrategy `
  --run-id phase3_paper_executor_plan_20260503 `
  --reviewer-note "Phase 3 paper process executor planning only; do not start, stop, poll, terminate, clean up, or manage paper trading."
```

Result: completed without bot startup, process polling, process stop,
termination, cleanup execution, or process management and returned
`status=blocked`, as expected. Artifacts were written under
`data/paper/LongOnlyFreqAIStrategy/phase3_paper_executor_plan_20260503/`.
The gate blocks because the execution request is still `blocked`, still has
blockers, is not eligible, no startup command preview exists while readiness
remains failed, and no `--confirm-process-executor-plan` or exact
`--requested-start-command` was supplied. It still verified local runtime paths,
manifest paths, no-startup/no-process-control scope, and wrote executor
planning artifacts for review.

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

Result: completed without bot startup, process polling, process stop,
termination, cleanup execution, or process management and returned
`status=blocked`, as expected. Artifacts were written under
`data/paper/LongOnlyFreqAIStrategy/phase3_paper_runtime_validation_20260504/`.
The validator blocks because the process executor plan is still `blocked`, has
blockers, is not eligible, no reviewed command preview exists, stdout/stderr
logs and paper metrics do not exist, and the available template runtime files
belong to the blocked startup preflight rather than a ready process executor
plan.

```powershell
.\.venv\Scripts\python.exe scripts\bot_factory_report_paper_drift.py `
  --historical-metrics-json data\freqai\LongOnlyFreqAIStrategy\phase2_safe_20250105_20250107\metrics.json `
  --walk-forward-metrics-json data\walk_forward\LongOnlyFreqAIStrategy\phase2_walk_forward_20250105_20250109\walk_forward_metrics.json `
  --training-manifest-json data\freqai_training\LongOnlyFreqAIStrategy\phase2_training_20250105_20250107\training_manifest.json `
  --paper-runtime-validation-json data\paper\LongOnlyFreqAIStrategy\phase3_paper_runtime_validation_20260504\paper_runtime_validation.json `
  --strategy LongOnlyFreqAIStrategy `
  --run-id phase3_paper_drift_report_20260504 `
  --reviewer-note "Phase 3 paper/backtest drift reporting path-integrity hardening only; do not start, stop, poll, terminate, clean up, promote, or manage paper trading."
```

Result: completed without bot startup, process polling, process stop,
termination, cleanup execution, promotion, or process management and returned
`status=blocked`, as expected. Artifacts were written under
`data/paper/LongOnlyFreqAIStrategy/phase3_paper_drift_report_20260504/`.
The reporter blocks because runtime validation is still `blocked`, the
referenced paper metrics file does not exist, and the current walk-forward and
training recommendations are still `fail`. The
`paper_metrics_path_matches_runtime_validation` check passes for the current
artifact chain.

Hard safety boundaries:

- Do not start `freqtrade trade`.
- Do not start paper trading, dry-run trading, canary live, live trading, or
  any bot startup process.
- Do not stop, poll, terminate, clean up, or manage any paper process unless a
  later runbook permits the exact no-secret process-control path and the user
  explicitly requests it.
- Do not use API keys, secrets, private environment values, exchange order
  endpoints, real order placement, leverage above `1.0`, or shorting.
- Do not promote `LongOnlyFreqAIStrategy` to paper while readiness remains
  `fail`.
- Local JSON, CSV, Markdown, and logs remain the source of truth. MLflow is
  optional and must not replace local artifacts.

Next safe Phase 3 direction:

- If continuing paper-path code work, harden the no-startup/no-process-control
  readiness, planning, startup preflight, monitoring, stop/cleanup, execution
  request, process executor planning, runtime artifact validation, or
  paper/backtest drift reporting gates only if a concrete gap is found.
- Safe next paper-path increment, if a concrete gap is found: harden the drift
  reporter's schema checks or documentation for future paper metrics while
  keeping it local-artifact-only. It must not start, stop, poll, terminate,
  clean up, promote, or manage any bot process, and it must not promote while
  readiness remains failed.
- Alternative safe direction: improve or replace the Phase 2 strategy candidate
  so a future readiness run can pass based on historical, walk-forward, and
  training artifacts. Any such work remains limited to dependency checks,
  static checks, OHLCV quality checks, FreqAI validation, historical
  `freqtrade backtesting`, walk-forward evaluation, training factory
  orchestration, and local reports. Do not promote the current failed
  `LongOnlyFreqAIStrategy` evidence to paper.
- Do not implement a real process-starting executor yet unless the user
  explicitly requests the exact preflight-approved paper start path and every
  upstream gate is `ready`. The current artifact chain is blocked, so a real
  executor must remain out of scope for the next agent unless new passing
  artifacts are produced first.
- Do not add `docs/BOT_FACTORY_PHASE3_RUNBOOK.md` for the current state. Add it
  only after an actual paper path has been implemented, explicitly requested,
  verified, and documented.
- A future actual process executor must require:
  - a passing `paper_readiness.json`
  - a `ready` `paper_run_plan.json`
  - a `ready` `paper_startup_preflight.json`
  - a `ready` `paper_monitoring_plan.json` or equivalent reviewed monitoring
    schema artifacts
  - a `ready` `paper_stop_cleanup_plan.json` or equivalent reviewed stop and
    cleanup artifacts
  - a `ready` `paper_execution_request.json`
  - a `ready` `paper_process_executor_plan.json`
  - explicit user request for the exact paper startup command
  - `--confirm-paper-execution` or equivalent acknowledgement
  - reviewer notes
  - sanitized metadata
  - process metadata path
  - stdout/stderr log paths
  - status snapshot path
  - stop/cleanup procedure before any start documentation
- Do not create `docs/BOT_FACTORY_PHASE3_RUNBOOK.md` until an actual paper
  path has been implemented and verified.
- Do not mark `Paper trading deployment` complete until an explicitly
  requested, preflight-approved paper path has been implemented, verified, and
  documented.

Suggested verification after changes:

```powershell
.\.venv\Scripts\python.exe -m py_compile `
  freqtrade_ext\bot_factory\paper.py `
  freqtrade_ext\bot_factory\paper_plan.py `
  freqtrade_ext\bot_factory\paper_startup.py `
  freqtrade_ext\bot_factory\paper_monitoring.py `
  freqtrade_ext\bot_factory\paper_stop_cleanup.py `
  freqtrade_ext\bot_factory\paper_execution.py `
  freqtrade_ext\bot_factory\paper_executor.py `
  freqtrade_ext\bot_factory\paper_runtime.py `
  freqtrade_ext\bot_factory\paper_drift.py `
  scripts\bot_factory_check_paper_readiness.py `
  scripts\bot_factory_plan_paper_run.py `
  scripts\bot_factory_prepare_paper_start.py `
  scripts\bot_factory_plan_paper_monitoring.py `
  scripts\bot_factory_plan_paper_stop_cleanup.py `
  scripts\bot_factory_request_paper_start.py `
  scripts\bot_factory_plan_paper_executor.py `
  scripts\bot_factory_validate_paper_runtime.py `
  scripts\bot_factory_report_paper_drift.py `
  tests\test_bot_factory.py

.\.venv\Scripts\python.exe -m pytest tests\test_bot_factory.py

.\.venv\Scripts\python.exe scripts\bot_factory_static_check.py user_data\strategies
```

Documentation requirement:

- Update `docs/BOT_FACTORY_MVP_TODO.md` after each completed Bot Factory
  increment with exact commands, results, artifacts, and remaining limitations.
````
