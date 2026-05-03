# Bot Factory Phase 3 Next Agent Prompt

Use the following prompt for the next coding agent.

````markdown
Continue Bot Factory Phase 3 from the current no-startup state.

First command, required:

```powershell
git status --short --untracked-files=all
```

Read these files before making changes:

- `AGENTS.md`
- `docs/BOT_FACTORY_MVP_TODO.md`
- `docs/BOT_FACTORY_PHASE3_AGENT_INSTRUCTIONS.md`
- `docs/BOT_FACTORY_PHASE3_PAPER_DESIGN.md`
- `docs/BOT_FACTORY_PHASE2_RUNBOOK.md`
- `docs/BOT_FACTORY_PHASE2_AGENT_INSTRUCTIONS.md`

Current branch:

- `develop`

Current worktree context:

- This handoff may include uncommitted Phase 3 no-startup changes:
  - `docs/BOT_FACTORY_MVP_TODO.md`
  - `docs/BOT_FACTORY_PHASE3_NEXT_AGENT_PROMPT.md`
  - `docs/BOT_FACTORY_PHASE3_PAPER_DESIGN.md`
  - `freqtrade_ext/bot_factory/paper_plan.py`
  - `freqtrade_ext/bot_factory/paper_startup.py`
  - `scripts/bot_factory_plan_paper_run.py`
  - `scripts/bot_factory_prepare_paper_start.py`
  - `tests/test_bot_factory.py`
  - `data/paper/LongOnlyFreqAIStrategy/phase3_paper_plan_20260503/`
  - `data/paper/LongOnlyFreqAIStrategy/phase3_paper_startup_preflight_20260503/`
- Existing readiness artifacts may exist under:
  - `data/paper_readiness/LongOnlyFreqAIStrategy/phase3_readiness_20260503/`
- Known Windows ACL warnings may appear in `git status`:
  - `.codex_tmp/pytest-of-yoro4/`
  - `bot_factory_pytest_tmp/`
  - `codex_tmp/pytest/`
- Do not remove ACL-blocked directories unless normal file access is available
  and the resolved path is safely inside the repository.

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

Latest verification commands and results:

```powershell
.\.venv\Scripts\python.exe -m py_compile `
  freqtrade_ext\bot_factory\paper.py `
  freqtrade_ext\bot_factory\paper_plan.py `
  freqtrade_ext\bot_factory\paper_startup.py `
  scripts\bot_factory_check_paper_readiness.py `
  scripts\bot_factory_plan_paper_run.py `
  scripts\bot_factory_prepare_paper_start.py `
  tests\test_bot_factory.py
```

Result: passed.

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_bot_factory.py
```

Result: sandbox temp/cache ACLs blocked the sandboxed run at
`C:\Users\yoro4\AppData\Local\Temp\pytest-of-yoro4`, producing 44 fixture setup
errors before test bodies ran. A normal filesystem permissions rerun was
requested but could not be approved in this environment because the Codex usage
limit was reached, so the focused pytest suite was not completed for this
increment. The previous normal-permissions run before this hardening passed
`42 passed`.

```powershell
.\.venv\Scripts\python.exe scripts\bot_factory_static_check.py user_data\strategies
```

Result: `ok=true`, 7 files checked, no errors. Existing review warnings remain
in `5mV1.py` and `FreqAICustomStrategy.py`. Report written to
`registry/strategies/checks/20260503T080954Z_static_check.json`.

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

Hard safety boundaries:

- Do not start `freqtrade trade`.
- Do not start paper trading, dry-run trading, canary live, live trading, or
  any bot startup process.
- Do not use API keys, secrets, private environment values, exchange order
  endpoints, real order placement, leverage above `1.0`, or shorting.
- Do not promote `LongOnlyFreqAIStrategy` to paper while readiness remains
  `fail`.
- Local JSON, CSV, Markdown, and logs remain the source of truth. MLflow is
  optional and must not replace local artifacts.

Next safe Phase 3 direction:

- If continuing code work, harden the no-startup startup preflight further only
  if a concrete gap is found, or design the later execution wrapper without
  running it.
- A future execution wrapper must require:
  - a passing `paper_readiness.json`
  - a `ready` `paper_run_plan.json`
  - a `ready` `paper_startup_preflight.json`
  - explicit user request for the exact paper startup command
  - `--confirm-paper-start` or equivalent acknowledgement
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
  scripts\bot_factory_check_paper_readiness.py `
  scripts\bot_factory_plan_paper_run.py `
  scripts\bot_factory_prepare_paper_start.py `
  tests\test_bot_factory.py

.\.venv\Scripts\python.exe -m pytest tests\test_bot_factory.py

.\.venv\Scripts\python.exe scripts\bot_factory_static_check.py user_data\strategies
```

Documentation requirement:

- Update `docs/BOT_FACTORY_MVP_TODO.md` after each completed Bot Factory
  increment with exact commands, results, artifacts, and remaining limitations.
````
