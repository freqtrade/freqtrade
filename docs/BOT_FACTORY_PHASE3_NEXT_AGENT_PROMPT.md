# Bot Factory Phase 3 Next Agent Prompt

Use the following prompt for the next coding agent.

````markdown
Continue Bot Factory Phase 3.

First command, required:

```powershell
git status --short --untracked-files=all
```

Source of truth:

- `AGENTS.md`
- `docs/BOT_FACTORY_MVP_TODO.md`
- `docs/BOT_FACTORY_PHASE3_AGENT_INSTRUCTIONS.md`
- `docs/BOT_FACTORY_PHASE2_RUNBOOK.md`
- `docs/BOT_FACTORY_PHASE2_AGENT_INSTRUCTIONS.md`

Current branch:

- `develop`

Recent commits:

- `ab1ad1b56 docs(bot-factory): add phase3 paper design handoff`
- `4dcb88aeb docs(bot-factory): mark phase2 complete`
- `530b3d49f chore(bot-factory): verify phase2 training factory`

Current status:

- Normal `git status` has no listed file changes.
- The following known ACL warnings may appear:
  - `.codex_tmp/pytest-of-yoro4/`
  - `bot_factory_pytest_tmp/`
  - `codex_tmp/pytest/`
- `bot_factory_pytest_tmp` could not be removed with normal file access in the
  previous session. The attempted command was:

  ```powershell
  Remove-Item -Recurse -Force -LiteralPath bot_factory_pytest_tmp
  ```

  It returned access denied. Remove it only if normal file access is available.

Completed work:

- Phase 1 Backtest Factory.
- Phase 2 backtesting-only FreqAI Factory.
- Phase 2 completion note:
  - `docs/BOT_FACTORY_MVP_TODO.md`
  - `docs/BOT_FACTORY_PHASE2_RUNBOOK.md`
- Phase 3 handoff instructions:
  - `docs/BOT_FACTORY_PHASE3_AGENT_INSTRUCTIONS.md`

Important constraints:

- The first Phase 3 increment is paper trading readiness design and preflight
  implementation only.
- Do not start `freqtrade trade`.
- Do not start paper trading, dry-run trading, canary live, live trading, or
  any bot startup process.
- Do not use API keys, secrets, credential-like config values, exchange order
  endpoints, real order placement, leverage above `1.0`, or shorting.
- The `LongOnlyFreqAIStrategy` Phase 2 results are pipeline verification, not
  profitable strategy approval. Recent gates are `fail`, so do not promote the
  strategy into paper trading.
- Local JSON, CSV, and Markdown artifacts are the source of truth. MLflow is
  optional.

First Phase 3 goal:

Implement a no-startup paper readiness layer that can decide, from local
artifacts and static/config checks, whether a candidate strategy is eligible for
a tightly scoped future paper trading run.

Recommended deliverables:

- `docs/BOT_FACTORY_PHASE3_PAPER_DESIGN.md`
- `freqtrade_ext/bot_factory/paper.py`
- `scripts/bot_factory_check_paper_readiness.py`
- focused tests in `tests/test_bot_factory.py` or a focused new test file

Minimum requirements:

- Read Phase 2 artifacts:
  - historical FreqAI backtest metrics/report/metadata/trades
  - walk-forward metrics/report
  - training factory manifest/report
- Treat failed Phase 2 gates as a paper readiness blocker.
- Document that a future human-approved infrastructure-only smoke test is a
  separate path, but do not implement bot startup in this first increment.
- Run or consume existing static safety checks.
- Validate that the candidate remains long-only:
  - `can_short = False`
  - no short entry or exit signals
  - no leverage hook, or no leverage above `1.0`
- Validate that any proposed paper config is `dry_run=true` only.
- Validate that any proposed paper config contains no secrets, API keys, private
  environment values, or credential-like values.
- Return readiness as one of `pass`, `fail`, or `blocked`.
- Write local artifacts as the source of truth.

Recommended artifact layout:

```text
data/paper_readiness/<strategy>/<run_id>/
```

Recommended artifact files:

- `paper_readiness.json`
- `paper_readiness_report.md`
- `candidate_artifacts.json`
- `config_safety.json`
- `command.txt`

Verification candidates:

```powershell
.\.venv\Scripts\python.exe -m py_compile `
  freqtrade_ext\bot_factory\paper.py `
  scripts\bot_factory_check_paper_readiness.py `
  tests\test_bot_factory.py

.\.venv\Scripts\python.exe -m pytest tests\test_bot_factory.py

.\.venv\Scripts\python.exe scripts\bot_factory_static_check.py user_data\strategies
```

Known issue:

- Pytest may fail in the sandbox due to Windows temp/cache ACLs at
  `C:\Users\yoro4\AppData\Local\Temp\pytest-of-yoro4`.
- If that happens, record the sandbox failure and rerun the same focused pytest
  command with normal filesystem permissions.

Documentation:

- Update `docs/BOT_FACTORY_MVP_TODO.md` after the increment with exact
  commands, results, artifacts, and remaining limitations.
- Do not create `docs/BOT_FACTORY_PHASE3_RUNBOOK.md` until an actual paper path
  has been implemented and verified.
- Do not mark `Paper trading deployment` complete until an explicitly requested,
  preflight-approved paper path has been implemented, verified, and documented.
````

