# Bot Factory Phase 3 Next Agent Prompt

Use the following prompt for the next coding agent.

````markdown
Continue Bot Factory Phase 3 after the first no-startup paper readiness
increment.

First command, required:

```powershell
git status --short --untracked-files=all
```

Source of truth:

- `AGENTS.md`
- `docs/BOT_FACTORY_MVP_TODO.md`
- `docs/BOT_FACTORY_PHASE3_AGENT_INSTRUCTIONS.md`
- `docs/BOT_FACTORY_PHASE3_PAPER_DESIGN.md`
- `docs/BOT_FACTORY_PHASE2_RUNBOOK.md`
- `docs/BOT_FACTORY_PHASE2_AGENT_INSTRUCTIONS.md`

Current branch:

- `develop`

Recent commits:

- `ab1ad1b56 docs(bot-factory): add phase3 paper design handoff`
- `4dcb88aeb docs(bot-factory): mark phase2 complete`
- `530b3d49f chore(bot-factory): verify phase2 training factory`

Current status:

- If this increment has not been committed, `git status` may list the Phase 3
  readiness files below. If it has been committed, status should be clean apart
  from the known ACL warnings.
- The following known ACL warnings may appear:
  - `.codex_tmp/pytest-of-yoro4/`
  - `bot_factory_pytest_tmp/`
  - `codex_tmp/pytest/`
- The first Phase 3 no-startup paper readiness layer is implemented and
  verified, and it has been hardened with stricter child evidence and config
  policy checks. Do not repeat that milestone unless you are fixing or
  extending it.
- Expected current file changes may include:
  - `docs/BOT_FACTORY_MVP_TODO.md`
  - `docs/BOT_FACTORY_PHASE3_NEXT_AGENT_PROMPT.md`
  - `docs/BOT_FACTORY_PHASE3_PAPER_DESIGN.md`
  - `freqtrade_ext/bot_factory/paper.py`
  - `scripts/bot_factory_check_paper_readiness.py`
  - `tests/test_bot_factory.py`
- Expected local readiness artifacts may exist under:
  - `data/paper_readiness/LongOnlyFreqAIStrategy/phase3_readiness_20260503/`
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
- Phase 3 no-startup paper readiness design and preflight:
  - `docs/BOT_FACTORY_PHASE3_PAPER_DESIGN.md`
  - `freqtrade_ext/bot_factory/paper.py`
  - `scripts/bot_factory_check_paper_readiness.py`
  - focused tests in `tests/test_bot_factory.py`
- Phase 3 no-startup readiness hardening:
  - walk-forward child window artifacts now require `metrics.json`,
    `trades.csv`, and `freqai_metadata.json`
  - training `freqai_backtest` child artifacts now require `metrics.json`,
    `trades.csv`, and `freqai_metadata.json`
  - historical, walk-forward child, and training child trade exports must have
    no shorts and no leverage above `1.0`
  - config policy rejects `force_entry_enable=true`, requires
    `initial_state=stopped`, requires explicit `cancel_open_orders_on_exit`,
    and enforces `max_open_trades <= 3`, `stake_amount <= 1000`,
    `dry_run_wallet <= 10000`, and `stake_amount <= dry_run_wallet`

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

Completed first Phase 3 goal:

Implement a no-startup paper readiness layer that can decide, from local
artifacts and static/config checks, whether a candidate strategy is eligible for
a tightly scoped future paper trading run.

Implemented deliverables:

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
  - historical exported trades contain no shorts and no leverage above `1.0`
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

Verified readiness result:

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

Result: completed without bot startup and returned `readiness=fail`, because
the historical, walk-forward, and training Phase 2 gates still recommend
`fail`. This is expected and blocks paper readiness for the current
`LongOnlyFreqAIStrategy` evidence. The hardened config policy, child evidence,
and long-only/leverage trade export checks pass for the current local
artifacts.

Latest verification from this handoff:

```powershell
.\.venv\Scripts\python.exe -m py_compile `
  freqtrade_ext\bot_factory\paper.py `
  scripts\bot_factory_check_paper_readiness.py `
  tests\test_bot_factory.py
```

Result: passed.

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_bot_factory.py
```

Result: the sandboxed run failed at `tmp_path` setup because
`C:\Users\yoro4\AppData\Local\Temp\pytest-of-yoro4` was ACL-blocked. The same
focused command was rerun with normal filesystem permissions and passed:
`36 passed`.

```powershell
.\.venv\Scripts\python.exe scripts\bot_factory_static_check.py user_data\strategies
```

Result: `ok=true`, 7 files checked, no errors. Existing review warnings remain
in `5mV1.py` and `FreqAICustomStrategy.py`. Report written to
`registry/strategies/checks/20260503T050639Z_static_check.json`.

Next safe Phase 3 direction:

- Improve or harden the readiness layer further only if a concrete gap is
  found, or design a future paper-run wrapper without starting it.
- Any future wrapper must require a passing readiness report,
  `--confirm-paper` or equivalent explicit acknowledgement, reviewer notes,
  sanitized metadata, and stop/cleanup documentation before any start
  documentation.
- Do not create `docs/BOT_FACTORY_PHASE3_RUNBOOK.md` until an actual paper path
  has been implemented and verified.

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
