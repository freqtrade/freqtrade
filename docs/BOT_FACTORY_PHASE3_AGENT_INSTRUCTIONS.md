# Bot Factory Phase 3 Agent Instructions

Use this document as the handoff prompt for moving Bot Factory from the
completed Phase 2 FreqAI backtesting factory into Phase 3 paper trading design.

Phase 3 begins with design and safety scaffolding. Do not start a paper,
dry-run, canary, or live bot as the first increment. A future paper-trading
startup requires explicit user request, successful preflight checks, and
phase documentation that permits that exact command.

## Required First Steps

1. Run:

   ```powershell
   git status --short --untracked-files=all
   ```

2. Read these files before making changes:

   - `AGENTS.md`
   - `docs/BOT_FACTORY_MVP_TODO.md`
   - `docs/BOT_FACTORY_PHASE1_RUNBOOK.md`
   - `docs/BOT_FACTORY_PHASE2_AGENT_INSTRUCTIONS.md`
   - `docs/BOT_FACTORY_PHASE2_RUNBOOK.md`
   - `docs/BOT_FACTORY_PHASE3_AGENT_INSTRUCTIONS.md`
   - `crypto_bot_factory_agent_instructions.md`

3. Treat `docs/BOT_FACTORY_MVP_TODO.md` as the current source of truth.
4. Preserve all existing user changes. Do not revert files you did not edit.
5. Keep Bot Factory changes scoped to:

   - `freqtrade_ext/bot_factory/`
   - `scripts/`
   - `docs/`
   - focused tests under `tests/`

## Current Starting Point

Phase 2 is complete for the backtesting-only FreqAI Factory scope.

Verified Phase 2 capabilities:

- FreqAI dependency audit.
- Static strategy checks.
- OHLCV parquet quality checks.
- FreqAI feature/label validation.
- Phase 2-safe historical FreqAI backtest artifacts.
- Two-window walk-forward artifacts.
- FreqAI training factory artifacts.
- Source-of-truth documentation.

Important limitation:

- The verified `LongOnlyFreqAIStrategy` runs are pipeline checks, not profitable
  strategy approvals. Recent Phase 2 gates still recommend `fail`, with too few
  trades and negative return. Do not promote that result into paper trading.

## Hard Boundaries

Forbidden unless a later Phase 3 document explicitly permits the exact command
and the user explicitly requests it:

- Starting `freqtrade trade`.
- Starting paper trading or dry-run trading.
- Starting canary live or live trading.
- Starting a bot process that manages simulated or real orders.

Always forbidden in Phase 3 until explicitly superseded by later human-approved
phase instructions:

- API key or secret usage.
- Real exchange order placement.
- Live or canary capital allocation.
- Leverage above `1.0`.
- Shorting.
- Human-approval bypasses.
- Promotion based only on failed, weak, or incomplete historical gates.

Any command that could place real orders, expose credentials, change exchange
account state, or bypass human approval must be rejected.

## Phase 3 Design Goal

Build the paper-trading readiness layer before running any bot.

Phase 3 should make it possible to answer, from local artifacts and static
checks, whether a strategy candidate is eligible for a tightly scoped paper
trading run. The first milestone should produce a deterministic design,
preflight checks, sanitized metadata, and clear block reasons without starting
paper trading.

## Recommended First Milestone

Start with a no-startup paper trading design increment.

Suggested deliverables:

- `docs/BOT_FACTORY_PHASE3_PAPER_DESIGN.md`
- `freqtrade_ext/bot_factory/paper.py`
- `scripts/bot_factory_check_paper_readiness.py`
- focused tests in `tests/test_bot_factory.py` or a new focused test file

Required behavior:

- Read Phase 2 local artifacts, including metrics, reports, walk-forward
  metrics, and training manifests.
- Require candidate evidence before paper readiness can pass:
  historical backtest artifacts, walk-forward artifacts, and explicit reviewer
  notes.
- Treat failed Phase 2 gates as a paper-readiness blocker unless a future
  human-approved instruction explicitly requests an infrastructure-only paper
  smoke test.
- Run or consume the existing static safety checks.
- Validate that the candidate remains long-only:
  `can_short = False`, no short entry/exit signals, and no leverage hook above
  `1.0`.
- Validate that proposed paper configuration is `dry_run=true` only.
- Validate that proposed paper configuration contains no API keys, secrets,
  credential-like values, private environment values, or exchange order
  endpoint overrides.
- Write local JSON and Markdown artifacts as the source of truth.
- Return clear `pass`, `fail`, or `blocked` readiness status.

## Suggested Artifact Shape

Use deterministic output directories such as:

```text
data/paper_readiness/<strategy>/<run_id>/
```

Suggested files:

- `paper_readiness.json`: machine-readable readiness result.
- `paper_readiness_report.md`: human-readable summary and block reasons.
- `candidate_artifacts.json`: paths and hashes or timestamps for historical
  evidence consumed by the readiness check.
- `config_safety.json`: sanitized config inspection result.
- `command.txt`: exact readiness command, not a bot startup command.

Do not write secrets, API keys, private environment values, or credential-like
config content to artifacts.

## Paper Config Safety Requirements

Any proposed paper config or config template must satisfy:

- `dry_run` is `true`.
- No API server credentials intended for remote control.
- No exchange API key, secret, password, UID, or token values.
- No live/canary order endpoints.
- No leverage experiments.
- No shorting.
- Explicit max open trades.
- Explicit stake cap suitable for simulation.
- Explicit pair allowlist.
- Explicit strategy and timeframe.
- Clear reviewer note that this is not live trading approval.

Prefer generating a sanitized template or validation report first. Do not
start a bot just because a template validates.

## Later Phase 3 Milestones

After the no-startup readiness layer is implemented and verified, a later agent
may design a paper-run wrapper. That wrapper must still default to no startup
unless the user explicitly requests a paper run.

Future wrapper requirements:

- Require the readiness report to pass.
- Require `--confirm-paper` or equivalent explicit acknowledgement.
- Require a reviewer note.
- Persist command, stdout/stderr logs, sanitized config metadata, status
  snapshots, and paper metrics under `data/paper/<strategy>/<run_id>/`.
- Provide a stop/cleanup procedure before any start procedure is documented.
- Document all limitations before adding a Phase 3 runbook.

Do not add `docs/BOT_FACTORY_PHASE3_RUNBOOK.md` until a paper path has been
implemented and verified.

## Verification Requirements

For docs-only handoff changes:

```powershell
git diff -- docs
```

For code changes in the first Phase 3 design increment:

```powershell
.\.venv\Scripts\python.exe -m py_compile `
  freqtrade_ext\bot_factory\paper.py `
  scripts\bot_factory_check_paper_readiness.py `
  tests\test_bot_factory.py

.\.venv\Scripts\python.exe -m pytest tests\test_bot_factory.py

.\.venv\Scripts\python.exe scripts\bot_factory_static_check.py user_data\strategies
```

If pytest hits the known Windows temp/cache ACL issue, record the sandbox
failure and re-run the same focused command only with normal filesystem
permissions.

## Documentation Updates

Update `docs/BOT_FACTORY_MVP_TODO.md` after every completed Phase 3 increment
with exact commands, results, artifacts, and remaining limitations.

Do not mark `Paper trading deployment` complete until an explicitly requested,
preflight-approved paper path has been implemented, verified, and documented.

