# Agent Instructions

These instructions apply to automated coding agents working in this repository.

## Bot Factory Source of Truth

- Treat `docs/BOT_FACTORY_MVP_TODO.md` as the current Bot Factory source of truth.
- Read the relevant phase runbook/instructions before changing Bot Factory code.
- Keep Bot Factory changes scoped to `freqtrade_ext/bot_factory/`, `scripts/`,
  `docs/`, and focused tests under `tests/` unless the task clearly requires more.
- Preserve existing user changes. Do not revert files you did not intentionally edit.

## Safety Boundaries

- Do not start `freqtrade trade`, paper trading, dry-run trading, canary live,
  live trading, or any exchange-facing order process unless the active phase
  documentation explicitly permits it and the user explicitly requests it.
- Do not use API keys, secrets, exchange order endpoints, leverage, or shorting
  while working on Phase 1 or Phase 2 Bot Factory tasks.
- Phase 2 FreqAI work is limited to historical `freqtrade backtesting`,
  dependency checks, static checks, OHLCV quality checks, metadata, reports,
  walk-forward evaluation, and optional MLflow logging.
- Generated metadata, logs, reports, and MLflow artifacts must not include
  secrets, API keys, private environment values, or credential-like config data.

## Verification Expectations

- Start Bot Factory handoffs with:

  ```powershell
  git status --short --untracked-files=all
  ```

- Before FreqAI work, confirm dependencies with:

  ```powershell
  .\.venv\Scripts\python.exe scripts\bot_factory_check_freqai_env.py
  ```

- Before any historical backtest, run static strategy checks and validate known
  OHLCV parquet inputs when paths can be resolved.
- Prefer the narrowest relevant verification first, then broaden as needed:

  ```powershell
  .\.venv\Scripts\python.exe -m py_compile <changed python files>
  .\.venv\Scripts\python.exe -m pytest tests\test_bot_factory.py
  .\.venv\Scripts\python.exe scripts\bot_factory_static_check.py user_data\strategies
  ```

- Keep local JSON/CSV/Markdown artifacts as the source of truth. MLflow is
  optional and must not erase or replace local artifacts if it fails.

## Documentation

- Update `docs/BOT_FACTORY_MVP_TODO.md` after each completed Bot Factory
  increment with exact commands, results, artifacts, and remaining limitations.
- Add or update a phase runbook only after the corresponding path has been
  implemented and verified.

## GitHub Review Follow-up

- When addressing GitHub review feedback, always reply to the corresponding
  review comment after the fix is implemented, verified, and pushed. The reply
  must briefly state what changed and include the relevant verification result.
