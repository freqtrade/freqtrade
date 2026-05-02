# Bot Factory Phase 2 Agent Instructions

Use this document as the handoff prompt for continuing the Crypto Bot Factory
after Phase 1. Phase 2 is limited to FreqAI training/backtesting, feature and
label validation, walk-forward evaluation, metadata capture, reports, and
optional MLflow logging.

Do not implement paper trading, canary live, live trading, exchange order
placement, API key/secret usage, leverage, or shorting in Phase 2.

## Required First Steps

1. Run:

   ```powershell
   git status --short --untracked-files=all
   ```

2. Read these files before making changes:

   - `docs/BOT_FACTORY_MVP_TODO.md`
   - `docs/BOT_FACTORY_PHASE1_RUNBOOK.md`
   - `docs/BOT_FACTORY_PHASE2_AGENT_INSTRUCTIONS.md`
   - `crypto_bot_factory_agent_instructions.md`

3. Treat `docs/BOT_FACTORY_MVP_TODO.md` as the current source of truth.
4. Preserve all existing user changes. Do not revert files you did not edit.
5. Keep implementation changes scoped to:

   - `freqtrade_ext/bot_factory/`
   - `scripts/`
   - `docs/`
   - focused tests under `tests/`

## Hard Boundaries

Allowed in Phase 2:

- FreqAI dependency audit and setup documentation.
- FreqAI strategy import/load checks.
- FreqAI backtesting/training through Freqtrade backtesting only.
- Public historical market data download and quality checks.
- Feature and label schema validation.
- Walk-forward backtest windows.
- Local metrics, model metadata, reports, and optional MLflow logging.

Forbidden in Phase 2:

- `freqtrade trade`
- paper bot startup
- dry-run bot startup
- live/canary startup
- API key or secret use
- exchange-facing order endpoints
- human-approval bypasses
- leverage or short implementation
- deployment or promotion beyond backtest/walk-forward review

Any command that could place, simulate live, or manage real orders must be
rejected or left as documentation only.

## Phase 2 Goals

Phase 2 is complete when the repository can safely:

- confirm FreqAI runtime dependencies are installed or clearly report what is
  missing;
- run static safety checks before any FreqAI backtest;
- validate existing OHLCV parquet inputs before training/backtesting;
- run a FreqAI-enabled backtest in an offline backtesting context;
- store FreqAI run outputs under deterministic Bot Factory output directories;
- capture model/run metadata without secrets;
- run walk-forward windows using historical data only;
- aggregate walk-forward metrics and generate a Markdown report;
- optionally log metrics and artifacts to MLflow while keeping local files as
  the source of truth.

## Recommended Implementation Order

### 1. Dependency Audit

Add a safe dependency audit command before attempting FreqAI runs.

Suggested deliverables:

- `freqtrade_ext/bot_factory/freqai_checks.py`
- `scripts/bot_factory_check_freqai_env.py`
- tests covering installed and missing dependency reporting

The checker should inspect imports such as:

- `lightgbm`
- `xgboost`
- `tensorboard`
- `datasieve`

It should fail clearly when required dependencies are missing, but it must not
start any bot process.

### 2. FreqAI Backtest Wrapper

Extend or add a wrapper for FreqAI-enabled backtests.

Suggested deliverable:

- `scripts/bot_factory_run_freqai_backtest.py`

Required behavior:

- run `scripts/bot_factory_static_check.py` or the underlying scanner first;
- run OHLCV quality checks before the backtest when input paths are known;
- call Freqtrade through `freqtrade_ext.bot_factory.freqtrade_cli`;
- use `backtesting` only;
- require explicit `--enable-freqai` or make the script FreqAI-specific by name;
- write outputs under `data/backtests/<strategy>/<run_id>/` or
  `data/freqai/<strategy>/<run_id>/`;
- write `command.txt`, `stdout.log`, `stderr.log`, `result.json`,
  `metrics.json`, `trades.csv`, `report.md`, and `static_check.json`;
- write `freqai_metadata.json` with strategy, model name, timerange, pairs,
  timeframe, dependency status, config paths, and generated artifact paths.

Do not include API keys, secrets, absolute private credentials, or environment
variable values in metadata.

### 3. Feature and Label Validation

Add validation for FreqAI feature/label conventions.

Suggested deliverables:

- helper functions in `freqtrade_ext/bot_factory/freqai_checks.py`
- tests in `tests/test_bot_factory.py` or a focused new test file

Minimum checks:

- features should use FreqAI feature naming conventions, such as `%`-prefixed
  columns where applicable;
- targets/labels should use FreqAI target naming conventions, such as
  `&`-prefixed columns where applicable;
- direct future data usage must not appear in entry/exit signal logic;
- supervised-learning future returns are allowed only inside target generation
  methods such as `set_freqai_targets`;
- generated reports must explicitly state that model labels are backtest labels,
  not live trading instructions.

### 4. Walk-Forward Runner

Add a backtest-only walk-forward runner.

Suggested deliverables:

- `freqtrade_ext/bot_factory/walk_forward.py`
- `scripts/bot_factory_run_walk_forward.py`

Required behavior:

- accept a list of fixed historical windows or generate rolling windows from
  `--start`, `--end`, `--train-days`, `--test-days`, and `--step-days`;
- run only historical backtests for each window;
- write each window under its own output directory;
- aggregate window metrics to `walk_forward_metrics.json`;
- generate `walk_forward_report.md`;
- include pass/fail checks such as:
  - minimum pass rate;
  - minimum profitable windows ratio;
  - maximum drawdown in any window;
  - no single window should dominate total profit.

This runner must not start paper, dry-run, canary, or live trading.

### 5. MLflow Logging

Reuse the existing optional MLflow behavior.

Required behavior:

- MLflow remains opt-in;
- local JSON and Markdown files remain the source of truth;
- MLflow failures write `mlflow_error.txt` and do not erase local artifacts;
- do not log secrets, API keys, or private environment values.

### 6. Documentation Updates

Update `docs/BOT_FACTORY_MVP_TODO.md` after every completed increment.

Add a Phase 2 runbook when the first FreqAI path is verified:

- `docs/BOT_FACTORY_PHASE2_RUNBOOK.md`

The runbook should include exact commands, expected outputs, known limitations,
and a clear statement that Phase 2 does not approve paper or live trading.

## Suggested First Milestone

Start with the smallest safe milestone:

1. Add the FreqAI dependency audit helper and script.
2. Add focused tests for dependency reporting.
3. Run:

   ```powershell
   .\.venv\Scripts\python.exe -m py_compile freqtrade_ext\bot_factory\*.py scripts\bot_factory_*.py tests\test_bot_factory.py
   .\.venv\Scripts\python.exe -m pytest tests\test_bot_factory.py
   .\.venv\Scripts\python.exe scripts\bot_factory_check_freqai_env.py
   ```

4. Update `docs/BOT_FACTORY_MVP_TODO.md` with commands and results.

Only after dependency status is clear should the agent attempt a FreqAI
backtest wrapper.

## Verification Requirements

For each Phase 2 change, run the narrowest relevant verification first.

Minimum recurring checks:

```powershell
.\.venv\Scripts\python.exe -m py_compile freqtrade_ext\bot_factory\*.py scripts\bot_factory_*.py tests\test_bot_factory.py
.\.venv\Scripts\python.exe -m pytest tests\test_bot_factory.py
.\.venv\Scripts\python.exe scripts\bot_factory_static_check.py user_data\strategies
```

When OHLCV data is involved:

```powershell
.\.venv\Scripts\python.exe scripts\bot_factory_check_ohlcv.py `
  user_data\data\bybit\futures\BTC_USDT_USDT-5m-futures.parquet `
  --timeframe 5m
```

When a FreqAI backtest path is implemented, verify it on a short historical
timerange first and record all artifacts in `docs/BOT_FACTORY_MVP_TODO.md`.

## Completion Criteria

Phase 2 can be marked complete only when:

- FreqAI dependencies are installed or missing dependencies are reported by a
  deterministic checker;
- at least one FreqAI-enabled historical backtest has completed without paper or
  live startup;
- generated artifacts include metrics, trades, report, static check result, and
  FreqAI metadata;
- walk-forward execution has been verified on at least two historical windows;
- reports clearly state that passing gates do not authorize paper or live
  trading;
- all verification commands and results are recorded in
  `docs/BOT_FACTORY_MVP_TODO.md`.
