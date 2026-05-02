# Bot Factory MVP TODO

This TODO is derived from `crypto_bot_factory_agent_instructions.md` and scoped to
Phase 1: Backtest Factory. The first milestone must not start live trading.

## Scope

- Goal: generate, check, backtest, save metrics, and report on strategy candidates.
- Non-goal: live trading, canary live, Hummingbot integration, production risk changes.
- Safety rule: every production-related action remains stubbed or human-approved.

## Current Reusable Work

- [x] Freqtrade repository and CLI structure exists.
- [x] Existing strategies are under `user_data/strategies/`.
- [x] Orderbook parquet feature store exists under `freqtrade_ext/feature_store.py`.
- [x] Bybit orderbook collector exists under `tools/ob_collector_ws.py`.
- [x] Risk/exit extension helpers exist under `freqtrade_ext/risk/`.
- [x] FreqAI strategy experiments exist under `user_data/strategies/FreqAICustomStrategy.py`.
- [x] Local `.venv` has Phase 1 runtime/test dependencies installed:
  `ccxt`, Freqtrade runtime requirements, `pytest`, pytest plugins, `duckdb`, and `mlflow`.
- [x] FreqAI dependency audit command exists and reports missing runtime dependencies
  deterministically without starting any bot process.
- [ ] FreqAI-specific runtime dependencies are not fully installed yet:
  `lightgbm`, `xgboost`, `tensorboard`, and `datasieve` are currently missing in the
  local `.venv`.

## Phase 1 MVP Tasks

### 1. Repository Skeleton

- [x] Add this TODO file.
- [x] Add `data/backtests/` for normalized backtest outputs.
- [x] Add `registry/strategies/checks/` for static check outputs.
- [x] Add `freqtrade_ext/bot_factory/` helper package.
- [x] Add strategy proposal templates under `registry/strategies/proposals/`.

### 2. Infrastructure

- [x] Add a separate Docker Compose overlay for PostgreSQL and MLflow.
- [x] Add `.env.example` entries for factory-only services.
- [x] Keep existing `docker-compose.yml` compatible with current Freqtrade usage.

### 3. Data Download

- [x] Add a safe wrapper for `freqtrade download-data`.
- [x] Verify OHLCV download after installing required dependencies.
- [x] Store OHLCV as parquet via `--data-format-ohlcv parquet`.
- [x] Verify parquet can be read by pandas and DuckDB.
- [x] Add data quality checks for OHLCV.

### 4. Static Safety Check

- [x] Add static strategy scanner.
- [x] Detect exact `shift(-1)` and `shift(periods=-1)`.
- [x] Detect dangerous `iloc[-1]` usage, including tuple row selectors such as
  `iloc[-1, column]`, in indicator/entry/exit generation.
- [x] Detect hardcoded secrets.
- [x] Detect direct order API calls.
- [x] Write JSON check reports to `registry/strategies/checks/`.

### 5. Backtest Runner

- [x] Add a safe wrapper for `freqtrade backtesting`.
- [x] Run static safety check before backtesting by default.
- [x] Save raw Freqtrade result under `data/backtests/<strategy>/<run_id>/`.
- [x] Support current Freqtrade zipped backtest result format.
- [x] Normalize metrics to `metrics.json`.
- [x] Export trades to `trades.csv`.
- [x] Generate `report.md`.
- [x] Verify runner with real OHLCV data after dependencies are installed.

### 6. MLflow Tracking

- [x] Add optional MLflow logging.
- [x] Keep local `metrics.json` as the source of truth if MLflow is unavailable.

### 7. Report and Gate Rules

- [x] Generate a Markdown report from Freqtrade result JSON.
- [x] Include initial pass/fail gate checks.
- [x] Add configurable gate thresholds.
- [x] Add reviewer notes and promotion recommendations.

## Latest Verification

Checked on 2026-04-26 JST.

- [x] Added Phase 2 FreqAI dependency audit helper and CLI:
  `freqtrade_ext/bot_factory/freqai_checks.py` and
  `scripts/bot_factory_check_freqai_env.py`.
- [x] Added focused tests for installed and missing dependency reporting.
- [x] `python -m py_compile` passed for Bot Factory scripts/helpers and focused tests
  using a PowerShell-expanded file list. The literal wildcard form shown in the
  handoff command is not expanded by this Windows shell.
- [x] `pytest tests/test_bot_factory.py` passed: 12 tests.
- [x] `scripts/bot_factory_check_freqai_env.py` ran and correctly exited with status
  `1` because required FreqAI dependencies are missing:
  `lightgbm`, `xgboost`, `tensorboard`, and `datasieve`.
- [x] FreqAI dependency report was written to
  `registry/strategies/checks/20260425T202159Z_freqai_env.json`.
- [x] `scripts/bot_factory_static_check.py user_data/strategies` passed with warnings only:
  6 files checked, no errors.
- [x] Strengthened static safety checks for `shift(periods=-1)` and tuple-form
  `iloc[-1, column]` lookahead patterns, while avoiding false positives for
  excluding slices such as `iloc[:-1]`.
- [x] `python -m py_compile` passed for Bot Factory scripts/helpers and focused tests.
- [x] `pytest tests/test_bot_factory.py` passed: 10 tests.
- [x] `scripts/bot_factory_static_check.py user_data/strategies` passed with warnings only:
  6 files checked, no errors.
- [x] `scripts/bot_factory_check_ohlcv.py user_data/data/bybit/futures/BTC_USDT_USDT-5m-futures.parquet --timeframe 5m`
  passed: 8995 rows, no duplicate timestamps, no missing intervals, no OHLCV
  integrity findings.
- [x] Downloaded Bybit futures OHLCV for `BTC/USDT:USDT`, timeframe `5m`,
  timerange `20250101-20250103`, stored as parquet at
  `user_data/data/bybit/futures/BTC_USDT_USDT-5m-futures.parquet`.
- [x] Verified the parquet file with pandas and DuckDB: 998 rows currently available locally.
- [x] `scripts/bot_factory_check_ohlcv.py` passed against the verified parquet file:
  998 rows, no duplicate timestamps, no missing intervals, no OHLCV integrity findings.
- [x] Ran real backtest for `SampleStrategy` with `BTC/USDT:USDT`, timeframe `5m`,
  timerange `20250101-20250103`.
- [x] Generated artifacts under
  `data/backtests/SampleStrategy/real_20250101_20250103/`:
  `result.json`, `metrics.json`, `trades.csv`, `report.md`, `static_check.json`.
- [x] Regenerated `metrics.json` and `report.md` with Sharpe/Sortino/Calmar fields,
  configurable gate thresholds, reviewer notes, and a promotion recommendation.
- [x] Verified opt-in MLflow logging with a local file tracking URI. This did not start
  paper trading, live trading, or any exchange-facing process.
- [x] `docker compose --profile bot-factory -f docker-compose.bot-factory.yml config`
  passed. This validated the Bot Factory service overlay without starting containers.
- [x] Downloaded a longer Bybit futures OHLCV window for `BTC/USDT:USDT`, timeframe `5m`,
  timerange `20250101-20250201`. The parquet now has 8995 rows locally.
- [x] The integrated post-download OHLCV quality check passed for the longer window:
  no duplicate timestamps, no missing intervals, no OHLCV integrity findings.
- [x] Ran a longer real backtest for `SampleStrategy` with `BTC/USDT:USDT`, timeframe `5m`,
  timerange `20250101-20250201`.
- [x] Generated artifacts under
  `data/backtests/SampleStrategy/real_20250101_20250201/`:
  `result.json`, `metrics.json`, `trades.csv`, `report.md`, `static_check.json`.
- [x] Longer-window metrics: 3 trades, 0.79% total return, 2.37 Sharpe, Sortino `-100`.
  The initial gate correctly remains `fail` because trade count and other thresholds are not met.
- [x] Added `docs/BOT_FACTORY_PHASE1_RUNBOOK.md` with the safe Phase 1 workflow.
- [x] Added `docs/BOT_FACTORY_PHASE2_AGENT_INSTRUCTIONS.md` as the Phase 2 handoff prompt,
  scoped to FreqAI dependency checks, historical FreqAI backtesting, feature/label validation,
  walk-forward evaluation, reports, and optional MLflow logging.
- [x] Verified `docs/BOT_FACTORY_PHASE2_AGENT_INSTRUCTIONS.md` exists and is readable:
  `Test-Path` returned `True`; `Measure-Object -Line` returned 178 lines.

## Implementation Notes

- Bot Factory wrappers disable FreqAI by default for Phase 1 runs using a small overlay config.
  Pass `--enable-freqai` when intentionally testing FreqAI behavior.
- `bot_factory_download_data.py` runs an OHLCV parquet quality check after successful
  downloads by default. Pass `--skip-data-quality-check` to skip it.
- Windows/aiohttp environments may select `aiodns` and fail public exchange DNS resolution.
  Bot Factory Freqtrade invocations use `freqtrade_ext.bot_factory.freqtrade_cli` to force
  aiohttp's threaded DNS resolver by default.
- Freqtrade now writes the latest backtest result as a zip and leaves a pointer JSON. The
  Bot Factory parser resolves that zip and writes expanded raw content to `result.json`.
- The verified `SampleStrategy` short timerange produced only 1 trade, so `report.md` correctly
  marks the initial gate as `fail`. This is a pipeline verification, not a strategy approval.
- Gate thresholds can be overridden with a JSON file such as
  `registry/strategies/gate_thresholds.example.json`.
- MLflow logging is opt-in with `--mlflow`. If MLflow is unavailable, the command records
  `mlflow_error.txt` and keeps the local `metrics.json`/`report.md` as the source of truth.
- The root `docker-compose.yml` was left unchanged. Use
  `docker-compose.bot-factory.yml` for Bot Factory PostgreSQL/MLflow infrastructure.
  Do not start paper or live bots in Phase 1.

## Phase 2 and Later

- [x] Add Phase 2 agent handoff instructions.
- [x] Add FreqAI dependency audit helper and script.
- [ ] FreqAI training factory.
- [ ] Walk-forward runner.
- [ ] Paper trading deployment.
- [ ] Risk Governor service.
- [ ] Execution Gateway service.
- [ ] Dashboard.
- [ ] Canary live workflow with mandatory human approval.

## Immediate Commands

Run static safety checks:

```powershell
.\.venv\Scripts\python.exe scripts\bot_factory_static_check.py user_data\strategies
```

Audit FreqAI runtime dependencies:

```powershell
.\.venv\Scripts\python.exe scripts\bot_factory_check_freqai_env.py
```

Run OHLCV quality checks:

```powershell
.\.venv\Scripts\python.exe scripts\bot_factory_check_ohlcv.py `
  user_data\data\bybit\futures\BTC_USDT_USDT-5m-futures.parquet `
  --timeframe 5m
```

Start Bot Factory services:

```powershell
docker compose --profile bot-factory -f docker-compose.bot-factory.yml up -d
```

Download OHLCV as parquet after installing Freqtrade dependencies:

```powershell
.\.venv\Scripts\python.exe scripts\bot_factory_download_data.py `
  --config user_data\config.json `
  --pairs BTC/USDT:USDT `
  --timeframes 5m `
  --timerange 20250101-20250201 `
  --trading-mode futures
```

Run a checked backtest:

```powershell
.\.venv\Scripts\python.exe scripts\bot_factory_run_backtest.py `
  --config user_data\config.json `
  --strategy SampleStrategy `
  --timeframe 5m `
  --timerange 20250101-20250201 `
  --pairs BTC/USDT:USDT
```

Run a checked backtest with custom gate thresholds and optional MLflow logging:

```powershell
.\.venv\Scripts\python.exe scripts\bot_factory_run_backtest.py `
  --config user_data\config.json `
  --strategy SampleStrategy `
  --timeframe 5m `
  --timerange 20250101-20250201 `
  --pairs BTC/USDT:USDT `
  --gate-config registry\strategies\gate_thresholds.example.json `
  --mlflow
```
