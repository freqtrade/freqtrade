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
- [x] FreqAI-specific runtime dependencies are installed in the local `.venv`:
  `lightgbm`, `xgboost`, `tensorboard`, and `datasieve`.

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

Checked on 2026-05-03 JST.

- [x] Marked Bot Factory Phase 2 complete for the backtesting-only FreqAI
  Factory scope. Phase 2 completion covers dependency audit, Phase 2-safe
  historical FreqAI backtest, required local artifacts, feature/label
  validation, two-window walk-forward verification, training factory
  verification, and documented results. This completion does not authorize
  paper trading, dry-run trading, live trading, canary live, exchange order
  placement, leverage, or shorting.
- [x] Started the handoff with:

  ```powershell
  git status --short --untracked-files=all
  ```

  Result: no file changes were listed, but the expected Windows ACL warnings
  remained for `.codex_tmp/pytest-of-yoro4/`, `bot_factory_pytest_tmp/`, and
  `codex_tmp/pytest/`.
- [x] Attempted to remove the workspace-local pytest temp directory after
  resolving it inside the repository:

  ```powershell
  Remove-Item -Recurse -Force -LiteralPath bot_factory_pytest_tmp
  ```

  Result: Windows returned access denied, so the directory was left untouched.
- [x] Re-ran the focused syntax check:

  ```powershell
  .\.venv\Scripts\python.exe -m py_compile `
    freqtrade_ext\bot_factory\freqai_training.py `
    scripts\bot_factory_run_freqai_training.py `
    tests\test_bot_factory.py
  ```

  Result: passed.
- [x] Re-ran focused pytest:

  ```powershell
  .\.venv\Scripts\python.exe -m pytest tests\test_bot_factory.py
  ```

  Result: the sandboxed run failed at `tmp_path` setup because
  `C:\Users\yoro4\AppData\Local\Temp\pytest-of-yoro4` was ACL-blocked. The
  same command was re-run with normal filesystem temp/cache permissions and
  passed: 26 tests.
- [x] Re-ran the FreqAI dependency audit:

  ```powershell
  .\.venv\Scripts\python.exe scripts\bot_factory_check_freqai_env.py
  ```

  Result: `ok=true` for `lightgbm==4.6.0`, `xgboost==3.0.5`,
  `tensorboard==2.20.0`, and `datasieve==0.1.9`. Report written to
  `registry/strategies/checks/20260503T033037Z_freqai_env.json`.
- [x] Re-ran static strategy checks:

  ```powershell
  .\.venv\Scripts\python.exe scripts\bot_factory_static_check.py user_data\strategies
  ```

  Result: `ok=true`, 7 files checked, no errors. The existing review warnings
  remain in `5mV1.py` and `FreqAICustomStrategy.py`. Report written to
  `registry/strategies/checks/20260503T033036Z_static_check.json`.
- [x] Re-ran the known OHLCV parquet quality check:

  ```powershell
  .\.venv\Scripts\python.exe scripts\bot_factory_check_ohlcv.py `
    user_data\data\bybit\futures\BTC_USDT_USDT-5m-futures.parquet `
    --timeframe 5m
  ```

  Result: passed with 8995 rows, no duplicate timestamps, no missing
  intervals, and no OHLCV integrity findings. Report written to
  `registry/strategies/checks/20260503T033036Z_ohlcv_quality.json`.
- [x] Completed a Phase 2-safe FreqAI training factory historical
  verification:

  ```powershell
  .\.venv\Scripts\python.exe scripts\bot_factory_run_freqai_training.py `
    --config user_data\config_freqai_phase2_safe.json `
    --strategy LongOnlyFreqAIStrategy `
    --timeframe 5m `
    --timerange 20250105-20250107 `
    --pairs BTC/USDT:USDT `
    --run-id phase2_training_20250105_20250107 `
    --python .\.venv\Scripts\python.exe `
    --reviewer-note "Phase 2 FreqAI training factory verification only; no paper or live promotion."
  ```

  The sandboxed attempt failed at the same public Bybit market metadata load
  seen previously. The same backtesting-only command was then re-run with
  normal network access for public metadata and completed successfully.
- [x] Updated FreqAI training artifacts under
  `data/freqai_training/LongOnlyFreqAIStrategy/phase2_training_20250105_20250107/`:
  parent `training_manifest.json`, `training_report.md`, `command.txt`,
  `freqai_env.json`, `logs/`, and child checked FreqAI backtest artifacts under
  `freqai_backtests/LongOnlyFreqAIStrategy/train_20250105_20250107/`.
  Child artifacts include `metrics.json`, `trades.csv`, `report.md`,
  `result.json`, `freqai_metadata.json`, `freqai_validation.json`,
  `static_check.json`, `ohlcv_quality.json`, `freqai_env.json`, and the raw
  Freqtrade zip/pointer files.
- [x] Training factory verification result: parent status `completed`, parent
  recommendation `fail`, child `freqai_backtest` status `completed`, child
  recommendation `fail`. Metrics: 2 trades, total return `-0.0617%`, profit
  factor `0.0`, max drawdown `0.0617%`, Sharpe/Sortino `-123.7515`. Exported
  trades remained `is_short=False` and `leverage=1.0`. Reports and metadata
  state that this is Phase 2 verification only, not paper/live promotion, and
  that FreqAI labels are backtest labels, not live trading instructions.

Checked on 2026-05-02 JST.

- [x] Added a FreqAI training factory orchestration helper:
  `freqtrade_ext/bot_factory/freqai_training.py`.
  It builds checked child commands for the existing FreqAI backtest and
  walk-forward wrappers, aggregates stage status/recommendations, and writes a
  local `training_manifest.json` plus `training_report.md` with Phase 2 safety
  scope. The factory does not call paper, dry-run, canary, live, order,
  leverage, or shorting paths.
- [x] Added `scripts/bot_factory_run_freqai_training.py`.
  The script runs a parent FreqAI dependency audit, requires FreqAI to be
  enabled, invokes `scripts/bot_factory_run_freqai_backtest.py` for the training
  stage, optionally invokes `scripts/bot_factory_run_walk_forward.py` when
  windows are supplied, and keeps local artifacts as the source of truth.
  Optional MLflow is pass-through to the checked child wrappers.
- [x] Added focused tests for training child run-id sanitization, checked child
  command construction, walk-forward command construction, and training
  manifest safety/source-of-truth metadata in `tests/test_bot_factory.py`.
- [x] `python -m py_compile` passed for:
  `freqtrade_ext/bot_factory/freqai_training.py`,
  `scripts/bot_factory_run_freqai_training.py`, and
  `tests/test_bot_factory.py`.

  ```powershell
  .\.venv\Scripts\python.exe -m py_compile `
    freqtrade_ext\bot_factory\freqai_training.py `
    scripts\bot_factory_run_freqai_training.py `
    tests\test_bot_factory.py
  ```
- [x] Direct helper verification passed with inline Python assertions for:
  `training_child_run_id`, checked FreqAI backtest command construction, and
  training manifest safety metadata.
- [ ] `pytest tests/test_bot_factory.py` could not complete in this session
  because Windows temp/cache ACLs blocked `tmp_path` setup under the local
  pytest temp root. A workspace-local
  `--basetemp bot_factory_pytest_tmp -p no:cacheprovider` retry was also
  blocked by ACLs, and normal-permission escalation was unavailable. The
  workspace-local temp directory remains ACL-blocked and now appears as a
  warning in `git status`; it should be removed when normal filesystem access
  is available.

  ```powershell
  .\.venv\Scripts\python.exe -m pytest tests\test_bot_factory.py
  .\.venv\Scripts\python.exe -m pytest tests\test_bot_factory.py `
    --basetemp bot_factory_pytest_tmp `
    -p no:cacheprovider
  ```
- [x] Re-ran `scripts/bot_factory_static_check.py user_data/strategies`; it
  passed with warnings only: 7 files checked, no errors. The warnings remain
  review-only findings in `5mV1.py` and `FreqAICustomStrategy.py`.

  ```powershell
  .\.venv\Scripts\python.exe scripts\bot_factory_static_check.py user_data\strategies
  ```
- [x] Re-ran
  `scripts/bot_factory_check_ohlcv.py user_data/data/bybit/futures/BTC_USDT_USDT-5m-futures.parquet --timeframe 5m`;
  it passed with 8995 rows, no duplicate timestamps, no missing intervals, and
  no OHLCV integrity findings.

  ```powershell
  .\.venv\Scripts\python.exe scripts\bot_factory_check_ohlcv.py `
    user_data\data\bybit\futures\BTC_USDT_USDT-5m-futures.parquet `
    --timeframe 5m
  ```
- [ ] Attempted a Phase 2-safe FreqAI training factory verification:

  ```powershell
  .\.venv\Scripts\python.exe scripts\bot_factory_run_freqai_training.py `
    --config user_data\config_freqai_phase2_safe.json `
    --strategy LongOnlyFreqAIStrategy `
    --timeframe 5m `
    --timerange 20250105-20250107 `
    --pairs BTC/USDT:USDT `
    --run-id phase2_training_20250105_20250107 `
    --python .\.venv\Scripts\python.exe `
    --reviewer-note "Phase 2 FreqAI training factory verification only; no paper or live promotion."
  ```

  Parent artifacts were written under
  `data/freqai_training/LongOnlyFreqAIStrategy/phase2_training_20250105_20250107/`:
  `training_manifest.json`, `training_report.md`, `command.txt`,
  `freqai_env.json`, and `logs/`. The child checked FreqAI backtest completed
  dependency, validation, and OHLCV prechecks, then failed while loading public
  Bybit market metadata under sandboxed network access:
  `Could not load markets, therefore cannot start.` A normal-network retry for
  this backtesting-only command was unavailable in this session. This is not a
  strategy promotion and does not authorize paper or live trading.
- [x] Added FreqAI feature/label validation helpers in
  `freqtrade_ext/bot_factory/freqai_checks.py`.
  The validation report checks `%`-prefixed FreqAI feature columns,
  `&`-prefixed target/label columns, records allowed negative shifts inside
  `set_freqai_targets`, and reports negative shifts in
  `populate_*`/`feature_engineering_*` signal logic as errors.
- [x] Added `scripts/bot_factory_validate_freqai_strategy.py` for standalone
  FreqAI feature/label/lookahead validation.
- [x] Updated `scripts/bot_factory_run_freqai_backtest.py` to write
  `freqai_validation.json`, block invalid FreqAI feature/label conventions
  before backtesting, and keep the note
  `FreqAI labels are backtest labels, not live trading instructions.` in both
  report reviewer notes and `freqai_metadata.json`.
- [x] Updated static safety scanning so `shift(-1)` remains blocked in
  indicator/entry/exit logic but is allowed in `set_freqai_targets` supervised
  label generation.
- [x] Added a backtest-only walk-forward runner:
  `freqtrade_ext/bot_factory/walk_forward.py` and
  `scripts/bot_factory_run_walk_forward.py`.
  It accepts repeated `--window` specs or generated rolling windows from
  `--start`, `--end`, `--train-days`, `--test-days`, and `--step-days`, runs
  only the checked FreqAI backtest wrapper per window, and writes
  `walk_forward_metrics.json` plus `walk_forward_report.md`.
- [x] `python -m py_compile` passed for:
  `freqtrade_ext/bot_factory/freqai_checks.py`,
  `freqtrade_ext/bot_factory/safety.py`,
  `freqtrade_ext/bot_factory/walk_forward.py`,
  `scripts/bot_factory_validate_freqai_strategy.py`,
  `scripts/bot_factory_run_freqai_backtest.py`,
  `scripts/bot_factory_run_walk_forward.py`, and
  `tests/test_bot_factory.py`.
- [x] `pytest tests/test_bot_factory.py` passed: 22 tests. The first sandboxed
  run was blocked by Windows temp/cache ACLs at
  `C:\Users\yoro4\AppData\Local\Temp\pytest-of-yoro4`; the same focused command
  was rerun with normal filesystem permissions and passed.
- [x] Ran standalone FreqAI validation:

  ```powershell
  .\.venv\Scripts\python.exe scripts\bot_factory_validate_freqai_strategy.py `
    user_data\strategies\LongOnlyFreqAIStrategy.py `
    --output registry\strategies\checks\phase2_freqai_validation_LongOnlyFreqAIStrategy.json
  ```

  Result: `ok=true`, 1 file checked, 9 `%` feature columns, 1 `&` target
  column, and the negative shift in `set_freqai_targets` recorded as allowed
  supervised target generation.
- [x] Re-ran `scripts/bot_factory_check_freqai_env.py`; it passed with `ok=true`
  for `lightgbm==4.6.0`, `xgboost==3.0.5`, `tensorboard==2.20.0`, and
  `datasieve==0.1.9`.
- [x] Re-ran `scripts/bot_factory_static_check.py user_data/strategies`; it
  passed with warnings only: 7 files checked, no errors. The warnings remain
  review-only findings in `5mV1.py` and `FreqAICustomStrategy.py`.
- [x] Re-ran
  `scripts/bot_factory_check_ohlcv.py user_data/data/bybit/futures/BTC_USDT_USDT-5m-futures.parquet --timeframe 5m`;
  it passed with 8995 rows, no duplicate timestamps, no missing intervals, and
  no OHLCV integrity findings.
- [x] Ran a two-window Phase 2 walk-forward verification:

  ```powershell
  .\.venv\Scripts\python.exe scripts\bot_factory_run_walk_forward.py `
    --config user_data\config_freqai_phase2_safe.json `
    --strategy LongOnlyFreqAIStrategy `
    --timeframe 5m `
    --pairs BTC/USDT:USDT `
    --window 20250105-20250107 `
    --window 20250107-20250109 `
    --run-id phase2_walk_forward_20250105_20250109 `
    --reviewer-note "Phase 2 walk-forward verification only; no paper or live promotion."
  ```

  The first sandboxed attempt completed parent artifact generation but both
  child windows failed while loading public Bybit market metadata. The same
  backtesting-only command completed after allowing normal network access for
  that public metadata load.
- [x] Generated walk-forward artifacts under
  `data/walk_forward/LongOnlyFreqAIStrategy/phase2_walk_forward_20250105_20250109/`:
  `walk_forward_metrics.json`, `walk_forward_report.md`, `command.txt`,
  `window_logs/`, and per-window FreqAI artifacts under
  `windows/LongOnlyFreqAIStrategy/wf_01_20250105_20250107/` and
  `windows/LongOnlyFreqAIStrategy/wf_02_20250107_20250109/`.
- [x] Walk-forward verification metrics: 2/2 windows completed, pass rate
  `0.00%`, profitable windows ratio `0.00%`, combined total return `-0.18%`,
  max drawdown in any window `0.1719%`, and recommendation `fail`. Window 1
  had 2 trades and `-0.0617%`; window 2 had 3 trades and `-0.1225%`.
  Exported trades in both windows remained `is_short=False`, `leverage=1.0`.
  This verifies the historical walk-forward pipeline, not promotion.
- [x] Installed FreqAI runtime dependencies from the existing
  `requirements-freqai.txt` into the local `.venv`.
- [x] Re-ran `scripts/bot_factory_check_freqai_env.py`; it passed with `ok=true`
  for `lightgbm==4.6.0`, `xgboost==3.0.5`, `tensorboard==2.20.0`, and
  `datasieve==0.1.9`.
- [x] Added a FreqAI backtest metadata/helper module:
  `freqtrade_ext/bot_factory/freqai_backtest.py`.
- [x] Added `scripts/bot_factory_run_freqai_backtest.py`, a FreqAI-specific
  wrapper that runs dependency audit, static strategy checks, known or explicitly
  supplied OHLCV parquet quality checks, `freqtrade backtesting` only, and writes
  `freqai_metadata.json` without secrets.
- [x] Added focused tests for FreqAI model-name resolution, FreqAI OHLCV input
  path resolution, and metadata path sanitization.
- [x] `python -m py_compile` passed for Bot Factory helpers, scripts, and
  `tests/test_bot_factory.py` using explicit file paths.
- [x] `pytest tests/test_bot_factory.py` passed: 15 tests. The sandboxed run
  could not create pytest temp directories, so the same command was rerun with
  normal filesystem permissions.
- [x] `scripts/bot_factory_static_check.py user_data/strategies` passed with
  warnings only: 6 files checked, no errors.
- [x] `scripts/bot_factory_check_ohlcv.py user_data/data/bybit/futures/BTC_USDT_USDT-5m-futures.parquet --timeframe 5m`
  passed: 8995 rows, no duplicate timestamps, no missing intervals, no OHLCV
  integrity findings.
- [x] Added a Phase 2-safe long-only FreqAI strategy:
  `user_data/strategies/LongOnlyFreqAIStrategy.py`.
  It sets `can_short = False`, emits no short entry/exit signals, and does not
  implement a `leverage()` hook.
- [x] Added a Phase 2-safe historical config:
  `user_data/config_freqai_phase2_safe.json`.
  It uses `LightGBMRegressor`, one local Bybit futures pair
  (`BTC/USDT:USDT`), no API server credentials, no orderbook pricing, no
  `ext_risk` leverage settings, `save_backtest_models = false`, and
  `freqtrade backtesting` only.
- [x] `python -m py_compile` passed for
  `user_data/strategies/LongOnlyFreqAIStrategy.py`,
  `freqtrade_ext/bot_factory/freqai_backtest.py`,
  `scripts/bot_factory_run_freqai_backtest.py`, and
  `tests/test_bot_factory.py`.
- [x] `python -m json.tool user_data/config_freqai_phase2_safe.json` passed.
- [x] `pytest tests/test_bot_factory.py` passed: 15 tests. The sandboxed run
  was blocked by Windows temp/cache ACLs, so the same focused command was
  rerun with normal filesystem permissions.
- [x] `scripts/bot_factory_static_check.py user_data/strategies` passed with
  warnings only: 7 files checked, no errors. The warnings are pre-existing
  review warnings in `5mV1.py` and `FreqAICustomStrategy.py`; the new
  long-only strategy added no findings.
- [x] `scripts/bot_factory_check_ohlcv.py user_data/data/bybit/futures/BTC_USDT_USDT-5m-futures.parquet --timeframe 5m`
  passed: 8995 rows, no duplicate timestamps, no missing intervals, no OHLCV
  integrity findings.
- [x] Ran a real Phase 2-safe FreqAI historical backtest:

  ```powershell
  .\.venv\Scripts\python.exe scripts\bot_factory_run_freqai_backtest.py `
    --config user_data\config_freqai_phase2_safe.json `
    --strategy LongOnlyFreqAIStrategy `
    --timeframe 5m `
    --timerange 20250105-20250107 `
    --pairs BTC/USDT:USDT `
    --run-id phase2_safe_20250105_20250107 `
    --reviewer-note "Phase 2 historical FreqAI verification only; no paper or live promotion."
  ```

  The first sandboxed attempt reached Freqtrade backtesting startup but was
  blocked while loading public Bybit market metadata. The same backtest-only
  command completed after allowing normal network access for that public market
  metadata load.
- [x] Generated FreqAI artifacts under
  `data/freqai/LongOnlyFreqAIStrategy/phase2_safe_20250105_20250107/`:
  `freqai_metadata.json`, `metrics.json`, `trades.csv`, `report.md`,
  `static_check.json`, `ohlcv_quality.json`, `freqai_env.json`, `result.json`,
  `command.txt`, `stdout.log`, and `stderr.log`.
- [x] FreqAI verification metrics: 2 trades, -0.06% total return, 0.00 profit
  factor, 0.0617% max drawdown, `is_short=False`, `leverage=1.0` in exported
  trades. The initial gate correctly remains `fail`; this verifies the
  historical FreqAI pipeline, not strategy promotion.
- [x] Added `docs/BOT_FACTORY_PHASE2_RUNBOOK.md` for the verified FreqAI
  historical backtest workflow and current Phase 2 limitations.

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
- Phase 2 FreqAI backtests may need public exchange market metadata during
  Freqtrade startup even when all OHLCV candles are local. This is not an order
  endpoint and must remain limited to `freqtrade backtesting`.
- The root `docker-compose.yml` was left unchanged. Use
  `docker-compose.bot-factory.yml` for Bot Factory PostgreSQL/MLflow infrastructure.
  Do not start paper or live bots in Phase 1.

## Phase 2 and Later

- [x] Add Phase 2 agent handoff instructions.
- [x] Add FreqAI dependency audit helper and script.
- [x] Install FreqAI runtime dependencies in the local `.venv`.
- [x] Add FreqAI backtest wrapper with dependency, static, OHLCV, and metadata
  prechecks.
- [x] Verify a FreqAI-enabled historical backtest on a Phase 2-safe config.
- [x] Add FreqAI feature/label validation and integrate it into FreqAI backtest
  prechecks.
- [x] Add and verify a backtest-only walk-forward runner on two historical
  windows.
- [x] Add FreqAI training factory orchestration helper and CLI.
- [x] Complete FreqAI training factory historical verification with public
  market metadata access.
- [x] Phase 2 complete for the FreqAI Factory backtesting-only scope. Remaining
  unchecked items below are later-phase work and are not required for Phase 2
  completion.
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

Validate FreqAI feature/label conventions:

```powershell
.\.venv\Scripts\python.exe scripts\bot_factory_validate_freqai_strategy.py `
  user_data\strategies\LongOnlyFreqAIStrategy.py `
  --output registry\strategies\checks\phase2_freqai_validation_LongOnlyFreqAIStrategy.json
```

Template for a checked FreqAI backtest wrapper run on a Phase 2-safe historical
config:

```powershell
.\.venv\Scripts\python.exe scripts\bot_factory_run_freqai_backtest.py `
  --config user_data\config_freqai_phase2_safe.json `
  --strategy LongOnlyFreqAIStrategy `
  --timeframe 5m `
  --timerange 20250101-20250103 `
  --pairs BTC/USDT:USDT
```

This command path is for `freqtrade backtesting` only. Do not use it to start
paper trading, dry-run trading, canary live, live trading, exchange order
placement, leverage experiments, or shorting in Phase 2.

Run a checked walk-forward verification:

```powershell
.\.venv\Scripts\python.exe scripts\bot_factory_run_walk_forward.py `
  --config user_data\config_freqai_phase2_safe.json `
  --strategy LongOnlyFreqAIStrategy `
  --timeframe 5m `
  --pairs BTC/USDT:USDT `
  --window 20250105-20250107 `
  --window 20250107-20250109 `
  --run-id phase2_walk_forward_20250105_20250109 `
  --reviewer-note "Phase 2 walk-forward verification only; no paper or live promotion."
```

This command runs the checked FreqAI backtest wrapper per window. It does not
authorize paper trading or live trading even if gates pass.

Run the FreqAI training factory wrapper:

```powershell
.\.venv\Scripts\python.exe scripts\bot_factory_run_freqai_training.py `
  --config user_data\config_freqai_phase2_safe.json `
  --strategy LongOnlyFreqAIStrategy `
  --timeframe 5m `
  --timerange 20250105-20250107 `
  --pairs BTC/USDT:USDT `
  --run-id phase2_training_20250105_20250107 `
  --python .\.venv\Scripts\python.exe `
  --reviewer-note "Phase 2 FreqAI training factory verification only; no paper or live promotion."
```

This command is an orchestration wrapper. It runs the checked FreqAI backtest
wrapper for the training stage and can run the checked walk-forward wrapper when
`--window` or rolling-window arguments are supplied. It remains limited to
historical `freqtrade backtesting`.

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
