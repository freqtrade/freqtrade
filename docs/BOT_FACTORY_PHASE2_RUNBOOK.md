# Bot Factory Phase 2 Runbook

This runbook covers the verified Phase 2 FreqAI historical backtest,
walk-forward, and training factory paths.

It must not be used to start paper trading, dry-run trading, canary live, live
trading, exchange order placement, leverage experiments, or shorting. The
verified path uses `freqtrade backtesting` only.

## Scope

Allowed:

- FreqAI dependency audits.
- Static safety checks for strategy source files.
- FreqAI feature/label convention validation.
- OHLCV parquet quality checks.
- FreqAI-enabled historical backtests through Freqtrade backtesting only.
- Walk-forward evaluation through checked historical backtest windows only.
- Local metrics, trades CSV, Markdown reports, and FreqAI metadata.
- Optional MLflow logging, with local files remaining the source of truth.

Not allowed in Phase 2:

- `freqtrade trade`.
- Paper or dry-run bot startup.
- Live/canary startup.
- API keys or secrets.
- Exchange order endpoints.
- Leverage or short implementation.
- Promotion beyond backtest or walk-forward review.

## 1. Dependency Audit

```powershell
.\.venv\Scripts\python.exe scripts\bot_factory_check_freqai_env.py
```

Expected result:

- Exit code `0`.
- `ok=true`.
- Required dependencies are installed: `lightgbm`, `xgboost`, `tensorboard`,
  and `datasieve`.

## 2. Static Safety Check

```powershell
.\.venv\Scripts\python.exe scripts\bot_factory_static_check.py user_data\strategies
```

Expected result:

- Exit code `0`.
- `ok=true`.
- Warnings may require review, but errors must block backtesting.

## 3. Validate FreqAI Feature/Label Conventions

```powershell
.\.venv\Scripts\python.exe scripts\bot_factory_validate_freqai_strategy.py `
  user_data\strategies\LongOnlyFreqAIStrategy.py `
  --output registry\strategies\checks\phase2_freqai_validation_LongOnlyFreqAIStrategy.json
```

Expected result:

- Exit code `0`.
- `ok=true`.
- Feature columns created by `feature_engineering_*` use the `%` prefix.
- Target/label columns created by `set_freqai_targets` use the `&` prefix.
- Negative `shift(-n)` is recorded as allowed only inside
  `set_freqai_targets` supervised label generation.
- Negative shifts in entry, exit, indicator, or feature signal logic are errors.
- The report includes:
  `FreqAI labels are backtest labels, not live trading instructions.`

## 4. Check Existing OHLCV

```powershell
.\.venv\Scripts\python.exe scripts\bot_factory_check_ohlcv.py `
  user_data\data\bybit\futures\BTC_USDT_USDT-5m-futures.parquet `
  --timeframe 5m
```

Expected result:

- Exit code `0`.
- No duplicate timestamps.
- No missing intervals.
- No OHLCV integrity findings.

## 5. Run FreqAI Historical Backtest

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

Outputs are written under:

```text
data/freqai/LongOnlyFreqAIStrategy/phase2_safe_20250105_20250107/
```

Important files:

- `freqai_metadata.json`: Phase 2 run metadata without secrets.
- `metrics.json`: normalized Bot Factory metrics.
- `trades.csv`: exported trades.
- `report.md`: gate checks, reviewer notes, and promotion recommendation.
- `static_check.json`: static strategy check captured before backtesting.
- `freqai_validation.json`: feature/label/lookahead validation captured before
  backtesting.
- `ohlcv_quality.json`: OHLCV quality report captured before backtesting.
- `freqai_env.json`: FreqAI dependency audit captured before backtesting.
- `result.json`: expanded raw Freqtrade backtest result.

`report.md` and `freqai_metadata.json` must retain this warning:

```text
FreqAI labels are backtest labels, not live trading instructions.
```

## 6. Run Walk-Forward Verification

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

Outputs are written under:

```text
data/walk_forward/LongOnlyFreqAIStrategy/phase2_walk_forward_20250105_20250109/
```

Important files:

- `walk_forward_metrics.json`: aggregated window metrics, pass/fail checks, and
  safety scope.
- `walk_forward_report.md`: Markdown summary of the walk-forward run.
- `command.txt`: exact child window commands.
- `window_logs/`: parent-captured command/stdout/stderr logs per window.
- `windows/LongOnlyFreqAIStrategy/<window_run_id>/`: per-window FreqAI backtest
  artifacts, including `metrics.json`, `trades.csv`, `report.md`,
  `freqai_metadata.json`, and `freqai_validation.json`.

Verified result on 2026-05-02 JST:

- 2/2 windows completed after allowing normal network access for public Bybit
  market metadata needed by Freqtrade startup.
- Walk-forward recommendation: `fail`.
- Pass rate: `0.00%`.
- Profitable windows ratio: `0.00%`.
- Combined total return: `-0.18%`.
- Max drawdown in any window: `0.1719%`.
- Window 1 (`20250105-20250107`): 2 trades, `-0.0617%`.
- Window 2 (`20250107-20250109`): 3 trades, `-0.1225%`.
- Exported trades in both windows remained `is_short=False`, `leverage=1.0`.

This verifies the walk-forward pipeline only. It is not promotion approval.

## 7. Run FreqAI Training Factory

The training factory is a parent orchestration wrapper. It runs the checked
FreqAI backtest wrapper for the training stage and can optionally run the
checked walk-forward wrapper when windows are supplied. Local artifacts remain
the source of truth.

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

Outputs are written under:

```text
data/freqai_training/LongOnlyFreqAIStrategy/phase2_training_20250105_20250107/
```

Important files:

- `training_manifest.json`: parent stage status, recommendations, dependency
  audit, local artifact paths, and Phase 2 safety scope.
- `training_report.md`: Markdown summary of the training factory run.
- `command.txt`: exact checked child command lines.
- `freqai_env.json`: parent FreqAI dependency audit.
- `logs/`: parent-captured child command/stdout/stderr logs.
- `freqai_backtests/`: child checked FreqAI backtest artifacts.

Verified result on 2026-05-03 JST:

- The training factory helper, CLI, command construction, and manifest/report
  generation are implemented.
- The first sandboxed historical training attempt failed while loading public
  Bybit market metadata during Freqtrade backtesting startup.
- The same backtesting-only command completed after allowing normal network
  access for that public metadata load.
- Parent dependency audit passed with `ok=true`.
- Parent status: `completed`; parent recommendation: `fail`.
- Child `freqai_backtest` status: `completed`; child recommendation: `fail`.
- Metrics: 2 trades, `-0.0617%` total return, `0.0` profit factor, and
  `0.0617%` max drawdown.
- Exported trades remained `is_short=False` and `leverage=1.0`.
- Reports and metadata state that this is Phase 2 verification only, not paper
  or live promotion, and that FreqAI labels are backtest labels, not live
  trading instructions.

## Current Limitations

- The verified run is a pipeline check, not a profitable strategy approval.
  The 2025-01-05 to 2025-01-07 run produced 2 trades and the initial gate
  failed as expected.
- The safe config uses futures OHLCV because that is the verified local data
  available. The strategy is long-only, exports `is_short=False`, and uses
  leverage `1.0`.
- Freqtrade may load public exchange market metadata during backtesting startup
  even when candle data is local. This must remain limited to public metadata
  needed by `freqtrade backtesting`; do not use order endpoints or credentials.
- The verified walk-forward run fails promotion gates because both windows are
  unprofitable and have too few trades. This is expected for pipeline
  verification.
- The verified training factory run also fails promotion gates because the
  brief verification timerange produced only 2 trades and negative return. This
  is expected for pipeline verification.
- Passing gates do not authorize paper trading or live trading.
