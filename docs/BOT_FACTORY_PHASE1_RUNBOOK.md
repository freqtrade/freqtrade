# Bot Factory Phase 1 Runbook

This runbook covers only the Phase 1 backtest factory workflow.

It must not be used to start paper trading, canary live, live trading, or any
exchange-facing order process. Do not provide API keys or secrets to these
commands.

## Scope

Allowed:

- Static safety checks for strategy source files.
- Public OHLCV download through the Freqtrade data downloader.
- OHLCV parquet quality checks.
- Freqtrade backtests.
- Local metrics, trades CSV, and Markdown report generation.
- Optional MLflow metric and artifact logging.

Not allowed in Phase 1:

- Paper bot startup.
- Live/canary startup.
- API key or secret usage.
- Leverage, shorting, or production deployment.
- Human-approval bypasses.

## 1. Static Safety Check

```powershell
.\.venv\Scripts\python.exe scripts\bot_factory_static_check.py user_data\strategies
```

Expected result:

- Exit code `0`.
- `ok=true`.
- Warnings may require review, but errors must block backtesting.

## 2. Download OHLCV

```powershell
.\.venv\Scripts\python.exe scripts\bot_factory_download_data.py `
  --config user_data\config.json `
  --pairs BTC/USDT:USDT `
  --timeframes 5m `
  --timerange 20250101-20250201 `
  --trading-mode futures
```

The wrapper disables FreqAI by default and runs an OHLCV quality check after a
successful download. Use `--skip-data-quality-check` only when debugging the
download path itself.

## 3. Check Existing OHLCV

```powershell
.\.venv\Scripts\python.exe scripts\bot_factory_check_ohlcv.py `
  user_data\data\bybit\futures\BTC_USDT_USDT-5m-futures.parquet `
  --timeframe 5m
```

The check validates required OHLCV columns, nulls, duplicate timestamps,
timestamp ordering, expected intervals, basic price bounds, and non-negative
volume.

## 4. Run Backtest

```powershell
.\.venv\Scripts\python.exe scripts\bot_factory_run_backtest.py `
  --config user_data\config.json `
  --strategy SampleStrategy `
  --timeframe 5m `
  --timerange 20250101-20250201 `
  --pairs BTC/USDT:USDT `
  --run-id real_20250101_20250201 `
  --reviewer-note "Phase 1 verification only; no paper or live promotion."
```

Outputs are written under:

```text
data/backtests/<strategy>/<run_id>/
```

Important files:

- `result.json`: expanded raw Freqtrade backtest result.
- `metrics.json`: normalized Bot Factory metrics.
- `trades.csv`: exported trades.
- `report.md`: summary, gate checks, reviewer notes, and promotion recommendation.
- `static_check.json`: static safety result captured before backtesting.

## 5. Optional Gate Config

```powershell
.\.venv\Scripts\python.exe scripts\bot_factory_run_backtest.py `
  --config user_data\config.json `
  --strategy SampleStrategy `
  --timeframe 5m `
  --timerange 20250101-20250201 `
  --pairs BTC/USDT:USDT `
  --gate-config registry\strategies\gate_thresholds.example.json
```

Default thresholds remain conservative:

- Minimum trades: `200`
- Minimum profit factor: `1.25`
- Maximum drawdown pct: `15.0`
- Minimum Sortino: `1.2`

Passing the backtest gate is not approval for paper or live trading.

## 6. Optional MLflow Logging

```powershell
.\.venv\Scripts\python.exe scripts\bot_factory_generate_report.py `
  data\backtests\SampleStrategy\real_20250101_20250201\result.json `
  --strategy SampleStrategy `
  --outdir data\backtests\SampleStrategy\real_20250101_20250201 `
  --mlflow
```

MLflow logging is optional. If MLflow fails, the command records
`mlflow_error.txt`; local `metrics.json` and `report.md` remain the source of
truth.

## 7. Service Overlay

Validate the Bot Factory service overlay without starting containers:

```powershell
docker compose --profile bot-factory -f docker-compose.bot-factory.yml config
```

Starting the overlay is only for PostgreSQL/MLflow infrastructure. Do not start
paper or live bots as part of Phase 1.
