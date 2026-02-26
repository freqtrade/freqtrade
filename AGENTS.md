# Agents

## Cursor Cloud specific instructions

### Overview

Freqtrade is a Python crypto trading bot. It is a single-process application with an embedded SQLite database — no external services (Redis, Postgres, etc.) are required for development or testing.

### Virtual environment

The project uses a Python 3.12 venv at `.venv/`. Always activate it before running commands:

```bash
source .venv/bin/activate
```

### Key commands

- **Lint:** `ruff check .` and `ruff format --check` (see `CONTRIBUTING.md` for full list including `isort --check .` and `mypy freqtrade scripts tests`)
- **Tests:** `pytest` (full suite), or `pytest tests/<file>.py` for a single file. Use `-n auto` for parallel runs. The CI uses `--random-order --durations 20 -n auto`.
- **Backtesting demo:** `freqtrade backtesting --datadir tests/testdata --strategy SampleStrategy -i 5m`
- **Type checking:** `mypy freqtrade scripts tests`

### Live deployment (Railway)

TrendRider v8 is deployed on Railway running **live** against MEXC with real funds.

- **Railway project:** `trendrider-bot` (auto-restarts, 24/7)
- **Exchange:** MEXC spot, $25 USDT wallet
- **Strategy:** TrendRiderStrategy v8 (1h trend-following)
- **Database:** Railway PostgreSQL — all trades logged to `trade_log`, hourly snapshots to `bot_status`
- **Telegram:** `@chatteraes_Bot` — trade alerts, 6h status, daily 8AM UTC report
- **Monitor:** background thread sends health checks + writes to Postgres

Railway env vars control live/dry mode:
- `FREQTRADE__DRY_RUN=false` — live trading (set on Railway)
- `FREQTRADE__EXCHANGE__KEY` / `SECRET` — MEXC API keys (Railway secrets)
- Config file always says `dry_run: true` as safe default; env var overrides it

Deploy updates: `cd user_data/deploy && railway up -d`

### Custom strategies

Three strategies in `user_data/strategies/`:

- **TrendRiderStrategy** — production strategy. 1h trend-following with bear market protection, 48h cooldown, stoploss guard. Profitable in backtests (+0.88% in -48% bear market). Writes trades to PostgreSQL, sends clean Telegram notifications.
- **DailyProfitStrategy** — dual-mode (bull pullback + bear bounce). Experimental.
- **MomentumScalpStrategy** — 1h momentum scalping. Experimental.

### Configs

- `user_data/config_mexc.json` — MEXC, TrendRider, $25 wallet, Telegram enabled (production)
- `user_data/config_trendrider.json` — Kraken, TrendRider, $1000 wallet
- `user_data/config_dryrun.json` — Kraken, SampleStrategy, $1000 wallet

API server on port 8080 (credentials: `freqtrader` / `SuperSecurePassword`). FreqUI installed.

### Backtesting data

Historical 1h data from Gate.io is stored at `user_data/data/gate/` for: BTC, ETH, SOL, XRP, ADA, DOT, LINK, AVAX, DOGE, SUI, PEPE, WIF (all /USDT). Refresh with:
```bash
freqtrade download-data --exchange gate --pairs BTC/USDT ETH/USDT SOL/USDT --timeframe 1h --timerange 20250201- --userdir user_data
```

### Exchange accessibility from this VM

- **Accessible:** MEXC, Gate.io, OKX, Bitget, Hyperliquid, Kraken
- **Geo-blocked:** Binance (HTTP 451), Bybit (CloudFront 403)
- **Futures support in Freqtrade:** Gate.io and OKX (spot + isolated margin). MEXC and Bitget are spot-only in Freqtrade.
- **Hyperliquid** uses USDC pairs (not USDT) and has partial Freqtrade support.

### Gotchas

- **TA-Lib C library** must be installed system-wide (`/usr/local/lib/libta_lib.so`). Already installed in the snapshot.
- **No test file at root for the bot module** — core bot tests are under `tests/freqtradebot/`, not `tests/test_freqtradebot.py`.
- **`pytest --dist loadscope`** is the default via `pyproject.toml`; `-n auto` works with `pytest-xdist`.
- **Exchange API calls are fully mocked** in the test suite. No API keys or network access are needed to run tests.
- The `user_data/` directory is gitignored and used for runtime data (strategies, configs, trade DBs). Not needed for tests.
- **Gate.io limits kline downloads** to 10,000 candles per request (~14 months for 1h data).
- **Telegram bot** is connected (token in config, chat_id `5216799062`). Protections (CooldownPeriod, StoplossGuard) are defined in the strategy's `protections` property and require `--enable-protections` for backtesting (live trading enables them automatically).
