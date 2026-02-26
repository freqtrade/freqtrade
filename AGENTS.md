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

### Custom strategies

Two custom strategies are available in `user_data/strategies/`:

- **TrendRiderStrategy** (`config_trendrider.json`): 1h trend-following, profitable in backtests (+0.88% in -48% bear market). Uses Kraken, 8 pairs, $1000 wallet, Telegram enabled. Run with:
  ```bash
  freqtrade trade --config user_data/config_trendrider.json --strategy TrendRiderStrategy --userdir user_data
  ```
- **MomentumScalpStrategy** (`config_50challenge.json`): 1h momentum scalping for small accounts ($50). Experimental.

### Dry-run (paper trading)

Multiple configs available:
- `user_data/config_dryrun.json` — SampleStrategy, Kraken, $1000 wallet
- `user_data/config_trendrider.json` — TrendRider v8, Kraken, $1000 wallet, Telegram connected
- `user_data/config_50challenge.json` — MomentumScalp, Gate.io, $50 wallet

API server runs on port 8080 (credentials: `freqtrader` / `SuperSecurePassword`). FreqUI is installed.

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
