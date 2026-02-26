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

### Dry-run (paper trading)

To run the bot in dry-run mode (simulated trading against a live exchange, no API keys needed):

```bash
freqtrade trade --config user_data/config_dryrun.json --strategy SampleStrategy --userdir user_data
```

The config at `user_data/config_dryrun.json` uses Kraken with a $1000 USDT paper wallet. The API server runs on port 8080 (credentials: `freqtrader` / `SuperSecurePassword`).

**Note:** Binance is geo-blocked from this cloud VM — use Kraken or Gate.io for dry-run mode.

### Gotchas

- **TA-Lib C library** must be installed system-wide (`/usr/local/lib/libta_lib.so`). The Python `ta-lib` wheel requires it at build and runtime. It is already installed in the snapshot.
- **No test file at root for the bot module** — core bot tests are under `tests/freqtradebot/`, not `tests/test_freqtradebot.py`.
- **`pytest --dist loadscope`** is the default via `pyproject.toml`; `-n auto` works with `pytest-xdist`.
- **Exchange API calls are fully mocked** in the test suite. No API keys or network access are needed to run tests.
- The `user_data/` directory is gitignored and used for runtime data (strategies, configs, trade DBs). It is not needed for tests.
- **Binance is geo-restricted** from this cloud VM. Use Kraken or Gate.io for live/dry-run exchange connectivity.
