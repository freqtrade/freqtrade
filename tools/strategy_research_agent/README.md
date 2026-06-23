# Strategy Research Agent

This directory contains the versioned source for the local Freqtrade strategy research agent.

Runtime files are installed into `user_data/` because Freqtrade expects strategies, reports, data, and local configs there. The repository should version the agent code and templates here, not local reports, market data, secrets, or backtest exports.

## Layout

- `strategy_research/`: research agent scripts, experiment templates, source registry, launchd templates, and documentation.
- `strategies/research_generated/`: generated strategy source files that are safe to version.
- `download_binance_um_1m.py`: incremental Binance USDT-M 1m OHLCV updater.
- `install_runtime.sh`: deploys the versioned source back into `user_data/`.
- `install_runtime.ps1`: Windows PowerShell runtime installer.
- `README_WINDOWS.md`: Windows PowerShell and Task Scheduler setup guide.

## Install Runtime Files

```bash
tools/strategy_research_agent/install_runtime.sh
```

Windows PowerShell:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\tools\strategy_research_agent\install_runtime.ps1
```

This copies source files to:

```text
user_data/strategy_research/
user_data/strategies/research_generated/
user_data/download_binance_um_1m.py
```

It does not copy local market data, reports, dashboards, API keys, Freqtrade configs, or backtest result archives into git.

## Run

Manual research cycle:

```bash
user_data/strategy_research/start_manual_research.sh --full
```

Quick manual refresh without rerunning backtests:

```bash
user_data/strategy_research/start_manual_research.sh --quick
```

Autonomous hypothesis generation plus short smoke backtests:

```bash
user_data/strategy_research/start_manual_research.sh --autonomous-smoke
```

Failure-driven V2 generation plus short smoke backtests:

```bash
user_data/strategy_research/start_manual_research.sh --iterate-smoke
```

Walk-forward validation across fixed calendar windows:

```bash
user_data/strategy_research/start_manual_research.sh --walk-forward
```

Promotion gate for manual dry-run review readiness:

```bash
user_data/strategy_research/start_manual_research.sh --promotion-gate
```

Preflight only:

```bash
user_data/strategy_research/start_manual_research.sh --preflight-only
```

Lower-level full cycle:

```bash
user_data/strategy_research/run_full_research_cycle.sh --skip-aux-fetch
```

Refresh report/dashboard without rerunning backtests:

```bash
PYTHONPATH=user_data/offline_exchange ./.venv/bin/python user_data/strategy_research/run_research_agent.py --skip-backtests
```

Install local launchd automation:

```bash
user_data/strategy_research/automation/install_launchd.sh
```

On Windows, use `README_WINDOWS.md` for the PowerShell cycle runner and Task Scheduler installation.

## Safety Boundary

- Research only.
- No live trading startup.
- No live API key access.
- No generated reports, market data, or local credentials should be committed.
- Autonomous strategies are generated from auditable local blueprints, not opaque external code.
- Iterated strategies must record the failed parent strategy, previous metrics, and the reason for the change.
- Walk-forward validation must reject strategies that only work in one favorable calendar window.
- Promotion gate only records readiness for manual dry-run review; it never starts dry-run/live trading.
