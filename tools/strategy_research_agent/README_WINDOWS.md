# Strategy Research Agent on Windows

This guide covers native Windows PowerShell usage. WSL and Git Bash can use the macOS/Linux shell scripts, but native Windows should use the `.ps1` scripts.

## Prerequisites

- Git for Windows.
- Python compatible with the Freqtrade checkout.
- Freqtrade dependencies installed in `.venv`.
- Local `user_data` configs created separately. Do not commit API keys, WebUI passwords, market data, reports, or backtest exports.

## Install Runtime Files

Run PowerShell from the repository root:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\tools\strategy_research_agent\install_runtime.ps1
```

This copies versioned source files into:

```text
user_data\strategy_research\
user_data\strategies\research_generated\
user_data\download_binance_um_1m.py
```

It protects local runtime state such as reports, dashboard files, candidates, rejected/watchlist records, data updates, external-source snapshots, and private configs.

## Run Manually

Refresh with local data only:

```powershell
$env:PYTHONPATH = "user_data\offline_exchange"
.\user_data\strategy_research\run_full_research_cycle.ps1 -SkipAuxFetch
```

This generates autonomous research hypotheses, runs a short smoke backtest for them, then runs the existing base/stress matrix checks.

Refresh with Binance funding/mark auxiliary data:

```powershell
$env:PYTHONPATH = "user_data\offline_exchange"
.\user_data\strategy_research\run_full_research_cycle.ps1
```

## Install Scheduled Tasks

Register two Windows Task Scheduler jobs:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\tools\strategy_research_agent\automation\windows_task_scheduler.ps1 -Action install
```

Installed tasks:

- `Freqtrade Strategy Research Daily`: daily at 08:30, skips aux download.
- `Freqtrade Strategy Research Weekly Aux`: Sunday at 09:15, refreshes funding/mark aux data.

Check status:

```powershell
.\tools\strategy_research_agent\automation\windows_task_scheduler.ps1 -Action status
```

Uninstall:

```powershell
.\tools\strategy_research_agent\automation\windows_task_scheduler.ps1 -Action uninstall
```

## Safety Boundary

- Research only.
- Does not start live trading.
- Does not read live API keys.
- Does not expose FreqUI or REST API.
- Does not commit generated reports, market data, backtest archives, or local secrets.
