# Strategy Research Agent

This directory contains the versioned source for the local Freqtrade strategy research agent.

Runtime files are installed into `user_data/` because Freqtrade expects strategies, reports, data, and local configs there. The repository should version the agent code and templates here, not local reports, market data, secrets, or backtest exports.

## Layout

- `strategy_research/`: research agent scripts, experiment templates, source registry, launchd templates, and documentation.
- `strategies/research_generated/`: generated strategy source files that are safe to version.
- `skills/`: Codex skills for strategy diagnosis, Freqtrade research loops, futures risk, scalping/microstructure, and promotion gates.
- `download_binance_um_1m.py`: incremental Binance USDT-M 1m OHLCV updater.
- `install_runtime.sh`: deploys the versioned source back into `user_data/`.
- `install_skills.sh`: installs versioned strategy research skills into `~/.agents/skills`.
- `install_runtime.ps1`: Windows PowerShell runtime installer.
- `install_skills.ps1`: Windows PowerShell skill installer.
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

## Install Strategy Research Skills

macOS/Linux:

```bash
tools/strategy_research_agent/install_skills.sh
```

Windows PowerShell:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\tools\strategy_research_agent\install_skills.ps1
```

By default, skills are installed into:

```text
~/.agents/skills
```

Override with `CODEX_AGENT_SKILLS_DIR` when needed.

## Run

Manual research cycle:

```bash
user_data/strategy_research/start_manual_research.sh --full
```

Quick manual refresh without rerunning backtests:

```bash
user_data/strategy_research/start_manual_research.sh --quick
```

External source discovery and review queue:

```bash
user_data/strategy_research/start_manual_research.sh --source-scout
```

Integrated strong researcher smoke loop:

```bash
user_data/strategy_research/start_manual_research.sh --strong-researcher-smoke
```

Build senior researcher diagnoses and next-experiment decisions:

```bash
user_data/strategy_research/start_manual_research.sh --mature-researcher
```

Convert those decisions into a safe response queue or execute one queued item:

```bash
user_data/strategy_research/start_manual_research.sh --mature-researcher-queue
user_data/strategy_research/start_manual_research.sh --execute-mature-researcher
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

Build the next research agenda from promotion blockers:

```bash
user_data/strategy_research/start_manual_research.sh --agenda
```

Select the next agenda item without executing it:

```bash
user_data/strategy_research/start_manual_research.sh --next-agenda
```

Analyze exported trades for behavior-level diagnostics:

```bash
user_data/strategy_research/start_manual_research.sh --trade-behavior
```

Plan follow-up experiments from behavior diagnostics:

```bash
user_data/strategy_research/start_manual_research.sh --behavior-experiments
```

Generate strategy variants from behavior experiment plans:

```bash
user_data/strategy_research/start_manual_research.sh --behavior-variants
```

Build cross-evidence failure attribution:

```bash
user_data/strategy_research/start_manual_research.sh --failure-attribution
```

Build strategy library lineage from registries and research evidence:

```bash
user_data/strategy_research/start_manual_research.sh --strategy-lineage
```

Build durable research memory for the next strategy-design loop:

```bash
user_data/strategy_research/start_manual_research.sh --research-memory
```

Plan next strategy hypotheses from research memory:

```bash
user_data/strategy_research/start_manual_research.sh --memory-guided-hypotheses
```

Generate isolated strategy variants from those hypotheses:

```bash
user_data/strategy_research/start_manual_research.sh --memory-guided-strategies
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
- Research agenda turns promotion blockers into auditable next experiments and commands.
- Agenda execution is allowlisted; the default `--next-agenda` mode writes a receipt without running a command.
- External source scouting queues untrusted online/open-source material for bounded snapshot, review, and isolated translation.
- Strong researcher smoke runs source scouting, lineage, memory, hypothesis planning, strategy generation, Freqtrade discovery, smoke backtesting, and report refresh in one research-only loop.
- Trade behavior analysis explains wins, losses, long/short skew, stop-loss exits, and entry excursion quality.
- Behavior-driven experiment planning turns those diagnostics into concrete next variants to test.
- Behavior variants turn experiment plans into isolated, auditable Freqtrade strategy subclasses.
- Failure attribution combines scorecards, promotion blockers, trade behavior, and experiment plans into ranked root causes.
- Strategy lineage links base strategies, generated variants, behavior experiments, candidate pools, promotion blockers, and failure modes into a reusable research library.
- Research memory turns current evidence into next-focus items, avoid patterns, knowledge gaps, and durable research rules.
- Memory-guided hypotheses convert that memory into auditable next strategy-design plans with explicit blockers and success gates.
- Memory-guided strategy variants turn actionable non-verification plans into isolated Freqtrade subclasses.
