# Strategy Research Agent

This directory contains the versioned source for the local Freqtrade strategy research agent.

Runtime files are installed into `user_data/` because Freqtrade expects strategies, reports, data, and local configs there. The repository should version the agent code and templates here, not local reports, market data, secrets, or backtest exports.

## Layout

- `strategy_research/`: research agent scripts, experiment templates, source registry, launchd templates, and documentation.
- `strategy_research/knowledge/knowledge_cards/`: short, traceable knowledge cards that are safe to version. These are summaries and testable hypotheses, not raw transcripts or book/article copies.
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

Quick manual refresh without rerunning backtests:

```bash
user_data/strategy_research/start_manual_research.sh --quick
```

External source discovery and review queue:

```bash
user_data/strategy_research/start_manual_research.sh --source-scout
```

Rebuild the Agent brain from the versioned knowledge cards, local research memory, and consolidation policy:

```bash
user_data/strategy_research/start_manual_research.sh --agent-brain
```

Weekly external knowledge update:

```bash
user_data/strategy_research/start_manual_research.sh --weekly-knowledge-update
```

This refreshes external/source knowledge, optionally refreshes Bilibili subtitles without downloading video, rebuilds the knowledge graph, rebuilds research memory, refreshes the consolidation layer, and updates the dashboard/report. It writes:

```text
user_data/strategy_research/knowledge_updates/latest_weekly_knowledge_update.md
```

Rebuild the Agent brain prerequisites:

```bash
user_data/strategy_research/start_manual_research.sh --agent-brain
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

The response queue records execution history and skips duplicate `strategy + experiment + command` items inside its cooldown window, so long research loops rotate instead of repeatedly executing the same action.

Walk-forward validation across fixed calendar windows:

```bash
user_data/strategy_research/start_manual_research.sh --walk-forward
```

Promotion gate for manual dry-run review readiness. This is a strategy-family
risk gate, not a naked all-regime strategy gate:

```bash
user_data/strategy_research/start_manual_research.sh --promotion-gate
```

Equivalent explicit entrypoint:

```bash
user_data/strategy_research/start_manual_research.sh --family-risk-gate
```

Analyze exported trades for behavior-level diagnostics:

```bash
user_data/strategy_research/start_manual_research.sh --trade-behavior
```

Build cross-evidence failure attribution:

```bash
user_data/strategy_research/start_manual_research.sh --failure-attribution
```

Run the mandatory post-backtest attribution gate and refresh research memory:

```bash
user_data/strategy_research/start_manual_research.sh --post-run-attribution
```

This is the same Strategy Research Agent, not a separate Agent. It must run after every backtest-driven strategy research round before the next experiment queue changes. The gate separates signal edge, entry timing, exit quality, cost/funding drag, fixed 50x risk amplification, regime dependency, and sample validity.

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

Refresh report/dashboard without rerunning backtests:

```bash
PYTHONPATH=user_data/offline_exchange ./.venv/bin/python user_data/strategy_research/run_research_agent.py --skip-backtests
```

Install local launchd automation:

```bash
user_data/strategy_research/automation/install_launchd.sh
```

The launchd templates include:

- weekly external knowledge update

On Windows, use `README_WINDOWS.md` for the PowerShell cycle runner and Task Scheduler installation.

## Safety Boundary

- Research only.
- No live trading startup.
- No live API key access.
- No generated reports, market data, or local credentials should be committed.
- Generated strategies must come from knowledge graph, research memory, factor/event evidence, and explicit strategy-family contracts.
- Memory-guided variants must lock the current futures risk policy: isolated USDT-M futures, 50x cap, ROI `{"0":1.20,"180":1.50,"360":1.00}`, and stoploss `-0.60`.
- Walk-forward validation must reject strategies that only work in one favorable calendar window.
- Promotion gate only records readiness for manual dry-run review; it never starts dry-run/live trading.
- Promotion is evaluated by strategy family: target-regime edge may be specialized, but hostile-regime losses must be contained by router, cooldown, drawdown, and consecutive stop-loss circuit breakers.
- External source scouting queues untrusted online/open-source material for bounded snapshot, review, and isolated translation.
- Trade behavior analysis explains wins, losses, long/short skew, stop-loss exits, and entry excursion quality.
- Failure attribution combines scorecards, promotion blockers, and trade behavior into ranked root causes.
- Strategy lineage links base strategies, generated variants, candidate pools, promotion blockers, and failure modes into a reusable research library.
- Research memory turns current evidence into next-focus items, avoid patterns, knowledge gaps, and durable research rules.
- Memory-guided hypotheses convert that memory into auditable next strategy-design plans with explicit blockers and success gates.
- Memory-guided strategy variants turn actionable non-verification plans into isolated Freqtrade subclasses.
- The knowledge graph and knowledge-guided hypotheses use only active short knowledge cards. Raw Bilibili subtitles, PDFs, browser cookies, web snapshots, generated graph artifacts, and weekly update reports remain local runtime data and should not be committed.
- Weekly external knowledge updates are an external iteration loop. They may create new research hypotheses, but they never promote strategies without backtesting, recursive-analysis, lookahead-analysis, regime matrix, cost stress, and promotion-gate review.
