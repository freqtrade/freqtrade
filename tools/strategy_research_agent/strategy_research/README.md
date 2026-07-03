# Strategy Research Agent Runtime

This directory contains the versioned source for the local Freqtrade strategy
research agent. Runtime copies live under `user_data/strategy_research/` after:

```bash
tools/strategy_research_agent/install_runtime.sh
```

`user_data/` remains the local runtime area for market data, reports,
dashboards, backtest exports, queues, and private config. Agent capabilities
that must survive a new machine belong here under `tools/strategy_research_agent/`.

## Fixed Research Contract

- Market: Binance USDT-M futures only.
- Margin: isolated.
- Leverage: fixed 50x; generated strategy classes must cap `leverage()` at 50x.
- ROI: `{"0": 1.20, "180": 1.50, "360": 1.00}`.
- Stoploss: `-0.60`.
- Entry timeframes: `3m`, `5m`, `15m`; current generated experiments default to `15m`.
- Background timeframes: `1h` may be used only as confirmation/context, not as primary entry.
- Promotion is family-level: target-regime edge plus hostile-regime loss containment, not naked all-regime performance.

## Required Preload

Every strategy research entrypoint first runs:

```bash
user_data/strategy_research/preflight_research_agent.py
user_data/strategy_research/enforce_agent_workflow_gate.py
```

The gate requires these fixed artifacts to be loadable:

- knowledge graph
- research memory
- consolidation policy
- workflow contract
- data-derived regime window manifest
- regime inference quarantine manifest
- weekly knowledge update layer

## Current Workflow

1. Preflight.
2. Load knowledge graph, research memory, and consolidation rules.
3. Run factor research on `3m`/`5m`/`15m` futures OHLCV.
4. Convert factor candidates into event-study hypotheses.
5. Run event study.
6. Refresh data-derived regime windows and quarantine legacy regime interpretations.
7. Generate memory-guided strategy variants only after the knowledge/memory layers are refreshed.
8. Backtest through Freqtrade.
9. Run post-run attribution.
10. Run failure attribution.
11. Run recursive-analysis and lookahead-analysis for candidates.
12. Run walk-forward validation.
13. Run fee/slippage/funding stress through the promotion/family gate.
14. Run family risk gate.
15. Run promotion gate.
16. Update strategy lineage, research memory, consolidation, dashboard, and registry.

Family-risk and promotion gate results are research evidence even when they
fail. A failed gate must still rebuild lineage, research memory, and
consolidation before the dashboard/report refresh so blockers become durable
experience for the next loop.

## Supported Entrypoints

```bash
user_data/strategy_research/start_manual_research.sh --preflight-only
user_data/strategy_research/start_manual_research.sh --quick
user_data/strategy_research/start_manual_research.sh --source-scout
user_data/strategy_research/start_manual_research.sh --price-action-knowledge
user_data/strategy_research/start_manual_research.sh --bilibili-transcripts
user_data/strategy_research/start_manual_research.sh --knowledge-graph
user_data/strategy_research/start_manual_research.sh --knowledge-guided-hypotheses
user_data/strategy_research/start_manual_research.sh --factor-research
user_data/strategy_research/start_manual_research.sh --factor-to-strategy
user_data/strategy_research/start_manual_research.sh --event-study
user_data/strategy_research/start_manual_research.sh --regime-windows
user_data/strategy_research/start_manual_research.sh --agent-brain
user_data/strategy_research/start_manual_research.sh --weekly-knowledge-update
user_data/strategy_research/start_manual_research.sh --walk-forward
user_data/strategy_research/start_manual_research.sh --promotion-gate
user_data/strategy_research/start_manual_research.sh --family-risk-gate
user_data/strategy_research/start_manual_research.sh --trade-behavior
user_data/strategy_research/start_manual_research.sh --failure-attribution
user_data/strategy_research/start_manual_research.sh --post-run-attribution
user_data/strategy_research/start_manual_research.sh --mature-researcher
user_data/strategy_research/start_manual_research.sh --mature-researcher-queue
user_data/strategy_research/start_manual_research.sh --execute-mature-researcher
user_data/strategy_research/start_manual_research.sh --strategy-lineage
user_data/strategy_research/start_manual_research.sh --research-memory
user_data/strategy_research/start_manual_research.sh --memory-guided-hypotheses
user_data/strategy_research/start_manual_research.sh --memory-guided-strategies
```

Removed legacy entrypoints must not be reintroduced without a new PR and a clear
workflow reason: broad smoke wrappers, all-in-one cycle wrappers, agenda
executors, manual playbook generators, behavior-plan generators, and separate
K-line lab wrappers.

## Current Evidence Outputs

- Dashboard: `user_data/strategy_research/dashboard/index.html`
- Reports: `user_data/strategy_research/reports/`
- Factor research: `user_data/strategy_research/factors/latest_factor_research.md`
- Factor-to-strategy plan: `user_data/strategy_research/factors/latest_factor_strategy_plan.md`
- Event study: `user_data/strategy_research/event_studies/latest_event_study.md`
- Regime windows: `user_data/strategy_research/regime_windows/latest_regime_windows.md`
- Regime quarantine: `user_data/strategy_research/regime_windows/regime_inference_quarantine.md`
- Walk-forward: `user_data/strategy_research/walk_forward_summaries/latest_walk_forward_summary.md`
- Family risk gate: `user_data/strategy_research/family_risk_gate/latest_family_risk_gate.md`
- Promotion report: `user_data/strategy_research/promotion_reports/latest_promotion_report.md`
- Trade behavior: `user_data/strategy_research/trade_behavior/latest_trade_behavior.md`
- Failure attribution: `user_data/strategy_research/failure_attribution/latest_failure_attribution.md`
- Mature researcher: `user_data/strategy_research/mature_researcher/latest_researcher_decision.md`
- Mature queue: `user_data/strategy_research/mature_researcher/latest_response_queue.md`
- Strategy lineage: `user_data/strategy_research/strategy_library/latest_strategy_lineage.md`
- Research memory: `user_data/strategy_research/research_memory/latest_research_memory.md`
- Weekly knowledge update: `user_data/strategy_research/knowledge_updates/latest_weekly_knowledge_update.md`
- Consolidation: `user_data/strategy_research/consolidation/latest_research_consolidation.md`

## Regime Window Builder

Regime labels are generated from local Binance USDT-M BTC/ETH futures OHLCV,
not from hardcoded historical examples. Refresh them with:

```bash
user_data/strategy_research/start_manual_research.sh --regime-windows
```

The builder prefers `1h` futures feather data and resamples `15m`/`5m`/`1m`
futures candles to `1h` when needed. It computes BTC/ETH returns, EMA gaps,
realized volatility, ATR%, BB width, trend strength, range score, and
directional agreement before selecting candidate `bull`, `bear`, `range`, and
`high_vol` windows.

Old manually named windows such as `bull_home`, `range_home`, `bear_home`, and
`high_vol_hostile` are quarantined. Their old reports may remain as raw
date-range backtests, but they must not be used as active regime truth,
promotion evidence, strategy-generation basis, or durable memory until relabeled
against `latest_regime_windows.json`.

## Dry-Run Runtime Safety

The futures dry-run helper is installed to:

```bash
user_data/start_futures_dryrun.sh
```

It sources `~/.freqtrade_telegram_env`, runs a Binance futures ccxt preflight
through the active proxy/VPN environment, and refuses to start if the futures
data path is unavailable. A running process or UI pong is not enough.

Before the helper starts the bot, it also runs:

```bash
user_data/strategy_research/dryrun_strategy_risk_preflight.py
```

The same check can be run manually for all registry candidates:

```bash
user_data/strategy_research/start_manual_research.sh --dryrun-risk-preflight
```

This preflight loads the strategy through Freqtrade, compares strategy-defined
values with config overrides, and blocks dry-run startup if the final effective
contract is not futures/isolated/50x with fixed ROI, fixed stoploss,
exchange-side stoploss, callable custom exits, and the three-stoploss guard.

Important runtime safety settings:

- `ccxt_config.requests_trust_env=true`
- `ccxt_async_config.aiohttp_trust_env=true`
- `order_types.stoploss=market`
- `order_types.stoploss_on_exchange=true`
- `order_types.stoploss_price_type=mark`

Before live review, config parsing is not enough. A tiny-size exchange-side
operation must verify that a filled futures position receives an exchange-side
stop order.

## Live-Review Candidate PR Boundary

If a dry-run passes, do not commit the running bot state. Open a secret-free
live-review candidate PR instead.

Versioned artifacts may include:

- strategy source code
- registry/candidate status such as `dryrun_candidate` or `live_review_candidate`
- promotion and family-risk gate summary evidence
- dry-run/live config templates with no secrets
- live-review checklist
- rollback and emergency-stop runbook

Local-only artifacts must not be committed:

- API keys, exchange secrets, Telegram token, or chat_id
- running process state
- trade/runtime sqlite databases
- full local dashboards, bulky backtest exports, and unsanitized private reports

Live activation always remains a separate manual approval step.
