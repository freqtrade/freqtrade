---
name: freqtrade-research-loop
description: Use when running or upgrading the user's Freqtrade strategy research agent, designing crypto strategy experiments, managing candidate/watchlist/rejected pools, or deciding which Freqtrade backtests, recursive-analysis, lookahead-analysis, walk-forward, matrix, and dashboard steps should run next.
---

# Freqtrade Research Loop

## Purpose

Run Freqtrade strategy research as a reproducible research loop, not one-off parameter guessing.

This skill is for the user’s local personal Freqtrade workspace, usually:

```text
/Users/wangsen/Documents/我的Projects/freqtrade
```

## Safety Boundary

Keep the agent research-only unless the user explicitly asks for dry-run/live configuration work.

Never:
- read or print private API keys
- start live trading
- modify live/dry-run default strategy without explicit approval
- import or execute external strategy repo code directly
- promote from one lucky backtest

## Loop

1. **Preflight**
   ```bash
   user_data/strategy_research/start_manual_research.sh --preflight-only
   ```

2. **Load Agent brain before strategy generation**
   ```bash
   user_data/strategy_research/start_manual_research.sh --agent-brain
   ```

3. **Generate or refresh hypotheses**
   ```bash
   user_data/strategy_research/start_manual_research.sh --factor-research
   user_data/strategy_research/start_manual_research.sh --event-study
   user_data/strategy_research/start_manual_research.sh --memory-guided-hypotheses
   user_data/strategy_research/start_manual_research.sh --memory-guided-strategies
   ```

4. **Backtest with Freqtrade**
   Use the repo agent or native Freqtrade. Always state timeframe, timerange, pair universe, fee, leverage, and config. For this workspace, default to Binance USDT-M futures, isolated margin, fixed 50x, ROI `{"0":1.20,"180":1.50,"360":1.00}`, stoploss `-0.60`, and primary entries on `3m`/`5m`/`15m`.

5. **Diagnose and consolidate**
   ```bash
   user_data/strategy_research/start_manual_research.sh --post-run-attribution
   user_data/strategy_research/start_manual_research.sh --trade-behavior
   user_data/strategy_research/start_manual_research.sh --failure-attribution
   user_data/strategy_research/start_manual_research.sh --mature-researcher
   ```

6. **Turn diagnosis into work**
   ```bash
   user_data/strategy_research/start_manual_research.sh --mature-researcher-queue
   user_data/strategy_research/start_manual_research.sh --execute-mature-researcher
   ```

7. **Validate**
   ```bash
   user_data/strategy_research/start_manual_research.sh --walk-forward
   user_data/strategy_research/start_manual_research.sh --family-risk-gate
   user_data/strategy_research/start_manual_research.sh --promotion-gate
   user_data/strategy_research/start_manual_research.sh --dryrun-risk-preflight
   ```

8. **Refresh dashboard**
   ```bash
   user_data/strategy_research/start_manual_research.sh --quick
   ```

## Promotion Meaning

- `rejected`: keep for learning, do not retest unless hypothesis changes.
- `watchlist`: interesting but incomplete; needs sample, cost, or robustness evidence.
- `research_candidate`: worth deeper validation, not dry-run permission.
- `dryrun_candidate`: only after explicit promotion gates and manual review.

Promotion/family-risk gate results are research evidence even when they fail. After either gate runs, the workflow must rebuild strategy lineage, research memory, and consolidation before refreshing the dashboard.

Before starting any dry-run helper, run the dry-run strategy risk preflight. It
loads the strategy through Freqtrade, lists config overrides, and blocks startup
unless final effective values still match the fixed futures contract: isolated
50x, ROI `{"0":1.20,"180":1.50,"360":1.00}`, stoploss `-0.60`, exchange-side
market stoploss on mark price, callable `custom_exit()` time-stop, and
StoplossGuard `trade_limit=3`.

## Completion Gate

Before saying the agent can do strategy research, verify:

```bash
user_data/strategy_research/start_manual_research.sh --preflight-only
```

For code changes, also run relevant `py_compile` checks and the specific mode touched.
