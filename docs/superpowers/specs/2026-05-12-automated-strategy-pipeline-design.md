# Automated Strategy Pipeline — Design Spec

**Date:** 2026-05-12
**Scope:** POC — Mean Reversion Strategy, All 6 Modules (Simplified)
**Status:** Approved

---

## Overview

An automated pipeline for cryptocurrency quantitative trading strategy development, built on top of freqtrade's existing capabilities. The pipeline covers the full lifecycle from strategy generation to deployment-ready output.

**Core approach:**
- freqtrade CLI and Python API serve as the execution engine
- A lightweight Orchestrator layer handles automation and scheduling
- Claude API generates strategy code; FreqAI handles ML signals
- Local development + cloud execution for heavy computation

---

## Architecture

Six independent modules communicate through a shared workspace directory. An Orchestrator coordinates execution order and error recovery.

```
┌─────────────────────────────────────────────────────────┐
│                      Orchestrator                        │
│          (CLI trigger / scheduled / single module)       │
└──┬──────┬──────┬──────┬──────┬──────────────────────────┘
   │      │      │      │      │
   ▼      ▼      ▼      ▼      ▼
[Data] [Gen] [Train] [BT] [Opt] [Report]
   │      │      │      │      │      │
   └──────┴──────┴──────┴──────┴──────┘
                    │
              workspace/
              ├── data/          ← historical OHLCV data
              ├── strategies/    ← LLM-generated strategy files
              ├── bt_results/    ← backtesting result JSON
              ├── models/        ← trained FreqAI models
              ├── hyperopt/      ← hyperopt result JSON
              └── reports/       ← leaderboard, analysis, deploy package
```

**Key principles:**
- Each module is an independent Python entry point, callable standalone
- Modules communicate only through the filesystem — no direct coupling
- Orchestrator maintains `pipeline_state.json` for checkpoint/resume
- Identical code runs locally and in cloud; only config differs

---

## Module Definitions

| Module | Input | Execution | Output |
|--------|-------|-----------|--------|
| **1. Data** | pairs, timerange, exchange | `freqtrade download-data` | `workspace/data/` |
| **2. Generator** | strategy direction description | Claude API → Python file | `workspace/strategies/*.py` |
| **3. FreqAI Trainer** | strategy file + data | `freqtrade trade --freqai-config` | `workspace/models/` |
| **4. Backtester** | strategy file + data | `freqtrade backtesting` | `workspace/bt_results/*.json` |
| **5. Hyperopt** | top-N strategies by backtest score | `freqtrade hyperopt` | `workspace/hyperopt/*.json` |
| **6. Reporter** | all result files | Python aggregation script | `workspace/reports/` |

### POC Constraints
- Pairs: BTC/USDT, ETH/USDT — Exchange: Binance
- Timeframe: 1h — Backtest range: 2025-01-01 to 2026-01-01
- LLM generates 3 strategy variants per run
- FreqAI uses LightGBMClassifier with mean reversion features
- Hyperopt runs only on top-1 backtest strategy (100 epochs)
- Reporter outputs Markdown leaderboard + Jupyter notebook

---

## Strategy Generation Module

### Mean Reversion Logic

Price deviates from mean → entry signal → price reverts → exit signal.

Three variants generated per run:

| Variant | Core Indicator | Entry Condition | Exit Condition |
|---------|---------------|-----------------|----------------|
| V1 | Bollinger Bands | Price breaks below lower band | Price returns to mid band |
| V2 | RSI + SMA | RSI < 30 and price below SMA | RSI > 50 |
| V3 | Z-score | Price z-score < -2 | z-score returns near 0 |

### LLM Generation Flow

```
System Prompt
  └── freqtrade IStrategy interface specification
  └── Mean reversion strategy skeleton template
  └── Code format requirements (directly executable)

User Prompt
  └── "Generate variant V1: Bollinger Bands mean reversion strategy"
  └── HyperOpt parameter space declaration

Output
  └── Complete Python strategy file
  └── Syntax validation (ast.parse)
  └── Load validation (freqtrade --strategy-list)
```

### Key Design Decisions
- **Fixed skeleton prompt:** LLM only fills indicator logic inside `populate_entry_trend` / `populate_exit_trend`, reducing generation errors
- **Two-step validation:** syntax check → freqtrade strategy load; retry up to 3 times on failure
- **Embedded parameter space:** each generated strategy file includes `class HyperOpt` declarations so the hyperopt module can use them directly

### FreqAI Features (LightGBM)
- `bb_position`: price position within Bollinger Bands (0–1)
- `rsi_zscore`: z-score of RSI over rolling 50-period window
- `price_deviation`: percentage deviation from 20-period SMA

---

## Orchestrator

### Pipeline State Machine

```
IDLE → DATA → GENERATE → TRAIN → BACKTEST → HYPEROPT → REPORT → DONE
```

`pipeline_state.json` tracks each step:

```json
{
  "run_id": "20260512_mean_reversion",
  "steps": {
    "data":      "done",
    "generate":  "done",
    "train":     "done",
    "backtest":  "running",
    "hyperopt":  "pending",
    "report":    "pending"
  }
}
```

### CLI Interface

```bash
# Run full pipeline
python pipeline.py run --config config/poc.yaml

# Resume from a specific step
python pipeline.py run --from backtest

# Run a single module only
python pipeline.py run --only generate

# Show current pipeline status
python pipeline.py status
```

### Configuration (`config/poc.yaml`)

```yaml
exchange: binance
pairs: ["BTC/USDT", "ETH/USDT"]
timeframe: "1h"
timerange: "20250101-20260101"

llm:
  model: claude-sonnet-4-6
  variants: 3
  strategy_type: mean_reversion

freqai:
  model: LightGBMClassifier
  features: [bb_position, rsi_zscore, price_deviation]

hyperopt:
  top_n: 1
  epochs: 100
  loss_function: SharpeHyperOptLoss
```

### Local vs Cloud Execution

| Scenario | Execution |
|----------|-----------|
| Local dev/debug | `python pipeline.py run --only generate` |
| Local full run | `python pipeline.py run` |
| Cloud scheduled | Same command, Docker container, cron trigger |

---

## Output & Reporting

### Report Structure

```
workspace/reports/
├── leaderboard.md          ← strategy leaderboard
├── best_strategy/
│   ├── BestStrategy.py     ← deployable strategy file
│   ├── best_params.json    ← optimal hyperopt parameters
│   └── config.json         ← complete freqtrade config
└── analysis.ipynb          ← detailed analysis notebook
```

### Strategy Leaderboard (`leaderboard.md`)

| Rank | Strategy | Sharpe | Sortino | Max DD | Total Return | Annual Return | Trades | Win Rate | Avg Win | Avg Loss | Profit Factor | Calmar | Status |
|------|----------|--------|---------|--------|-------------|---------------|--------|----------|---------|----------|---------------|--------|--------|
| 1 | BB_V1 | 1.82 | 2.31 | -12.3% | +34% | +28% | 187 | 58% | +2.1% | -1.4% | 1.50 | 2.28 | ✅ Tuned |
| 2 | RSI_SMA_V2 | 1.41 | 1.76 | -18.1% | +27% | +22% | 243 | 54% | +1.8% | -1.5% | 1.20 | 1.22 | — |
| 3 | ZScore_V3 | 0.93 | 1.12 | -24.5% | +19% | +15% | 312 | 51% | +1.5% | -1.6% | 0.94 | 0.61 | — |

All metrics extracted directly from freqtrade backtest result JSON.

### Deployable Strategy Package (top-1)
- Strategy `.py` with optimal parameters written into `buy_params` / `sell_params`
- freqtrade `config.json` ready to run
- `deploy.sh` one-click start script

### Jupyter Notebook Contents
- Equity curve chart
- Monthly returns heatmap
- Trade distribution analysis
- Simple walk-forward validation (quarterly split of backtest range)

---

## Directory Layout

```
pipeline/
├── pipeline.py             ← Orchestrator entry point
├── config/
│   └── poc.yaml
├── modules/
│   ├── data.py
│   ├── generator.py        ← Claude API integration
│   ├── trainer.py          ← FreqAI wrapper
│   ├── backtester.py
│   ├── hyperopt.py
│   └── reporter.py
├── prompts/
│   └── mean_reversion.md   ← LLM system prompt template
└── workspace/              ← runtime data (gitignored)
```

---

## Out of Scope (POC)

- Live paper trading or dry-run validation
- Multi-exchange support
- Portfolio-level optimization across multiple strategies
- Advanced walk-forward or Monte Carlo simulation
- Web UI or dashboard
