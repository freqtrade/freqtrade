# Genetic Algorithm Documentation Index

This folder contains all documentation for the FreqTrade Genetic Algorithm Strategy Optimizer.

## 📂 Directory Structure

```
docs/
├── INDEX.md                  # This file
├── features/                 # Feature documentation
├── plans/                    # Roadmaps and improvement plans
├── plots/                    # Generated visualization plots
└── troubleshooting/          # Bug fixes and debugging guides
```

---

## 📖 Features Documentation (`features/`)

### Core Features

| Document | Description | Status |
|----------|-------------|--------|
| [CONFIG_REFERENCE.md](features/CONFIG_REFERENCE.md) | Complete configuration reference for ga_config.yaml | ✅ Complete |
| [VISUALIZATION_GUIDE.md](features/VISUALIZATION_GUIDE.md) | How to visualize evolution progress and results | ✅ Complete |
| [MAX_OPEN_TRADES_FEATURE.md](features/MAX_OPEN_TRADES_FEATURE.md) | Per-strategy max_open_trades evolution | ✅ Complete |

### Walk-Forward Optimization

| Document | Description | Status |
|----------|-------------|--------|
| [WALK_FORWARD_GUIDE.md](features/WALK_FORWARD_GUIDE.md) | User guide for walk-forward optimization | ✅ Complete |

### Parallel Evaluation

| Document | Description | Status |
|----------|-------------|--------|
| [PARALLEL_EVALUATION_GUIDE.md](features/PARALLEL_EVALUATION_GUIDE.md) | Multi-process parallel backtesting | ✅ Complete |

### Market Regime Detection

| Document | Description | Status |
|----------|-------------|--------|
| [MARKET_REGIME_DATASET_SELECTION.md](features/MARKET_REGIME_DATASET_SELECTION.md) | Concepts and design for regime-aware evaluation | ✅ Complete |
| [REGIME_DETECTION_IMPLEMENTATION.md](features/REGIME_DETECTION_IMPLEMENTATION.md) | Implementation details and methods | ✅ Complete |

### Tier 3: Robustness & Anti-Overfitting

| Document | Description | Status |
|----------|-------------|--------|
| [TIER3_ROBUSTNESS_FEATURES.md](features/TIER3_ROBUSTNESS_FEATURES.md) | Monte-Carlo, Parsimony, Pareto Archive, Dynamic Bounds | ✅ Complete |

### LLM Strategy Designer (Phase 1A)

| Document | Description | Status |
|----------|-------------|--------|
| [LLM_STRATEGY_DESIGNER_GUIDE.md](features/LLM_STRATEGY_DESIGNER_GUIDE.md) | How to use LLMs to seed and diversify the GA population | ✅ Complete |

---

## 📋 Plans & Roadmaps (`plans/`)

| Document | Description |
|----------|-------------|
| [MASTER_PLAN.md](plans/MASTER_PLAN.md) | **Consolidated roadmap** with all planned improvements |
| [TODO_ga_improvements.md](plans/TODO_ga_improvements.md) | Detailed TODO list for GA improvements |
| [PHASE_6_PROGRESS.md](plans/PHASE_6_PROGRESS.md) | Phase 6: Regime Detection Accuracy (Complete) |

---

## 🔧 Troubleshooting (`troubleshooting/`)

| Document | Description |
|----------|-------------|
| [TROUBLESHOOTING.md](troubleshooting/TROUBLESHOOTING.md) | Consolidated troubleshooting guide |

---

## 📊 Implementation Status Overview

| Feature | Status | Notes |
|---------|--------|-------|
| Walk-Forward Optimization | ✅ Complete | Time-based train/validation splits |
| Multi-Timeframe Strategies | ✅ Complete | Higher timeframe indicators |
| NSGA-II Multiobjective | ✅ Complete | Pareto-optimal strategy selection |
| Parallel Evaluation | ✅ Complete | Multi-process backtesting |
| Market Regime Detection | ✅ Complete | Phase 5/5 implemented |
| **Regime Detection Accuracy** | ✅ Complete | Phase 6 - adx_di_hysteresis method |
| Elite Fitness Caching | ✅ Complete | Prevents fitness degradation |
| **Monte-Carlo Robustness** | ✅ Complete | Tier 3 - Trade permutation testing |
| **Parsimony Pressure** | ✅ Complete | Tier 3 - Strategy simplification |
| **Pareto Archive** | ✅ Complete | Tier 3 - Non-dominated solution preservation |
| **Dynamic Bounds** | ✅ Complete | Tier 3 - Evolvable parameter ranges |
| Island Model | 📋 Planned | Future enhancement |
| **LLM Strategy Designer** | ✅ Complete | Phase 1A — multi-provider seeding & immigrants |

---

## 🔧 Configuration Files

| Config | Purpose | Runtime |
|--------|---------|---------|
| [ga_config_fast.yaml](../config/ga_config_fast.yaml) | Quick testing (15 pop, 5 gen) | 2-5 min |
| [ga_config_medium.yaml](../config/ga_config_medium.yaml) | Balanced search (40 pop, 15 gen) | 15-30 min |
| [ga_config_deep.yaml](../config/ga_config_deep.yaml) | Production search (100 pop, 50 gen) | 4-8 hours |

---

*Last Updated: March 2026*
