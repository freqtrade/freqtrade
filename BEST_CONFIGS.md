# Best GA Configurations — Production Tracker

> Living document. Updated as new experiments complete.
> Goal: Find 2-3 battle-tested configs for 100+ generation production runs.

---

## Current Best: R5 Multi-Pair (Benchmark v2)

**Status**: PROVEN — 5/5 SAFE, best robustness of all benchmark runs

| Metric | Value |
|--------|-------|
| Best Fitness | 0.4127 |
| Best Profit | +0.17% |
| Sharpe | 4.98 |
| Trades | 85 |
| Overfit Analysis | **5/5 SAFE** (0 overfit, 0 warning) |
| Holdout Degradation | Rank 1: 38.2%, Ranks 2-5: **negative** (holdout > training) |
| Runtime | ~55 min (pop=25, gen=20) |

**Why it works**: Multi-pair training (BTC+ETH+SOL+BNB+XRP) forces cross-asset generalization. Walk-forward + holdout + deflated Sharpe + parsimony all enabled.

**Key parameters**:
- `population_size: 25`, `generations: 20`
- `pairs: [BTC/USDT, ETH/USDT, SOL/USDT, BNB/USDT, XRP/USDT]`
- `walk_forward: enabled, train_days=90, validation_days=30, step_days=15, aggregation=harmonic_mean`
- `holdout_validation: enabled, holdout_pct=0.15`
- `holdout_monitoring: enabled, interval=5, penalty_factor=0.5, trend_early_stop=true`
- `deflated_sharpe: enabled, min_trades=10`
- `parsimony: enabled, tolerance=0.02`
- `strategy TF: 15m`
- Config file: `genetic_algorithm/config/benchmark_v2/run5_multi_pair.yaml`

---

## Runner-Up: R8 Island+Ensemble (Benchmark v2)

**Status**: PROMISING — 3/5 SAFE, master island strongest

| Metric | Value |
|--------|-------|
| Best Fitness (master) | 0.4871 |
| Best Profit | +0.15% |
| Sharpe | 1.33 |
| Trades | 34 |
| Overfit Analysis | **3/5 SAFE**, 2 overfit |
| Holdout Degradation (safe) | ~3.5% |
| Runtime | ~70 min (4×15 pop, 20 gen) |

**Why it's interesting**: Island model with ensemble regime detection. Master island generalizes well; specialist islands overfit. Potential if specialists are constrained better.

**Key parameters**:
- `islands: bullish(15), bearish(15), sideways(15), master(15)`
- `regime_detection: method=ensemble, pair=BTC/USDT, timeframe=1h`
- `deflated_sharpe: enabled`
- `parsimony: enabled`
- Config file: `genetic_algorithm/config/benchmark_v2/run8_island_ensemble.yaml`

---

## Runner-Up: R2 Walk-Forward (Benchmark v2)

**Status**: TRUSTWORTHY — Only 5.2% WF gap, most statistically reliable

| Metric | Value |
|--------|-------|
| Best Fitness | 0.3941 |
| Best Profit | -0.02% |
| WF Gap | 5.2% |
| Overfit Analysis | **1/5 SAFE**, 4 unknown |
| Runtime | ~23 min |

**Why it matters**: Walk-forward validation provides the strongest statistical confidence. The lower fitness is a feature, not a bug — it reflects honest out-of-sample performance.

---

## Experiment Log

### Benchmark v2 (2026-03-12)

| Run | Config | Fitness | Profit | Overfit | Notes |
|-----|--------|---------|--------|---------|-------|
| R1 Baseline | Single-pop, BTC, no validation | 0.4553 | +0.14% | 5 UNKNOWN | No holdout configured; ceiling test |
| R2 Walk-Forward | Single-pop, BTC, WF on | 0.3941 | -0.02% | 1 SAFE, 4 UNK | Most honest result; 5.2% WF gap |
| R4v2 Island+SMA | 4 islands, sma_slope regime | 0.6133 (sideways) | +0.01% | 0 SAFE, 3 OVERFIT | Specialists overfit narrow data |
| R5 Multi-Pair | 5 pairs, WF+holdout+DSR | **0.4127** | **+0.17%** | **5/5 SAFE** | **WINNER** |
| R6 NSGA-II | Multi-objective, BTC | 1.2914 | +1.12% | 0 SAFE, 3 OVERFIT | Degenerate (3 trades); needs min_trades |
| R7 Short Selling | Futures, shorts | 0.0000 | 0.00% | N/A | 0 trades — strategy gen bug |
| R8 Island+Ensemble | 4 islands, ensemble regime | 0.4871 | +0.15% | 3 SAFE, 2 OVERFIT | Master island strong |

### Phase 1: Timeframe Exploration (pending)

| Run | Config | Fitness | Profit | Overfit | Notes |
|-----|--------|---------|--------|---------|-------|
| A1 | R5-improved, 15m, pop=40, gen=30 | — | — | — | Baseline extension |
| A2 | R5-improved, 5m, pop=40, gen=30 | — | — | — | Shorter TF test |
| A3 | R5-improved, 1h, pop=40, gen=30 | — | — | — | Longer TF test |

---

## Configuration Space Notes

### What works (proven)
- Multi-pair training is the single strongest regularizer
- Walk-forward with `harmonic_mean` aggregation catches overfitting
- Holdout monitoring with penalty keeps evolution honest during training
- Deflated Sharpe + parsimony are low-cost improvements

### What doesn't work (proven)
- NSGA-II without min_trades floor → degenerate 3-trade solutions
- Short selling → 0 trades (strategy gen bug, not config issue)
- Island model specialists with < 5 regime segments → overfit narrow data
- SMA-slope regime on 1h without auto-scaling → all-bearish (FIXED)

### Untested combinations
- Multi-pair + Island model (should work but never tested)
- NSGA-II + walk-forward + multi-pair
- CPCV on any multi-pair config
- Monte Carlo robustness on multi-pair
- MTF regime detection (1h+4h+1d) with island model
- 5m or 1h strategy timeframes

### Feature incompatibilities (code-level)
- Island model + Walk-forward: **INCOMPATIBLE** (WF force-disabled in island sub-GAs)
- Island model + Monte Carlo: **NOT SUPPORTED** (skipped with warning)
- Island model + CPCV: **NOT SUPPORTED** (skipped with warning)
- NSGA-II + Fitness sharing: **SHOULD BE OFF** (NSGA-II has its own diversity mechanism)
