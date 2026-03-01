# Genetic Algorithm Trading System — Roadmap

**Created:** 2026-03-01  
**Status:** Foundation laid. Core GA engine is stable and runs end-to-end.  
**Last validated run:** 9 gens, 0 eval failures, convergence early-stop, 410/411 tests passing.

---

## Current State Summary

The GA engine is **production-stable for iteration**. It can:
- Generate, evaluate, and evolve trading strategies across multiple pairs
- Walk-forward validation with train/val windows
- Holdout monitoring with early-stop on overfitting
- Monte Carlo robustness analysis
- Hall of Fame persistence with injection
- Parsimony pressure (parallel component removal trials)
- Adaptive mutation, fitness sharing, convergence detection
- Feature importance tracking

**Known limitations:**
- Strategies converge quickly to low-fitness plateaus (best ~0.21 fitness, ~0.3% profit)
- Monte Carlo robustness is fragile for most evolved strategies
- Regime detection is implemented but not yet producing meaningful differentiation
- Some generated strategy code can be syntactically invalid (handled gracefully — zero fitness)

---

## Branch 1: Web Interface & Monitoring Dashboard

**Goal:** Real-time visual monitoring, parameter exploration, and live evolution steering.

### Phase 1 — Read-Only Dashboard
- [ ] **Web server skeleton** — FastAPI/Flask backend serving a React/Vue frontend
- [ ] **WebSocket real-time feed** — Push generation stats, fitness curves, diversity metrics as they happen
- [ ] **Fitness evolution chart** — Line chart of best/avg/worst fitness per generation
- [ ] **Population overview** — Scatter plot of individuals (fitness vs complexity, or 2D PCA)
- [ ] **Strategy inspector** — Click on a strategy to see its gene (indicators, conditions, parameters)
- [ ] **Hall of Fame viewer** — Browse and compare top strategies across runs
- [ ] **Log viewer** — Streaming log output with severity filtering
- [ ] **Run history** — List past runs with summary stats, compare runs side-by-side

### Phase 2 — Parameter Exploration
- [ ] **Config editor** — Edit `ga_config.yaml` through the UI with validation
- [ ] **Parameter impact visualization** — Show how fitness weights, mutation rates, etc. affect outcomes
- [ ] **Indicator analysis** — Which indicators appear most in top strategies? Feature importance heatmap
- [ ] **Walk-forward window inspector** — Per-window train/val breakdown, identify problematic windows
- [ ] **Overfitting dashboard** — Holdout degradation, Monte Carlo robustness, composite score gauges

### Phase 3 — Live Steering & State Management
- [ ] **Checkpoint save/load** — Save full evolutionary state (population, stats, RNG state) on demand
- [ ] **Graceful interrupt** — Signal handler to snapshot state before shutdown
- [ ] **Resume from checkpoint** — Start evolution from any saved state
- [ ] **Live parameter adjustment** — Change mutation rate, crossover rate, population size mid-run
- [ ] **Manual individual injection** — Add hand-crafted strategies to the population
- [ ] **Pause/resume evolution** — Pause, inspect, modify, resume
- [ ] **Kill/restart individual evaluations** — Cancel stuck backtests

### Foundation needed now (before branch starts):
- [ ] Expose evolution engine state via a clean API (not just logging)
- [ ] Decouple evolution loop from console output — emit structured events
- [ ] Ensure checkpoint serialization captures ALL state (best_fitness_ever, no_improvement_count, feature_tracker, etc.)

---

## Branch 2: Regime Detection & Overfitting Prevention

**Goal:** Dramatically improve strategy robustness through smarter regime awareness and anti-overfitting mechanisms.

### Phase 1 — Improved Regime Detection
- [ ] **ADX/DI hysteresis method** — Already configured, needs validation on real data
- [ ] **Hidden Markov Model (HMM) regime detector** — Probabilistic regime classification
- [ ] **Volatility-regime clustering** — K-means on rolling volatility + trend features
- [ ] **Multi-asset regime consensus** — Combine regime signals across pairs for more robust detection
- [ ] **Regime-aware fitness** — Weight fitness by regime performance (penalize strategies that only work in bull markets)
- [ ] **Regime transition detection** — Identify regime changes and measure adaptation speed
- [ ] **Regime visualization** — Color-coded price chart overlays showing detected regimes

### Phase 2 — Anti-Overfitting Mechanisms
- [ ] **Anchored walk-forward** — Expanding training window instead of rolling
- [ ] **Combinatorial purged cross-validation (CPCV)** — Gold standard for financial time series
- [ ] **Probability of backtest overfitting (PBO)** — De Prado's method for measuring overfit probability
- [ ] **White's Reality Check / SPA test** — Statistical test for strategy significance
- [ ] **Deflated Sharpe Ratio** — Adjust for multiple testing (number of strategies tried)
- [ ] **Embargo periods** — Larger gaps between train/test to prevent leakage
- [ ] **Noise injection during evaluation** — Add random slippage/fee variation to backtest
- [ ] **Out-of-distribution detection** — Flag when market conditions differ from training data

### Phase 3 — Fitness Function Evolution
- [ ] **Multi-objective Pareto optimization** — Optimize profit AND robustness simultaneously (NSGA-II)
- [ ] **Robustness-weighted fitness** — Monte Carlo median replaces single backtest result
- [ ] **Time-decay fitness** — More recent performance weighted higher
- [ ] **Drawdown-duration penalty** — Penalize long underwater periods, not just depth
- [ ] **Tail-risk metrics** — CVaR, max consecutive losses, recovery time

### Foundation needed now:
- [ ] Clean up `regime_aware` config — remove `sma_adx` default, validate `adx_di_hysteresis` works
- [ ] Ensure regime detection method is pluggable (strategy pattern)
- [ ] Add regime metadata to backtest results for downstream analysis

---

## Branch 3: Code Quality & Tech Debt

**Priority items to address across branches.**

### Pre-existing Issues
- [ ] **Fix `test_code_generation_informative_condition_suffix`** — Multi-TF condition suffix test (1 failing test)
- [ ] **Strategy code validation** — Pre-flight `compile()` check before submitting for backtest (some generated strategies produce invalid Python)
- [ ] **CDL indicator availability mismatch** — `CDL_MORNINGSTAR`/`CDL_EVENINGSTAR` have param configs but are NOT in the `available` list in `ga_config.yaml`; can appear via HOF injection
- [ ] **Stale checkpoint data** — `checkpoints_feature_test/` contains CDL types with cascading `_0_0_0` suffixes. Won't crash (stripped at load time) but should be cleaned or regenerated
- [ ] **Config: set `max_mutation_rate` explicitly** — Currently defaults to 0.5 with info log; should be explicit in `ga_config.yaml`
- [ ] **Clean up `tmpum08mhsb.yaml`** — Stale smoke-test config in repo root

### Code Improvements
- [ ] **Ensure `ga_generated` directory auto-creation** — Log warning when missing
- [ ] **Rate-limit zero-trade warnings** — Avoid log spam from walk-forward windows with no trades
- [ ] **Add type hints** — Core modules (evolution.py, crossover.py, mutation.py) lack some type annotations
- [ ] **Structured logging** — Emit JSON-structured events alongside human-readable logs (needed for web dashboard)
- [ ] **Test coverage** — Add tests for: `_enforce_max_indicators`, holdout convergence fix, parsimony ID restore

---

## Bug Fix History (Completed)

### Session: 2026-03-01

| # | Severity | Fix | File |
|---|----------|-----|------|
| BUG-1 | Critical | Holdout penalty corrupts `best_individual` tracking → premature convergence. Fixed by snapshotting `best_fitness_ever` before holdout runs. | `evolution.py` |
| BUG-2 | High | Indicator count exceeds `max_per_strategy` after crossover. Added `_enforce_max_indicators()` in all 3 crossover methods. | `crossover.py` |
| BUG-3 | Medium | Zero-trade WF windows report 100% drawdown (sentinel `1.0`). Changed to `0.0`. | `fitness.py` |
| MINOR-1 | Low | Extreme Sharpe/Sortino values (43M+) displayed unclamped. Clamped to [-10, 50]. | `fitness.py` |
| MINOR-2 | Low | Parsimony trial IDs (9000+) not restored after acceptance. Now restores original ID. | `parallel.py` |
| BONUS | Medium | `StrategyGene.copy()` lost all `instance_id` values — broke parsimony cascade removal. Fixed by calling `assign_instance_ids()` after `from_dict()`. | `strategy_gene.py` |
| TEST | — | Updated stale `composite_warning` assertion from 0.35 → 0.25. | `test_overfit_analysis.py` |

### Previous Session

| # | Fix | File |
|---|-----|------|
| Atomic writes | Race condition in parallel strategy file writes. Added tempfile + `os.replace`. | `direct_backtester.py` |
| Parsimony IDs | Filename collisions from duplicate parsimony trial names. Added 9000+ namespace. | `parallel.py` |
| Regime default | Broken regime detection — `sma_adx` method referenced non-existent function. Changed default to `adx_di_hysteresis`. | `dataset_policy.py`, `ga_config.yaml` |
| CDL corruption | Cascading `_0` suffixes on candlestick indicator types. Added `_strip_cdl_suffixes()`. | `strategy_gene.py` |

---

## Priority Order

1. **Branch 3 quick wins** — Fix the remaining test, add strategy code validation, clean up config (1-2 hours)
2. **Branch 1 Phase 1** — Read-only dashboard with WebSocket streaming (1-2 days)
3. **Branch 2 Phase 1** — Regime detection improvements (1-2 days)
4. **Branch 1 Phase 2** — Parameter exploration UI (1-2 days)
5. **Branch 2 Phase 2** — Anti-overfitting mechanisms (2-3 days)
6. **Branch 1 Phase 3** — Live steering (2-3 days, depends on solid checkpoint system)
7. **Branch 2 Phase 3** — Fitness function evolution (2-3 days)
