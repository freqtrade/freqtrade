# GA Improvements TODO

Last Updated: 2026-02-18 (Session 3)  
Based on: Comprehensive code audit + deep code review of all core modules

---

## ⚠️ CRITICAL BUGS (Fix Immediately)

These bugs silently corrupt the evolutionary process. **Fix before any other work.**

### Active Bugs

**No active critical bugs remaining!** All critical bugs have been fixed. 🎉

### Previously Fixed (Verified ✅)

- [x] **Fix shallow copy corruption in crossover operators** (Commit: 095490b)  
  Location: `crossover.py` — `single_point_crossover()`, `uniform_crossover()`, `component_crossover()`  
  Fixed: Added `copy.deepcopy()` for all IndicatorGene and ConditionGene objects during crossover
  
- [x] **Fix shallow copy in StrategyGene.to_dict() / copy()** (Commit: 095490b)  
  Location: `strategy_gene.py:100` — `to_dict()` now uses `dict(ind.parameters)` to copy parameter dicts  
  Fixed: Parameters are now properly copied during serialization
  
- [x] **Fix fitness weight sum not normalized** (Commit: b9bc89d)  
  Location: `fitness.py:181-210` — weights now normalized to sum to 1.0 at runtime  
  Fixed: All fitness weights are normalized, preventing inflation from default values

- [x] Fix mutate_adaptive_per_gene() crash when Individual.fitness is None
- [x] Normalize minimal_roi keys to strings everywhere (Commit: 9f6375c)
- [x] Preserve trailing_stop parameters in StrategyGene serialization (Commit: 9f6375c)
- [x] Replace unsafe generator fallback condition (Commit: 9f6375c)
- [x] Remove mutation double-gating (Commit: 9f6375c)
- [x] Guard random.sample() in indicator selection (Commit: 9f6375c)

---

## 🎯 NEXT STEP: Medium-Scope Improvements

> **All critical bugs and quick wins completed!** The genetic algorithm now has a solid foundation. Next priorities are performance improvements and enhanced representation.

---

## 🎯 QUICK WINS (High Impact, Low Effort)

**All quick wins completed!** 🎉

### Completed ✅

- [x] **Stop re-evaluating elite individuals** (Commit: bf295b9)  
  Location: `evolution.py:257-267`  
  Fixed: Elite copies now carry over fitness/metrics and are marked as evaluated  
  Impact: Significant speedup — saves ~3 backtests/generation × N generations

- [x] **Fix population size overshoot** (Commit: bf295b9)  
  Location: `evolution.py:340-350`  
  Fixed: Check size before adding each child individually  
  Impact: Population size now exactly matches configuration

- [x] **Normalize fitness weights at runtime** (Commit: b9bc89d)  
  Fixed: See critical bugs section above  
  Impact: Fitness scores now comparable across different configurations

- [x] **Remove dead strategy_name logic in FitnessEvaluator** (Commit: b9bc89d)  
  Location: `fitness.py:45-70`  
  Fixed: Removed unreachable code, use `generated_name` directly  
  Impact: Cleaner code, slightly improved performance

### Previously Completed ✅

- [x] Separate raw_fitness from shared_fitness
- [x] Restrict indicators.available to fully-supported indicators only
- [x] Add deterministic seeding support
- [x] Fix strategy_name duplication
- [x] Add parent uniqueness check
- [x] Complete logging configuration

---

## 🏗️ MEDIUM SCOPE (2-5 days each)

### Performance

- [x] **Cache pairwise distances per generation** (Commit: d6c48a2)  
  Location: `population.py` — `apply_fitness_sharing()` and `calculate_genetic_diversity()`  
  Fixed: Created `calculate_pairwise_distances()` helper that computes O(n²) distances once  
  Impact: Both functions now accept optional `distance_matrix` parameter; evolution.py computes once and reuses  
  Result: 2x speedup on diversity/sharing calculations; critical for pop_size > 50

### Encoding & Representation

- [ ] **Upgrade to instance-based indicator encoding**  
  Current: Indicators referenced by type name (ambiguous if same type used twice)  
  New: Unique instance IDs: `RSI_0(period=7)`, `RSI_1(period=21)`  
  Benefits: Clear crossover semantics, better genetic distance metrics, no ambiguity in condition references  
  Effort: 2-3 days

### Previously Completed ✅

- [x] Add complexity penalty to fitness (Commit: 5384ed0)
- [x] Unit tests for mutation operators (Commit: f6f9be0)
- [x] Integration test: run 1 generation on test data (Commit: c59483d)

---

## 🚀 MAJOR FEATURES (1-2 weeks each)

### 🏆 TOP PRIORITY: Walk-Forward Optimization

**Why**: Critical anti-overfitting measure; dramatically improves real-world performance. Without this, evolved strategies are almost certainly overfit to the training data.  
**Effort**: 4-7 days  
**Impact**: ⭐⭐⭐⭐⭐

- [ ] **Implement walk-forward validation**  
  Split backtest timerange into rolling windows  
  Train on N days, validate on next M days, slide forward  
  Fitness = average/min validation score across all windows  
  Config options:  
    - `walk_forward.enabled: true`  
    - `walk_forward.train_days: 60`  
    - `walk_forward.validation_days: 15`  
    - `walk_forward.step_days: 15`  
    - `walk_forward.aggregation: 'mean' | 'min' | 'harmonic_mean'`  
  - **Variants**: Anchored (expanding window) and Rolling (fixed window)

**Implementation steps**:
1. Add timerange splitting logic
2. Modify fitness evaluator to run multiple backtests per strategy
3. Aggregate results across validation windows
4. Add progress tracking (current window X/Y)
5. Cache train-window results for efficiency

---

### 🛡️ HIGH PRIORITY: Multi-Timeframe Strategies

**Why**: Industry standard for robust strategies; huge quality improvement  
**Effort**: 3-5 days  
**Impact**: ⭐⭐⭐⭐⭐

- [ ] **Implement multi-timeframe genome + codegen**  
  Extend `StrategyGene` with `informative_timeframes: List[str]`  
  Generate `@informative()` decorators in strategy code  
  Use Freqtrade's `merge_informative_pair()` for higher-TF data  
  Allow conditions to reference informative columns: `close_1h`, `rsi_1h`  
  - **Example**: Trade 5m, confirm trend on 1h, filter market regime on 4h

**Implementation steps**:
1. Add `informative_timeframes` field to StrategyGene
2. Mutation operator to add/remove/change informative timeframes
3. Condition generator to create cross-timeframe conditions
4. Strategy codegen to emit `@informative()` decorated methods
5. Test with 5m base + 1h informative

---

### 🎨 HIGH PRIORITY: Multiobjective Evolution (NSGA-II)

**Why**: Retain Pareto front of diverse optimal strategies; no fitness weight tuning needed  
**Effort**: 5-10 days  
**Impact**: ⭐⭐⭐⭐

- [ ] **Implement NSGA-II for multiobjective optimization**  
  Replace single fitness scalar with multiple objectives:  
    - Objective 1: Total profit %  
    - Objective 2: Max drawdown (minimize)  
    - Objective 3: Sharpe ratio  
    - Objective 4: Number of trades (Goldilocks)  
  - Implement non-dominated sorting (Pareto fronts)  
  - Implement crowding distance for diversity  
  - Return Pareto front (10-20 strategies) instead of single best  
  - **Libraries**: Consider `pymoo` or implement NSGA-II directly

---

### Other Major Features

- [ ] **Island model with migration** (3-6 days)  
  Run N islands (populations) in parallel  
  Migrate top K individuals every M generations  
  Config already has `island_model` placeholder

- [ ] **Parallel evaluation** (2-4 days)  
  Multiprocessing worker pool for backtest evaluation  
  Each worker gets own DirectBacktester instance  
  Benefits: 4-8x speedup on multi-core systems

- [ ] **Strategy grammar / strongly-typed conditions** (5-10 days)  
  Grammar-based genetic programming (GGP)  
  Prevent semantically invalid rules: `(RSI > 70) AND (RSI < 30)`  
  Type system: `Indicator → Comparison → Condition`

---

## 🔬 RESEARCH & EXPERIMENTAL

**Low priority; explore after core features stable**

- [ ] Multi-exchange evolution (evolve across Binance + Kraken + Coinbase)
- [ ] Portfolio-aware evolution (optimize portfolio Sharpe, not individual profit)
- [ ] Meta-learning (evolve on multiple timeranges, extract generalizable meta-strategy)
- [ ] Transfer learning (seed population with known good strategies)
- [ ] Ensemble strategies (voting/stacking from multiple evolved strategies)
- [ ] Adaptive mutation scheduling (start high, decay over generations)
- [ ] Lexicase selection (instead of tournament/roulette)
- [ ] Archive of high-quality strategies (novelty search + quality)

---

## 📚 References

- [Freqtrade Informative Pairs](https://www.freqtrade.io/en/stable/strategy-customization/#informative-pairs)
- [Freqtrade Hyperopt](https://www.freqtrade.io/en/stable/hyperopt/)
- [Freqtrade Backtesting](https://www.freqtrade.io/en/stable/backtesting/)
- [Freqtrade Data Downloading](https://www.freqtrade.io/en/stable/data-download/)
- [NSGA-II Paper (IEEE)](https://ieeexplore.ieee.org/document/996017)