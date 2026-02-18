# GA Improvements TODO

Last Updated: 2026-02-18 21:41:11  
Based on: Comprehensive code audit + deep code review of all core modules

---

## ⚠️ CRITICAL BUGS (Fix Immediately)

These bugs silently corrupt the evolutionary process. **Fix before any other work.**

### Active Bugs

- [ ] **🔴 Fix shallow copy corruption in crossover operators**  
  Location: `crossover.py` — `single_point_crossover()`, `uniform_crossover()`, `component_crossover()`  
  Issue: List slicing (`indicators[:point]`) creates new lists but **shares IndicatorGene/ConditionGene object references** between parent and child. When a child is mutated, the parent's genes are corrupted too.  
  Impact: **Catastrophic** — silently destroys population diversity across generations, undermines entire evolution  
  Fix: Deep copy each element during crossover: `[copy.deepcopy(ind) for ind in parent.indicators[:point]]`  
  Effort: 1-2 hours  
  Priority: **FIX FIRST — blocks all other work**

- [ ] **🔴 Fix shallow copy in StrategyGene.to_dict() / copy()**  
  Location: `strategy_gene.py:99` — `to_dict()` references `ind.parameters` dict directly instead of copying  
  Issue: `copy()` → `to_dict()` → `from_dict()` produces a "copy" that **shares mutable parameter dicts** with the original. Mutation of copy corrupts original.  
  Impact: **Catastrophic** — same as above. `copy()` is used in elitism, crossover, and mutation — hit on every generation  
  Fix: `'parameters': dict(ind.parameters)` or `copy.deepcopy(ind.parameters)` in `to_dict()`  
  Effort: 30 minutes

- [ ] **🟠 Fix fitness weight sum not normalized**  
  Location: `fitness.py:182-189` — code defaults sum to 1.0 but configs may omit `sortino_ratio`/`profit_factor`  
  Issue: `ga_config_example.yaml` only has 5 weights (sum=1.0), but code fills in 2 extra defaults (0.15+0.10), making effective total=1.25. Fitness scores inflate by 25% and are not comparable across configs.  
  Fix: Normalize weights to sum to 1.0 at runtime: `total = sum(weights.values()); weights = {k: v/total for k, v in weights.items()}`  
  Effort: 30 minutes

### Previously Fixed (Verified ✅)

- [x] Fix mutate_adaptive_per_gene() crash when Individual.fitness is None
- [x] Normalize minimal_roi keys to strings everywhere (Commit: 9f6375c)
- [x] Preserve trailing_stop parameters in StrategyGene serialization (Commit: 9f6375c)
- [x] Replace unsafe generator fallback condition (Commit: 9f6375c)
- [x] Remove mutation double-gating (Commit: 9f6375c)
- [x] Guard random.sample() in indicator selection (Commit: 9f6375c)

---

## 🎯 NEXT STEP: Fix Shallow Copy Bugs

> **The two shallow copy bugs (crossover + `StrategyGene.copy()`) are the single most impactful fix you can make right now.** They silently corrupt every generation and undermine all evolutionary progress. Every other improvement is pointless until these are fixed.

**Action plan:**
1. Add `import copy` to `crossover.py` and `strategy_gene.py`
2. In `strategy_gene.py:to_dict()`, change `'parameters': ind.parameters` → `'parameters': dict(ind.parameters)`
3. In all three crossover functions, wrap indicator/condition list assignments with deep copies
4. Write a test: create parent, copy, mutate copy's indicator params, assert parent is unchanged
5. Run existing test suite to verify no regressions

---

## 🎯 QUICK WINS (High Impact, Low Effort)

**Target: Complete within 1-2 hours each**

- [ ] **Stop re-evaluating elite individuals**  
  Location: `evolution.py:257-261`  
  Issue: Elite copies are new `Individual` objects with `evaluated=False`, causing redundant backtests every generation  
  Fix: Carry over fitness/metrics to elite copies, or mark as pre-evaluated  
  Effort: 1 hour  
  Impact: Significant speedup (saves ~3 backtests/generation × N generations)

- [ ] **Fix population size overshoot**  
  Location: `evolution.py:301-338`  
  Issue: Crossover produces 2 children per iteration; the `while len < size` loop can overshoot by 1  
  Fix: Check size before adding each child individually, not both at once  
  Effort: 30 minutes

- [ ] **Normalize fitness weights at runtime**  
  See critical bugs section above  
  Effort: 30 minutes

- [ ] **Remove dead strategy_name logic in FitnessEvaluator**  
  Location: `fitness.py:56-68`  
  Issue: Line 67 `if strategy_name is None` is unreachable (already set on line 58)  
  Fix: Remove the dead branch, use `generated_name` directly  
  Effort: 15 minutes

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

- [ ] **Cache pairwise distances per generation**  
  Location: `population.py` — `apply_fitness_sharing()` and `calculate_genetic_diversity()`  
  Issue: Both compute O(n²) pairwise distances independently — computed twice per generation  
  Fix: Compute once, pass distance matrix to both functions  
  Effort: 2-3 hours  
  Impact: 2x speedup on diversity/sharing calculations; critical for pop_size > 50

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