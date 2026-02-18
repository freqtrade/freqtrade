# GA Improvements TODO

Last Updated: 2026-02-18  
Based on: Comprehensive code audit + current implementation analysis

---

## ⚠️ CRITICAL BUGS (Fix Immediately)

These bugs can crash evolution or corrupt strategy output. **Fix before any other work.**

- [x] **Fix mutate_adaptive_per_gene() crash when Individual.fitness is None**
  - Location: `genetic_algorithm/core/mutation.py:~618`
  - Error: `TypeError: '>' not supported between instances of 'NoneType' and 'int'`
  - Status: **ALREADY FIXED** - Code properly handles None with `if individual.fitness is None or individual.fitness <= 0`
  - No patch needed
  
- [x] **Normalize minimal_roi keys to strings everywhere**
  - Locations: `generator.py:66-72`, `mutation.py:108-112, 411-415, 511-516`
  - Error: KeyError and inconsistent strategy attributes
  - Fix: Use `Dict[str, float]` for minimal_roi, normalize in from_dict()
  - Status: **FIXED** - All locations now use string keys consistently
  - Commit: 9f6375c
  
- [x] **Preserve trailing_stop parameters in StrategyGene serialization**
  - Locations: `strategy_gene.py:124` (to_dict), `:168` (from_dict)
  - Issue: `trailing_stop_positive` and `_offset` silently lost on copy/mutation
  - Fix: Add both fields to to_dict() and from_dict()
  - Status: **FIXED** - Both parameters now serialized/deserialized correctly
  - Commit: 9f6375c
  
- [x] **Replace unsafe generator fallback condition**
  - Location: `generator.py:503`
  - Issue: Returns literal "True" causing scalar mask behavior
  - Fix: Return `"(dataframe['volume'] > 0)"`
  - Status: **FIXED** - Now returns vectorized condition
  - Commit: 9f6375c
  
- [x] **Remove mutation double-gating**
  - Location: `evolution.py:223`
  - Issue: Outer random.random() gate + internal mutation sampling = very low effective rate
  - Fix: Call mutate() unconditionally or ensure at least one operator fires
  - Status: **FIXED** - Removed outer gating, mutate() now handles all probability checks internally
  - Commit: 9f6375c
  
- [x] **Guard random.sample() in indicator selection**
  - Location: `generator.py:58`
  - Issue: ValueError if num_indicators > len(available_indicators)
  - Fix: `num_indicators = min(num_indicators, len(self.available_indicators))`
  - Status: **FIXED** - Guard added before random.sample()
  - Commit: 9f6375c

---

## 🎯 QUICK WINS (High Impact, Low Effort)

**Target: Complete within 1-2 hours each**

- [ ] **Separate raw_fitness from shared_fitness**
  - Store both in Individual class
  - Use raw_fitness for reporting/convergence, shared_fitness for selection only
  - Benefits: Accurate best strategy reporting, proper convergence detection
  
- [ ] **Restrict indicators.available to fully-supported indicators only**
  - Audit which indicators have full codegen + condition support
  - Remove unsupported indicators from config
  - Benefits: No wasted genes, more meaningful variation
  
- [ ] **Add deterministic seeding support**
  - Add `random_seed: int` to config
  - Seed Python random, NumPy, strategy generation
  - Benefits: Reproducible experiments, easier debugging
  
- [ ] **Fix strategy_name duplication**
  - Location: `evolution.py:142`
  - Change `f"Gen{gen}_Ind{individual.id}"` to just `individual.id`
  - Benefits: Cleaner logging, better caching keys
  
- [ ] **Add parent uniqueness check**
  - Config option: `allow_self_crossover: false`
  - Ensure parent1 != parent2 in selection
  - Benefits: More diverse offspring
  
- [ ] **Complete logging configuration**
  - Location: `evolution.py:_setup_logging()`
  - Add file handler with configured format/path
  - Benefits: Better diagnosability for long runs

---

## 🏗️ MEDIUM SCOPE (2-5 days each)

### Encoding & Representation

- [ ] **Upgrade to instance-based indicator encoding**
  - Current: Indicators referenced by type name (ambiguous)
  - New: Unique instance IDs: `RSI_0(period=7)`, `RSI_1(period=21)`
  - Benefits:
    - Clear crossover semantics
    - Better genetic distance metrics
    - No ambiguity in condition references
  - Effort: 2-3 days

### Fitness & Robustness

- [ ] **Add complexity penalty to fitness**
  - Formula: `fitness -= complexity_weight * (num_indicators + num_conditions)`
  - Config: `fitness_penalties.complexity_weight: 0.01`
  - Benefits: Reduces overfitting, promotes simpler strategies
  - Effort: 2-3 hours

### Testing & Validation

- [x] **Unit tests for mutation operators**
  - Test: `test_roi_keys_are_strings_after_mutations` ✅
  - Test: `test_mutate_adaptive_per_gene_handles_none_fitness` ✅ (already fixed)
  - Test: `test_generator_fallback_condition_is_vectorized` ✅
  - Test: `test_strategy_gene_roundtrip_preserves_trailing_stop` ✅
  - Status: **COMPLETED** - See `test_critical_fixes.py`
  - Commit: f6f9be0
  
- [ ] **Integration test: run 1 generation on test data**
  - Validate full pipeline completes
  - Check result parsing
  - Verify caching doesn't break
  - Effort: 4-6 hours
  
- [ ] **Add CI/CD pipeline**
  - pytest + coverage
  - ruff/flake8 + black
  - mypy (GA package only initially)
  - Optional: scheduled weekly smoke run
  - Effort: 1 day

---

## 🚀 MAJOR FEATURES (1-2 weeks each)

### 🏆 TOP PRIORITY: Multi-Timeframe Strategies

**Why**: Industry standard for robust strategies; huge quality improvement  
**Effort**: 3-5 days  
**Impact**: ⭐⭐⭐⭐⭐

- [ ] **Implement multi-timeframe genome + codegen**
  - Extend `StrategyGene` with `informative_timeframes: List[str]`
  - Generate `@informative()` decorators in strategy code
  - Use Freqtrade's `merge_informative_pair()` for higher-TF data
  - Allow conditions to reference informative columns: `close_1h`, `rsi_1h`
  - **Example**: Trade 5m, confirm trend on 1h, filter market regime on 4h
  - **Freqtrade docs**: Already has native support via `@informative()` decorator
  - **Reference**: https://www.freqtrade.io/en/stable/strategy-customization/#informative-pairs

**Implementation steps**:
1. Add `informative_timeframes` field to StrategyGene
2. Mutation operator to add/remove/change informative timeframes
3. Condition generator to create cross-timeframe conditions
4. Strategy codegen to emit `@informative()` decorated methods
5. Test with 5m base + 1h informative

---

### 🛡️ HIGH PRIORITY: Walk-Forward Optimization

**Why**: Critical anti-overfitting measure; dramatically improves real-world performance  
**Effort**: 4-7 days  
**Impact**: ⭐⭐⭐⭐⭐

- [ ] **Implement walk-forward validation**
  - Split backtest timerange into rolling windows
  - Train on N days, validate on next M days, slide forward
  - Fitness = average/min validation score across all windows
  - Config options:
    - `walk_forward.enabled: true`
    - `walk_forward.train_days: 60`
    - `walk_forward.validation_days: 15`
    - `walk_forward.step_days: 15`
    - `walk_forward.aggregation: 'mean' | 'min' | 'harmonic_mean'`
  - **Variants**:
    - Anchored: Expanding training window
    - Rolling: Fixed training window size
  - **Reference**: https://www.freqtrade.io/en/stable/hyperopt/

**Implementation steps**:
1. Add timerange splitting logic
2. Modify fitness evaluator to run multiple backtests per strategy
3. Aggregate results across validation windows
4. Add progress tracking (current window X/Y)
5. Cache train-window results for efficiency

---

### 🎨 HIGH PRIORITY: Multiobjective Evolution (NSGA-II)

**Why**: Retain Pareto front of diverse optimal strategies; no fitness weight tuning needed  
**Effort**: 5-10 days  
**Impact**: ⭐⭐⭐⭐

- [ ] **Implement NSGA-II for multiobjective optimization**
  - Replace single fitness scalar with multiple objectives:
    - Objective 1: Total profit %
    - Objective 2: Max drawdown (minimize)
    - Objective 3: Sharpe ratio
    - Objective 4: Number of trades (Goldilocks: not too few, not too many)
  - Implement non-dominated sorting (Pareto fronts)
  - Implement crowding distance for diversity
  - Return Pareto front (10-20 strategies) instead of single best
  - **Benefits**:
    - User chooses from diverse optimal strategies
    - Natural diversity preservation
    - No need to tune fitness weights
  - **Libraries**: Consider `pymoo` or implement NSGA-II directly
  - **Reference**: https://ieeexplore.ieee.org/document/996017

**Implementation steps**:
1. Refactor Individual to store `objectives: List[float]`
2. Implement fast non-dominated sort
3. Implement crowding distance calculation
4. Modify selection to use Pareto rank + crowding distance
5. Update visualization to show Pareto front
6. Save all Pareto-optimal strategies, not just top-1

---

### Other Major Features

- [ ] **Island model with migration** (3-6 days)
  - Run N islands (populations) in parallel
  - Migrate top K individuals every M generations
  - Config already has `island_model` placeholder
  - Benefits: Better diversity, natural parallelism, escape local optima
  
- [ ] **Parallel evaluation** (2-4 days)
  - Multiprocessing worker pool for backtest evaluation
  - Careful with Freqtrade backtesting global state
  - Each worker gets own DirectBacktester instance
  - Benefits: 4-8x speedup on multi-core systems
  
- [ ] **Strategy grammar / strongly-typed conditions** (5-10 days)
  - Grammar-based genetic programming (GGP)
  - Prevent semantically invalid rules: `(RSI > 70) AND (RSI < 30)`
  - Type system: `Indicator → Comparison → Condition`
  - Benefits: Higher quality strategies, faster convergence

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
- [pymoo - Multi-objective Optimization](https://pymoo.org/)

---

## 📊 Progress Tracking

**Last Audit Date**: 2026-02-18  
**Critical Bugs Fixed**: 6/6 ✅  
**Quick Wins Completed**: 0/6  
**Medium Features Completed**: 1/4 (unit tests)  
**Major Features Completed**: 0/3  

**Next Sprint Focus**: Implement quick wins, then multi-timeframe strategies

---

## ✅ Completed Features

### Advanced Mutation Operators ✅
- Gaussian mutation for smooth parameter tuning
- Swap mutation for component reordering
- Adaptive per-gene mutation based on fitness
   
### Enhanced Fitness Function ✅
- Added Sortino ratio (downside risk focus)
- Added profit factor (win/loss ratio)
- Robustness bonuses for consistent performers
- Risk-adjusted excellence bonuses

### Diversity Preservation ✅
- Fitness sharing/niching implementation
- Genetic diversity tracking
- Configurable sharing radius

### Richer Strategy Grammar ✅
- Added 7 new indicators: MFI, WILLR, ROC, TEMA, KAMA, SAR, AROON
- Expanded indicator parameter ranges

---

## 🐛 Bug Fixes Applied (2026-02-18)

All critical bugs from the audit have been fixed:

1. ✅ **ROI Key Consistency** - All minimal_roi dictionaries now use string keys
2. ✅ **Trailing Stop Serialization** - trailing_stop_positive and _offset now preserved
3. ✅ **Generator Fallback** - Returns vectorized condition instead of scalar "True"
4. ✅ **Mutation Double-Gating** - Removed outer probability check
5. ✅ **Random.sample Guard** - Protected against requesting more indicators than available
6. ✅ **Comprehensive Tests** - Added test_critical_fixes.py with 5 passing tests

See commits: 9f6375c, f6f9be0

---

Last Updated: 2026-02-18

