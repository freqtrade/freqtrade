# GA Improvements TODO

Last Updated: 2026-02-22  
Status: Walk-Forward, Multi-Timeframe, NSGA-II, and Parallel Evaluation complete, next up is Island Model

---

## 🎯 NEXT STEP: Major Features (Production Quality Strategies)

> **Current Status:** All foundational work is complete! The genetic algorithm has:
> - ✅ No critical bugs
> - ✅ Good performance optimizations
> - ✅ Clear indicator encoding
> - ✅ Comprehensive tests
> - ✅ Walk-Forward Optimization (prevents overfitting)
> - ✅ Multi-Timeframe Strategies (industry-standard quality boost)
> - ✅ NSGA-II Multiobjective Evolution (diverse strategy portfolios)
> - ✅ Parallel Evaluation (3-4x speedup on multi-core systems)
> 
> **What's Next:** Island Model with Migration for even more diversity and parallel evolution.

### Recommended Implementation Order

**Phase 1: Anti-Overfitting (MUST DO FIRST)**
1. ✅ **Walk-Forward Optimization** (⭐⭐⭐⭐⭐) - COMPLETE
   - Rolling & anchored window modes
   - Configurable train/validation/step sizes
   - Multiple aggregation methods (mean, min, harmonic_mean, weighted)
   - Intelligent caching with strategy_hash + window_index
   - See `WALK_FORWARD_SUMMARY.md` and `WALK_FORWARD_GUIDE.md`

**Phase 2: Strategy Quality Improvements**
2. ✅ **Multi-Timeframe Strategies** (⭐⭐⭐⭐⭐) - COMPLETE
   - `IndicatorGene.timeframe` field for informative timeframe indicators
   - `StrategyGene.informative_timeframes` for tracking active higher TFs
   - Multi-TF instance ID naming (e.g., `RSI_1h_0`, `EMA_4h_0`)
   - Code generation with `informative_pairs()`, `merge_informative_pair()`, TF-suffixed columns
   - `mutate_timeframes()` operator: add/remove TFs, change indicator TFs
   - Crossover preserves informative_timeframes
   - Full configuration in `ga_config.yaml` under `multi_timeframe:` section
   - 24 comprehensive tests in `test_multi_timeframe.py`
   
3. ✅ **NSGA-II Multiobjective Evolution** (⭐⭐⭐⭐) - COMPLETE
   - Non-dominated sorting with Pareto fronts
   - Crowding distance for diversity
   - NSGA-II selection (rank + crowding distance)
   - Configurable objectives (profit, drawdown, sharpe_ratio)
   - See `genetic_algorithm/core/nsga2.py`

**Phase 3: Performance Scaling**
4. ✅ **Parallel Evaluation** (⭐⭐⭐) - COMPLETE
   - ProcessPoolExecutor for parallel backtesting
   - 3.62x speedup on 6-core system (measured)
   - See `PARALLEL_EVALUATION_GUIDE.md`

**Phase 4: Advanced Features**
5. **Island Model with Migration** (⭐⭐⭐) - 3-6 days ← **NEXT STEP**
   - **Why next:** More diversity through isolated populations
   - **Benefit:** Better exploration of strategy space
6. **Strategy Grammar / Strongly-Typed Conditions** (⭐⭐) - 5-10 days

---

## 🚀 MAJOR FEATURES (1-2 weeks each)

### 🏆 Walk-Forward Optimization

**Status:** ✅ COMPLETE  
**Completed:** February 19, 2026  
**Details:** See `WALK_FORWARD_SUMMARY.md` and `WALK_FORWARD_GUIDE.md`

#### The Problem
Current implementation trains on entire backtest period and selects the best strategy. This leads to:
- **Data snooping bias**: Strategy sees all data during evolution
- **Overfitting**: High backtest performance, poor live performance
- **No out-of-sample validation**: Can't estimate real-world performance

#### The Solution: Walk-Forward Validation

Split backtest timerange into rolling windows:
1. **Train** on N days → Evolve population
2. **Validate** on next M days → Evaluate performance on unseen data
3. **Slide forward** by S days
4. **Repeat** for entire timerange
5. **Fitness** = Aggregate validation score (not training score!)

**Key Insight:** Evolution sees only training data, but fitness is measured on validation data.

#### Implementation Checklist

- [x] **Step 1: Timerange Splitting Logic** (Day 1)
  - Add `walk_forward` section to config
  - Implement `create_walk_forward_windows()` function
  - Config options:
    - `walk_forward.enabled: true/false`
    - `walk_forward.train_days: 60` (training window size)
    - `walk_forward.validation_days: 15` (validation window size)
    - `walk_forward.step_days: 15` (how far to slide forward)
    - `walk_forward.mode: 'rolling'` or `'anchored'`
    - `walk_forward.aggregation: 'mean' | 'min' | 'harmonic_mean'`
  - Example: Train on days 0-60, validate on days 60-75, slide to 15-75 train, 75-90 validate
  
- [x] **Step 2: Multi-Window Fitness Evaluator** (Days 2-3)
  - Extend `FitnessEvaluator` to support walk-forward mode
  - For each strategy:
    1. Run backtest on each training window
    2. Evaluate on corresponding validation window
    3. Aggregate validation results
  - Update progress tracking to show "Window X/Y"
  - Handle edge cases (insufficient data for window)
  
- [x] **Step 3: Train/Validate Window Caching** (Day 4)
  - Cache training window results to avoid re-evaluation
  - Key insight: Same strategy evaluated on same train window = same result
  - Implement cache with (strategy_hash, train_window) as key
  - Significant speedup for elite individuals across generations
  
- [x] **Step 4: Aggregation Strategies** (Day 5)
  - **Mean**: `fitness = mean(validation_scores)` - Balanced
  - **Min**: `fitness = min(validation_scores)` - Conservative (worst-case)
  - **Harmonic Mean**: `fitness = harmonic_mean(validation_scores)` - Penalizes inconsistency
  - **Weighted**: More weight to recent windows
  - Make aggregation configurable
  
- [x] **Step 5: Integration & Testing** (Days 6-7)
  - Write comprehensive test suite:
    - `test_window_creation()`
    - `test_walk_forward_fitness()`
    - `test_walk_forward_caching()`
    - `test_aggregation_methods()`
  - Integration test: Run full GA with walk-forward enabled
  - Compare walk-forward vs standard evolution on same data
  - Document performance differences
  
- [x] **Step 6: Visualization & Reporting** (Optional, Day 8)
  - Add walk-forward results to visualization
  - Show train vs validation performance per window
  - Plot performance degradation (train vs validation gap)
  - Generate report with per-window breakdown

#### Configuration Example

```yaml
walk_forward:
  enabled: true
  train_days: 60          # 60 days for training
  validation_days: 15     # 15 days for validation
  step_days: 15           # Slide forward by 15 days
  mode: 'rolling'         # 'rolling' (fixed window) or 'anchored' (expanding)
  aggregation: 'mean'     # 'mean', 'min', 'harmonic_mean', 'weighted'
  min_train_trades: 10    # Skip window if < 10 trades in training
```

#### Expected Outcomes

**Before Walk-Forward:**
- Training fitness: 15.0%
- Live performance: 3.0% (massive overfitting)

**After Walk-Forward:**
- Training fitness: 10.0%
- Validation fitness: 8.5%
- Live performance: 7.0% (much closer to validation)

**Trade-off:** Lower training fitness, but much better real-world performance.

#### Files to Modify

1. `genetic_algorithm/config/ga_config.yaml` - Add walk_forward section
2. `genetic_algorithm/evaluation/fitness.py` - Extend FitnessEvaluator
3. `genetic_algorithm/core/evolution.py` - Integrate with main loop
4. `genetic_algorithm/utils/timerange.py` (NEW) - Window creation logic
5. `genetic_algorithm/test_walk_forward.py` (NEW) - Test suite

#### Alternative Approaches Considered

❌ **K-Fold Cross-Validation**: Doesn't respect time order (look-ahead bias)  
❌ **Single Train/Test Split**: Not enough validation data, no robustness check  
✅ **Walk-Forward**: Industry standard, respects time order, multiple validation windows

---

### 🛡️ Multi-Timeframe Strategies

**Status:** ✅ COMPLETE  
**Completed:** February 19, 2026  
**Impact:** ⭐⭐⭐⭐⭐

#### The Concept

Trade on one timeframe (e.g., 5m) but use indicators from higher timeframes (e.g., 1h, 4h) for:
- **Trend confirmation**: Only buy on 5m when 1h trend is bullish
- **Market regime filtering**: Avoid trades during 4h consolidation
- **Stronger signals**: Higher timeframe = less noise

**Example Strategy:**
```python
# Base timeframe: 5m
# Entry: RSI_5m < 30 (oversold on 5m)
#    AND EMA_1h > EMA_4h (bullish trend on 1h)
#    AND ATR_4h > threshold (volatility filter on 4h)
```

#### Implementation Checklist

- [x] **Step 1: Extend StrategyGene** (Day 1)
  - Added `informative_timeframes: List[str]` field to StrategyGene
  - Added `timeframe: Optional[str]` field to IndicatorGene
  - Updated serialization (to_dict/from_dict)
  - Added timeframe ordering helpers (`timeframe_to_minutes`, `is_higher_timeframe`)
  
- [x] **Step 2: Multi-TF Indicator Genes** (Day 1-2)
  - Extended IndicatorGene to include `timeframe` field
  - Format: `RSI_1h_0` (type + timeframe + instance)
  - Updated assign_instance_ids() to handle multi-TF
  - Example: `[RSI_0, RSI_1h_0, EMA_5m_0, EMA_1h_0]`
  
- [x] **Step 3: Multi-TF Condition Generation** (Day 2)
  - Extended condition code generator to produce TF-suffixed columns
  - Conditions like: `dataframe['rsi_14_1h'] < 30`
  - Informative TF conditions use `AND` logic for trend filtering
  
- [x] **Step 4: Strategy Code Generation** (Day 3-4)
  - Generate `informative_pairs()` method listing all higher TFs
  - Use Freqtrade's `merge_informative_pair()` helper
  - Informative indicators calculated via `self.dp.get_pair_dataframe()`
  - Merged columns with suffix: `rsi_14_1h`, `ema_20_4h`
  
- [x] **Step 5: Genetic Operators** (Day 4)
  - Mutation: `mutate_timeframes()` — add/remove informative TFs, change indicator TF
  - Crossover: All 3 methods (single_point, uniform, component) preserve informative_timeframes
  - Mutation dispatch updated in `mutate()` with 20% probability when multi-TF enabled
  
- [x] **Step 6: Testing & Validation** (Day 5)
  - 24 comprehensive tests in `test_multi_timeframe.py`
  - Tests cover: gene model, instance IDs, serialization, generation, code gen, mutation, crossover
  - All existing tests continue to pass (backward compatible)

#### Configuration Example

```yaml
strategy:
  base_timeframe: '5m'
  informative_timeframes:
    enabled: true
    available: ['15m', '1h', '4h']  # Higher TFs allowed
    max_timeframes: 2               # Max 2 informative TFs per strategy
    
indicators:
  # Some indicators work better on higher TFs
  higher_timeframe_preference:
    - 'EMA'      # Trend indicators
    - 'BBANDS'   # Volatility bands
    - 'ATR'      # Volatility
```

#### Expected Benefits

- **Better win rate**: Higher TF filters reduce false signals
- **Larger average win**: Catches stronger trends
- **More robust**: Less sensitivity to noise
- **Standard practice**: All professional strategies use multi-TF

#### Files Modified

1. ✅ `genetic_algorithm/core/strategy_gene.py` - Added informative_timeframes, IndicatorGene.timeframe, multi-TF instance IDs
2. ✅ `genetic_algorithm/strategies/generator.py` - Multi-TF indicator generation + code generation with merge_informative_pair
3. ✅ `genetic_algorithm/core/mutation.py` - Added `mutate_timeframes()`, integrated into `mutate()` dispatch
4. ✅ `genetic_algorithm/core/crossover.py` - All 3 crossover methods preserve informative_timeframes
5. ✅ `genetic_algorithm/config/ga_config.yaml` - Added `multi_timeframe:` configuration section
6. ✅ `genetic_algorithm/test_multi_timeframe.py` (NEW) - 24 comprehensive tests

---

### 🎨 Multiobjective Evolution (NSGA-II)

**Status:** ✅ COMPLETE  
**Completed:** February 22, 2026  
**Why Important:** Removes need for fitness weight tuning; returns diverse strategy portfolio  
**Effort:** 5-10 days  
**Impact:** ⭐⭐⭐⭐  
**Files:** `genetic_algorithm/core/nsga2.py`, `genetic_algorithm/core/individual.py`, `genetic_algorithm/core/selection.py`, `genetic_algorithm/core/evolution.py`

#### The Problem with Single-Objective Optimization

Current approach uses weighted sum of objectives:
```python
fitness = w1*profit + w2*sharpe + w3*drawdown + w4*trades
```

**Issues:**
- **Weight sensitivity**: Results change drastically with different weights
- **Single solution**: Returns only one strategy
- **No trade-off visibility**: Can't see profit vs risk trade-offs
- **Manual tuning**: Need to experiment with many weight combinations

#### The NSGA-II Solution

**Multiobjective optimization** finds the **Pareto front** - all strategies where:
- No other strategy is better in ALL objectives
- Trade-off between objectives (high profit might have high drawdown)

**Output:** Portfolio of 10-20 diverse optimal strategies, not just one.

**Example Pareto Front:**
| Strategy | Profit | Drawdown | Sharpe | Trades |
|----------|--------|----------|--------|--------|
| A        | 25%    | 15%      | 1.8    | 150    |
| B        | 20%    | 10%      | 2.1    | 120    |
| C        | 15%    | 5%       | 2.5    | 80     |

**User picks** based on their risk tolerance!

#### Implementation Checklist

- [x] **Step 1: Multi-Objective Fitness** (Day 1-2)
  - Added `objectives: List[float]` to Individual class
  - `extract_objectives_from_metrics()` converts metrics to objectives vector
  - Configurable objectives in `ga_config.yaml` under `nsga2.objectives`
  - Supports maximize, minimize, and goldilocks (target value) types
  
- [x] **Step 2: Non-Dominated Sorting** (Day 2-3)
  - `dominates()` function for Pareto dominance check
  - `fast_non_dominated_sort()` implements NSGA-II algorithm (O(M*N²))
  - Each individual gets `rank` attribute (1 = Pareto front)
  
- [x] **Step 3: Crowding Distance** (Day 3-4)
  - `crowding_distance_assignment()` calculates diversity metric
  - Individuals on front boundaries get infinite distance
  - Each individual gets `crowding_distance` attribute
  
- [x] **Step 4: NSGA-II Selection** (Day 4-5)
  - `nsga2_selection()` in selection.py
  - Binary tournament: prefer lower rank, then higher crowding distance
  - Integrated into `select_parents()` with `method='nsga2'`
  
- [x] **Step 5: Update Evolution Logic** (Day 5-6)
  - `evolution.py` supports `mode: 'nsga2'` config option
  - Automatic selection method override when NSGA-II enabled
  - `get_pareto_front()` extracts rank-1 individuals
  
- [x] **Step 6: Multi-Objective Reporting** (Day 6-8)
  - Pareto front logged at each generation
  - `nsga2_crowded_comparison_sort()` for sorted output
  
- [x] **Step 7: Testing & Integration** (Day 8-10)
  - Non-dominated sorting tests
  - Crowding distance tests
  - Integration with evolution loop

#### Configuration Example

```yaml
genetic_algorithm:
  mode: 'nsga2'  # 'single_objective' or 'nsga2'
  
nsga2:
  objectives:
    - name: 'total_profit'
      type: 'maximize'
      weight: 1.0  # For normalization only
    - name: 'max_drawdown'
      type: 'minimize'
      weight: 1.0
    - name: 'sharpe_ratio'
      type: 'maximize'
      weight: 1.0
    - name: 'num_trades'
      type: 'goldilocks'  # Penalty if too high or too low
      target: 100
      tolerance: 50
  
  pareto_front_size: 20  # Number of strategies to return
  crowding_distance_percentile: 0.1  # Preserve diversity
```

#### Expected Benefits

**Before (Single-Objective):**
- One strategy with profit=20%, drawdown=12%
- Need to re-run with different weights to explore trade-offs

**After (NSGA-II):**
- 20 strategies spanning profit=10%-30%, drawdown=3%-15%
- User can pick conservative (low profit, low DD) or aggressive (high profit, high DD)
- No weight tuning required

#### Files to Modify

1. ✅ `genetic_algorithm/core/individual.py` - Added objectives, rank, crowding_distance fields
2. ✅ `genetic_algorithm/core/nsga2.py` - Non-dominated sorting + crowding distance
3. ✅ `genetic_algorithm/core/selection.py` - Added `nsga2_selection()` function
4. ✅ `genetic_algorithm/core/evolution.py` - Integrated NSGA-II mode with `mode: 'nsga2'`
5. ✅ `genetic_algorithm/evaluation/fitness.py` - Objects extracted via `extract_objectives_from_metrics()`
6. ✅ `genetic_algorithm/config/ga_config.yaml` - NSGA-II config section

#### Libraries to Consider

- **pymoo**: Professional multi-objective optimization library
  - Pros: Well-tested, many algorithms (NSGA-II, NSGA-III, MOEA/D)
  - Cons: Additional dependency
  
- **Custom Implementation**: NSGA-II from scratch
  - Pros: No dependencies, full control
  - Cons: More implementation work, need thorough testing

**Recommendation**: Start with custom NSGA-II (it's not that complex), consider pymoo if expanding to other algorithms.

---

### ⚡ Parallel Evaluation

**Status:** ✅ COMPLETE  
**Completed:** February 22, 2026  
**Impact:** ⭐⭐⭐  
**Files:** `genetic_algorithm/evaluation/parallel.py`, `genetic_algorithm/core/evolution.py`

#### The Problem

Evolution with walk-forward optimization is slow:
- Each strategy requires multiple backtests (one per window)
- Sequential processing: 20 strategies × 12 windows = 240 backtests one-by-one
- With 3s per backtest: 12 minutes per generation

#### The Solution: Multi-Process Parallelism

Use Python's `ProcessPoolExecutor` to run backtests in parallel:
- Each worker process has its own `FitnessEvaluator` instance
- Strategies distributed across workers via queue
- Near-linear speedup with number of workers

#### Benchmark Results

| Strategies | Workers | Sequential | Parallel | Speedup |
|------------|---------|------------|----------|---------|
| 6          | 4       | 22.73s     | 7.41s    | 3.07x   |
| 12         | 6       | 33.02s     | 9.11s    | 3.62x   |

#### Configuration

```yaml
parallel_evaluation:
  enabled: true        # Enable parallel backtesting
  num_workers: null    # Auto-detect (CPU cores - 1)
```

#### Implementation Checklist

- [x] **Step 1: Worker Function** - `_evaluate_strategy_in_worker()`
- [x] **Step 2: Worker Initialization** - `_init_worker()` creates FitnessEvaluator per process
- [x] **Step 3: ParallelEvaluator Class** - Manages worker pool and batch evaluation
- [x] **Step 4: Integration** - `evolution.py` uses parallel when enabled
- [x] **Step 5: Configuration** - Added `parallel_evaluation:` section to config
- [x] **Step 6: Benchmark Script** - `benchmark_parallel.py` measures speedup
- [x] **Step 7: Documentation** - `PARALLEL_EVALUATION_GUIDE.md`

#### Files Added/Modified

1. ✅ `genetic_algorithm/evaluation/parallel.py` (NEW) - Parallel evaluation module
2. ✅ `genetic_algorithm/core/evolution.py` - Integration with parallel evaluator
3. ✅ `genetic_algorithm/config/ga_config.yaml` - Added parallel_evaluation config
4. ✅ `genetic_algorithm/benchmark_parallel.py` (NEW) - Benchmark script
5. ✅ `genetic_algorithm/PARALLEL_EVALUATION_GUIDE.md` (NEW) - Documentation
6. ✅ `tests/test_parallel_evaluation.py` (NEW) - Test suite

---

### Other Major Features

- [ ] **Island model with migration** (3-6 days)  
  Run N islands (populations) in parallel  
  Migrate top K individuals every M generations  
  Config already has `island_model` placeholder

- [ ] **Parallel evaluation** (2-4 days) ← **COMPLETE**  
  Multiprocessing worker pool for backtest evaluation  
  Each worker gets own DirectBacktester instance  
  Benefits: 3-4x speedup on 6-core systems (measured)  
  See: `PARALLEL_EVALUATION_GUIDE.md`

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

## 🎯 IMMEDIATE ACTION ITEMS

**For the next session:**

1. ✅ ~~**Walk-Forward Optimization**~~ — COMPLETE
2. ✅ ~~**Multi-Timeframe Strategies**~~ — COMPLETE
3. ✅ ~~**NSGA-II Multiobjective Evolution**~~ — COMPLETE
4. ✅ ~~**Parallel Evaluation**~~ — COMPLETE (February 22, 2026)
   - ProcessPoolExecutor for parallel backtesting
   - 3-4x speedup on 6 workers
   - See `PARALLEL_EVALUATION_GUIDE.md`

5. **START HERE**: Island Model with Migration
   - Multiple populations evolving in parallel
   - Migration of top individuals between islands
   - Expected completion: 3-6 days

**Success Criteria:**
- [x] Walk-forward validation shows <20% degradation from train to validation
- [x] Multi-TF strategies show improved Sharpe ratio vs single-TF
- [x] NSGA-II returns diverse Pareto front with visible trade-offs
- [x] All features have comprehensive test coverage
- [x] Parallel evaluation shows 3-4x speedup on 6-core system (measured: 3.62x)
- [ ] CodeQL security scan passes with 0 vulnerabilities

---

## 🔗 REFERENCES & RESOURCES

- [Freqtrade Informative Pairs](https://www.freqtrade.io/en/stable/strategy-customization/#informative-pairs)
- [Freqtrade Hyperopt](https://www.freqtrade.io/en/stable/hyperopt/)
- [Freqtrade Backtesting](https://www.freqtrade.io/en/stable/backtesting/)
- [Freqtrade Data Downloading](https://www.freqtrade.io/en/stable/data-download/)
- [NSGA-II Paper (IEEE)](https://ieeexplore.ieee.org/document/996017)
