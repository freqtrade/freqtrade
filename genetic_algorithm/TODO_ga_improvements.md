# Genetic Algorithm Improvements TODO

This document outlines larger, more complex improvements to the GA system that require significant implementation effort. These are organized by priority and estimated complexity.

## Completed Quick Wins ✅

1. **Advanced Mutation Operators** ✅
   - Gaussian mutation for smooth parameter tuning
   - Swap mutation for component reordering
   - Adaptive per-gene mutation based on fitness
   
2. **Enhanced Fitness Function** ✅
   - Added Sortino ratio (downside risk focus)
   - Added profit factor (win/loss ratio)
   - Robustness bonuses for consistent performers
   - Risk-adjusted excellence bonuses

3. **Diversity Preservation** ✅
   - Fitness sharing/niching implementation
   - Genetic diversity tracking
   - Configurable sharing radius

4. **Richer Strategy Grammar** ✅
   - Added 7 new indicators: MFI, WILLR, ROC, TEMA, KAMA, SAR, AROON
   - Expanded indicator parameter ranges

---

## High Priority (High Impact, Medium Effort)

### 1. Walk-Forward Optimization (WFO)

**Rationale:**
Walk-forward optimization is critical for preventing overfitting. It validates strategies on out-of-sample data by training on one period and testing on the next, mimicking real-world deployment.

**Implementation Approach:**
1. Split timerange into multiple windows (e.g., 3-month train, 1-month test)
2. For each generation:
   - Train strategies on current window
   - Validate on next window
   - Calculate combined fitness (weighted: 70% in-sample, 30% out-of-sample)
3. Roll window forward after N generations
4. Track consistency across windows

**Files to Modify:**
- `genetic_algorithm/evaluation/fitness.py` - Add WFO evaluation mode
- `genetic_algorithm/core/evolution.py` - Integrate WFO into evolution loop
- `genetic_algorithm/config/ga_config.yaml` - Add WFO configuration

**Estimated Complexity:** Medium (2-3 days)
- Need to handle data splitting
- Manage multiple backtest windows
- Balance in-sample vs out-of-sample scores

**Benefits:**
- Dramatically reduces overfitting
- Produces more robust, deployable strategies
- Better reflects real-world performance

---

### 2. Multi-Timeframe Strategy Support

**Rationale:**
Professional traders use multiple timeframes for confirmation. A strategy that considers both 1h and 4h charts is often more robust than single-timeframe strategies.

**Implementation Approach:**
1. Extend `StrategyGene` to support multiple timeframe indicators:
   ```python
   indicators_1h: List[IndicatorGene]
   indicators_4h: List[IndicatorGene]
   ```
2. Modify `StrategyGenerator` to:
   - Generate indicators for each timeframe
   - Create cross-timeframe conditions (e.g., "1h RSI < 30 AND 4h trend up")
3. Update code generation to:
   - Call `resample()` in FreqTrade for higher timeframes
   - Combine multi-timeframe conditions properly
4. Add timeframe alignment logic

**Files to Modify:**
- `genetic_algorithm/core/strategy_gene.py` - Add multi-timeframe fields
- `genetic_algorithm/strategies/generator.py` - Multi-timeframe generation
- `genetic_algorithm/strategies/template.py` - Code generation for MTF
- `genetic_algorithm/core/mutation.py` - Mutate across timeframes
- `genetic_algorithm/core/crossover.py` - Cross timeframes properly

**Estimated Complexity:** Medium-High (3-5 days)
- Need to handle timeframe synchronization
- More complex strategy representation
- Increased strategy space (more indicators/conditions)

**Benefits:**
- More sophisticated strategies
- Better trend/momentum confirmation
- Reduced false signals
- Professional-grade strategies

---

### 3. Strategy Complexity Penalty

**Rationale:**
Simpler strategies generalize better. Overly complex strategies with many indicators and conditions often overfit to historical data.

**Implementation Approach:**
1. Define complexity metric:
   ```python
   complexity = (
       num_indicators * 0.3 +
       num_entry_conditions * 0.4 +
       num_exit_conditions * 0.3
   )
   ```
2. Add penalty to fitness:
   ```python
   if complexity > threshold:
       fitness *= (1.0 - penalty_factor * (complexity - threshold))
   ```
3. Make configurable via config:
   ```yaml
   fitness_penalties:
     complexity_threshold: 8
     complexity_penalty: 0.05  # 5% penalty per unit over threshold
   ```

**Files to Modify:**
- `genetic_algorithm/evaluation/fitness.py` - Add complexity calculation and penalty
- `genetic_algorithm/config/ga_config.yaml` - Add complexity settings

**Estimated Complexity:** Low (0.5-1 day)
- Simple calculation
- Easy to integrate into existing fitness function

**Benefits:**
- Encourages simpler, more robust strategies
- Reduces overfitting risk
- Faster backtesting (fewer indicators to calculate)
- Easier to understand and maintain strategies

---

## Medium Priority (Good Impact, High Effort)

### 4. Island Model / Speciation

**Rationale:**
Island models maintain multiple sub-populations (islands) that evolve independently with occasional migration. This preserves diversity and explores different regions of the solution space simultaneously.

**Implementation Approach:**
1. Split population into N islands (e.g., 4 islands of 25 individuals each)
2. Each island evolves independently:
   - Different selection pressures (aggressive vs conservative)
   - Different mutation rates
   - Different fitness weight profiles
3. Periodic migration:
   - Every K generations, migrate top M individuals between islands
   - Migration topology: ring, fully-connected, or random
4. Final integration: Combine all islands for final ranking

**Architecture:**
```python
class IslandModel:
    def __init__(self, num_islands, population_per_island):
        self.islands = [
            GeneticAlgorithm(config_variant) 
            for _ in range(num_islands)
        ]
    
    def evolve_generation(self):
        # Evolve each island in parallel (could use multiprocessing)
        for island in self.islands:
            island.evolve_one_generation()
        
        # Migrate if needed
        if self.generation % migration_interval == 0:
            self.migrate()
```

**Files to Create:**
- `genetic_algorithm/core/island_model.py` - Island management
- `genetic_algorithm/core/migration.py` - Migration strategies

**Files to Modify:**
- `genetic_algorithm/core/evolution.py` - Extract single generation logic
- `genetic_algorithm/run_ga.py` - Support island mode flag
- `genetic_algorithm/config/ga_config.yaml` - Island configuration

**Estimated Complexity:** High (5-7 days)
- Significant refactoring of evolution loop
- Parallel execution challenges
- Migration logic complexity
- Visualization updates for multi-island view

**Benefits:**
- Massive diversity improvement
- Explores multiple solution strategies simultaneously
- Natural parallelization opportunity
- Often finds better solutions than single population

---

### 5. NSGA-II Multi-Objective Optimization

**Rationale:**
Trading has inherently conflicting objectives (maximize profit vs minimize risk). NSGA-II finds the Pareto front of non-dominated solutions, giving users a range of strategies with different risk/reward profiles.

**Implementation Approach:**
1. Define multiple objectives (not combined into single fitness):
   - Maximize: Profit, Sharpe ratio, Win rate
   - Minimize: Drawdown, Volatility
2. Implement NSGA-II selection:
   - Fast non-dominated sorting
   - Crowding distance for diversity
   - Tournament selection based on rank + distance
3. Report Pareto front instead of single "best" strategy

**Key Algorithm Components:**
```python
def fast_non_dominated_sort(population):
    """Assign rank to each individual based on domination."""
    fronts = [[]]
    for individual in population:
        individual.domination_count = 0
        individual.dominated_solutions = []
        for other in population:
            if dominates(individual, other):
                individual.dominated_solutions.append(other)
            elif dominates(other, individual):
                individual.domination_count += 1
        
        if individual.domination_count == 0:
            individual.rank = 0
            fronts[0].append(individual)
    # ... continue for other ranks
    return fronts

def crowding_distance(front):
    """Calculate crowding distance for diversity."""
    # Distance based on neighbors in objective space
    # Higher distance = more isolated = preserve for diversity
```

**Files to Create:**
- `genetic_algorithm/core/nsga2.py` - NSGA-II implementation
- `genetic_algorithm/core/pareto.py` - Pareto front utilities

**Files to Modify:**
- `genetic_algorithm/core/selection.py` - Add NSGA-II selection
- `genetic_algorithm/evaluation/fitness.py` - Return multi-objective scores
- `genetic_algorithm/core/individual.py` - Add rank, crowding distance fields
- `genetic_algorithm/visualization/` - Visualize Pareto front

**Estimated Complexity:** High (7-10 days)
- Complex algorithm implementation
- Careful handling of multi-objective space
- Significant changes to selection logic
- New visualization requirements

**Benefits:**
- No need to manually tune fitness weights
- Presents range of solutions (aggressive to conservative)
- User can choose risk/reward profile
- More scientifically rigorous approach
- Better for production deployment (multiple strategies for different market conditions)

---

### 6. Meta-Learning for Hyperparameter Optimization

**Rationale:**
GA hyperparameters (mutation rate, population size, elite size, etc.) significantly affect performance. Meta-learning automatically tunes these based on evolution dynamics.

**Implementation Approach:**
1. Track meta-metrics during evolution:
   - Convergence speed
   - Diversity trends
   - Improvement rate
2. Adjust hyperparameters dynamically:
   ```python
   if diversity < threshold:
       mutation_rate *= 1.2  # Increase exploration
   if no_improvement > 5 and diversity > threshold:
       elite_size *= 1.1  # Increase exploitation
   ```
3. Or use external optimizer (Optuna, Bayesian optimization) to tune:
   - Run short GA experiments with different hyperparameters
   - Learn which settings work best for this problem
   - Apply learned settings to full run

**Levels of Implementation:**
1. **Simple** (adaptive rules): Hard-coded rules based on metrics
2. **Medium** (Bayesian optimization): Use Optuna to tune hyperparameters
3. **Advanced** (Reinforcement learning): RL agent learns to adjust hyperparameters

**Files to Modify:**
- `genetic_algorithm/core/evolution.py` - Add adaptive logic
- Create `genetic_algorithm/meta/` directory for meta-learning
  - `meta_optimizer.py` - Hyperparameter tuning
  - `meta_rules.py` - Adaptive rule engine

**Estimated Complexity:** Medium-High (4-6 days)
- Simple adaptive rules: 1-2 days
- Bayesian optimization: 3-4 days
- RL-based: 6+ days

**Benefits:**
- Removes manual hyperparameter tuning burden
- Adapts to problem characteristics automatically
- Better performance with less expert knowledge required
- Can discover non-obvious parameter combinations

---

## Lower Priority (Nice to Have)

### 7. Ensemble Strategies

Combine multiple evolved strategies into an ensemble that votes on trades. Implementation: evolve strategies normally, then create a meta-strategy that combines top N strategies with weighted voting.

**Estimated Complexity:** Medium (2-3 days)

---

### 8. Transfer Learning from Previous Runs

Save and reuse good strategy "building blocks" from previous evolution runs. Seeds new populations with proven components.

**Estimated Complexity:** Medium (2-3 days)

---

### 9. Coevolution of Strategy and Market Regime Detector

Evolve both strategies AND the logic to detect when each strategy should be active (bull/bear/sideways market detection).

**Estimated Complexity:** High (7+ days)

---

### 10. GPU-Accelerated Backtesting

Offload backtest calculations to GPU for massive parallelization. Requires rewriting backtest engine in CUDA/OpenCL or using existing GPU libraries.

**Estimated Complexity:** Very High (14+ days)

---

## Implementation Priority Recommendation

Based on impact vs effort, suggested implementation order:

1. **Walk-Forward Optimization** - Critical for production readiness
2. **Strategy Complexity Penalty** - Quick win with good impact
3. **Multi-Timeframe Support** - Professional-grade feature
4. **Island Model** - Significant quality improvement
5. **Meta-Learning** - Automation and optimization
6. **NSGA-II** - Advanced users, research applications

---

## Testing Strategy

For each improvement:

1. **Unit Tests**: Test individual components in isolation
2. **Integration Tests**: Run small GA (10 individuals, 5 generations) to verify end-to-end
3. **Benchmark Tests**: Compare results with and without the feature
4. **Performance Tests**: Measure execution time and memory impact

---

## Documentation Requirements

Each implemented feature should include:

1. **Code comments**: Inline documentation of complex logic
2. **Config documentation**: Update `config/README.md` with new parameters
3. **User guide**: Add section to main README or separate guide
4. **Research notes**: Document assumptions, limitations, and future improvements

---

## Notes

- All improvements should be configurable (enable/disable via config)
- Maintain backward compatibility with existing configs
- Consider computational cost vs benefit for each feature
- Prioritize features that reduce overfitting and improve robustness

---

Last Updated: 2026-02-18
