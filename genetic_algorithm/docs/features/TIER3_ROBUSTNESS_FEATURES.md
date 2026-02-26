# Tier 3: Robustness & Anti-Overfitting Features

**Implemented:** February 26, 2026  
**Status:** ✅ Complete  
**Tests:** 31 tests passing in `tests/test_tier3_improvements.py`

---

## Overview

Tier 3 introduces four advanced features to ensure evolved strategies are robust and not overfit to historical data. These features address the critical gap between backtest profitability and real-world trading performance.

| Feature | Purpose | File |
|---------|---------|------|
| Monte-Carlo Robustness | Test stability across trade permutations | `evaluation/monte_carlo.py` |
| Parsimony Pressure | Simplify strategies, remove redundant rules | `core/parsimony.py` |
| Pareto Archive | Preserve best non-dominated solutions | `core/pareto_archive.py` |
| Dynamic Bounds | Evolve indicator parameter ranges | `utils/dynamic_bounds.py` |

---

## 1. Monte-Carlo Robustness Analysis

### What It Does
After evolution completes, the top N strategies are subjected to Monte-Carlo permutation testing. This shuffles trade order, applies slippage jitter, and bootstraps trade samples to verify that profitability is not dependent on specific trade sequences.

### Why It Matters
A strategy that is profitable only because trades happened in a specific order (e.g., lucky early wins) will likely fail in live trading where order is unpredictable.

### Key Metrics
- **Robustness Score**: Percentage of permutations that remain profitable
- **Mean Profit**: Average profit across all permutations
- **P5/P95**: 5th and 95th percentile profits (confidence bounds)

### Configuration
```yaml
monte_carlo:
  enabled: true
  num_perms: 30           # Number of permutations to run
  slippage_jitter: 0.001  # Random slippage ±0.1%
  bootstrap_ratio: 0.8    # Sample 80% of trades per permutation
  min_robustness: 0.7     # Warn if robustness < 70%
```

### Usage
Monte-Carlo analysis runs automatically after evolution if `monte_carlo.enabled: true`. Results are displayed in the final output:

```
MONTE-CARLO ROBUSTNESS ANALYSIS
  ✓ Rank 1: robustness=100.0%, mean_profit=12.69%, p5=2.09%, p95=30.08%
  ⚠️ Rank 2: robustness=0.0%, mean_profit=-97.96%, p5=-152.64%, p95=-41.99%
```

### API
```python
from genetic_algorithm.evaluation.monte_carlo import run_monte_carlo, MonteCarloResult

result: MonteCarloResult = run_monte_carlo(
    trades=trade_list,
    num_perms=30,
    slippage_jitter=0.001,
    bootstrap_ratio=0.8
)

print(f"Robustness: {result.robustness_score:.1%}")
print(f"Mean Profit: {result.mean_profit:.2%}")
print(f"95% CI: [{result.profit_p5:.2%}, {result.profit_p95:.2%}]")
```

---

## 2. Parsimony Pressure

### What It Does
After each generation, elite strategies are analyzed for redundant components. Indicators and conditions that don't meaningfully contribute to fitness are removed, producing simpler strategies.

### Why It Matters
- Simpler strategies are less likely to be overfit
- Fewer indicators = faster backtesting
- Easier to understand and debug
- More likely to generalize to unseen data

### How It Works
1. For each elite strategy, identify removal candidates (indicators, conditions)
2. Temporarily remove each candidate and re-evaluate fitness
3. If fitness drop ≤ epsilon (tolerance), permanently remove the component
4. Log complexity reduction

### Configuration
```yaml
parsimony:
  enabled: true
  epsilon: 0.01        # Max fitness drop to allow removal (1%)
  max_removals: 3      # Max components to remove per strategy
  apply_to_elites: true
  min_complexity: 2    # Don't simplify below this many indicators
```

### Example Output
```
[PARSIMONY] Simplified strategy: removed 1 component(s), complexity 11 → 9
[PARSIMONY] Simplified strategy: removed 1 component(s), complexity 4 → 3
[PARSIMONY] Removed 2 total component(s) from elites
```

### API
```python
from genetic_algorithm.core.parsimony import simplify_strategy, apply_parsimony_to_elites

# Simplify a single strategy
simplified = simplify_strategy(
    strategy=strategy_gene,
    evaluate_fn=fitness_evaluator,
    epsilon=0.01,
    max_removals=3
)

# Apply to all elites after selection
total_removed = apply_parsimony_to_elites(
    elites=elite_population,
    evaluate_fn=fitness_evaluator,
    config=parsimony_config
)
```

---

## 3. Pareto Archive

### What It Does
Maintains an external archive of the best non-dominated solutions found across all generations. Uses crowding-distance decay to manage archive size and preserve diversity along the Pareto front.

### Why It Matters
- Prevents loss of good solutions during selection pressure
- Maintains diversity in multi-objective optimization
- Enables selection of strategies with different trade-offs (e.g., high profit vs low drawdown)

### How It Works
1. After each generation, new solutions are compared against the archive
2. Dominated solutions are removed from the archive
3. If archive exceeds max size, crowding distance is used to prune
4. Archive is serializable for checkpointing

### Configuration
```yaml
pareto_archive:
  enabled: true
  max_size: 100              # Maximum archive size
  crowding_decay: 0.95       # Decay factor for crowded regions
  objectives:                # Objectives to optimize
    - profit
    - sharpe_ratio
    - max_drawdown
```

### API
```python
from genetic_algorithm.core.pareto_archive import ParetoArchive

archive = ParetoArchive(max_size=100, objectives=['profit', 'sharpe', 'drawdown'])

# Update with new solutions
archive.update(population)

# Get current Pareto front
pareto_front = archive.get_archive()

# Get single best by weighted sum
best = archive.get_best(weights={'profit': 0.5, 'sharpe': 0.3, 'drawdown': 0.2})

# Serialize for checkpointing
state = archive.to_dict()
archive = ParetoArchive.from_dict(state)
```

---

## 4. Dynamic Bounds

### What It Does
Allows indicator parameter ranges (e.g., RSI period 5-30) to evolve alongside the strategy. Instead of fixed bounds, each indicator's parameter range can widen or narrow based on what works best.

### Why It Matters
- Fixed parameter ranges may be suboptimal
- Different market conditions may favor different parameter scales
- Self-adapting ranges can discover unexpected effective values

### How It Works
1. Each `IndicatorGene` has an optional `param_bounds` field
2. Bounds can be initialized from config or evolved
3. Mutation can adjust bounds (widen/narrow) with constraints
4. Crossover can blend bounds from parents

### Configuration
```yaml
dynamic_bounds:
  enabled: true
  min_range: 5           # Minimum parameter range width
  max_range: 100         # Maximum parameter range width
  mutation_rate: 0.1     # Probability of mutating bounds
  bound_change_max: 0.2  # Max change per mutation (20%)
```

### API
```python
from genetic_algorithm.utils.dynamic_bounds import (
    initialise_bounds,
    mutate_bounds,
    sample_from_bounds,
    crossover_bounds
)

# Initialize bounds for an indicator
bounds = initialise_bounds(indicator_type='RSI', config=config)
# Returns: {'period': {'min': 5, 'max': 30}}

# Mutate bounds
new_bounds = mutate_bounds(bounds, change_max=0.2, min_range=5, max_range=100)

# Sample a value from bounds
period = sample_from_bounds(bounds, 'period')

# Crossover bounds from two parents
child_bounds = crossover_bounds(parent1_bounds, parent2_bounds, alpha=0.5)
```

---

## Integration with Evolution

Tier 3 features are integrated into the main evolution loop:

### In `evolution.py`
```python
# After elite selection (line ~694)
if self.parsimony_config.get('enabled'):
    removed = apply_parsimony_to_elites(elites, evaluate_fn, self.parsimony_config)
    logger.info(f"[PARSIMONY] Removed {removed} component(s) from elites")

# In evolve() loop (line ~885)
if self.pareto_archive:
    self.pareto_archive.update(population)
```

### In `run_ga.py`
```python
# After evolution completes (line ~604)
if monte_carlo_config.get('enabled'):
    for strategy in top_strategies:
        result = run_monte_carlo(strategy.trades, ...)
        print(f"Robustness: {result.robustness_score:.1%}")
```

### In `mutation.py`
```python
# Dynamic bounds mutation (line ~810)
if dynamic_bounds_config.get('enabled'):
    if random.random() < dynamic_bounds_config.get('mutation_rate', 0.1):
        gene.param_bounds = mutate_bounds(gene.param_bounds, ...)
```

---

## Test Coverage

All Tier 3 features have comprehensive unit tests in `tests/test_tier3_improvements.py`:

| Test Class | Tests | Coverage |
|------------|-------|----------|
| `TestMonteCarlo` | 6 | Bootstrap, shuffle, jitter, full analysis |
| `TestParsimony` | 5 | Simplification, removal, edge cases |
| `TestParetoArchive` | 6 | Update, dominance, crowding, serialization |
| `TestDynamicBounds` | 10 | Init, mutate, sample, crossover |
| `TestTier3Integration` | 4 | End-to-end integration |
| **Total** | **31** | All passing ✅ |

Run tests:
```bash
python -m pytest tests/test_tier3_improvements.py -v --tb=short --noconftest -o "addopts=" --rootdir=/tmp
```

---

## Verification Results

A real GA run with Tier 3 features enabled produced:

### Parsimony
```
[PARSIMONY] Simplified strategy: removed 1 component(s), complexity 11 → 9
[PARSIMONY] Simplified strategy: removed 1 component(s), complexity 4 → 3
[PARSIMONY] Removed 2 total component(s) from elites
```

### Monte-Carlo
```
MONTE-CARLO ROBUSTNESS ANALYSIS
  ✓ Rank 1: robustness=100.0%, mean_profit=12.69%, p5=2.09%, p95=30.08%
  ⚠️ Rank 2: robustness=0.0%, mean_profit=-97.96%
  ⚠️ Rank 3: robustness=0.0%, mean_profit=-317.59%
  ⚠️ Rank 4: robustness=0.0%, mean_profit=-1203.91%
  ⚠️ Rank 5: robustness=0.0%, mean_profit=-2638.69%
```

**Key Finding:** Monte-Carlo correctly identified that only 1 of 5 top strategies was truly robust. The other 4 appeared decent by fitness but would fail in live trading due to trade-order dependency.

---

## Next Steps (Tier 4)

Planned improvements building on Tier 3:

| Feature | Priority | Impact |
|---------|----------|--------|
| Automated Lookahead Analysis | High | Detect strategies using future data |
| Volume/Liquidity Constraints | High | Prevent execution failures |
| Trailing Stop Evolution | Medium | Better risk management |
| Per-Pair Stability Penalty | Medium | Reduce tail risk |
| Portfolio Correlation | Medium | Diversification |
| Data Quality Validation | Medium | Prevent garbage-in |

---

*Last Updated: February 26, 2026*
