# GA Bug Tracking Log
# Created: 2026-02-21
# Status: Active testing

## Test Results Summary

### Successful Tests (2026-02-21)

1. **Walk-Forward Optimization Test** (`ga_test_walkforward.yaml`)
   - 4 generations, 6 population
   - Walk-forward: train_days=60, validation_days=30, step_days=30
   - Fitness sharing enabled
   - Result: ✅ Completed successfully, 5 strategies saved

2. **NSGA-II Multi-Objective Test** (`ga_test_nsga2.yaml`)
   - 3 generations, 10 population
   - Multi-objective: profit + sharpe
   - Result: ✅ Completed successfully, strategy saved

3. **Basic GA with Progress Bar** (`ga_production.yaml`)
   - 2 generations, 2 population (quick test)
   - Progress bar with tqdm
   - Result: ✅ Completed successfully

4. **Visualization Test** (`ga_test_visualization.yaml`)
   - 2 generations, 4 population
   - Visualization enabled (non-interactive mode)
   - Result: ✅ Completed successfully, plot saved to output/plots/

---

## Bugs Found During Testing

---

### BUG-001: NSGA-II Fitness Display Inconsistency
**Severity**: Low (Cosmetic/Confusing)
**Status**: Open
**Found**: 2026-02-21

**Description**:
In NSGA-II mode, the progress bar shows single-objective weighted fitness (e.g., 0.477),
but after `set_objectives()` is called, fitness is overwritten with the first objective
value (scaled profit, e.g., 0.0036). This causes:
- Progress bar shows: `best_fit=0.477`
- Stats log shows: `[STATS] Best: 0.0036`

**Root Cause**:
In `evolution.py:evaluate_population()`:
1. `fitness, metrics = fitness_evaluator.evaluate()` computes single-objective fitness
2. `individual.set_fitness(fitness, metrics)` sets fitness
3. Progress bar reads fitness and displays it
4. `individual.set_objectives(objectives, metrics)` OVERWRITES fitness with objectives[0]
5. Stats compute from the overwritten value

**Location**: 
- `genetic_algorithm/core/individual.py` line 100: `self.fitness = objectives[0]`
- `genetic_algorithm/core/evolution.py` line 290: Progress bar reads fitness

**Suggested Fix**:
Either:
1. Don't overwrite fitness in set_objectives for backwards compatibility
2. Or update progress bar to use objectives[0] in NSGA-II mode
3. Or add a separate `nsga2_fitness` property

---

### BUG-002: Walk-Forward Window Logging Repetition
**Severity**: Very Low (Cosmetic)
**Status**: Open
**Found**: 2026-02-21

**Description**:
For each strategy evaluation, the walk-forward window creation logs are repeated:
```
Creating walk-forward windows from 20250601 to 20260101
Parameters: train_days=60, validation_days=30, step_days=30, mode=rolling
Stopping: validation window would exceed available data
Created 5 walk-forward windows
```

This could be reduced to only log once per generation or cached.

**Location**: `genetic_algorithm/utils/timerange.py`

---

## Features Tested ✅

1. **Walk-Forward Optimization** - Working correctly
2. **NSGA-II Multi-Objective** - Working, minor display issue (BUG-001)
3. **Fitness Sharing** - Working correctly
4. **Progress Bar (tqdm)** - Working correctly
5. **Visualization** - Working with non-interactive fallback
6. **Elite Preservation** - Working correctly
7. **Random Immigrants** - Working correctly
8. **Tournament Selection** - Working correctly

## Features Not Yet Tested

1. **Multi-timeframe indicators** - Not used in test configs
2. **Different selection methods** - Only tournament tested
3. **High population sizes** - Only tested with 4-10
4. **Roulette wheel selection** - Not tested
5. **Interactive visualization** - Requires display (TkAgg)

---

### BUG-003: Generator Crashes with Empty Indicators List
**Severity**: Medium (Causes crash)
**Status**: Open
**Found**: 2026-02-21

**Description**:
In `_generate_random_conditions()`, if the indicators list is empty, the code
tries to access `indicators[0]` causing an IndexError.

**Error Message**:
```
IndexError: list index out of range
  File "generator.py", line 171, in _generate_random_conditions
    indicator = indicators[0]
```

**Root Cause**:
When `valid_indicators` is empty (e.g., no indicators of type RSI/MACD/etc exist),
the fallback code assumes `indicators` itself has at least one element.

**Location**: `genetic_algorithm/strategies/generator.py` line 171

**Suggested Fix**:
```python
if not valid_indicators:
    if not indicators:
        # Create a default indicator if none exist
        indicator = IndicatorGene(type='RSI', period=14)
        indicators.append(indicator)
    indicator = indicators[0]
```

---

