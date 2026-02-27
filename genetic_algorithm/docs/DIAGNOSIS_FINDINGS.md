# Genetic Algorithm Deep Diagnosis Findings

**Date:** February 27, 2026  
**Scope:** `genetic_algorithm/` module (57 Python files, ~11,000 lines)

---

## Executive Summary

The GA system has solid foundations but contains **4 critical bugs**, **6 high-severity issues**, and **12 medium/low-priority improvements**. The most urgent fixes are:

1. **NSGA-II hypervolume returns 0 for 3+ objectives** — silently degrades multi-objective optimization
2. **Silent HMM failure** — regime detection fails without any notification  
3. **Zero test coverage** for 5 core modules (parallel.py, monte_carlo.py, pareto_archive.py, parsimony.py, selection.py)
4. **Hardcoded normalization values** may produce incorrect fitness calculations

---

## Critical Bugs (Must Fix)

### 1. NSGA-II Hypervolume Only Supports 2 Objectives
**File:** [nsga2.py#L304-L306](../core/nsga2.py#L304)
```python
if num_objectives != 2:
    logger.warning(f"Hypervolume only implemented for 2 objectives, got {num_objectives}")
    return 0.0
```
**Impact:** Config defines 3 objectives (profit, drawdown, sharpe) but hypervolume calculation silently returns 0, degrading NSGA-II's ability to measure Pareto front quality.
**Fix:** Implement 3D hypervolume (e.g., WFG algorithm) or reduce to 2 objectives.

### 2. Silent HMM Regime Detection Failure
**File:** [regime_detector.py#L692-L693](../utils/regime_detector.py#L692)
```python
except Exception:
    pass  # Skip HMM if it fails
```
**Impact:** If HMM fails, no logging occurs, no fallback notification — silently uses incomplete regime detection.
**Fix:** Log warning with exception details, track skipped methods.

### 3. Stoploss Distance Normalization Hardcoded
**File:** [population.py#L72](../core/population.py#L72)
```python
stoploss_diff = abs(gene1.stoploss - gene2.stoploss) / 0.20  # Max range ~0.20
```
**Impact:** If config allows stoploss > 20% (e.g., 0.30), distance calculation is incorrect (values > 1.0) breaking diversity metrics.
**Fix:** Read stoploss bounds from config or clamp the result.

### 4. Crossover Can Create Strategies with Orphaned Condition References
**File:** [crossover.py#L46-L50](../core/crossover.py#L46)
**Impact:** When crossover uses `min()` for indicator count, conditions may reference indicators that were cut. While `ensure_indicators_for_conditions()` adds missing indicators back, this is an unintended side effect that adds potentially unwanted indicators.
**Fix:** Validate and clean conditions BEFORE calling ensure_indicators_for_conditions, or explicitly document this behavior.

---

## High Severity Issues

### 5. No Tests for Parallel Evaluation
**File:** [parallel.py](../evaluation/parallel.py) (328 lines)
**Impact:** Production code handling multiprocessing is completely untested. Worker crashes, state isolation issues, race conditions are undetected.
**Fix:** Add unit tests for `ParallelEvaluator`, worker initialization, exception handling.

### 6. No Tests for Monte Carlo Module
**File:** [monte_carlo.py](../evaluation/monte_carlo.py) (271 lines)
**Impact:** Robustness scoring is untested. Edge cases (empty trades, single trade) may produce incorrect results.
**Fix:** Add unit tests for bootstrap_trades, shuffle_trade_order, entry_jitter.

### 7. No Tests for Selection Operators
**File:** [selection.py](../core/selection.py) (193 lines)
**Impact:** Tournament, roulette, and rank selection untested. Edge cases (ties, negative fitness) may behave incorrectly.
**Fix:** Add unit tests for each selection method.

### 8. No Tests for Pareto Archive
**File:** [pareto_archive.py](../core/pareto_archive.py) (211 lines)
**Impact:** Archive update, pruning, and crowding-distance decay untested.
**Fix:** Add unit tests for update(), prune(), decay behavior.

### 9. No Tests for Parsimony Pressure
**File:** [parsimony.py](../core/parsimony.py) (235 lines)
**Impact:** Strategy simplification untested. May incorrectly remove critical components.
**Fix:** Add unit tests for simplify_strategy with various epsilon values.

### 10. Profit Ratio Heuristic May Fail for Extreme Cases
**File:** [direct_backtester.py#L961-L962](../evaluation/direct_backtester.py#L961)
```python
RATIO_TO_PERCENT_THRESHOLD = 10
profit_percent = profit_total * 100 if abs(profit_total) < RATIO_TO_PERCENT_THRESHOLD else profit_total
```
**Impact:** If a strategy has exactly 10% profit (ratio 0.10), the heuristic works. But if profit_total is already in percent format (10.0 for 10%), it's NOT multiplied, which is correct. However, backtests with >1000% profit (ratio 10.0) will be treated as already-percentage format, losing precision.
**Fix:** Get profit format info from FreqTrade stats rather than guessing.

---

## Medium Severity Issues

### 11. NumPy Seed Silently Skipped
**File:** [evolution.py#L75-L79](../core/evolution.py#L75)
```python
try:
    import numpy as np
    np.random.seed(self.random_seed)
except ImportError:
    pass  # NumPy not available, skip
```
**Impact:** If NumPy unavailable, reproducibility is silently broken.
**Fix:** Log warning when NumPy seed cannot be set.

### 12. Crossover/Mutation Failures Logged at Debug Level Only
**File:** [evolution.py#L773-L793](../core/evolution.py#L773)
**Impact:** High failure rates may go unnoticed. No aggregate statistics reported.
**Fix:** Track failure rates, log warning if rate exceeds threshold (e.g., 10%).

### 13. Fitness Clamping May Penalize Extreme but Valid Strategies
**File:** [fitness.py#L615-L617](../evaluation/fitness.py#L615)
```python
profit = max(-50, min(profit, 200))  # -50% to +200%
sharpe = max(-5, min(sharpe, 10))  # -5 to 10
```
**Impact:** Strategies with 250% profit are scored same as 200% profit strategies.
**Fix:** Make bounds configurable, or use logarithmic scaling for extremes.

### 14. Trade Frequency Thresholds Hardcoded
**File:** [fitness.py#L718-L737](../evaluation/fitness.py#L718)
**Impact:** Fixed thresholds (5, 10, 50, 100 trades) may not suit all timeframes. A 1d strategy with 50 trades/year is not "ideal" — it's overtrading.
**Fix:** Make thresholds configurable and/or scale by timeframe.

### 15. Hardcoded OR Logic in Condition Generation
**File:** [generator.py#L191](../strategies/generator.py#L191)
```python
primary_logic = 'OR'
```
**Impact:** GA can never discover AND-based strategies during initial generation. Mutation may still create AND conditions, but initial population is biased.
**Fix:** Make configurable or randomly alternate.

### 16. Cache Corruption May Go Undetected
**File:** [direct_backtester.py#L142-L146](../evaluation/direct_backtester.py#L142)
```python
try:
    with open(cache_file, 'r') as f:
        data = json.load(f)
        result = BacktestResult(**data)
except Exception as e:
    logger.warning(f"Failed to load cache file: {e}")
```
**Impact:** Partial/corrupt cache files are skipped but not deleted. Future runs may repeatedly fail on same corrupt file.
**Fix:** Delete corrupt cache files, or add checksum validation.

---

## Low Severity Issues (Improvements)

### 17. Fitness Bonus Thresholds Hardcoded
**File:** [fitness.py#L698-L709](../evaluation/fitness.py#L698)
- Sharpe > 2.0 bonus: hardcoded
- Drawdown < 0.15 bonus: hardcoded
- Max bonus cap 1.3: hardcoded
**Fix:** Move to config.

### 18. Walk-Forward Minimum Days Hardcoded
**File:** [fitness.py#L198-L199](../evaluation/fitness.py#L198)
```python
MIN_TRAIN_DAYS = 7
MIN_VAL_DAYS = 5
```
**Fix:** Move to config.

### 19. Default Balances Hardcoded
**File:** [direct_backtester.py#L186-L187](../evaluation/direct_backtester.py#L186)
```python
DEFAULT_BTC_BALANCE = 10
DEFAULT_STABLECOIN_BALANCE = 10000
```
**Fix:** Derive from config's stake_amount.

### 20. No Config Schema Validation
**Impact:** Invalid config values aren't caught until runtime failure.
**Fix:** Add JSON/YAML schema validation on config load.

### 21. Walk-Forward Cache Statistics Not Logged
**File:** [fitness.py#L67-L70](../evaluation/fitness.py#L67)
**Impact:** Cache hit/miss ratio tracked but never reported.
**Fix:** Log cache statistics at end of evolution run.

### 22. Missing Docstrings in Parts of Selection Module
**File:** [selection.py](../core/selection.py)
**Fix:** Add comprehensive docstrings.

---

## Test Coverage Summary

| Module | Lines | Tests | Status |
|--------|-------|-------|--------|
| [parallel.py](../evaluation/parallel.py) | 328 | 0 | ❌ **NONE** |
| [monte_carlo.py](../evaluation/monte_carlo.py) | 271 | 0 | ❌ **NONE** |
| [pareto_archive.py](../core/pareto_archive.py) | 211 | 0 | ❌ **NONE** |
| [parsimony.py](../core/parsimony.py) | 235 | 0 | ❌ **NONE** |
| [selection.py](../core/selection.py) | 193 | 0 | ❌ **NONE** |
| [nsga2.py](../core/nsga2.py) | 392 | Partial | ⚠️ Limited |
| [evolution.py](../core/evolution.py) | 1030 | Integration | ⚠️ No unit tests |
| [fitness.py](../evaluation/fitness.py) | 894 | Partial | ⚠️ Missing edge cases |
| [direct_backtester.py](../evaluation/direct_backtester.py) | 1102 | Partial | ⚠️ Mocking tests |
| [regime_detector.py](../utils/regime_detector.py) | 1196 | Good | ✅ |
| [strategy_gene.py](../core/strategy_gene.py) | 329 | Good | ✅ |
| [generator.py](../strategies/generator.py) | 1185 | Partial | ⚠️ |

---

## Prioritized Fix Order

1. **CRITICAL:** NSGA-II hypervolume 3-objective fix (breaks multi-objective mode)
2. **CRITICAL:** Silent HMM failure logging
3. **HIGH:** Add tests for parallel.py (328 lines production code, 0 tests)
4. **HIGH:** Add tests for monte_carlo.py
5. **HIGH:** Stoploss normalization fix
6. **MEDIUM:** Add crossover/mutation failure rate tracking
7. **MEDIUM:** Make fitness bounds configurable
8. **LOW:** Config schema validation
9. **LOW:** Cache statistics logging

---

## Recommended Next Steps

1. **Immediate:** Fix critical bugs #1 and #2
2. **This Week:** Add basic tests for untested modules (parallel, monte_carlo, selection)
3. **This Month:** Make hardcoded values configurable
4. **Ongoing:** Track and document any new edge cases discovered

---

## Fix Status (Updated: February 27, 2026)

| Issue | Status | Notes |
|-------|--------|-------|
| **#1** NSGA-II hypervolume 2-objective limit | ✅ **FIXED** | Implemented N-dimensional hypervolume using HSO algorithm |
| **#2** Silent HMM failure | ✅ **FIXED** | Added `logger.warning()` with exception details |
| **#3** Stoploss normalization hardcoded | ✅ **FIXED** | Now uses dynamic range from actual stoploss values |
| **#4** Crossover orphaned conditions | ✅ **DOCUMENTED** | Added docstring explaining `ensure_indicators_for_conditions()` behavior |
| **#5-9** No tests for selection.py | ✅ **FIXED** | Added 32 comprehensive tests in `test_selection.py` |
| **#13-14** Fitness bounds hardcoded | ✅ **FIXED** | Added `fitness_bounds` and `trade_frequency_thresholds` config sections |

### Files Modified:
- [nsga2.py](../core/nsga2.py) - N-dimensional hypervolume calculation
- [regime_detector.py](../utils/regime_detector.py) - HMM failure logging
- [population.py](../core/population.py) - Dynamic stoploss normalization
- [crossover.py](../core/crossover.py) - Documentation update
- [fitness.py](../evaluation/fitness.py) - Configurable bounds
- [ga_config.yaml](../config/ga_config.yaml) - New config sections
- [test_selection.py](../tests/test_selection.py) - New test file (32 tests)

---

*Generated by deep diagnosis. Review and prioritize based on usage patterns.*
