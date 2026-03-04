# GA Project Comprehensive Audit Report

**Date:** 2025-01-XX  
**Scope:** Full codebase audit of `genetic_algorithm/` (~28,400 lines of Python)  
**Coverage:** All `.py` files read and analyzed. React frontend NOT analyzed.  

---

## Table of Contents

1. [Critical Bugs](#1-critical-bugs)
2. [High Severity Issues](#2-high-severity-issues)
3. [Medium Severity Issues](#3-medium-severity-issues)
4. [Low Severity Issues](#4-low-severity-issues)
5. [Dead Code](#5-dead-code)
6. [Areas Needing Deeper Analysis](#6-areas-needing-deeper-analysis)

---

## 1. Critical Bugs

Issues that WILL cause crashes, incorrect results, or major functional failures.

---

### C-01: TerminalMonitor Missing Required Methods
**File:** `monitor/terminal_monitor.py`  
**Impact:** `AttributeError` at runtime when evolution triggers these events  

`TerminalMonitor` is missing 3 methods that `NullMonitor` implements:
- `on_checkpoint_saved()`
- `on_log()`
- `on_error()`

When the evolution engine calls these methods, it will crash if using the terminal monitor.

**Fix:** Add stub implementations matching `NullMonitor`'s signatures.

---

### C-02: N-Dimensional Hypervolume Calculation Is Broken
**File:** `core/nsga2.py`, function `_hypervolume_nd()` (line ~375)  
**Impact:** Multi-objective optimization with 3+ objectives produces near-zero hypervolume values  

The code sorts points by first objective **descending** (`key=lambda x: -x[0]`), then computes `slice_width = point[0] - prev_slice` where `prev_slice` starts at `reference_point[0]`.

- First point (highest x): `slice_width = max_x - ref_x` → positive ✓
- Second point (lower x): `slice_width = lower_x - max_x` → **NEGATIVE** → skipped by `if slice_width > 0`
- All subsequent points: also negative → skipped

**Result:** Only the first point's contribution is counted. The hypervolume is massively underestimated.

**Note:** The 2D version (`_hypervolume_2d`) uses a different (staircase) approach and IS correct.

**Fix:** Change sort to ascending (`key=lambda x: x[0]`), OR reverse the slice accumulation logic.

---

### C-03: Generated Strategies Missing `startup_candle_count`
**File:** `strategies/generator.py`  
**Impact:** NaN indicator values at the start of backtesting cause incorrect signals  

The generator never sets `startup_candle_count` in generated strategy classes. FreqTrade needs this to know how many candles to skip before trading. Without it, indicators like SMA-200 will produce NaN for the first 199 candles and the strategy may generate bogus entry signals.

`template.py` has the attribute but is dead code (see D-01).

**Fix:** Calculate `startup_candle_count` from the maximum indicator period in the gene and write it into the generated class.

---

### C-04: `_generate_single_indicator_code` Returns `pass` for Unsupported Indicators
**File:** `strategies/generator.py`, line ~766  
**Impact:** Informative timeframe indicators silently broken for many indicator types  

Only RSI, MACD, BBANDS, EMA, SMA, STOCH, ATR, ADX, CCI are handled. All others (SUPERTREND, ICHIMOKU, MFI, OBV, WILLR, ROC, TEMA, KAMA, AROON, and all CDL_* patterns) cause `pass # Unsupported indicator: {type}` to be generated.

This only affects **informative timeframe** indicators. Base timeframe indicators use a separate, more complete `_generate_indicator_code()` method (line ~768+) which handles many more types.

**Fix:** Merge the two code paths, or extend `_generate_single_indicator_code` to cover all indicator types that `_generate_indicator_code` handles.

---

## 2. High Severity Issues

Issues that cause incorrect behavior, serious performance problems, or data corruption.

---

### H-01: SuperTrend Implementation Uses Python For-Loop Over All Rows
**File:** `strategies/generator.py` (SuperTrend code generation)  
**Impact:** Backtesting is extremely slow when SuperTrend is used  

The generated SuperTrend code uses a Python `for i in range(1, len(dataframe)):` loop to iterate over every row. On a typical dataset of 50K+ candles, this takes orders of magnitude longer than vectorized alternatives.

**Fix:** Use a vectorized SuperTrend implementation (e.g., from `pandas_ta` or a numpy-based version).

---

### H-02: Crossover Subsumption Assumes AND Logic
**File:** `core/crossover.py`, `_deduplicate_conditions()`  
**Impact:** Condition deduplication may incorrectly remove conditions when OR logic is used  

The subsumption check (`_is_subsumed_by`) assumes all conditions are AND-combined. If a strategy uses OR logic (which the `.logic` attribute supports), removing a "subsumed" condition changes the strategy semantics.

**Fix:** Skip subsumption deduplication when OR logic is active, or make deduplication logic-aware.

---

### H-03: `_enforce_max_indicators` Condition Cleanup Uses Wrong Key
**File:** `core/crossover.py`  
**Impact:** Conditions referencing removed indicators may survive, causing KeyError at runtime  

The cleanup filters conditions by matching `condition.indicator.instance_id`, but conditions may reference indicators by `.type` instead. If the matching fails, orphaned conditions referencing non-existent indicators remain.

**Fix:** Ensure condition cleanup matches on all possible reference paths (instance_id, type, and indicator object identity).

---

### H-04: `feature_importance.py` - `generations_seen` Overwritten Not Incremented
**File:** `evaluation/feature_importance.py`  
**Impact:** Feature importance decay and statistics are wrong  

`generations_seen` is set to the current generation number (`= gen`) instead of being incremented (`+= 1`). This means the value represents the last generation seen, not how many generations the feature has been active.

**Fix:** Change to `self._indicator_stats[key]['generations_seen'] += 1` (or track first/last seen separately).

---

### H-05: Hall of Fame Fingerprint Too Coarse
**File:** `core/hall_of_fame.py`  
**Impact:** Genuinely different strategies treated as duplicates  

The fingerprint is based only on the sorted set of indicator types and condition types. It ignores:
- Parameter values (e.g., RSI-14 vs RSI-7)
- Thresholds (RSI > 70 vs RSI > 30)
- Timeframes (informative 1h vs 4h)

**Fix:** Include parameter hashes, threshold values, and timeframes in the fingerprint.

---

### H-06: `regime_aware.py` Skips Zero-Fitness Segments
**File:** `evaluation/regime_aware.py`, `_aggregate_results()`  
**Impact:** Aggregate fitness is biased upward (bad segments hidden)  

When a segment has fitness 0.0, it's skipped in aggregation. But 0.0 may be a legitimate poor result, not a failure. This inflates the apparent performance.

**Fix:** Distinguish between "evaluation failed" (skip) and "zero fitness" (include). Use `None` or a sentinel for failures.

---

### H-07: `regime_aware.py` Averages `num_trades` Instead of Summing
**File:** `evaluation/regime_aware.py`, `_aggregate_metrics()`  
**Impact:** `num_trades` metric is incorrect (divided by segment count)  

Trade counts should be summed across segments, not averaged. If one segment has 100 trades and another has 200, the aggregate should be 300, not 150.

**Fix:** Sum `num_trades` instead of averaging.

---

### H-08: Parallel Parsimony Creates Ephemeral Process Pools
**File:** `evaluation/parallel.py`, `parallel_parsimony()`  
**Impact:** Unnecessary process creation overhead, potential resource exhaustion  

`parallel_parsimony` creates a new `ProcessPoolExecutor` each time instead of reusing the persistent pool from `ParallelEvaluator`. This wastes resources and adds startup latency.

**Fix:** Reuse the existing persistent pool.

---

### H-09: `direct_backtester.py` Returns `success=True` With 0 Trades
**File:** `evaluation/direct_backtester.py`  
**Impact:** Strategies that never trade are treated as successful evaluations  

When a backtest produces 0 trades, the result is `success=True` with all metrics at 0. Downstream fitness calculation may not properly penalize this.

**Fix:** Return `success=True` with a clear flag (e.g., `no_trades=True`), or return `success=False` with a reason.

---

### H-10: Mutation Only Handles RSI and CCI Thresholds
**File:** `core/mutation.py`, `_mutate_condition_threshold()`  
**Impact:** Threshold mutation silently skips all other indicator types  

Only RSI and CCI conditions get their thresholds mutated. STOCH, ADX, MFI, WILLR, and all other indicators with meaningful thresholds are silently skipped, reducing the GA's ability to explore the parameter space.

**Fix:** Add threshold ranges for all indicator types, or use a generic approach based on indicator config.

---

### H-11: `position_stacking: True` Hardcoded
**File:** `evaluation/direct_backtester.py`, line ~974  
**Impact:** Backtesting allows multiple positions in the same pair, which may not match live trading  

`position_stacking` is hardcoded to `True`. Most users run live trading without position stacking. This means backtest results don't reflect real-world behavior.

**Fix:** Make configurable via GA config, default to `False`.

---

### H-12: `dataformat_ohlcv: "feather"` Hardcoded
**File:** `evaluation/direct_backtester.py`, line ~963  
**Impact:** Fails if user data is in JSON, HDF5, or other formats  

The OHLCV data format is hardcoded to "feather". Users with data in other formats will get silent failures or crashes.

**Fix:** Read from FreqTrade config or make configurable.

---

### H-13: No `volume > 0` Check in Generated Strategies
**File:** `strategies/generator.py`  
**Impact:** Strategies may trade on zero-volume candles (bad data)  

Standard FreqTrade practice is to include `dataframe['volume'] > 0` as an entry condition. Generated strategies don't include this check.

**Fix:** Add `& (dataframe['volume'] > 0)` to all entry conditions.

---

### H-14: ROI Values Not Guaranteed Monotonically Decreasing
**File:** `core/strategy_gene.py` (random ROI generation)  
**Impact:** ROI table may have illogical values (higher ROI at later timepoints)  

ROI values are generated randomly without ensuring they decrease over time. A valid ROI table should have decreasing target profit as time increases.

**Fix:** Sort/enforce monotonically decreasing ROI values.

---

### H-15: Config Key Naming Inconsistency for Regime Detection
**File:** Multiple config files + `regime_detector.py`  
**Impact:** Regime detection may use wrong method or wrong defaults depending on config path  

Some configs use `method` while others use `detection_method`. The code has different defaults depending on which key is looked up, leading to inconsistent behavior.

**Fix:** Standardize on one key name and add migration/validation.

---

### H-16: LLM Provider Double-Exponential Retry
**File:** `llm/provider.py`  
**Impact:** API calls retry `max_retries²` times instead of `max_retries` times  

`generate_json()` has its own retry loop that calls `generate()`, which also has a retry loop. If `max_retries=3`, total attempts can be up to 9.

**Fix:** Remove the inner retry loop from `generate()` or the outer one from `generate_json()`.

---

### H-17: EventBus Lock Can Deadlock
**File:** `web/event_bus.py`  
**Impact:** Web dashboard freezes when sync subscriber triggers another event  

Uses `threading.Lock()` but sync subscribers are called while the lock is held. If a subscriber emits another event (re-entering `emit()`), it will deadlock.

**Fix:** Use `threading.RLock()` (reentrant lock), or release lock before calling subscribers.

---

### H-18: matplotlib.use() Called After pyplot Import
**File:** `visualization/visualizer.py`  
**Impact:** Backend setting has no effect; may crash on headless servers  

`matplotlib.use('Agg')` must be called BEFORE `import matplotlib.pyplot as plt`. If pyplot is imported first (directly or transitively), the backend switch is ignored.

**Fix:** Move `matplotlib.use('Agg')` before any pyplot imports.

---

### H-19: Twin Axes Leak in Visualizer
**File:** `visualization/visualizer.py`  
**Impact:** Memory leak and overlapping axes every generation  

A new `twinx()` axis is created each generation without removing the old one. After 50+ generations, the plot becomes unreadable and memory usage grows.

**Fix:** Store and reuse the twin axis, or clear the figure properly between generations.

---

### H-20: TradeVisualizer Backend Override
**File:** `visualization/trade_visualizer.py`  
**Impact:** Overrides matplotlib backend globally, breaking interactive plots  

Calling `matplotlib.use('Agg')` in this module affects ALL matplotlib usage in the process, not just this module. If the main visualizer or user code expects an interactive backend, it breaks.

**Fix:** Use a non-global approach (e.g., save figures without switching backend, or use separate processes).

---

## 3. Medium Severity Issues

Issues that cause edge-case failures, subtle bugs, or code quality problems.

---

### M-01: `parallel.py` - `timed_out` Variable Potentially Unbound
**File:** `evaluation/parallel.py`  
**Impact:** `UnboundLocalError` if `BrokenProcessPool` exception occurs  

If the pool breaks before `timed_out` is set, the variable is referenced in the `finally` block without having been assigned.

**Fix:** Initialize `timed_out = False` before the try block.

---

### M-02: `parallel.py` - Double-Counting Failed Individuals
**File:** `evaluation/parallel.py`  
**Impact:** Failed count inflated in logs/stats  

Failed individuals may be counted in both the exception handler and the post-processing loop, inflating the failure count.

**Fix:** Track failures in a set and count once.

---

### M-03: Parsimony ID Namespace Collision
**File:** `core/parsimony.py`  
**Impact:** Parsimony-added indicators could collide with real indicator IDs  

Parsimony generates IDs starting at `9000 + i`. If the main GA also assigns IDs in this range, collisions occur.

**Fix:** Use a dedicated namespace (e.g., negative IDs or UUID-based IDs).

---

### M-04: BacktestCache Schema Mismatch
**File:** `evaluation/direct_backtester.py`  
**Impact:** Crash on loading cached results from a different code version  

If the cache was written by a different version of the code with different metric fields, deserialization may crash.

**Fix:** Add a version/schema hash to the cache key.

---

### M-05: Cache Key Missing `strategy_max_open_trades`
**File:** `evaluation/direct_backtester.py`  
**Impact:** Cache may return stale results when max_open_trades changes  

The cache key doesn't include `strategy_max_open_trades`, so changing this config value still returns old cached results.

**Fix:** Include all backtest-affecting parameters in the cache key.

---

### M-06: Selection Adds Duplicates Despite `allow_duplicates=False`
**File:** `core/selection.py`, `select_parents()`  
**Impact:** Duplicate parents in breeding pool reduce genetic diversity  

After `max_attempts` is reached, the code falls back to adding a random individual which may be a duplicate. The `allow_duplicates=False` contract is violated.

**Fix:** Force uniqueness in the fallback path, or reduce the requested count.

---

### M-07: Duplicate `nsga2_tournament_selection` Functions
**File:** `core/selection.py` AND `core/nsga2.py`  
**Impact:** Code duplication, risk of divergence  

The same tournament selection logic exists in two files. Changes to one won't be reflected in the other.

**Fix:** Remove one and import from the other.

---

### M-08: Parsimony Can Remove ALL Exit Conditions
**File:** `core/parsimony.py`  
**Impact:** Strategy has no exit logic; trades never close  

The parsimony pressure can remove exit conditions one by one until none remain. A strategy with no exit conditions relies solely on stoploss/ROI, which may not be desirable.

**Fix:** Enforce a minimum of 1 exit condition.

---

### M-09: `regime_detector.py` Warmup Hardcoded for 15m Timeframes
**File:** `utils/regime_detector.py`  
**Impact:** Incorrect warmup period for other timeframes  

The warmup calculation assumes 15-minute candles. Using 1h or 5m data produces wrong warmup periods.

**Fix:** Accept timeframe as a parameter and compute warmup dynamically.

---

### M-10: HMM `min_dwell` Default Inconsistency
**File:** `utils/regime_detector.py`  
**Impact:** Different default dwell times depending on code path  

Some paths default to `min_dwell=10`, others to `min_dwell=1`. The effective minimum dwell time is unpredictable.

**Fix:** Use a single default value, ideally from config.

---

### M-11: `regime_walk_forward.py` Balance Score Bias
**File:** `utils/regime_walk_forward.py`  
**Impact:** Regime balance score penalizes strategies with fewer regimes  

The balance score calculation is biased: having fewer regime types yields a lower score even if the regimes are perfectly balanced.

**Fix:** Normalize the balance score by the number of regimes actually present.

---

### M-12: `overfit_analysis.py` - Signal Score Weight Ordering Fragile
**File:** `utils/overfit_analysis.py`  
**Impact:** Incorrect signal weights if metric order changes  

Signal score weights are applied by position (first metric × w1, second × w2, etc.) rather than by metric name. Reordering metrics silently breaks the weights.

**Fix:** Apply weights by metric name, not position.

---

### M-13: `dataset_policy.py` - `model_selection_ratio` Always 0
**File:** `utils/dataset_policy.py`  
**Impact:** Model selection split is non-functional  

`model_selection_ratio` is hardcoded to 0, making the model selection feature unused.

**Fix:** Make configurable or remove if not planned.

---

### M-14: `dynamic_bounds.py` - Bounds Can Drift Outside Meaningful Ranges
**File:** `utils/dynamic_bounds.py`  
**Impact:** GA parameters may drift to meaningless values (e.g., negative periods)  

Dynamic bounds expansion has no absolute limits. Over many generations, bounds can expand to include invalid parameter values.

**Fix:** Add hard minimum/maximum limits for each parameter type.

---

### M-15: `run_ga.py` - Temp Config File Not Cleaned Up on Exception
**File:** `run_ga.py`  
**Impact:** Orphaned temp files accumulate on disk  

The temp config file created for FreqTrade is not cleaned up if an exception occurs during setup.

**Fix:** Use a `try/finally` block or `tempfile.NamedTemporaryFile` with automatic cleanup.

---

### M-16: `run_manager.py` - `generation_stats` Grows Unbounded
**File:** `web/run_manager.py`  
**Impact:** Memory leak for long-running evolutions  

Every generation's stats are appended to a list that's never pruned. Over thousands of generations, this consumes significant memory.

**Fix:** Add a rolling window (e.g., keep last N generations) or write to disk.

---

### M-17: `run_manager.py` - Race Condition in Stop/Pause
**File:** `web/run_manager.py`  
**Impact:** Evolution may not stop correctly if pause and stop are called in quick succession  

The pause/stop state transitions aren't atomic. Rapid pause→stop can leave the evolution in an inconsistent state.

**Fix:** Use a proper state machine with lock-protected transitions.

---

### M-18: CORS Wildcard + Credentials
**File:** `web/server.py`  
**Impact:** CORS configuration is invalid per HTTP spec  

`allow_origins=["*"]` combined with `allow_credentials=True` is rejected by browsers per the CORS spec. Credentials-enabled CORS requires explicit origin listing.

**Fix:** Specify allowed origins explicitly, or disable credentials.

---

### M-19: `timerange.py` - Weighted Aggregation Loses Weight from NaN Scores
**File:** `utils/timerange.py`  
**Impact:** Total weight doesn't sum to 1.0 when some scores are NaN  

When a metric returns NaN, its weight is effectively lost (not redistributed). The total weighted score underestimates performance.

**Fix:** Renormalize weights after removing NaN entries.

---

### M-20: Indicator Factory Missing Code Generation for Multiple Indicators
**File:** `strategies/indicator_factory.py`  
**Impact:** These indicators can be selected by the GA but generate broken code  

The following indicators have entries in `indicator_factory.py` but lack code generation support in `_generate_single_indicator_code()`: MFI, OBV, WILLR, ROC, TEMA, KAMA, AROON.

They work in `_generate_indicator_code()` (base timeframe) but fail for informative timeframes.

**Fix:** Extend `_generate_single_indicator_code()` for these types (or unify the two code paths per C-04).

---

### M-21: `run_diagnostics.py` - CSV File Handle Leak
**File:** `utils/run_diagnostics.py`  
**Impact:** Open file handles accumulate on crash  

The CSV output file is opened but not closed in a `finally` block. If the diagnostic run crashes, the file handle leaks.

**Fix:** Use a `with` statement.

---

### M-22: `ensemble.py` - Vote Column Not Initialized
**File:** `strategies/ensemble.py`  
**Impact:** KeyError if fallback condition is triggered without prior voting  

The fallback condition references a vote column that may not exist if no strategies contributed votes.

**Fix:** Initialize vote column to 0 before the voting loop.

---

### M-23: Config - Indicators in Param Configs But Not in `available` List
**File:** Various config YAML files  
**Impact:** Some indicators have parameter definitions but can't be selected by the GA  

Several indicators have `indicator_params` entries but aren't listed in `available_indicators`, making the param configs dead.

**Fix:** Audit indicator lists and sync between `available_indicators` and `indicator_params`.

---

### M-24: `uniform_crossover` Doesn't Deep-Copy Scalar Fallbacks
**File:** `core/crossover.py`  
**Impact:** Parent and child may share mutable references  

When falling back to parent values, scalars are fine but any mutable objects (lists, dicts) would be shared between parent and child.

**Fix:** Use `copy.deepcopy()` for all fallback assignments.

---

### M-25: `regime_detector.py` - JSON Fallback Assumes Exactly 6 Columns
**File:** `utils/regime_detector.py`  
**Impact:** Crash or silent data corruption if OHLCV data has different column count  

The JSON fallback path hardcodes column names assuming a specific layout.

**Fix:** Use column names from the data or validate column count.

---

### M-26: `regime_detector.py` - Ensemble Doesn't Propagate Custom Params
**File:** `utils/regime_detector.py`  
**Impact:** Ensemble regime detection ignores user-configured parameters  

Custom detection parameters are not forwarded to sub-detectors in ensemble mode.

**Fix:** Pass `detection_params` through to each sub-detector.

---

### M-27: Fitness 0.0 Is Ambiguous
**File:** Multiple files  
**Impact:** "Failed evaluation" and "genuinely zero fitness" are indistinguishable  

Several code paths use `fitness = 0.0` both for "evaluation failed" and "strategy performed at exactly break-even". This makes it impossible to distinguish between the two in downstream code.

**Fix:** Use `None` or `float('-inf')` for failures, reserve 0.0 for legitimate zero fitness.

---

### M-28: No Config Schema Validation
**File:** `run_ga.py`, config loading  
**Impact:** Typos and invalid config keys silently ignored  

There's no JSON Schema or pydantic validation for the GA config. Users can misspell keys or use wrong types without any error message.

**Fix:** Add a config validation layer (pydantic model or JSON Schema).

---

### M-29: `pareto_archive.py` - `_generation_added` Uses `id()` (Memory Address)
**File:** `core/pareto_archive.py`  
**Impact:** Generation tracking breaks after garbage collection/object recreation  

Using `id(member)` as a hash key is unstable—Python can reuse memory addresses for new objects after old ones are GC'd.

**Fix:** Use a stable identifier (e.g., a UUID or fingerprint).

---

### M-30: `pareto_archive.py` - `from_dict` Doesn't Restore `_generation_added`
**File:** `core/pareto_archive.py`  
**Impact:** Archive restored from checkpoint loses generation tracking  

Deserialization doesn't reconstruct the `_generation_added` mapping, so all restored members appear as newly added.

**Fix:** Serialize and deserialize `_generation_added`.

---

### M-31: `direct_backtester.py` Config Path Inconsistency
**File:** `evaluation/direct_backtester.py`  
**Impact:** Timeframe may come from wrong config section  

Sometimes reads from `config['strategy']` and sometimes from `config['strategy_constraints']` for timeframe-related settings.

**Fix:** Standardize on one config path.

---

### M-32: Short Selling Exit Conditions Not Fully Handled
**File:** `strategies/generator.py`  
**Impact:** Short exit logic may be incomplete or missing in some generated code paths  

When `can_short` is enabled, the exit condition generation may not cover all code paths properly, potentially leaving short positions without proper exit signals.

**Fix:** Audit all exit condition code paths for short selling support.

---

## 4. Low Severity Issues

Minor issues, style problems, or cosmetic bugs.

---

### L-01: `EventType.RUN_COMPLETED` Defined But Never Used
**File:** `web/event_bus.py`  
**Impact:** Dead enum value  

---

### L-02: LLM Prompts Reference STOCH But It's Not in INDICATOR_REFERENCE
**File:** `llm/prompts.py`  
**Impact:** LLM may generate references to indicators not in the reference table  

---

### L-03: `run_ga.py` - `print_configuration` Only Prints Subset of Fitness Weights
**File:** `run_ga.py`  
**Impact:** User doesn't see all active configuration  

---

### L-04: `run_ga.py` - Dashboard Ctrl+C Doesn't Gracefully Stop Runs
**File:** `run_ga.py`  
**Impact:** Abrupt shutdown may lose current generation progress  

---

### L-05: Condition Logic Override Makes Per-Condition Logic Meaningless
**File:** `core/strategy_gene.py` / `core/crossover.py`  
**Impact:** Individual condition `.logic` attributes are set but overridden by strategy-level logic  

Each condition has a `.logic` attribute (AND/OR), but the overall strategy uses a single global logic mode. The per-condition values are noise.

**Fix:** Either honor per-condition logic or remove the attribute.

---

### L-06: Variable Shadowing in `mutation.py`
**File:** `core/mutation.py`  
**Impact:** Readability issue; outer `available_indicators` and `indicator_config` are shadowed in inner scope  

---

### L-07: `parsimony.py` Inconsistent Fitness Setting
**File:** `core/parsimony.py`  
**Impact:** Bypasses `set_fitness()` method, potentially skipping validation  

Directly sets `individual.fitness` instead of using the setter method.

---

### L-08: `run_manager.py` - No Mechanism to Remove Finished Runs
**File:** `web/run_manager.py`  
**Impact:** `_runs` dict grows indefinitely in long-running web servers  

---

### L-09: `parallel.py` - Health Check Timeout Hardcoded to 10s
**File:** `evaluation/parallel.py`  
**Impact:** May be too short on heavily loaded machines  

---

### L-10: `feature_importance.py` - No Persistence/Serialization
**File:** `evaluation/feature_importance.py`  
**Impact:** Feature importance data lost on restart  

---

### L-11: `feature_importance.py` - `importance_score` Formula Has Asymmetric Scaling
**File:** `evaluation/feature_importance.py`  
**Impact:** Positive and negative importance aren't weighted symmetrically  

---

### L-12: `pareto_archive.py` - O(n²) Pruning
**File:** `core/pareto_archive.py`  
**Impact:** Slow for large archives (>500 members)  

---

### L-13: `DEFAULT_BTC_BALANCE` / `DEFAULT_STABLECOIN_BALANCE` Hardcoded
**File:** Various  
**Impact:** Users with different balances must manually override  

---

### L-14: `hall_of_fame.py` - Thread-Unsafe Save/Load
**File:** `core/hall_of_fame.py`  
**Impact:** Concurrent save/load can corrupt the file  

---

### L-15: `nsga2.py` - `fast_non_dominated_sort` O(MN²) With No Size Guard
**File:** `core/nsga2.py`  
**Impact:** Very slow for populations > 500  

---

### L-16: `run_diagnostics.py` - `0.0 or 0` Loses Precision
**File:** `utils/run_diagnostics.py`  
**Impact:** Falsy float values (0.0) are replaced with integer 0  

---

---

## 5. Dead Code

Files or functions that exist but are never used.

---

### D-01: `strategies/template.py` Is Dead Code
**Impact:** Maintenance burden; misleading to developers  

`STRATEGY_TEMPLATE` and `get_strategy_template()` are defined but never imported or used anywhere in the codebase. `generator.py` builds strategy code entirely with inline f-strings.

**Action:** Either integrate `template.py` into `generator.py` (which would be cleaner) or remove it.

---

### D-02: `strategies/components.py` Is Dead Code
**Impact:** Parallel indicator definition system causes confusion  

`components.py` defines indicator configurations but is never imported anywhere. `generator.py` and `indicator_factory.py` have their own indicator definitions. This creates confusion about which definitions are authoritative.

**Action:** Remove or integrate into the main indicator system.

---

### D-03: `parallel_walk_forward_validation` Function (Original Version)
**File:** `evaluation/parallel.py`  
**Impact:** Superseded by flat version  

The original `parallel_walk_forward_validation` has been superseded by `_flat_wf_parallel` but still exists.

---

### D-04: `_auto_detect_segments` in `regime_aware.py`
**File:** `evaluation/regime_aware.py`  
**Impact:** Unused function  

---

## 6. Areas Needing Deeper Analysis

These areas were reviewed at the code level but would benefit from runtime testing or deeper investigation.

---

### DA-01: Generated Strategy Code End-to-End Testing
**Priority:** HIGH  

No generated `.py` strategy files were tested with FreqTrade. The generator produces complex code with many branches. Specific concerns:
- Do all indicator combinations produce valid Python?
- Do informative timeframe merges work correctly?
- Are all column names consistent between indicator generation and condition generation?
- Do CDL_* patterns work correctly after suffix sanitization?

**Recommendation:** Generate strategies with diverse gene combinations and run them through FreqTrade's backtest to verify they execute without errors.

---

### DA-02: Web Dashboard Frontend (React)
**Priority:** MEDIUM  

The React frontend code was not analyzed. Potential concerns:
- WebSocket reconnection handling
- State management correctness
- Error boundary coverage
- Build/bundle configuration

---

### DA-03: Test Suite Effectiveness
**Priority:** MEDIUM  

The `tests/` directory was not analyzed for coverage gaps. Key concerns:
- Are edge cases covered (empty populations, zero trades, NaN metrics)?
- Do tests catch the bugs identified in this report?
- Is there integration testing (full evolution cycle)?

---

### DA-04: Performance Profiling
**Priority:** MEDIUM  

No performance profiling was done. Potential bottlenecks:
- Strategy file I/O (write → import → delete per evaluation)
- Process pool creation/destruction in parallel parsimony (H-08)
- O(MN²) non-dominated sort for large populations (L-15)
- Memory growth from unbounded generation_stats (M-16)
- SuperTrend Python loop (H-01)

---

### DA-05: LLM Integration End-to-End
**Priority:** LOW  

The LLM module was analyzed at code level, but the actual quality of LLM-generated strategies was not assessed. Concerns:
- Do LLM-generated genes conform to all GA invariants?
- Are parsed thresholds always within valid ranges?
- Does the retry logic handle all API error types?

---

## Summary Statistics

| Severity | Count |
|----------|-------|
| Critical | 4 |
| High | 20 |
| Medium | 32 |
| Low | 16 |
| Dead Code | 4 |
| Needs Deeper Analysis | 5 |
| **Total** | **81** |

## Recommended Fix Order

1. **C-01** (TerminalMonitor crash) — trivial 5-minute fix
2. **C-03** (startup_candle_count) — important for backtest accuracy
3. **C-04** (informative indicator code generation) — unify the two code paths
4. **C-02** (ND hypervolume) — fix sort order to ascending
5. **H-13** (volume > 0 check) — standard practice, easy add
6. **H-11, H-12** (hardcoded position_stacking, dataformat) — make configurable
7. **H-01** (SuperTrend loop) — significant performance impact
8. **H-10** (mutation threshold) — limits GA exploration
9. **D-01, D-02** (dead code cleanup) — reduces confusion
10. All remaining High issues
11. Medium issues by impact
12. Low issues as time permits
