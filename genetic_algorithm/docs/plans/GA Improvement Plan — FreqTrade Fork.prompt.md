# GA Improvement Plan — FreqTrade Fork

## Goal

Improve the Genetic Algorithm evolution outcome so that evolved strategies generalise to real-life trading scenarios instead of overfitting to historical data.

---

## Tier 1 — Anti-Overfitting & Realism (Highest Impact)

### 1. Walk-Forward Embargo Period ✅
**File:** `genetic_algorithm/utils/timerange.py`
Add an `embargo_days` parameter to `create_walk_forward_windows()` that inserts a gap between training and validation windows. This prevents autocorrelated data (e.g. momentum carry-over) from leaking between the two periods.

### 2. Train-Validation Gap Penalty ✅
**File:** `genetic_algorithm/evaluation/fitness.py`
Penalise strategies whose training fitness is significantly higher than their validation fitness. A large gap is a strong overfit signal. The penalty is progressive: small gaps are tolerated, large gaps are penalised up to a configurable cap.

### 3. Out-of-Sample Holdout Validation ✅
**Files:** `genetic_algorithm/evaluation/fitness.py`, `genetic_algorithm/run_ga.py`
Reserve the final N% of data (default 15%) as a completely unseen holdout set. After evolution finishes, the top strategies are evaluated on this holdout. Strategies with >30% fitness degradation are flagged. This is the single strongest guard against overfitting.

### 4. Realistic Slippage Modelling ✅
**File:** `genetic_algorithm/evaluation/direct_backtester.py`, `genetic_algorithm/config/ga_config.yaml`
Add a `slippage_pct` config value that is summed onto the exchange fee before backtesting. This models spread, market impact, and execution delays that are absent from idealised backtests.

### 5. Per-Pair Performance Breakdown & Penalty ✅
**Files:** `genetic_algorithm/evaluation/direct_backtester.py`, `genetic_algorithm/evaluation/fitness.py`
Extract per-pair profit from backtest results. Penalise strategies where any single pair has an outsized loss (beyond `pair_loss_threshold`). This prevents aggregate-level profit from masking catastrophic pair-concentration risk.

---

## Tier 2 — Genetic Operator Quality

### 6. AND/OR Condition Logic in Strategy Template ✅
**File:** `genetic_algorithm/strategies/generator.py`
Rewrite `_generate_condition_code()` to honour each condition's `logic` field individually. AND conditions become required terms joined with `&`; OR conditions are grouped into a single `(A | B | C)` block. Mixed strategies get `(AND1) & (AND2) & (OR1 | OR2)`. This dramatically expands the strategy search space beyond all-AND or all-OR.

### 7. Expose Crossover Method in Config ✅
**Files:** `genetic_algorithm/core/evolution.py`, `genetic_algorithm/config/ga_config.yaml`
Read `crossover_method` (`single_point`, `uniform`, `component`) from YAML config and pass it through to the `crossover()` call. Previously the method was hard-coded.

### 8. Fix Failed Walk-Forward Window Handling ✅
**File:** `genetic_algorithm/evaluation/fitness.py`
Previously, failed windows appended `0.0` to `validation_fitness_scores`, dragging the average down unfairly. Now failed windows are tracked separately via a counter and the final fitness is scaled by `success_ratio = successful / total`. If all windows fail, fitness is zero.

### 9. Checkpoint Save / Load / Resume ✅
**Files:** `genetic_algorithm/core/evolution.py`, `genetic_algorithm/run_ga.py`
Add `save_checkpoint()`, `load_checkpoint()`, `restore_from_checkpoint()` methods. Checkpoints serialise the full population, generation stats, best individual, and adaptive parameters to JSON with atomic writes. The `--resume` CLI flag restores the last checkpoint and continues evolution.

### 10. Update Config with New Settings ✅
**File:** `genetic_algorithm/config/ga_config.yaml`
Add all new config keys: `crossover_method`, `slippage_pct`, `pair_loss_threshold`, `embargo_days`, `gap_penalty.*`, `holdout_validation.*`, `storage.checkpoint_dir`, `storage.checkpoint_interval`.

---

## Tier 3 — Advanced (Not Yet Implemented)

### 11. Monte-Carlo Robustness Scoring
Run each top strategy through N randomised permutations (shuffled trade order, jittered entry/exit times, varied slippage). The fraction of permutations that remain profitable becomes a robustness multiplier on fitness. This catches strategies that are fragile to minor execution differences.

### 12. Dynamic Indicator Parameter Ranges
Instead of fixed parameter ranges (e.g. RSI period 10–30), allow the GA to meta-evolve the bounds themselves. This avoids the human bias of choosing "reasonable" ranges and lets the algorithm discover non-obvious parameter regions.

### 13. Strategy Simplification Pressure (Parsimony)
Beyond the existing additive complexity penalty, add an Occam's-razor operator: after each generation, attempt to remove one random indicator/condition from each elite strategy and keep the simpler version if fitness doesn't drop by more than ε. This actively pushes toward minimal strategies that are less likely to overfit.

### 14. Multi-Objective Pareto Archive with Crowding Decay
Enhance NSGA-II by maintaining a separate external Pareto archive. Apply crowding-distance decay over generations to gradually shrink the archive toward the most robust region of the front, rather than keeping all non-dominated solutions indefinitely.

---

## Implemented Test Suite

All 10 implemented features are covered by **42 unit tests** in `tests/test_ga_improvements.py`:

| # | Feature | Tests |
|---|---------|-------|
| 1 | Embargo period | 5 |
| 2 | Train-val gap penalty | 4 |
| 3 | Holdout split | 3 |
| 4 | AND/OR conditions | 5 |
| 5 | Slippage modelling | 3 |
| 6 | Per-pair metrics & penalty | 4 |
| 7 | Checkpointing | 5 |
| 8 | Crossover method config | 3 |
| 9 | Failed window handling | 4 |
| 10 | Config new settings | 6 |

### Bug Found During Testing

`restore_from_checkpoint()` in `evolution.py` was missing the required `size` field when reconstructing `PopulationStats`. Fixed by adding `size` to both the save and restore paths (with fallback for old checkpoints).

---

## Next Steps

1. Pick one Tier 3 item to implement next (Monte-Carlo robustness is recommended as highest-impact).
2. Run a full GA evolution end-to-end with all Tier 1+2 improvements active and compare holdout results against a baseline run without them.
3. Tune the new config knobs (`embargo_days`, `gap_penalty.threshold`, `holdout_pct`, `slippage_pct`) on real market data.
