# Feature Status & Configuration Guide

> **Living reference** — update after each validation run.  
> Last updated: 2026-03-02 (post valRunRegimeTest analysis)

---

## Quick Reference Table

| Feature | Status | Verdict | Notes |
|---------|--------|---------|-------|
| Core GA (selection, crossover, mutation) | ✅ WORKING | USE | Stable; uniform crossover + tournament recommended |
| Walk-Forward Validation | ✅ WORKING | USE | Essential for preventing overfitting |
| Fitness Sharing | ⚠️ PARTIAL | USE WITH CARE | Works but over-penalizes at high sharing_radius |
| Holdout Monitoring | ⚠️ PARTIAL | DIAGNOSTIC ONLY | Fitness penalty compounds with sharing — disable penalty |
| Regime-Aware Evaluation | ⚠️ PARTIAL | USE WITH FIXES | harmonic_mean + long-only = broken; use mean aggregation |
| DSR (Deflated Sharpe) | ⚠️ PARTIAL | USE WITH LOW THRESHOLD | Inactive at min_trades=20 under regime segmentation |
| CPCV | 🔬 UNTESTED | SKIP FOR NOW | Expensive; use for final validation only |
| Monte Carlo | 🔬 UNTESTED | SKIP FOR NOW | Implemented but not integrated into fitness loop |
| Parsimony Pressure | ✅ WORKING | USE | Simplifies elites without fitness loss |
| Dynamic Bounds | 🔬 UNTESTED | OPTIONAL | Evolves parameter ranges; low priority |
| Parallel Evaluation | ✅ WORKING | USE | 3-5x speedup; walk-forward disabled in workers |
| LLM Seeding | ⚠️ PARTIAL | OPTIONAL | Generates valid seeds; quality varies by provider |
| Hall of Fame | ✅ WORKING | USE | Persists best strategies across runs |
| NSGA-II Multi-Objective | 🔬 UNTESTED | SKIP FOR NOW | Implemented; needs validation run |
| Feature Importance Tracking | ✅ WORKING | USE | Tracks which indicators contribute to fitness |
| Adaptive Mutation | ✅ WORKING | USE | Auto-increases mutation when stuck |
| Trade Visualization | ✅ WORKING | OPTIONAL | Generates charts per generation or on improvement |

Status key: ✅ WORKING = tested & reliable, ⚠️ PARTIAL = works but has known issues, 🔬 UNTESTED = implemented but not validated, ❌ BROKEN = do not use

---

## Detailed Feature Analysis

### 1. Core GA Engine

**Status: ✅ WORKING**

The core genetic algorithm loop (population init → evaluate → select → crossover → mutate → next gen) is stable and well-tested across multiple runs.

**Recommended Config:**
```yaml
genetic_algorithm:
  population_size: 40         # 30 min for quick tests, 40-60 for validation, 100+ for production
  generations: 12             # 12 for validation, 30-50 for production
  mutation_rate: 0.20         # Good default
  max_mutation_rate: 0.35     # Cap for adaptive mutation
  crossover_rate: 0.75
  crossover_method: 'uniform' # Best diversity preservation
  elite_size: 4               # ~10% of population
  tournament_size: 3
  selection_method: 'tournament'
  convergence_patience: 8
  adaptive_mutation: true
  allow_self_crossover: false # Better diversity
  random_immigrants: 4-5
```

**Key Parameters:**
- `population_size`: Directly affects search space coverage. Below 30 produces poor diversity.
- `elite_size`: Set to ~10% of population. Too high = stagnation, too low = best strategies lost.
- `crossover_method`: `uniform` > `single_point` > `component` for diversity.
- `random_immigrants`: Injects fresh genetic material. Doubled automatically when diversity drops below threshold.

**What works:** Tournament selection, uniform crossover, adaptive mutation, random immigrants.
**What doesn't:** Roulette selection (fitness-proportional is too noisy with our fitness landscape).

---

### 2. Walk-Forward Validation

**Status: ✅ WORKING**

Splits the time range into rolling train/validation windows. Fitness = aggregated validation score (NOT training score). Critical for preventing overfitting.

**Recommended Config:**
```yaml
walk_forward:
  enabled: true
  train_days: 120
  validation_days: 30
  step_days: 15
  mode: 'rolling'            # 'anchored' also works but slower
  aggregation: 'mean'        # 'harmonic_mean' too punitive; avoid
  min_train_trades: 5
```

**Key Insight:** Walk-forward auto-adjusts when data is shorter than configured windows. Check logs for `[WALK-FORWARD] Auto-adjusted parameters`.

**What works:** Rolling mode with mean aggregation produces reliable fitness scores.
**What doesn't:** harmonic_mean aggregation punishes any single bad window disproportionately.

---

### 3. Fitness Sharing

**Status: ⚠️ PARTIAL — use with care**

Reduces fitness of strategies in crowded niches to preserve diversity. Formula: `shared_fitness = raw_fitness / niche_count`.

**Recommended Config:**
```yaml
genetic_algorithm:
  fitness_sharing: true
  sharing_radius: 0.12       # Was 0.25, caused over-penalization
  diversity_threshold: 0.15
```

**Known Issues:**
- **Over-penalization at high radius**: With `sharing_radius: 0.25`, ~80% of the population gets penalized and ~43% of raw fitness is eaten. This flattens the fitness landscape and removes selection pressure.
- **Stacking with holdout penalty**: When holdout monitoring also penalizes `raw_fitness`, fitness sharing amplifies the penalty in subsequent generations (now fixed with `_pre_holdout_raw_fitness` restoration on elite carry-over).

**What works:** `sharing_radius: 0.10-0.15` gives meaningful diversity pressure without crushing fitness.
**What doesn't:** `sharing_radius: 0.25+` in combination with any other penalty layer.

**Anti-pattern ⛔:** `fitness_sharing + holdout_penalty + harmonic_mean` = triple penalty stacking → fitness collapses to near-zero regardless of strategy quality.

---

### 4. Holdout Monitoring

**Status: ⚠️ PARTIAL — use as diagnostic only**

Evaluates top-N elites on held-out data each generation. Can optionally penalize fitness based on train→holdout degradation.

**Recommended Config:**
```yaml
holdout_monitoring:
  enabled: true
  top_n: 5
  fitness_penalty: false      # ← KEEP OFF with fitness sharing enabled
  penalty_factor: 0.5
  early_stop: true
  early_stop_threshold: 0.5
  early_stop_checks: 4
```

**Known Issues:**
- **Penalty compounds with fitness sharing**: Holdout penalty modifies `raw_fitness` in-place. When elites carry over, fitness sharing uses the already-penalized `raw_fitness` as base → double penalty each generation. **Code fix applied** (2026-03-02): Elite carry-over now restores `_pre_holdout_raw_fitness`.
- **Use for early stopping instead**: The `early_stop` mechanism (stop when degradation exceeds threshold for N consecutive checks) works well without modifying fitness.

**What works:** Holdout monitoring as READ-ONLY diagnostic + early stopping trigger.
**What doesn't:** `fitness_penalty: true` when `fitness_sharing: true` (compounding issue).

---

### 5. Regime-Aware Evaluation

**Status: ⚠️ PARTIAL — needs config tuning for long-only**

Divides backtesting period into market regime segments (bullish, bearish, sideways) and evaluates strategies on each. Aggregates per-regime fitness into a single score.

**Recommended Config (long-only strategies):**
```yaml
regime_aware:
  enabled: true
  method: 'ensemble'
  aggregation: 'mean'         # NOT harmonic_mean — see below
  segments_per_regime: 2      # More trades per segment
  min_period_days: 90         # Ensure sufficient data per segment
  min_segment_trades: 5       # Skip segments with too few trades
  holdout_ratio: 0.20
  regime_weights:
    bullish: 1.2              # Reward strong bull performance  
    bearish: 0.6              # Long-only can't profit here — lower weight
    sideways: 1.0
```

**Known Issues:**
- **harmonic_mean + long-only = broken**: Without short selling, bearish regime fitness is structurally near-zero. Harmonic mean of `[0.45, 0.02, 0.30]` → `0.054` instead of `mean = 0.257`. This collapses the entire fitness landscape.
- **Low-trade segments add noise**: With many small segments, some have <5 trades producing noisy fitness. **Code fix applied** (2026-03-02): Added `min_segment_trades` filter in `_aggregate_results()`.
- **Bearish over-weighting**: Equal regime weights penalize long-only strategies for being unable to profit in bear markets.

**What works:** `mean` aggregation + asymmetric regime weights + min_segment_trades filter.
**What doesn't:** `harmonic_mean` aggregation with any long-only configuration.

**When to use harmonic_mean:** Only when short selling is enabled AND you truly require proficiency across ALL regimes.

---

### 6. DSR (Deflated Sharpe Ratio)

**Status: ⚠️ PARTIAL — needs low min_trades**

Implements Bailey & López de Prado's DSR to penalize strategies whose Sharpe Ratio is likely a statistical fluke given the number of strategies tested.

**Recommended Config:**
```yaml
deflated_sharpe:
  enabled: true
  penalty_weight: 0.15
  min_trades: 8               # Was 20; too high for regime-segmented evals
  significance: 0.05
```

**Known Issues:**
- **min_trades too high**: Default `min_trades: 20` means DSR penalty = 1.0 (inactive) for most strategies when regime segmentation reduces trade counts per segment. `dsr_penalty = 1.0000` was constant across all 12 generations in `valRunRegimeTest`.
- **Depends on scipy**: Requires `scipy.stats.norm` for p-value calculations.

**What works:** `min_trades: 8` makes DSR engage meaningfully even under regime segmentation.
**What doesn't:** `min_trades: 20+` when using regime-aware evaluation (most segments have <20 trades).

---

### 7. CPCV (Combinatorially Purged Cross-Validation)

**Status: 🔬 UNTESTED**

Implements PBO (Probability of Backtest Overfitting) via CPCV paths. Expensive — `C(6,2) = 15` paths, each requiring a full backtest.

**Config:**
```yaml
cpcv:
  enabled: false              # Opt-in only
  n_groups: 6
  n_test_groups: 2
  purge_pct: 0.01
  embargo_pct: 0.01
  max_paths: 100
  pbo_threshold: 0.5
  penalty_weight: 0.20
```

**Recommendation:** Keep disabled during evolution. Use as a post-hoc validation step on the final champion strategy. Can be run via `--validate-only` mode.

---

### 8. Monte Carlo Robustness

**Status: 🔬 UNTESTED**

Implements bootstrap resampling, trade-order shuffling, and slippage jitter to estimate robustness scores.

**Config:**
```yaml
monte_carlo:
  enabled: false
  num_permutations: 100
  confidence_level: 0.95
  penalty_weight: 0.10
```

**Recommendation:** Not currently integrated into the main fitness loop. Functions exist in `evaluation/monte_carlo.py` but are only called in post-hoc validation. Low priority for now.

---

### 9. Parsimony Pressure

**Status: ✅ WORKING**

After each generation, attempts to remove indicators/conditions from elites. Keeps the simpler version if fitness drop < epsilon.

**Recommended Config:**
```yaml
parsimony:
  enabled: true
  epsilon: 0.02               # 2% fitness drop tolerance
  max_removals: 1             # Try removing 1 component per gen
```

**What works:** Gradually simplifies strategies, reducing from 5-6 indicators to 3-4. Reduces overfitting.
**What doesn't:** `max_removals > 2` can strip strategies too aggressively in early generations.

---

### 10. Dynamic Parameter Bounds

**Status: 🔬 UNTESTED**

Evolves indicator parameter ranges alongside the parameters themselves. E.g., RSI period range `[7, 21]` might tighten to `[10, 16]` if values around 14 keep winning.

**Config:**
```yaml
dynamic_bounds:
  enabled: false
  mutation_strength: 0.1
```

**Recommendation:** Low priority. The standard fixed parameter ranges work fine for most indicators. Enable only for long production runs (50+ generations) where parameter convergence matters.

---

### 11. Parallel Evaluation

**Status: ✅ WORKING**

Uses `ProcessPoolExecutor` to evaluate multiple strategies simultaneously.

**Recommended Config:**
```yaml
parallel_evaluation:
  enabled: true
  num_workers: null            # null = auto-detect (CPU count - 1)
  backtest_timeout: 120
```

**Key Details:**
- Walk-forward is disabled in worker processes (applied post-hoc on elites only)
- Includes orphan worker cleanup via `atexit` handlers
- Benchmark: ~3.5x speedup with 4 workers, ~4.7x with 8 workers
- `num_workers: null` → auto-detects; on 6-core machine → 5 workers

**What works:** Stable with automatic worker cleanup. Significant speedup.
**What doesn't:** May have issues if FreqTrade holds file locks inside workers (rare).

---

### 12. LLM Strategy Seeding

**Status: ⚠️ PARTIAL**

Uses an LLM (GPT-4, Claude, etc.) to generate initial population seeds and periodic immigrants with diverse trading styles.

**Config:**
```yaml
advanced:
  llm:
    enabled: true
    provider: 'openai'
    model: 'gpt-4'
    seed_ratio: 0.4            # 40% of initial pop from LLM
    immigrant_ratio: 0.5       # 50% of random_immigrants from LLM
    min_call_interval: 1.0
```

**What works:** Generates structurally valid strategies with diverse styles (momentum, mean-reversion, breakout, etc.). Good for initial population diversity.
**What doesn't:** LLM strategies don't consistently outperform random initialization. API costs add up. Quality varies significantly by provider/model.

---

### 13. Hall of Fame

**Status: ✅ WORKING**

Persistent JSON archive of top strategies. Updated each generation. Can re-inject champions into future runs.

**Config:**
```yaml
hall_of_fame:
  enabled: true
  max_size: 50
  min_fitness: 0.0
```

**Location:** `genetic_algorithm/data/hall_of_fame/`

---

### 14. Feature Importance Tracking

**Status: ✅ WORKING**

Tracks which indicators and conditions appear in high-fitness strategies. Logged every 5 generations and at the end.

Automatically feeds back into mutation via `_indicator_weights` — indicators that appear in top strategies are more likely to be added during mutation.

---

### 15. Adaptive Mutation

**Status: ✅ WORKING**

When no improvement is detected for consecutive generations, the mutation rate increases by `adaptation_step` (up to `max_mutation_rate`). Resets on improvement.

**Config:**
```yaml
genetic_algorithm:
  adaptive_mutation: true
  max_adaptation_factor: 2.5
  adaptation_step: 0.15
  max_mutation_rate: 0.35
```

**What works:** Prevents premature convergence. Automatically explores more when stuck.
**What doesn't:** With `max_mutation_rate > 0.40`, search becomes too random.

---

## Anti-Patterns (What NOT to Do)

### ⛔ 1. Triple Penalty Stacking
```yaml
# DO NOT combine all three of these with strong settings:
regime_aware:
  aggregation: 'harmonic_mean'    # Punishes worst regime
genetic_algorithm:
  fitness_sharing: true
  sharing_radius: 0.25            # High sharing radius
holdout_monitoring:
  fitness_penalty: true           # Also penalizes fitness
```
**Why it breaks:** Each penalty layer compounds. A strategy with raw_fitness=0.50 can get reduced to:
- After harmonic_mean: 0.15 (bearish drags it down)
- After fitness sharing: 0.08 (niche count ~2)
- After holdout penalty: 0.05 (20% degradation)
- Net result: 10% of true quality → no selection pressure

### ⛔ 2. High min_trades with Regime Segmentation
```yaml
deflated_sharpe:
  min_trades: 20    # Too high
regime_aware:
  enabled: true     # Splits data into segments
```
**Why it breaks:** Regime segmentation divides trades across segments. Most segments end up with <20 trades → DSR is permanently inactive (penalty = 1.0).

### ⛔ 3. harmonic_mean with Long-Only Strategies
```yaml
regime_aware:
  aggregation: 'harmonic_mean'
# + no short selling configured
```
**Why it breaks:** Long-only strategies can't profit in bear markets. Harmonic mean of `[good, ~zero, okay]` → `~zero`. Use `mean` with lowered bearish weight instead.

### ⛔ 4. Holdout Fitness Penalty + Fitness Sharing
```yaml
holdout_monitoring:
  fitness_penalty: true
genetic_algorithm:
  fitness_sharing: true
```
**Why it breaks:** Holdout penalty modifies `raw_fitness`. Fitness sharing uses `raw_fitness` as base. Elites that survive to the next gen have penalties compound. **Code fix is in place** but it's still safer to keep `fitness_penalty: false`.

---

## Recommended Config Templates

### Template A: Quick Validation Run (~30 min)
```yaml
genetic_algorithm:
  population_size: 30
  generations: 8
  mutation_rate: 0.20
  crossover_method: 'uniform'
  elite_size: 3
  fitness_sharing: true
  sharing_radius: 0.12
  random_immigrants: 4

walk_forward:
  enabled: true
  train_days: 90
  validation_days: 21

regime_aware:
  enabled: false              # Skip for quick validation

holdout_monitoring:
  enabled: true
  fitness_penalty: false

parallel_evaluation:
  enabled: true

parsimony:
  enabled: true
```

### Template B: Regime-Aware Validation (~60 min)
```yaml
genetic_algorithm:
  population_size: 40
  generations: 12
  mutation_rate: 0.20
  crossover_method: 'uniform'
  elite_size: 4
  fitness_sharing: true
  sharing_radius: 0.12
  random_immigrants: 4

walk_forward:
  enabled: true
  train_days: 120
  validation_days: 30

regime_aware:
  enabled: true
  method: 'ensemble'
  aggregation: 'mean'
  segments_per_regime: 2
  min_period_days: 90
  min_segment_trades: 5
  regime_weights:
    bullish: 1.2
    bearish: 0.6
    sideways: 1.0

holdout_monitoring:
  enabled: true
  fitness_penalty: false

deflated_sharpe:
  enabled: true
  min_trades: 8

parallel_evaluation:
  enabled: true

parsimony:
  enabled: true
```

### Template C: Production Run (~4-8 hours)
```yaml
genetic_algorithm:
  population_size: 100
  generations: 40
  mutation_rate: 0.18
  crossover_method: 'uniform'
  elite_size: 10
  fitness_sharing: true
  sharing_radius: 0.15
  random_immigrants: 8
  convergence_patience: 12

walk_forward:
  enabled: true
  train_days: 150
  validation_days: 45
  step_days: 20

regime_aware:
  enabled: true
  method: 'ensemble'
  aggregation: 'mean'
  segments_per_regime: 3
  min_period_days: 90
  min_segment_trades: 8
  regime_weights:
    bullish: 1.2
    bearish: 0.6
    sideways: 1.0

holdout_monitoring:
  enabled: true
  fitness_penalty: false
  early_stop: true
  early_stop_threshold: 0.5
  early_stop_checks: 5

deflated_sharpe:
  enabled: true
  min_trades: 8

parallel_evaluation:
  enabled: true

parsimony:
  enabled: true
  epsilon: 0.02
  max_removals: 1
```

---

## Run History & Observations

### valRunRegimeTest (2026-03-02)
- **Config:** v1 regime validation (harmonic_mean, equal weights, sharing_radius=0.25)
- **Generations:** 12 (completed)
- **Result:** ❌ FAILED — fitness stagnant, bearish dominance, DSR inactive
- **Key findings:**
  - Best fitness: 0.3663 (gen 7), dropped to 0.2726 by gen 11
  - Population avg profit: -1.48% (net negative)
  - DSR penalty: 1.0000 across all 12 gens (inactive)
  - Fitness sharing ate ~43% of raw fitness
  - Bullish regime avg: 0.41, bearish: 0.21, sideways: 0.21
  - Holdout degradation: 23-48% range
- **Diagnosis:** Triple penalty stacking (harmonic_mean + fitness_sharing@0.25 + holdout_penalty)
- **Fixes applied:** See v2 config (ga_config_val_regime.yaml)

### AntiOverfitingValidationRun (incomplete)
- **Config:** Anti-overfitting validation (regime disabled, WF + holdout + CPCV enabled)
- **Generations:** 2 of 12 (interrupted)
- **Status:** Incomplete — did not run long enough for analysis

---

## Indicator Reference

### Tier 1 — Well-Tested, Reliable
| Indicator | Parameters | Best For |
|-----------|-----------|----------|
| RSI | period: 7-21 | Overbought/oversold detection |
| MACD | fast: 8-21, slow: 21-50, signal: 5-14 | Trend direction + momentum |
| BBANDS | period: 15-30, std_dev: 1.5-3.0 | Volatility bands |
| EMA/SMA | period: 10-50 | Trend following |
| STOCH | k: 5-21, d: 3-14 | Momentum oscillator |
| ATR | period: 10-20 | Volatility measurement |
| ADX | period: 10-20 | Trend strength |

### Tier 2 — Available, Less Tested
| Indicator | Parameters | Best For |
|-----------|-----------|----------|
| MFI | period: 10-20 | Volume-weighted RSI |
| OBV | (none) | Volume trend |
| WILLR | period: 10-20 | Williams %R oscillator |
| ROC | period: 5-20 | Rate of change |
| TEMA | period: 10-30 | Smoothed trend |
| KAMA | period: 10-30 | Adaptive moving average |
| SAR/PSAR | accel: 0.01-0.05, max: 0.1-0.3 | Trend reversals |
| CCI | period: 10-20 | Cyclical oscillator |
| AROON | period: 10-25 | Trend identification |
| SUPERTREND | period: 7-14, mult: 2.0-4.0 | Trend following with dynamic stop |
| ICHIMOKU | tenkan: 7-12, kijun: 20-30, senkou: 40-60 | Comprehensive trend system |
| DONCHIAN | period: 10-30 | Breakout detection |
| CMF | period: 10-25 | Volume-based momentum |

### Tier 3 — Candlestick Patterns (No Parameters)
CDL_ENGULFING, CDL_HAMMER, CDL_DOJI, CDL_MORNINGSTAR, CDL_EVENINGSTAR, CDL_SHOOTINGSTAR, CDL_HARAMI, CDL_PIERCING, CDL_DARKCLOUD, CDL_3WHITESOLDIERS, CDL_3BLACKCROWS

**Note:** Candlestick patterns work best as entry signals combined with Tier 1 trend indicators.

---

## Code Fixes Applied (2026-03-02)

### Fix 1: min_segment_trades guard
- **File:** `evaluation/regime_aware.py`
- **What:** Segments with fewer trades than `min_segment_trades` are skipped during aggregation
- **Why:** Prevents noisy 2-3 trade segment fitness from dominating the aggregated score

### Fix 2: Holdout penalty compounding
- **File:** `core/evolution.py`
- **What:** Stores `_pre_holdout_raw_fitness` before penalty; restores it on elite carry-over
- **Why:** Prevents holdout penalties from accumulating across generations via fitness sharing

### Fix 3: Skipped segment tracking
- **File:** `evaluation/regime_aware.py`
- **What:** `skipped_low_trade_segments` count added to returned metrics
- **Why:** Visibility into how many segments were excluded from aggregation
