# Genetic Algorithm Trading System — Roadmap

**Created:** 2026-03-01  
**Updated:** 2026-03-04  
**Status:** Web dashboard implemented (backend + frontend). Core GA engine stable. Branch 3 quick wins done. **Branch 4 (LLM improvements) implemented.**  
**Last validated run:** 229/229 GA tests passing.

---

## Current State Summary

The GA engine is **production-stable for iteration**. It can:
- Generate, evaluate, and evolve trading strategies across multiple pairs
- Walk-forward validation with train/val windows
- Holdout monitoring with early-stop on overfitting
- Monte Carlo robustness analysis
- Hall of Fame persistence with injection
- Parsimony pressure (parallel component removal trials)
- Adaptive mutation, fitness sharing, convergence detection
- Feature importance tracking
- **LLM-guided mutation** — targeted patches to existing strategies *(NEW)*
- **Batched LLM calls** — request multiple strategies per API call *(NEW)*
- **Provider router with fallback** — automatic failover across providers *(NEW)*
- **Stagnation escalation** — increase LLM involvement when GA plateaus *(NEW)*

**Known limitations:**
- Strategies converge quickly to low-fitness plateaus (best ~0.21 fitness, ~0.3% profit)
- Monte Carlo robustness is fragile for most evolved strategies
- Regime detection is implemented but not yet producing meaningful differentiation
- Some generated strategy code can be syntactically invalid (handled gracefully — zero fitness)

---

## Branch 1: Web Interface & Monitoring Dashboard

**Goal:** Real-time visual monitoring, parameter exploration, and live evolution steering.

### Phase 1 — Read-Only Dashboard ✅
- [x] **Web server skeleton** — FastAPI backend + React/TypeScript/Vite frontend
- [x] **WebSocket real-time feed** — EventBus → SubprocessEventRelay → async Queue → WS endpoint
- [x] **Fitness evolution chart** — Recharts area/line chart of best/avg/worst fitness per generation
- [x] **Population overview** — Scatter plot of individuals (fitness vs complexity)
- [x] **Strategy inspector** — Gene tree visualization (indicators, conditions, parameters)
- [x] **Hall of Fame viewer** — Browse top strategies with metrics table
- [x] **Log viewer** — Event stream with severity filtering (WS-based)
- [x] **Run history** — List past runs with summary stats, run detail pages

### Phase 2 — Parameter Exploration (partial)
- [x] **Config editor** — Config list/load via REST API + frontend page
- [ ] **Parameter impact visualization** — Show how fitness weights, mutation rates, etc. affect outcomes
- [ ] **Indicator analysis** — Which indicators appear most in top strategies? Feature importance heatmap
- [ ] **Walk-forward window inspector** — Per-window train/val breakdown, identify problematic windows
- [ ] **Overfitting dashboard** — Holdout degradation, Monte Carlo robustness, composite score gauges

### Phase 3 — Live Steering & State Management (partial)
- [ ] **Checkpoint save/load** — Save full evolutionary state (population, stats, RNG state) on demand
- [ ] **Graceful interrupt** — Signal handler to snapshot state before shutdown
- [ ] **Resume from checkpoint** — Start evolution from any saved state
- [ ] **Live parameter adjustment** — Change mutation rate, crossover rate, population size mid-run
- [x] **Manual individual injection** — API endpoint + mp.Queue → evolution loop drain
- [x] **Pause/resume evolution** — mp.Event flags + RunManager API
- [ ] **Kill/restart individual evaluations** — Cancel stuck backtests

### Foundation (completed):
- [x] Expose evolution engine state via a clean API (EventBus + REST + WS)
- [x] Decouple evolution loop from console output — WebSocketMonitor + EventBus structured events
- [ ] Ensure checkpoint serialization captures ALL state (best_fitness_ever, no_improvement_count, feature_tracker, etc.)

---

## Branch 2: Regime Detection & Overfitting Prevention

**Goal:** Dramatically improve strategy robustness through smarter regime awareness and anti-overfitting mechanisms.

### Phase 1 — Improved Regime Detection
- [ ] **ADX/DI hysteresis method** — Already configured, needs validation on real data
- [ ] **Hidden Markov Model (HMM) regime detector** — Probabilistic regime classification
- [ ] **Volatility-regime clustering** — K-means on rolling volatility + trend features
- [ ] **Multi-asset regime consensus** — Combine regime signals across pairs for more robust detection
- [ ] **Regime-aware fitness** — Weight fitness by regime performance (penalize strategies that only work in bull markets)
- [ ] **Regime transition detection** — Identify regime changes and measure adaptation speed
- [ ] **Regime visualization** — Color-coded price chart overlays showing detected regimes

### Phase 2 — Anti-Overfitting Mechanisms
- [ ] **Anchored walk-forward** — Expanding training window instead of rolling
- [ ] **Combinatorial purged cross-validation (CPCV)** — Gold standard for financial time series
- [ ] **Probability of backtest overfitting (PBO)** — De Prado's method for measuring overfit probability
- [ ] **White's Reality Check / SPA test** — Statistical test for strategy significance
- [ ] **Deflated Sharpe Ratio** — Adjust for multiple testing (number of strategies tried)
- [ ] **Embargo periods** — Larger gaps between train/test to prevent leakage
- [ ] **Noise injection during evaluation** — Add random slippage/fee variation to backtest
- [ ] **Out-of-distribution detection** — Flag when market conditions differ from training data

### Phase 3 — Fitness Function Evolution
- [ ] **Multi-objective Pareto optimization** — Optimize profit AND robustness simultaneously (NSGA-II)
- [ ] **Robustness-weighted fitness** — Monte Carlo median replaces single backtest result
- [ ] **Time-decay fitness** — More recent performance weighted higher
- [ ] **Drawdown-duration penalty** — Penalize long underwater periods, not just depth
- [ ] **Tail-risk metrics** — CVaR, max consecutive losses, recovery time

### Foundation needed now:
- [ ] Clean up `regime_aware` config — remove `sma_adx` default, validate `adx_di_hysteresis` works
- [ ] Ensure regime detection method is pluggable (strategy pattern)
- [ ] Add regime metadata to backtest results for downstream analysis

---

## Branch 3: Code Quality & Tech Debt

**Priority items to address across branches.**

### Pre-existing Issues
- [x] **Fix `test_code_generation_informative_condition_suffix`** — Root cause: `assign_instance_ids()` creates `EMA_1h_0` instance IDs, `rsplit('_',1)` misparses as type `EMA_1h`. Fixed by using `target_indicator.type` after match.
- [x] **Strategy code validation** — Pre-flight `compile()` check added to `generate_strategy_code()`. Falls back to zero-trade strategy on syntax error.
- [x] **CDL indicator availability mismatch** — Added `CDL_MORNINGSTAR`, `CDL_EVENINGSTAR`, `CDL_SHOOTINGSTAR`, `CDL_HARAMI` to `available` list in `ga_config.yaml`
- [x] **Stale checkpoint data** — Removed `checkpoints_feature_test/` directory
- [x] **Config: set `max_mutation_rate` explicitly** — Set to `0.35` in `ga_config.yaml`
- [x] **Clean up `tmpum08mhsb.yaml`** — Already removed (not present)

### Code Improvements
- [x] **Ensure `ga_generated` directory auto-creation** — Already handled by `DirectBacktester.__init__` (`mkdir(parents=True, exist_ok=True)`)
- [x] **Rate-limit zero-trade warnings** — Only first 3 consecutive warnings logged per evaluation
- [ ] **Add type hints** — Core modules (evolution.py, crossover.py, mutation.py) lack some type annotations
- [x] **Structured logging** — EventBus provides structured events; WebSocketMonitor bridges to WS clients
- [ ] **Test coverage** — Add tests for: `_enforce_max_indicators`, holdout convergence fix, parsimony ID restore

---

## Bug Fix History (Completed)

### Session: 2026-03-01

| # | Severity | Fix | File |
|---|----------|-----|------|
| BUG-1 | Critical | Holdout penalty corrupts `best_individual` tracking → premature convergence. Fixed by snapshotting `best_fitness_ever` before holdout runs. | `evolution.py` |
| BUG-2 | High | Indicator count exceeds `max_per_strategy` after crossover. Added `_enforce_max_indicators()` in all 3 crossover methods. | `crossover.py` |
| BUG-3 | Medium | Zero-trade WF windows report 100% drawdown (sentinel `1.0`). Changed to `0.0`. | `fitness.py` |
| MINOR-1 | Low | Extreme Sharpe/Sortino values (43M+) displayed unclamped. Clamped to [-10, 50]. | `fitness.py` |
| MINOR-2 | Low | Parsimony trial IDs (9000+) not restored after acceptance. Now restores original ID. | `parallel.py` |
| BONUS | Medium | `StrategyGene.copy()` lost all `instance_id` values — broke parsimony cascade removal. Fixed by calling `assign_instance_ids()` after `from_dict()`. | `strategy_gene.py` |
| TEST | — | Updated stale `composite_warning` assertion from 0.35 → 0.25. | `test_overfit_analysis.py` |

### Previous Session

| # | Fix | File |
|---|-----|------|
| Atomic writes | Race condition in parallel strategy file writes. Added tempfile + `os.replace`. | `direct_backtester.py` |
| Parsimony IDs | Filename collisions from duplicate parsimony trial names. Added 9000+ namespace. | `parallel.py` |
| Regime default | Broken regime detection — `sma_adx` method referenced non-existent function. Changed default to `adx_di_hysteresis`. | `dataset_policy.py`, `ga_config.yaml` |
| CDL corruption | Cascading `_0` suffixes on candlestick indicator types. Added `_strip_cdl_suffixes()`. | `strategy_gene.py` |

---

## Branch 4: LLM Integration Improvements

**Status:** ✅ Implemented (2026-03-04)

### Phase 1: LLM-as-Mutation Operator ✅

**Implementation:**
- [x] **`llm/diagnostics.py`** (NEW) — Diagnose strategy failure modes from metrics
  - `diagnose_failure_mode()` — Returns targeted improvement hint based on metrics (zero trades, excessive drawdown, low win rate, etc.)
  - `diagnose_all_failure_modes()` — Returns all applicable failure diagnoses
  - `select_mutation_objective()` — Picks single most relevant mutation objective
- [x] **`llm/prompts.py`** — Extended with mutation prompts
  - `build_mutation_prompt()` — Creates LLM prompt with parent strategy JSON, metrics, and objective-conditioned improvement hint
  - `build_batch_mutation_prompt()` — Batch version requesting N mutations per call
- [x] **`llm/designer.py`** — Extended `StrategyDesigner` with mutation methods
  - `mutate_strategy()` — Single strategy mutation via LLM
  - `mutate_strategies_batch()` — Batch mutation of multiple strategies
  - New config: `mutation_enabled`, `mutation_probability`, `mutation_top_k`, `mutation_stagnation_threshold`
- [x] **`core/evolution.py`** — Wired LLM mutation into offspring loop
  - Step 4 after normal GA mutation: LLM-guided mutation on top-K offspring
  - Only applies when `mutation_enabled=True` and random < `mutation_probability`

### Phase 2: Batching ✅

**Implementation:**
- [x] **`llm/designer.py`** — Added batch methods
  - `generate_seed_strategies_batch()` — Request multiple seeds in single LLM call
  - `generate_immigrants_batch()` — Request multiple immigrants per call
  - Config: `batch_enabled`, `max_batch_size`
- [x] **`core/evolution.py`** — Uses batch mode when enabled
  - Seed population uses batch generation
  - Immigrant injection uses batch generation

### Phase 3: Provider Router with Fallback ✅

**Implementation:**
- [x] **`llm/router.py`** (NEW) — Multi-provider failover
  - `LLMProviderRouter(LLMProvider)` — Wraps multiple providers, auto-failover on 429/errors
  - 60-second cooldown for failed providers
  - `create_provider_or_router()` — Factory function for backward compatibility
- [x] **`llm/designer.py`** — Uses `create_provider_or_router()` for provider creation
- [x] **`llm/__init__.py`** — Exports `LLMProviderRouter`, `create_provider_or_router`

### Phase 4: Objective-Conditioned Prompts ✅

**Implementation:**
- [x] `build_mutation_prompt()` includes `objective` parameter
- [x] `diagnose_failure_mode()` returns specific improvement hints:
  - `ZERO TRADES` → "loosen entry conditions, widen indicator thresholds"
  - `EXCESSIVE DRAWDOWN` → "add trend filter, tighten stops, reduce position size"
  - `LOW WIN RATE` → "improve entry timing, add confirmation indicator"
  - `HIGH TRADE FREQUENCY` → "add cooldown logic, stricter conditions"
  - etc.

### Phase 5: Stagnation Escalation ✅

**Implementation:**
- [x] **`core/evolution.py`** — `check_convergence()` enhanced
  - Detects stagnation via `no_improvement_count >= escalation_threshold`
  - Increases `immigrant_ratio` and `mutation_probability` when GA plateaus
  - Config: `escalation_threshold`, `escalation_immigrant_ratio`, `escalation_mutation_probability`

### Phase 6: LLM Usage Metrics ✅

**Implementation:**
- [x] **`llm/designer.py`** — Enhanced `stats` dictionary
  - `calls_by_type`: tracks seed, immigrant, mutation calls separately
  - `improvements_by_type`: tracks successful improvements by type
  - `_budget_available()`, `_record_call()`, `reset_generation_budget()` for budget tracking
  - Config: `max_calls_per_generation`, `max_calls_per_run`
- [x] **`core/evolution.py`** — End-of-run LLM usage report
  - Per-type breakdown (seeds, immigrants, mutations)
  - Token usage, cost estimates
  - Budget tracking

### Config Keys Added

```yaml
advanced:
  llm:
    enabled: true
    # Mutation
    mutation_enabled: true
    mutation_probability: 0.3
    mutation_top_k: 5
    mutation_stagnation_threshold: 3
    # Batching
    batch_enabled: true
    max_batch_size: 5
    # Budget
    max_calls_per_generation: 20
    max_calls_per_run: 200
    # Escalation
    escalation_threshold: 5
    escalation_immigrant_ratio: 0.3
    escalation_mutation_probability: 0.5
```

---

## Priority Order

1. ~~**Branch 3 quick wins**~~ ✅ Done
2. ~~**Branch 1 Phase 1**~~ ✅ Done (full React + FastAPI dashboard)
3. ~~**Branch 4 LLM improvements**~~ ✅ Done (LLM mutation, batching, router, escalation)
4. **Branch 2 Phase 1** — Regime detection improvements (1-2 days)
5. **Branch 1 Phase 2** — Parameter exploration UI polish (1-2 days)
6. **Branch 2 Phase 2** — Anti-overfitting mechanisms (2-3 days)
7. **Branch 1 Phase 3** — Checkpoint save/load, live parameter adjustment (2-3 days)
8. **Branch 2 Phase 3** — Fitness function evolution (2-3 days)
