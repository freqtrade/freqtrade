# Exploration Log — GA Configuration Laboratory

## Overview

Systematic exploration of GA evolution parameters to find robust configurations before committing to production evolution. Runs short "wave" experiments (6 parallel, ~2-3h each), analyzes results, then designs the next wave based on findings.

**Hardware:** i5-9500T, 6 cores, 16 GB RAM, NVMe SSD  
**Data:** BTC/USDT + SOL/USDT, 15m candles, 2023-03-01 → 2026-02-28  
**Base sizing:** pop=15, gen=12, 1 worker/run, no dashboard  

---

## Wave 1 — Baseline Characterization

**Status:** NOT STARTED  
**Focus:** Establish baselines across 4 focus areas  
**Launch:** `./run_parallel_wave.sh wave1`  
**Monitor:** `./wave_monitor.sh wave1`  
**Analyze:** `python genetic_algorithm/scripts/wave_comparison.py wave1`  

| # | Experiment | Focus | Key Variation | Status | Best Fit | SAFE | Notes |
|---|-----------|-------|--------------|--------|----------|------|-------|
| E1 | baseline_control | Control | Standard single-obj, WF+MC ON | — | — | — | Reference for all comparisons |
| E2 | nsga2_multi_obj | NSGA-II vs single | mode=nsga2, fitness_sharing=OFF | — | — | — | Tests Pareto vs weighted fitness |
| E3 | island_wf_stress | Island bug #4 | Island ON + WF ON (known conflict) | — | — | — | Diagnostic: expect failure |
| E4 | island_no_validation | Island proper | Island ON, WF OFF, MC OFF | — | — | — | Clean island vs E1 comparison |
| E5 | aggressive_mutation | Mutation tuning | mut=0.40, max=0.85, component XO | — | — | — | High exploration vs E1 conservative |
| E6 | llm_guided | LLM integration | Groq LLaMA, seed=40%, mut guided | — | — | — | LLM value-add test |

### Wave 1 Findings
<!-- Fill after wave completes -->

### Wave 1 Decisions
<!-- What we learned, what to test next -->

---

## Wave 2 — Hypothesis Refinement

**Status:** NOT STARTED (depends on Wave 1 findings)  
**Planned experiments:** TBD based on Wave 1 analysis  

| # | Experiment | Hypothesis | What Changed | Status | Best Fit | SAFE | Notes |
|---|-----------|-----------|-------------|--------|----------|------|-------|
| E7 | TBD | Optimal mutation rate between E1/E5 | mut=0.30, max=0.60 | — | — | — | |
| E8 | TBD | NSGA-II + proper WF validation | E2 + tune WF params | — | — | — | |
| E9 | TBD | Fix island overfitting | E4 + tighter holdout | — | — | — | |
| E10 | TBD | LLM seed + conservative evolve | E6 seed + E1 mutation | — | — | — | |
| E11 | TBD | Defensive fitness weights | drawdown↑, profit↓ | — | — | — | |
| E12 | TBD | Cross-wave HoF injection | Wave 1 best → fresh run | — | — | — | |

---

## Wave 3+ — Convergence & Scale Testing

Planned after Wave 2 analysis.

---

## Parameter Sensitivity Tracker

| Parameter | Tested Values | Best So Far | 95% CI | Notes |
|-----------|--------------|-------------|--------|-------|
| mutation_rate | 0.20 (E1), 0.40 (E5) | TBD | — | |
| crossover_method | uniform (E1), component (E5) | TBD | — | |
| mode | single_obj (E1), nsga2 (E2) | TBD | — | |
| population structure | single (E1), island (E4) | TBD | — | |
| LLM seeding | off (E1), 40% (E6) | TBD | — | |

---

## Bug Tracker

| # | Source | Issue | Severity | Status | Fix |
|---|--------|-------|----------|--------|-----|
| B1 | E3 | Island + WF interaction | — | PENDING | — |
| B2 | E6 | LLM API reliability | — | PENDING | — |

---

## Cross-Wave Hall of Fame

Best strategies ever found across all waves:

| Rank | Wave | Experiment | Fitness | Profit% | SAFE | Strategy File |
|------|------|-----------|---------|---------|------|---------------|
| — | — | — | — | — | — | — |

---

## Quick Reference

```bash
# Launch wave
./run_parallel_wave.sh wave1

# Monitor live
./wave_monitor.sh wave1

# One-shot status
./wave_monitor.sh wave1 --once

# Compare results
python genetic_algorithm/scripts/wave_comparison.py wave1

# Diff two configs
python genetic_algorithm/scripts/config_diff.py \
  genetic_algorithm/config/exploration/wave1/E1_baseline_control.yaml \
  genetic_algorithm/config/exploration/wave1/E5_aggressive_mutation.yaml --compact

# Merge Hall of Fame across experiments
python genetic_algorithm/scripts/merge_hall_of_fame.py wave1

# Launch only specific experiments
./run_parallel_wave.sh wave1 E1 E2 E5

# Validate configs without launching
./run_parallel_wave.sh wave1 --dry-run
```
