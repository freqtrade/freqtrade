# GA Fixes & Improvements Tracker

> Discovered issues, improvement opportunities, and recommendations from 79 experiments across 10 waves.
> Updated: 2026-03-17

---

## Critical Design Constraints

### DC-1: Monte Carlo + Island Model Incompatible
- **Severity**: Design constraint (not fixable without major refactor)
- **Evidence**: E14 configured `monte_carlo.enabled: true` with island model → MC results show `N/A` in output
- **Root cause**: Island model splits data by regime segments; MC requires full temporal data for permutation testing. Two incompatible data partitioning strategies.
- **Code location**: `island_model.py` prints "not supported yet" warning (see KNOWN_ISSUES.md #5)
- **Impact**: Any island experiment with `monte_carlo.enabled: true` silently ignores MC validation
- **Recommendation**: Never configure MC with island model. Use MC only with standard GA. Remove MC config from island experiment templates to prevent confusion.

### DC-2: Walk-Forward + Island Model Incompatible
- **Severity**: Design constraint
- **Evidence**: `island_model.py` ~L1264 force-disables walk-forward
- **Root cause**: Same as DC-1 — conflicting data partitioning approaches
- **Recommendation**: Choose one: island (regime-based) OR walk-forward (temporal). Despite this, some island experiments paradoxically produced good results with WF in config — investigate if WF is partially applied.

---

## Performance Anti-Patterns (Confirmed by Experiments)

### AP-1: Elite Size > 1 Destroys Island Diversity
- **Severity**: High
- **Evidence**: E16 (`elite_size: 3`, island pop=6) → 1S/2W/2O, score 0.533. Compare to E7 (`elite_size: 1`, same island pop=6) → 4S/1W/0O, score 0.136
- **Root cause**: With pop=6 and elite=3, 50% of each island is frozen per generation. Effectively halves the search space on each island.
- **Recommendation**: Keep `elite_size: 1` for island model (pop ≤ 8). For standard GA (pop ≥ 15), `elite_size: 2-3` is acceptable (E8 uses 3 with pop=15 successfully).

### AP-2: Feature Stacking Degrades Results
- **Severity**: High
- **Evidence**: ALL Wave 3 experiments combining best features scored WORSE than parents:
  - E9 (island+WF+LLM) = 0.326 vs E7=0.136 and E6=0.188
  - E10 (island+WF+highmut) = 0.315 vs E7=0.136 and E5=0.228
  - E13 (island+WF+180d) = 0.503 vs E7=0.136
- **Root cause**: GA features interact non-linearly. LLM guidance may conflict with island model's regime specialization. High mutation disrupts island convergence.
- **Recommendation**: Change only ONE variable per experiment. Treat E8 and E7 as proven templates — derive experiments with single-variable changes only.

### AP-3: Component Crossover on Island Model = Overfit
- **Severity**: Medium-High
- **Evidence**: E18 (island + component crossover) → ~0S/4W/1O. Top 5 strategies: 33-53% degradation.
- **Root cause**: Component crossover preserves coherent indicator blocks, but on island model these blocks become regime-specialized. When transferred to holdout (full temporal range), they fail.
- **Recommendation**: Use `uniform` crossover for island model. Component crossover works better with standard GA (E5: 4S/1O, score 0.228).

### AP-4: 180d Training Windows Worse Than 120d
- **Severity**: Medium
- **Evidence**: E13 (180d train, 45d val) = score 0.503 vs E7 (120d train, 30d val) = 0.136
- **Root cause**: Longer windows provide less validation diversity (fewer WF folds in the same date range). May also cause strategies to overfit to dominant market regimes.
- **Recommendation**: Stick with 120d train / 30d validation windows as the sweet spot.

### AP-5: LLM + High Mutation = All WARNING
- **Severity**: Medium
- **Evidence**: E12 (LLM + mut=0.40, standard GA) = 0S/5W/0O, score 0.330. All 5 strategies WARNING.
- **Root cause**: LLM guidance (seed_ratio=0.40, immigrant_ratio=0.60) combined with high mutation (0.40) creates excessive exploration. Strategies never converge to robust solutions.
- **Recommendation**: Use LLM with conservative mutation (≤0.20). E6 (LLM + mut=0.20) = 4S/1W/0O, best holdout 0.4039.

### AP-6: Island Pop Scaling > 6 Causes Severe Overfitting
- **Severity**: High
- **Evidence**: E3 (pop=4) score 0.183, E7 (pop=6) score 0.136, E11 (pop=8 + gen=20) score 0.366, **E24 (pop=8, island) = EXTREME OVERFIT (62-100% holdout degradation, 0.8240 fitness)**
- **Root cause**: Each island trains on ~30% of data (3-year split into regimes). Larger population on limited data doesn't help — it just produces more variants of the same overfit patterns.
- **Recommendation**: Keep island pop=6. pop=8 is catastrophic (E24 confirms). For more search power, use standard GA instead.

### AP-8: Population > 15 in Standard GA = Overfitting
- **Severity**: Medium-High
- **Evidence**: E19 (pop=20) = 59-65% holdout degradation (OVERFIT). Compare E8 (pop=15) = 5/5 SAFE.
- **Root cause**: With strict constraints (parsimony, max_indicators=3), larger populations generate more strategies that exploit narrow patterns. pop=15 provides sufficient diversity without overfit risk.
- **Recommendation**: Keep `population_size: 15` for standard GA with strict constraints.

### AP-9: Component Crossover Harmful with Strict Constraints
- **Severity**: Medium-High
- **Evidence**: E20 (E8 + component_cx) = 52-61% holdout degradation (OVERFIT). Compare E8 (uniform cx) = 5/5 SAFE.
- **Root cause**: Component crossover preserves coherent indicator blocks. With strict parsimony (max_indicators=3), these blocks become the entire strategy — no mixing occurs, leading to convergence on overfit patterns.
- **Recommendation**: Always use `crossover_method: 'uniform'` with strict constraints. Component cx only viable without parsimony.

### AP-10: Short Selling Doubles Search Space Without Benefit
- **Severity**: Medium
- **Evidence**: E25 (E8 + short_selling.enabled) = Only 1 HoF strategy from 15×12 evolution. Fitness 0.2388.
- **Root cause**: Enabling shorts doubles the indicator/signal search space. With pop=15 and gen=12, insufficient exploration to find robust long+short strategies.
- **Recommendation**: If testing shorts, increase pop to 25+ or gen to 20+. Or use a dedicated short-only mode.

### AP-11: MC Permutations > 15 Causes Premature Convergence
- **Severity**: Low-Medium
- **Evidence**: E26 (MC=30) early stopped at gen 8, fitness 0.4006, 37.6% degradation trend. E8 (MC=15) ran full 12 gens, 5/5 SAFE.
- **Root cause**: More MC permutations means stricter robustness filtering, which penalizes exploration and triggers convergence/early-stop sooner.
- **Recommendation**: Keep `num_permutations: 15`. MC=30 adds no value.

### AP-12: Multi-Pair Expansion Degrades Generalization
- **Severity**: Medium-High
- **Evidence**: E32 (3 pairs: BTC+SOL+ETH) = 0S/5W, all MC=0.0 OVERFIT. Compare E21 (2 pairs: BTC+SOL) = SAFE holdout.
- **Root cause**: GA optimizes to the average across all pairs, producing generic strategies that don't survive MC permutation testing on individual pairs. More pairs = more noise, less signal.
- **Recommendation**: Stick with 2 pairs. Multi-pair needs a fundamentally different approach (e.g., per-pair specialists with ensemble).

### AP-13: Pop=18 Confirms Pop Ceiling (AP-8 Revalidation)
- **Severity**: Medium-High
- **Evidence**: E30 (pop=18) = 0S/5W, all MC=0.0 OVERFIT. Worst W6 fitness (max 0.1781). Catastrophic restart at Gen 8.
- **Root cause**: Same as AP-8 — more individuals with strict constraints generates more strategies exploiting narrow patterns.
- **Recommendation**: Do not exceed pop=15 for standard GA. This is the third confirmation (E8=15 SAFE, E19=20 OVERFIT, E30=18 all WARNING).

### AP-14: patience=4 + elite=2 Causes Premature Early Stop
- **Severity**: Medium-High
- **Evidence**: E45 (rank + elite=2 + patience=4) early-stopped at Gen 8 with 37.9% holdout degradation. Best raw fitness 0.6105 → penalized 0.3794.
- **Root cause**: Smaller elite (2) preserves less diversity between generations. Combined with aggressive patience (4), the early stop triggers before the GA has enough exploration time. Compare: E36 (rank + elite=3 + patience=4) ran full 12 gens and broke through at Gen 10.
- **Recommendation**: Use patience=4 only with elite ≥ 3. With elite=2, use default patience or patience ≥ 6.

### AP-15: LLM + Rank Selection = Negative Synergy
- **Severity**: Medium
- **Evidence**: E44 (rank + LLM) fitness 0.4285 vs E43 (tournament + LLM) fitness 0.6350. LLM disadvantage of -0.0653 with rank selection.
- **Root cause**: Rank selection provides smooth, gradient-like selection pressure. LLM immigrants inject strategies that disrupt this gradient — their diverse gene combinations scatter the fitness landscape that rank selection relies on for incremental improvement.
- **Recommendation**: Use LLM only with tournament selection. Tournament's stochastic nature accommodates LLM immigrants better.

### AP-20: mut=0.18 + Tournament + Elite=2 Triggers Holdout Early Stop
- **Severity**: Medium-High
- **Evidence**: E77 (tournament + elite=2 + mut=0.18 + patience=8) holdout early stop at gen 8. Only 4/5 SAFE (1 WARNING).
- **Root cause**: Higher mutation (0.18 vs 0.15) with small elite (2) creates too much exploration churn. Tournament selection amplifies this — top-performing strategies are mutated aggressively, lose their edge on holdout. With patience=8, the system tolerates degradation longer but eventually triggers early stop.
- **Recommendation**: Use mut=0.18 only with elite ≥ 3 (E78 with rank+elite3+mut018 was 5/5 SAFE). Or keep mut=0.15 with elite=2.

### AP-7: NSGA-II Produces Near-Zero Fitness
- **Severity**: High (if using NSGA-II)
- **Evidence**: E2 (NSGA-II) = fitness 0.0008. "SAFE" only because training fitness was nearly zero.
- **Root cause**: Current Pareto objective formulation produces degenerate solutions. Crowding distance + current fitness weight configuration don't converge.
- **Status**: Needs investigation before NSGA-II can be used again.
- **Recommendation**: Do not use `mode: 'nsga2'` until the fitness formulation is fixed.

---

## Optimal Parameter Ranges (Validated)

### Mutation Rate
| Context | Best Range | Evidence |
|---------|-----------|----------|
| Standard GA, strict constraints | **0.15-0.20** | E8 (0.15) = 0.114, **E21 (0.20) = SAFE/0.5412 fitness**, E15 (0.25) = 0.216 |
| Standard GA, default constraints | 0.20-0.40 | E5 (0.40) = 0.228, E1 (0.20) = 0.275 |
| Island model | 0.20-0.25 | E7 (0.25) = 0.136, E10 (0.40) = 0.315 |
| LLM-guided | ≤ 0.20 | E6 (0.20) = 0.188, E12 (0.40) = 0.330 |

### Population Size
| Context | Best Size | Evidence |
|---------|----------|----------|
| Standard GA | **15** | E8 (15) = 0.114. **E19 (20) = OVERFIT (59-65%)**. E30 (18) = 0S/5W. 15 is the ceiling. |
| Island model (per island) | **6** | E7 (6) = 0.136, E3 (4) = 0.183, E11 (8) = 0.366, **E24 (8) = EXTREME OVERFIT**. Never exceed 6. |

### Walk-Forward Windows
| Parameter | Optimal | Evidence |
|-----------|---------|----------|
| train_days | 120 | E7/E8 (120d) >> E3 (90d). E13 (180d) = worse. |
| validation_days | 30 | Standard across best experiments |
| max_windows | 8 | E7/E8 both use 8 |

### Elite Size
| Context | Best Size | Evidence |
|---------|----------|----------|
| Standard GA + tournament | **3** | E8 (3) = 5/5 SAFE, score 0.114. Proven gold standard. |
| Standard GA + tournament + patience=8 | **3** (safest) | **E76 (3+patience=8) = 5/5 SAFE, avg composite 0.189 — safest non-island config in 79 experiments.** |
| Standard GA + rank | **3** (with patience≥6) | E51 (3+patience=8) = HoF 0.7218, new champion. E40 (2) was seed-dependent (AP-16). |
| Standard GA + rank + patience=8 | **3** (mandatory) | AP-17: E50 (2+patience=8) = 55.6% degradation. E51 (3+patience=8) = 5/5 SAFE. |
| Standard GA + rank + mut=0.20 | **2** | E49 (2+mut=0.20) = 85% MC robustness, 5/5 SAFE. Higher mutation compensates smaller elite. |
| Standard GA + tournament + mut=0.18 | **≥3** (mandatory) | AP-20: E77 (elite=2+mut=0.18) early stop gen 8. E78 (elite=3+mut=0.18+rank) was 5/5 SAFE. |
| Island model | **1** | AP-1: elite > 1 destroys island diversity. |

### Selection Method
| Context | Best Method | Evidence |
|---------|------------|----------|
| Standard GA | **rank** | E33 (rank) = 85% MC robustness. E40 (rank+elite=2) = HoF#1 0.6953. Consistently top performer in W6-W7. |
| Standard GA + LLM | **tournament** | AP-15: LLM + rank = negative synergy (E44). E43 (tournament+LLM) = 0.6350 fitness. **W10 confirmed**: E68-E70 groq working, E69 +0.1396 advantage (best ever). |
| Standard GA + patience=8 | **tournament** (W9-W10) or **rank** (W8) | E63 (tournament+patience=8) = 0.6704 (W9 champion). E76 (tournament+elite=3+patience=8) = safest W10 config (0.189 composite). |
| Island model | tournament | Default for island; rank not yet tested on island. |

### Convergence Patience
| Value | Result | Evidence |
|-------|--------|----------|
| default (6) | Standard evolution | E8: 5/5 SAFE but no positive MC profit |
| **4** (with elite ≥ 3) | Late breakthrough possible | E36 (rank+elite=3+patience=4): Gen 10 breakthrough, fitness 0.6250, 81.6% win rate |
| 4 (with elite=2) | **CONFLICT — early stop** | AP-14: E45 early-stopped Gen 8 with 37.9% degradation. Do NOT combine patience=4 with elite=2. |
| **8** (with elite=3) | **BEST FITNESS** | E51 (rank+elite=3+patience=8): HoF=0.7218, Sharpe=7.45, 5/5 SAFE. New all-time champion. |
| 8 (with elite=2) | **OVERFIT RISK** | AP-17: E50 (rank+elite=2+patience=8): 55.6% degradation warning. Use elite≥3 with patience=8. |
| 8 (tournament+elite=2) | **W9 CHAMPION** | E63 (tournament+elite=2+patience=8): HoF=0.6704, 5/5 SAFE. Tournament+patience=8 is strong. |

### Generations
| Value | Result | Evidence |
|-------|--------|----------|
| **12** | Sufficient | E8, E40, E43, E36, E51 — all produced best results within 12 gens |
| 15 | **MC-robust discovery** | E61 (15 gens): Gen14_Ind9 = 100% MC robustness. E41 had lower fitness. |
| **15 + LLM(groq)** | **MC BREAKTHROUGH** | **E70 (15 gens + LLM): Gen14_Ind2 = 95% MC, +75.63% MC profit — only strategy with positive MC profit in 79 experiments.** LLM is the key: E73 (gen=15 no LLM) got 0% MC. |
| 15 (with patience=8) | Worth exploring | E61 had catastrophic restart at gen 14, found MC-robust right after. More gens + patience=8 may find more. |

---

## Missing Features / Untested

### MF-1: HMM Regime Detection Not Installed
- **Severity**: Low
- **Status**: `hmmlearn` package not installed. Falls back to `ensemble` method silently.
- **Impact**: Minimal — ensemble method works fine for all experiments.
- **Action**: Install `hmmlearn` only if ensemble detection proves insufficient.

### MF-2: Short Selling Never Tested in Exploration
- **Severity**: Medium (opportunity cost)
- **Status**: Code is confirmed working (KNOWN_ISSUES F0a), but no exploration experiment has used `short_selling.enabled: true`
- **Impact**: Missing potential alpha from short signals
- **Action**: Test with E8's strict constraints to prevent overfit from doubled search space.

### MF-3: Multi-Timeframe Untested
- **Severity**: Unknown
- **Status**: `multi_timeframe.enabled: true` never tested at scale
- **Impact**: Could provide better signals but may explode runtime
- **Action**: Low priority. Test in isolation with small pop/gen first.

### MF-4: E17 Overfit Analysis Missing
- **Severity**: Medium
- **Status**: RESOLVED — root cause identified as MF-5 (post-evolution code not wrapped in try-except).
- **Impact**: Cannot properly compare E17 to E8 for reproducibility validation.
- **Action**: Will be fixed by MF-5 fix for future experiments.

### MF-5: Post-Evolution Analysis Crash Bug (FIXED)
- **Severity**: **Critical**
- **Status**: **FIXED** (2026-03-16) — wrapped post-evolution code in try-except
- **Evidence**: ALL Wave 5 experiments (E19-E26) and E17 (Wave 4) ended at "EVOLUTION COMPLETE" without SAFE/WARNING/OVERFIT classification. No "GA RUN COMPLETE!" printed.
- **Root cause**: In `run_ga.py`, the evolution phase (lines ~750-856) was wrapped in try-except, but all post-evolution code (holdout validation, CPCV, Monte Carlo, save_summary_report with overfitting classification) was UNPROTECTED. An exception in any post-evolution step crashed the process silently with buffered stdout — no error message visible in logs.
- **Code location**: `genetic_algorithm/run_ga.py` lines 874-1140 (after fix)
- **Fix applied**: Wrapped entire post-evolution analysis block (holdout → CPCV → MC → save_summary_report → "GA RUN COMPLETE!") in try-except with `traceback.print_exc()` and `sys.stdout.flush()`/`sys.stderr.flush()`. Exception now prints clear "ERROR IN POST-EVOLUTION ANALYSIS" banner.
- **Impact**: 9 experiments affected (E17, E19-E26). All have evolution results saved but no formal overfit classification. Wave 6+ will have proper error reporting even if post-evolution analysis fails.

### MF-7: None Fitness TypeError in Evolution Summary (FIXED)
- **Severity**: **Critical**
- **Status**: **FIXED** (2026-03-16) — filter None fitness before sorting
- **Evidence**: E37 and E38 (Wave 7) crashed at Gen 4 with `TypeError: '<' not supported between instances of 'NoneType' and 'NoneType'`
- **Root cause**: `sorted(population.individuals, key=lambda x: x.fitness)` at `evolution.py:1800` crashed when LLM immigrants had `None` fitness because they were injected into the population but not yet evaluated.
- **Code location**: `genetic_algorithm/core/evolution.py` line 1800
- **Fix applied**: Changed to `sorted([ind for ind in population.individuals if ind.fitness is not None], key=lambda x: x.fitness, reverse=True)[:5]`
- **Impact**: Any experiment using LLM integration would crash mid-evolution when LLM immigrants were present in the population summary. 2 experiments lost (E37, E38).

### MF-8: ga_monitor.sh Shows STALE for Completed Experiments (FIXED)
- **Severity**: **Medium** (monitoring only)
- **Status**: **FIXED** (2026-03-16) — updated completion detection + added queue log fallback
- **Evidence**: All W5-W7 experiments showed "? STALE" in monitor dashboard since Wave 5.
- **Root cause**: TWO issues:
  1. Monitor greps for `'GA RUN COMPLETE'` but the wave7_*.log (RotatingFileHandler output) only contains `'EVOLUTION COMPLETE'` (from print() in run_ga.py). The `'GA RUN COMPLETE!'` message is printed at the end of post-evolution analysis which goes to stdout → queue_*.log.
  2. SAFE/WARNING/OVERFIT counts and composite scores are only in queue_*.log, not in the experiment's wave7_*.log.
- **Code location**: `ga_monitor.sh` function `get_metrics()`
- **Fix applied**: (a) Changed grep to match `'GA RUN COMPLETE|EVOLUTION COMPLETE'`. (b) Added fallback to search corresponding queue_*.log for results when the experiment log doesn't have SAFE/WARNING/OVERFIT stats.
- **Impact**: Monitor now properly shows DONE status and extracts results for all completed experiments.

### MF-6: numpy UnboundLocalError in Strategy Generator (FIXED)
- **Severity**: **High**
- **Status**: **FIXED** (2026-03-16) — removed local import from SuperTrend code generation
- **Evidence**: E30 (9 errors), E32 (9 errors), E34 (3 errors). Strategies with SuperTrend + CMF or VWAP indicators crash during backtesting.
- **Root cause**: SuperTrend indicator code generation in `generator.py` (~line 1138) added `import numpy as np` inside the `populate_indicators()` method body. Python scoping marks `np` as a local variable for the ENTIRE function scope when it sees any assignment to `np` anywhere in the function. When CMF/VWAP indicators reference `np.nan` before the SuperTrend import line executes, Python raises `UnboundLocalError: cannot access local variable 'np'`.
- **Code location**: `genetic_algorithm/strategies/generator.py` line 1123 (before fix)
- **Fix applied**: Removed the redundant `import numpy as np` line from SuperTrend generation. numpy is already imported at module level (line 743 in the strategy template).
- **Impact**: Any strategy combining SuperTrend with CMF, VWAP, or other `np.nan`-using indicators would crash. This affected ~20% of generated strategies in E30/E32/E34, causing evaluation fallbacks and degraded fitness scores.

---

## Key Learnings Summary

1. **E8 (strict standard GA) is the gold standard**: 5/5 SAFE, score 0.114, negative degradation on all strategies
2. **E21 (mut=0.20) is the best raw performer**: 0.5412 fitness with SAFE holdout — mutation sweet spot [0.15-0.20]
3. **Single-variable experiments only**: W3 definitively proved that combining features fails
4. **Walk-forward is mandatory**: Every experiment without WF has OVERFIT strategies (E4: 3/5 OVERFIT)
5. **Island model is fast but fragile**: 12-16 min vs 50-57 min, but more sensitive to parameter choices
6. **MC validation adds safety**: But only works with standard GA (not island), and 15 perms is sufficient
7. **LLM guidance is quality-focused**: Best holdout ever (0.4039) but needs conservative mutation and GROQ_API_KEY
8. **Negative degradation is the goal**: E8's strategies perform BETTER on holdout than training
9. **Population scaling doesn't help**: pop=20 (E19) OVERFITS for standard GA; pop=8 (E24) CATASTROPHIC for island
10. **Component crossover is risky**: Overfits even with standard GA + strict constraints (E20)
11. **Post-evolution code must be protected**: Bug MF-5 caused 9 experiments to lose overfit classification
12. **Rank selection > tournament for MC robustness**: E33 produced 85% MC robustness and +23.9% MC profit — the most actionable discovery since E8
13. **numpy import bug in strategy generator**: MF-6 caused ~20% of SuperTrend+CMF strategies to crash. Fixed by removing local import.
14. **Multi-pair scaling doesn't improve generalization**: E32 (3 pairs) produced 0S/5W vs E21 (2 pairs) SAFE. Strategies become too generic.
15. **elite_size=2 is optimal for rank selection**: E40 (rank+elite=2) produced HoF#1=0.6953, the best single strategy ever. E33 (rank+elite=3) had best MC but lower fitness.
16. **None fitness bug crashed LLM experiments**: MF-7 — sorted() on population with None fitness LLM immigrants caused TypeError. Fixed by filtering. E37/E38 lost.
17. **patience=4 conflicts with elite=2**: AP-14 — E45 early-stopped at Gen 8. Too much diversity pressure without enough elite preservation. Use default patience with elite=2.
18. **LLM works great standalone but hurts rank selection**: AP-15 — E43 (tournament+LLM) = 0.6350 vs E44 (rank+LLM) = 0.4285. LLM immigrants disrupt rank's smooth gradient.
19. **E40 (rank+elite=2) is NOT reproducible**: AP-16 — 2/3 seed variations failed (E48: 0% MC, E54: early stop). Seed-dependent local optimum. Demoted from champion.
20. **E51 (rank+elite=3+patience=8) is the new fitness champion**: HoF=0.7218, Sharpe=7.45, 5/5 SAFE. Must validate with seed checks in Wave 9.
21. **patience=8 requires elite≥3**: AP-17 — E50 (elite=2+patience=8) hit 55.6% degradation; E51 (elite=3+patience=8) was 5/5 SAFE. Bigger elite preserves solutions through delayed restarts.
22. **mut=0.20 produces MC-robust strategies**: E49 found 85% MC robustness — best ever. Negative holdout degradation (holdout BETTER than training). mut=0.20 is the robustness sweet spot.
23. **LLM benefit is reproducible**: E52 (+0.1865 advantage) confirmed E43's benefit. Tournament+LLM+groq consistently helps. ~~E55~~ was NOT an LLM test (openrouter bug).
24. **Wave 8 safety rate excellent**: 38/45 strategies SAFE (84%), 7 WARNING, 0 OVERFIT across all 8 experiments.
25. **OpenRouter provider was never registered**: AP-18 — `provider: 'openrouter'` silently disabled LLM in E55, E58, E64, E67 (4 wasted experiments). Fixed by adding OpenRouterProvider class. Use `groq` provider for reliable LLM.
26. **E51 is seed-dependent**: E56 (0.6436) and E57 (0.6533) confirmed the config works (4/5 SAFE) but doesn't reach 0.7218 with other seeds. Original seed was lucky.
27. **tournament+patience=8 is the W9 champion config**: E63 (0.6704, 5/5 SAFE) beat all rank-based configs in Wave 9. Tournament selection benefits from patience=8 just as rank does.
28. **gen=15 enables MC-robust discovery**: E61 Gen14_Ind9 achieved 100% MC robustness — the only such strategy in 67 experiments. The discovery came at gen 14 (after catastrophic restart), proving more generations can find MC-robust solutions.
29. **"Kitchen sink" approach underperforms**: E66 (rank+elite3+mut020+patience8, combined all W8 winners) only got 0.6191. Individual features work better in isolation.
30. **Wave 9 safety rate 91.7%**: 55/60 SAFE, 5 WARNING, 0 OVERFIT — best ever. The system consistently produces safe strategies.
31. **E70 Gen14_Ind2 = 95% MC robustness + POSITIVE MC profit (+75.63%)**: The only strategy with positive MC mean profit in 79 experiments. gen=15 + LLM(groq) is the recipe for MC-robust discovery. E73 (gen=15 without LLM) got 0% MC — LLM is the key ingredient.
32. **LLM groq provider confirmed working across 3 experiments**: E68 (7/12 success), E69 (4/15 success, +0.1396 advantage), E70 (8/15 success, +0.063). The openrouter fix from W9 was not needed — groq works natively.
33. **E69 = highest LLM advantage ever (+0.1396)**: tournament+elite3+patience8+LLM produces the strongest LLM benefit, even with 0 seeds (API reliability issue caused all seed generation to fail, but mid-evolution immigrants compensated).
34. **E76 = safest non-island config in 79 experiments**: tournament+elite3+patience8 (no LLM) produced avg composite 0.189 and 5/5 SAFE. Rock-solid safe baseline.
35. **All Wave 10 strategies show negative holdout profit**: -1.81% to -3.36% across the wave. Concerning trend — may indicate market regime shift in holdout period. Monitor in future waves.
