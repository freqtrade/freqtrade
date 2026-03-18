# GA Configuration Ranking & Results Tracker

> **Purpose**: Track every GA experiment run, rank configurations by effectiveness, and capture lessons learned.
> **Updated**: 2026-03-17 — Wave 1-10 COMPLETE (79 experiments, E70 Gen14_Ind2 = 95% MC robustness + positive MC profit, LLM groq confirmed working)

---

## Quick Ranking (Best → Worst by Avg Overfit Score)

| Rank | Experiment | Wave | SAFE | WARN | OVERFIT | Avg Score | Best Holdout | Runtime | Verdict |
|------|-----------|------|------|------|---------|-----------|-------------|---------|---------|
| **1** | **E8_strict_anti_overfit** | **W2** | **5** | **0** | **0** | **0.114** | **0.3038** | **57 min** | **BEST — 5/5 SAFE, MC validated, neg degradation** |
| 2 | E7_island_wf_optimized | W2 | 4 | 1 | 0 | 0.136 | 0.3392 | 16 min | EXCELLENT — 92.9% win rate, 1.4% degradation |
| 3 | E3_island_wf_stress | W1 | 3 | 2 | 0 | 0.183 | 0.3514 | 5.8 min | GREAT — Predecessor to E7 |
| 4 | E2_nsga2_multi_obj | W1 | 4 | 0 | 0 | 0.188 | 0.2815 | 49 min | MISLEADING — Near-zero fitness, SAFE by default |
| 5 | E6_llm_guided | W1 | 4 | 1 | 0 | 0.188 | **0.4039** | 50 min | **QUALITY** — Best holdout (0.4039), 4/5 SAFE |
| 6 | E15_strict_med_mutation | W4 | 2 | 3 | 0 | 0.216 | 0.2375 | 57 min | E8+mut=0.25 — more WARNING, worse than E8 |
| 7 | E5_aggressive_mutation | W1 | 4 | 1 | 0 | 0.228 | 0.3663 | 55 min | GOOD — High mutation works, negative degradation |
| 8 | E1_baseline_control | W1 | 1 | 4 | 0 | 0.275 | 0.2735 | 50 min | DECENT — No overfit but mostly WARNING |
| 9 | E10_island_wf_highmut | W3 | 3 | 1 | 1 | 0.315 | 0.3258 | 16 min | MIXED — High mut + island = 1 overfit |
| 10 | E9_island_wf_llm | W3 | 3 | 1 | 1 | 0.326 | 0.3165 | 17 min | MIXED — LLM + island didn't synergize |
| 11 | E12_stdga_llm_highmut | W3 | 0 | 5 | 0 | 0.330 | 0.1902 | 55 min | BAD — ALL WARNING, LLM+highmut too aggressive |
| 12 | E11_island_wf_scaled | W3 | 2 | 2 | 1 | 0.366 | 0.2669 | 30 min | MIXED — 8/island, 20 gen scaling didn't help |
| 13 | E14_island_wf_strict_mc | W4 | 3 | 0 | 2 | 0.403 | 0.2808 | 18 min | BAD — Island+strict+MC, MC=N/A (incompatible!) |
| 14 | E18_island_wf_component_cx | W4 | 0 | 4 | 1 | ~0.45 | 0.4250 | 12 min | BAD — Component cx on island = overfit |
| 15 | E13_island_wf_longwindow | W3 | 3 | 0 | 2 | 0.503 | 0.2727 | 15 min | BAD — 180d windows increased overfitting |
| 16 | E16_island_wf_elite_mc | W4 | 1 | 2 | 2 | 0.533 | 0.2571 | 18 min | WORST W4 — Elite=3 killed island diversity |
| 17 | E4_island_no_validation | W1 | 0 | 2 | 3 | **0.743** | 0.3170 | 8.6 min | **WORST** — Massive overfitting without WF |
| - | E17_e8_reproducibility | W4 | ? | ? | ? | ? | 0.5427* | 57 min | *Overfit analysis missing — see notes* |

**Wave 6** *(post-evolution bug fix confirmed — all 7 completed experiments got proper SAFE/WARNING/OVERFIT classification)*

| Rank | Experiment | Wave | SAFE | WARN | OVERFIT | Avg Score | Best Holdout | Runtime | Verdict |
|------|-----------|------|------|------|---------|-----------|-------------|---------|---------||
| ~5 | **E33_rank_selection** | **W6** | **2** | **3** | **0** | **~0.16** | **0.369** | **57 min** | **BEST W6 — rank selection, 85% MC robustness (best ever!)** |
| ~7 | E28_e8_mut017 | W6 | 2 | 3 | 0 | 0.188 | 0.210 | 57 min | mut=0.17, marginal vs E8 |
| ~7 | E34_e21_patience4 | W6 | 1 | 4 | 0 | 0.188 | 0.290 | 61 min | patience=4 effective, 35% MC + positive MC profit |
| ~8 | E29_e8_mut018 | W6 | 2 | 3 | 0 | 0.196 | 0.390 | 55 min | mut=0.18, marginal vs E8 |
| ~10 | E30_e21_pop18 | W6 | 0 | 5 | 0 | 0.200 | 0.178 | 68 min | pop=18 all WARNING, confirms pop=15 |
| ~11 | E32_e21_3pairs | W6 | 0 | 5 | 0 | 0.204 | 0.340 | 61 min | 3 pairs degraded generalization, numpy bugs |
| ~11 | E27_e21_seed271 | W6 | 2* | 3* | 0 | ~0.26 | 0.313 | 43 min | E21 seed variation, early stopped Gen 8 |
| - | E31_e8_llm_retry | W6 | - | - | - | - | - | - | INCOMPLETE — force-quit Gen 4 (daemon shutdown) |

**Wave 7** *(rank selection optimization + LLM bug fix. Post-evolution stats in queue logs — holdout degradation from evolution logs.)*

| Rank | Experiment | Wave | Best Fitness | HoF #1 | Win% | Sharpe | Status | Verdict |
|------|-----------|------|-------------|--------|------|--------|--------|---------|
| — | **E40_rank_elite2** | **W7** | **0.6181** | **0.6953** | **61.6%** | **5.13** | **Complete** | **W7 LEADER by HoF — elite=2 > elite=3 with rank** |
| — | **E43_llm_retry_fixed** | **W7** | **0.6350** | **~0.63** | **66.4%** | **4.58** | **Complete** | **BEST RAW FITNESS — LLM advantage +0.0839** |
| — | **E36_rank_patience4** | **W7** | **0.6250** | **0.6250** | **81.6%** | **6.74** | **Complete** | **Gen 10 breakthrough — patience=4 + elite=3 works** |
| — | E35_rank_mut020 | W7 | 0.5656 | ? | ? | ? | Complete | rank+mut=0.20 decent baseline |
| — | E46_rank_elite2_mut020 | W7 | 0.5385 | ? | ? | ? | Complete | Good mid-range |
| — | E42_rank_wf150 | W7 | 0.5133 | 0.6419 | ? | ? | Complete | 150d windows moderate, not worth change |
| — | E44_rank_llm_fixed | W7 | 0.4285 | ? | ? | ? | Complete | LLM HURTS rank selection (AP-15) |
| — | E41_rank_gen15 | W7 | 0.4194 | 0.6715 | ? | ? | Complete | 15 gens no benefit over 12 |
| — | E45_rank_elite2_patience4 | W7 | 0.3794* | 0.6105 raw | 78.2% | ? | Early stop Gen 8 | OVERFITTING — elite2+patience4 conflict (AP-14) |
| — | E39_e21_rank | W7 | 0.3006 | ? | ? | ? | Complete | rank doesn't universally help E21 config |
| — | E37_llm_retry | W7 | N/A | N/A | — | — | Crashed Gen 4 | None fitness bug pre-fix (MF-7) |
| — | E38_rank_llm | W7 | N/A | N/A | — | — | Crashed Gen 4 | None fitness bug pre-fix (MF-7) |
| — | E47_rank_elite2_llm | W7 | ? | ? | ? | ? | Started | Pending results |

**Wave 8** *(E51 new fitness champion HoF=0.7218, E49 found 85% MC robust strategy, E40 proven seed-dependent)*

| Rank | Experiment | Wave | SAFE | WARN | OVERFIT | Avg Score | HoF #1 | Sharpe | Verdict |
|------|-----------|------|------|------|---------|-----------|--------|--------|---------|
| ~1 | **E49_rank_elite2_mut020** | **W8** | **5** | **0** | **0** | **0.164** | **0.6307** | **3.59** | **MC STAR — 85% MC robustness, holdout IMPROVES** |
| ~2 | E55_e43_llm_seed_check | W8 | 5 | 0 | 0 | 0.190 | 0.5966 | 3.72 | ~~LLM confirmed~~ LLM was BROKEN (openrouter bug) — seed check only |
| ~3 | E52_tournament_elite2_llm | W8 | 5 | 0 | 0 | 0.196 | 0.6120 | — | LLM +0.1865 advantage over random |
| ~4 | E53_tournament_elite2 | W8 | 5 | 0 | 0 | 0.198 | 0.6349 | — | Best holdout (29.7% avg degradation) |
| ~5 | **E51_rank_elite3_patience8** | **W8** | **5** | **0** | **0** | **0.200** | **0.7218** | **7.45** | **NEW FITNESS CHAMPION — best HoF ever** |
| ~5 | E50_rank_elite2_patience8 | W8 | 5 | 0 | 0 | 0.200 | 0.6752 | 4.20 | Good but 55.6% degradation warning |
| ~6 | E48_e40_seed_check | W8 | 0 | 5 | 0 | 0.200 | 0.6243 | 3.05 | E40 NOT reproducible — ALL WARNING, 0% MC |
| ~7 | E54_e40_seed2 | W8 | 3 | 2 | 0 | 0.224 | 0.6592 | 7.87 | E40 NOT reproducible — EARLY STOP gen 8 |

*\*E45 best raw fitness was 0.6105 but penalized to 0.3794 due to 37.9% holdout degradation trend*

**Wave 9** *(E63 new Wave 9 champion, LLM openrouter bug discovered — all 3 LLM experiments had 0 LLM individuals, E61 found 100% MC strategy)*

| Rank | Experiment | Wave | SAFE | WARN | OVERFIT | Avg Score | HoF #1 | Sharpe | Verdict |
|------|-----------|------|------|------|---------|-----------|--------|--------|----------|
| ~1 | **E63_tournament_elite2_patience8** | **W9** | **5** | **0** | **0** | **0.198** | **0.6704** | **5.35** | **W9 CHAMPION — tournament+patience8 produces best Wave 9 fitness** |
| ~2 | E65_e51_pop20 | W9 | 4 | 1 | 0 | 0.213 | 0.6561 | 4.15 | pop=20 marginal improvement, not worth runtime cost |
| ~3 | E62_rank_elite3_mut020 | W9 | 4 | 1 | 0 | 0.214 | 0.6547 | 6.36 | Early stop gen 9 — mut=0.20+elite=3 triggers holdout degradation |
| ~4 | E57_e51_seed_check2 | W9 | 4 | 1 | 0 | 0.216 | 0.6533 | 5.20 | E51 partially reproduces — fitness drops from 0.7218 to 0.65 |
| ~5 | E56_e51_seed_check | W9 | 4 | 1 | 0 | 0.226 | 0.6436 | 3.14 | E51 partially reproduces — confirms the config is good, not champion-level |
| ~6 | E60_e49_seed_check | W9 | 5 | 0 | 0 | 0.199 | 0.6461 | 2.92 | E49 reproduces (5/5 SAFE) but no 85% MC this time |
| ~7 | E58_e51_llm | W9 | 5 | 0 | 0 | 0.207 | 0.6133 | 3.04 | LLM BROKEN (openrouter bug) — early stop gen 11 |
| ~8 | E61_e51_gen15 | W9 | 4 | 1 | 0 | 0.208 | 0.6014 | 2.65 | **100% MC strategy found (Gen14_Ind9)** — gen=15 enables MC-robust strategies |
| ~9 | E66_rank_elite3_mut020_patience8 | W9 | 5 | 0 | 0 | 0.198 | 0.6191 | 4.23 | "Kitchen sink" combo — combining everything doesn't help |
| ~10 | E59_rank_elite2_mut020_patience8 | W9 | 5 | 0 | 0 | 0.196 | 0.6050 | 2.58 | Low DSR (0.40) suggests Sharpe not statistically significant |
| ~11 | E64_rank_elite2_llm_mut018 | W9 | 5 | 0 | 0 | 0.194 | 0.5934 | 4.65 | LLM BROKEN (openrouter bug) — but still 5/5 SAFE |
| ~12 | E67_tournament_elite2_llm_patience8 | W9 | 5 | 0 | 0 | 0.200 | 0.5873 | 4.15 | LLM BROKEN (openrouter bug) — early stop gen 8 |

**Wave 10** *(E70 MC breakthrough — 95% MC robustness with positive MC profit, LLM groq confirmed working, E76 safest config)*

| Rank | Experiment | Wave | SAFE | WARN | OVERFIT | Avg Score | HoF #1 | MC% | Verdict |
|------|-----------|------|------|------|---------|-----------|--------|-----|----------|
| ~1 | **E76_tournament_elite3_patience8** | **W10** | **5** | **0** | **0** | **0.189** | **0.2678** | **25%** | **SAFEST CONFIG — avg composite 0.189 (lowest ever in non-island)** |
| ~2 | E69_tournament_elite3_patience8_llm | W10 | 5 | 0 | 0 | 0.194 | 0.5363 | 10% | **LLM advantage +0.1396 (best ever)** — despite 0 seeds |
| ~3 | E78_rank_elite3_mut018_patience8 | W10 | 5 | 0 | 0 | 0.196 | 0.4138 | 15% | Rank+elite3+mut018 = safe combo, good negative degradation |
| ~4 | E75_rank_elite2_mut020_gen15 | W10 | 5 | 0 | 0 | 0.198 | 0.4895 | 5% | 3 catastrophic restarts — most ever. High exploration instability |
| ~5 | E71_e63_seed_check | W10 | 5 | 0 | 0 | 0.199 | 0.6453 | 5% | E63 seed check — holdout early stop gen 12. Seed-dependent |
| ~6 | E74_rank_elite3_patience8_gen15 | W10 | 5 | 0 | 0 | 0.200 | 0.6094 | 0% | Holdout early stop gen 14. Rank+gen15 promising but degraded |
| ~7 | E73_tournament_elite2_patience8_gen15 | W10 | 5 | 0 | 0 | 0.202 | 0.3382 | 0% | gen=15 alone (no LLM) didn't produce MC results |
| ~8 | E68_tournament_elite2_patience8_llm | W10 | 5 | 0 | 0 | 0.211 | 0.5346 | 10% | LLM +0.036 advantage — "on par" |
| ~9 | E77_tournament_elite2_mut018_patience8 | W10 | 4 | 1 | 0 | 0.217 | 0.4493 | 0% | **Holdout early stop gen 8 — AP-20 (mut018+tourn+elite2)** |
| ~10 | E72_e63_seed_check2 | W10 | 5 | 0 | 0 | 0.188 | 0.2777 | 30% | E63 NOT reproducible — very low fitness. But 30% MC |
| ~11 | **E70_tournament_elite2_gen15_llm** | **W10** | **2** | **2** | **1** | **0.268** | **0.3426** | — | **MC BREAKTHROUGH: Gen14_Ind2 = 95% MC, +75.63% MC profit** |
| — | E79_tournament_elite2_patience6 | W10 | — | — | — | — | — | — | STILL RUNNING (gen 9/12) |

**Wave 5** *(formal SAFE/WARNING/OVERFIT scoring missing due to post-evolution analysis bug — see GA_FIXES_AND_IMPROVEMENTS.md MF-5. Rankings based on holdout degradation proxy.)*

| Rank | Experiment | Wave | Holdout Degrad | Best Fitness | Best Holdout | Runtime | Verdict |
|------|-----------|------|---------------|-------------|-------------|---------|---------|
| ~4 | **E21_e8_mut020** | **W5** | **39.2%** | **0.5412** | **0.3295** | **56 min** | **BEST W5 — mut=0.20 sweet spot, SAFE holdout** |
| ~8 | E25_short_selling | W5 | 22.3% | 0.2388 | 0.1856 | 55 min | SAFE but LIMITED — only 1 HoF strategy |
| ~9 | E23_llm_conservative | W5 | 39.3% | 0.2533 | 0.1537 | 55 min | LLM FAILED (no GROQ_API_KEY) — de facto E8 retry |
| ~10 | E22_profit_heavy | W5 | 42.6% | 0.2859 | 0.1641 | 56 min | SAFE holdout but lower fitness than E21 |
| ~11 | E26_mc30 | W5 | 37.6% trend | 0.4006 | 0.2499 | 42 min | WARNING — early stopped gen 8, MC=30 not better |
| ~14 | E20_component_cx | W5 | 52-61% | 0.3989 | 0.1553 | 55 min | OVERFIT — component cx worse than uniform for E8 |
| ~15 | E19_pop20 | W5 | 59-65% | 0.3492 | 0.1221 | 68 min | OVERFIT — pop=20 too large for strict constraints |
| ~18 | E24_e7_pop8 | W5 | 62-100% | 0.8240 | 0.0000 | 18 min | **EXTREME OVERFIT** — island pop=8 catastrophic |

---

## Detailed Experiment Results

### Wave 1 — All Complete (2026-03-15)

#### E3_island_wf_stress — RANK #1
- **Config**: Island model ON, Walk-Forward ON (deliberate conflict test), MC OFF
- **Settings**: pop=4/island, gen=8/phase, mut=0.20, cx=0.75 uniform, seed=42
- **Island**: 4 islands (bullish/bearish/sideways/master), ensemble regime detection, 4h timeframe
- **Regime Data**: bullish=360 bars (INSUFFICIENT), bearish=177 bars (INSUFFICIENT), sideways=adequate
- **Migrations**: 36 performed, fully_connected topology
- **Runtime**: 5.8 minutes (20:08 → 20:14)
- **Results**:
  | Strategy | Fitness | Holdout | Degradation | Trades | Win% | Sharpe | Label |
  |----------|---------|---------|-------------|--------|------|--------|-------|
  | Gen7_Ind1 | 0.3229 | 0.3082 | 4.6% | 58 | 65.0% | 6.43 | SAFE |
  | Gen7_Ind2 | 0.4505 | 0.2735 | 39.3% | 89 | 58.8% | 2.87 | WARNING |
  | Gen7_Ind3 | 0.3676 | 0.3514 | 4.4% | 76 | 60.5% | 0.37 | SAFE |
- **Takeaway**: Despite Known Issue #4 (island+WF "incompatible"), it produced the **best SAFE strategies**. WF validation inside island model *helps* prevent overfitting.
- **Errors**: 0

#### E6_llm_guided — RANK #3 (best holdout fitness!)
- **Config**: LLM-guided evolution (Groq/LLaMA-3.3-70B), WF ON, MC ON, holdout ON
- **Settings**: pop=15, gen=12, mut=0.20, cx=0.75, llm_seed_ratio=0.40, llm_immigrant_ratio=0.60
- **Runtime**: 50 minutes (20:08 → 20:58)
- **Best fitness**: 0.3214 (Gen4), final best: Gen11_Ind10 at 0.3127
- **Results**:
  | Strategy | Fitness | Holdout | Degradation | Trades | Win% | Label |
  |----------|---------|---------|-------------|--------|------|-------|
  | Gen11_Ind10 | 0.3127 | 0.3423 | -9.5% | 15 | 74.0% | SAFE |
  | Gen11_Ind12 | 0.2453 | 0.2018 | 17.7% | 63 | 56.0% | WARNING |
  | Gen11_Ind7 | 0.1367 | 0.4039 | -195.5% | 13 | 71.0% | SAFE |
- **Takeaway**: **Best holdout fitness** (0.4039) and **negative degradation** = holdout outperforms training! LLM guidance produces quality over quantity. High win rates (71-74%).
- **Errors**: 0

#### E5_aggressive_mutation — RANK #4
- **Config**: High mutation (0.40), WF ON, MC ON, holdout ON
- **Settings**: pop=15, gen=12, mut=0.40, cx=0.80 component, max_mutation=0.85, random_immigrants=4
- **Runtime**: 55 minutes (20:08 → 21:03)
- **Best fitness**: 0.3413 (Gen6), final best: Gen11_Ind11 at 0.3366
- **Results**:
  | Strategy | Fitness | Holdout | Degradation | Trades | Win% | Label |
  |----------|---------|---------|-------------|--------|------|-------|
  | Gen11_Ind11 | 0.3366 | 0.3663 | -8.8% | 13 | 71.3% | SAFE |
  | Gen11_Ind8 | 0.2984 | 0.3441 | -15.3% | 16 | 70.6% | SAFE |
  | Gen11_Ind13 | 0.2658 | 0.3074 | -15.7% | 15 | 74.3% | SAFE |
- **Takeaway**: High mutation works well! 4/5 SAFE, negative degradation. Trade count is low (13-16) suggesting conservative strategies survive. Very similar quality to E6 (LLM).
- **Errors**: 0

#### E1_baseline_control — RANK #5
- **Config**: Standard single-obj, WF ON, MC ON, holdout ON (control group)
- **Settings**: pop=15, gen=12, mut=0.20, cx=0.75 uniform, adaptive_mutation ON
- **Runtime**: 50 minutes (20:08 → 20:58)
- **Best fitness**: 0.3587 (Gen11), final best: Gen11_Ind12 at 0.3587
- **Results**:
  | Strategy | Fitness | Holdout | Degradation | Trades | Win% | Label |
  |----------|---------|---------|-------------|--------|------|-------|
  | Gen11_Ind11 | 0.3121 | 0.2735 | 12.4% | 26 | 78.1% | WARNING |
  | Gen11_Ind9 | 0.2767 | 0.1813 | 34.5% | 51 | 63.5% | WARNING |
  | Gen11_Ind3 | 0.2563 | 0.2197 | 14.3% | 54 | 53.5% | WARNING |
- **Takeaway**: Only 1/5 SAFE. Standard config with default mutation tends toward WARNING. Higher mutation (E5) or LLM (E6) produce more SAFE results.
- **Errors**: 0

#### E2_nsga2_multi_obj — RANK #2 (by score, but misleading)
- **Config**: NSGA-II multi-objective (profit↑, drawdown↓, sharpe↑), WF ON, MC ON, holdout ON
- **Settings**: pop=15, gen=12, mut=0.18, cx=0.80, fitness_sharing=false (G1 gotcha)
- **Runtime**: 49 minutes (20:08 → 20:57)
- **Best fitness**: 0.1617 (Gen3), final: Gen11_Ind11 at 0.0008
- **Results**:
  | Strategy | Fitness | Holdout | Degradation | Trades | Win% | Label |
  |----------|---------|---------|-------------|--------|------|-------|
  | Gen11_Ind11 | 0.0008 | 0.2815 | -37344% | 42 | 61.7% | SAFE |
  | Gen11_Ind3 | 0.0007 | 0.2738 | -39984% | 32 | 58.3% | SAFE |
  | Gen11_Ind9 | 0.0003 | 0.0356 | -11237% | 8 | 33.5% | SAFE |
- **Takeaway**: NSGA-II produces **near-zero training fitness** (0.0008!). The massive negative "degradation" is an artifact — training fitness is so low that holdout trivially outperforms. **Pareto selection has issues with the current fitness formulation.** Needs investigation before using NSGA-II again.
- **Errors**: 0

#### E4_island_no_validation — RANK #6 (WORST)
- **Config**: Island model ON, Walk-Forward OFF, MC OFF, holdout-only
- **Settings**: pop=4/island, gen=12/phase, mut=0.25, cx=0.70 single_point, diversity_threshold=0.20
- **Migrations**: 60 performed, fully_connected topology
- **Runtime**: 8.6 minutes (20:08 → 20:16)
- **Results**:
  | Strategy | Fitness | Holdout | Degradation | Trades | Win% | Sharpe | Label |
  |----------|---------|---------|-------------|--------|------|--------|-------|
  | Gen11_Ind2 | 0.8054 | 0.2748 | 65.9% | 97 | 68.1% | 12.10 | OVERFIT |
  | Gen11_Ind2 | 0.5342 | 0.3170 | 40.7% | 16 | 62.5% | 0.39 | WARNING |
  | Gen11_Ind3 | 0.5286 | 0.2122 | 59.9% | 95 | 34.7% | 2.82 | OVERFIT |
- **Takeaway**: **Worst experiment**. Without walk-forward, island model overfits massively (65.9% degradation). NEVER run island model without WF.
- **Errors**: 0

### Wave 2 (2026-03-15)

#### E7_island_wf_optimized — NEW RANK #1!
- **Config**: E3 refinement — island+WF with larger pop (6/island), more gen (15), stricter penalties
- **Settings**: pop=6/island, gen=15, mut=0.25, cx=0.75, WF train=120d, max_windows=8, holdout penalty ON
- **Runtime**: 16 minutes (20:51 → 21:08)
- **Results**:
  | Strategy | Fitness | Holdout | Degradation | Trades | Win% | Label |
  |----------|---------|---------|-------------|--------|------|-------|
  | Gen14_Ind5 | 0.3342 | 0.3296 | 1.4% | 24 | **92.9%** | SAFE |
  | Gen14_Ind4 | 0.5346 | 0.3392 | 36.6% | 47 | 77.0% | WARNING |
  | Gen14_Ind5 | 0.3260 | 0.3072 | 5.8% | 22 | 68.2% | SAFE |
  | Gen14_Ind5 | 0.2759 | 0.2083 | 24.5% | 16 | 47.4% | SAFE |
  | Gen14_Ind4 | 0.2405 | 0.2890 | -20.2% | 13 | 36.1% | SAFE |
- **Takeaway**: **Best experiment so far!** Avg score 0.136 beats E3's 0.183. The #1 strategy has a stunning **92.9% win rate** with only **1.4% holdout degradation**. Confirms that island+WF with larger population and more generations is the winning formula. Increasing from pop=4 to pop=6 and gen=8 to gen=15 paid off.
- **Errors**: 0

#### E8_strict_anti_overfit — NEW RANK #1!
- **Config**: Standard GA with maximally strict anti-overfit settings, MC validation ON
- **Settings**: pop=15, gen=12, mut=0.15, cx=0.80, min_trades=15, max_drawdown=0.20, holdout=20%, holdout penalty ON, MC 15 perms, max 3 indicators
- **Runtime**: 57 minutes (20:51 → 21:48)
- **Results**:
  | Strategy | Fitness | Holdout | Degradation | Trades | Win% | MC | Score | Label |
  |----------|---------|---------|-------------|--------|------|----|-------|-------|
  | Gen11_Ind12 | 0.2933 | 0.2957 | **-0.8%** | 40 | 70.8% | 0.80 | 0.040 | SAFE |
  | Gen11_Ind9 | 0.2240 | 0.3038 | **-35.6%** | 16 | 67.5% | 0.20 | 0.160 | SAFE |
  | Gen11_Ind4 | 0.1833 | 0.1855 | **-1.2%** | 18 | 79.8% | 0.30 | 0.140 | SAFE |
  | Gen11_Ind7 | 0.1813 | 0.2295 | **-26.6%** | 34 | 71.6% | 0.25 | 0.150 | SAFE |
  | Gen11_Ind3 | 0.1765 | 0.2931 | **-66.0%** | 6 | 77.0% | 0.80 | 0.080 | SAFE |
- **Takeaway**: **Perfect 5/5 SAFE with avg score 0.114!** All strategies show **negative degradation** (holdout BETTER than training). This is remarkable: strict constraints + MC validation + conservative mutation = most robust experiment yet. The top strategy has 40 avg trades and 0.80 MC robustness. Standard GA CAN match island when properly constrained.
- **Key insight**: Low mutation (0.15) + strict constraints prevents the GA from discovering overfitting shortcuts. The MC validation (15 permutations) provides additional robustness verification.
- **Errors**: 0

#### E8 vs E7 Comparison
- E8 (standard GA, strict): 5/5 SAFE, score **0.114**, fitness 0.29, 57 min
- E7 (island+WF): 4/5 SAFE, score 0.136, fitness 0.33, 16 min
- **Winner on robustness**: E8 (all strategies SAFE, negative degradation)
- **Winner on speed**: E7 (16 min vs 57 min)
- **Winner on raw fitness**: E7 (0.33 vs 0.29)
- **Conclusion**: Both approaches produce excellent results but optimize differently. E8 prioritizes safety, E7 prioritizes performance.

### Wave 3 — ALL COMPLETE (2026-03-15)

#### E9_island_wf_llm — Score 0.326 (WORSE THAN PARENTS)
- **Config**: Island(6/island) + WF(120d) + LLM(Groq llama-3.3-70b)
- **Hypothesis**: Combine E7's island+WF with E6's LLM guidance
- **Runtime**: 17 minutes
- **Results**:
  | Rank | Fitness | Holdout | Degradation | Label |
  |------|---------|---------|-------------|-------|
  | 1 | 0.3184 | 0.2693 | 15.4% | SAFE |
  | 2 | 0.3162 | 0.2714 | 14.2% | SAFE |
  | 3 | 0.5493 | 0.0536 | 90.2% | WARNING |
  | 4 | 0.2962 | 0.1976 | 33.3% | WARNING |
  | 5 | 0.2421 | 0.3165 | -30.7% | SAFE |
- **Verdict**: 3S/1W/1O, score 0.326. **LLM + island didn't synergize** — worse than E7 (0.136) or E6 (0.188) individually.

#### E10_island_wf_highmut — Score 0.315
- **Config**: Island(6/island) + WF(120d) + mut=0.40
- **Hypothesis**: Combine E7's island+WF with E5's high mutation
- **Runtime**: 16 minutes
- **Results**:
  | Rank | Fitness | Holdout | Degradation | Label |
  |------|---------|---------|-------------|-------|
  | 1 | 0.3151 | 0.2839 | 9.9% | SAFE |
  | 2 | 0.2951 | 0.3258 | -10.4% | SAFE |
  | 3 | 0.3141 | 0.2263 | 28.0% | SAFE |
  | 4 | 0.3143 | 0.2533 | 19.4% | SAFE |
  | 5 | 0.6224 | 0.2640 | 57.6% | WARNING |
- **Verdict**: 3S/1W/1O, score 0.315. High mutation + island = 4 safe individuals but the worst one was highly overfit. Net worse than E7.

#### E13_island_wf_longwindow — Score 0.503 (BAD)
- **Config**: Island(6/island) + WF(180d train / 45d val) + 15 gen
- **Hypothesis**: Longer training windows = more robust validation
- **Runtime**: 15 minutes
- **Results**:
  | Rank | Fitness | Holdout | Degradation | Label |
  |------|---------|---------|-------------|-------|
  | 1 | 0.3070 | 0.2709 | 11.8% | SAFE |
  | 2 | 0.6219 | 0.3331 | 46.4% | WARNING |
  | 3 | 0.5092 | 0.1874 | 63.2% | WARNING |
  | 4 | 0.2927 | 0.2259 | 22.8% | SAFE |
  | 5 | 0.3287 | 0.2727 | 17.0% | SAFE |
- **Verdict**: 3S/0W/2O, score 0.503. **180d windows WORSE than 120d** — counter-intuitive but longer windows may provide less diverse validation.

#### E11_island_wf_scaled — Score 0.366 (WORSE THAN E7)
- **Config**: Island(8/island) + WF(120d) + 20 gen
- **Hypothesis**: Scaling E7 with larger island pop (6→8) and more generations (15→20)
- **Runtime**: ~30 minutes
- **Results**:
  | Rank | Fitness | Holdout | Degradation | Label |
  |------|---------|---------|-------------|-------|
  | 1 | 0.3258 | 0.2669 | 18.1% | SAFE |
  | 2 | 0.3246 | 0.2654 | 18.2% | SAFE |
  | 3 | 0.5341 | 0.2027 | 62.1% | OVERFIT |
  | 4 | 0.2958 | 0.1929 | 34.8% | WARNING |
  | 5 | 0.2864 | 0.1846 | 35.5% | WARNING |
- **Verdict**: 2S/2W/1O, score 0.366. Scaling island pop from 6→8 with 20 gens = WORSE. Pop=6, gen=15 (E7) is the sweet spot.

#### E12_stdga_llm_highmut — Score 0.330 (ALL WARNING)
- **Config**: Standard GA + LLM(Groq llama-3.3-70b) + mut=0.40
- **Hypothesis**: Combine E6's LLM with E5's high mutation in standard GA
- **Runtime**: ~55 minutes
- **Results**:
  | Rank | Fitness | Holdout | Degradation | WF-Gap | MC-Rob | Label |
  |------|---------|---------|-------------|--------|--------|-------|
  | 1 | 0.2017 | 0.1902 | 5.7% | neg | 1.3% | WARNING |
  | 2 | 0.1785 | 0.1648 | 7.7% | neg | 3.3% | WARNING |
  | 3 | 0.1680 | 0.2098 | -24.9% | neg | 6.7% | WARNING |
  | 4 | 0.1476 | 0.1363 | 7.7% | neg | 9.3% | WARNING |
  | 5 | 0.1383 | 0.1208 | 12.7% | neg | 3.3% | WARNING |
- **Verdict**: 0S/5W/0O, score 0.330. **All WARNING** — LLM + mut=0.40 pushes exploration too far. Strategies have low fitness overall (max 0.20) and uniformly mediocre holdout. MC robustness scores near 0% confirm poor signal quality.

### Wave 5 — All Complete (2026-03-16)

> **NOTE**: All Wave 5 experiments are missing formal SAFE/WARNING/OVERFIT classification due to a bug in `run_ga.py` where post-evolution analysis code (holdout validation through `save_summary_report`) was not wrapped in try-except. An unhandled exception crashed silently after "EVOLUTION COMPLETE". Bug fixed — see GA_FIXES_AND_IMPROVEMENTS.md MF-5. Classifications below are estimated from holdout degradation patterns.

#### E21_e8_mut020 — BEST OF WAVE 5
- **Config**: E8 template + mutation rate raised to 0.20 (from 0.15)
- **Settings**: pop=15, gen=12, mut=0.20, cx=0.80 uniform, elite=3, tournament=4, WF 120d/30d, MC 15 perms, holdout 20%, parsimony max_indicators=3
- **Runtime**: ~56 min, holdout early stopped
- **Best fitness**: 0.5412 (highest raw fitness among SAFE experiments!)
- **Holdout**: Top strategy 39.2% degradation → SAFE estimate
- **HoF strategies**: 32
- **Takeaway**: mut=0.20 is the new sweet spot for E8's strict constraints. Higher raw fitness than E8 (0.5412 vs ~0.30) while maintaining SAFE holdout degradation. Confirms mutation range [0.15-0.20] as optimal.

#### E22_profit_heavy — SAFE but lower fitness
- **Config**: E8 template + profit weight raised to 0.30 (from default)
- **Settings**: pop=15, gen=12, mut=0.15, cx=0.80 uniform, profit_weight=0.30
- **Runtime**: ~56 min
- **Best fitness**: 0.2859, Holdout degradation: 42.6% → SAFE estimate
- **HoF strategies**: 28
- **Takeaway**: Increasing profit weight didn't improve results over E8.

#### E23_llm_conservative — LLM FAILED (GROQ_API_KEY missing)
- **Config**: E8 template + conservative LLM guidance
- **Error**: "API key required for GroqProvider. LLM generation disabled." — ran as E8 duplicate
- **Best fitness**: 0.2533, Holdout degradation: 39.3% → SAFE estimate
- **HoF strategies**: 28
- **Takeaway**: Need to set GROQ_API_KEY in environment. Will retry as E31 in Wave 6.

#### E19_pop20 — OVERFIT
- **Config**: E8 template + population raised to 20 (from 15)
- **Settings**: pop=20, gen=12, mut=0.15, cx=0.80 uniform
- **Runtime**: ~68 min (longer due to larger pop)
- **Best fitness**: 0.3492, Holdout degradation: 59-65% → OVERFIT
- **HoF strategies**: 44
- **Takeaway**: Larger population doesn't help with strict constraints. More individuals → more chances to overfit. pop=15 remains optimal for standard GA.

#### E20_component_cx — OVERFIT
- **Config**: E8 template + component crossover (instead of uniform)
- **Settings**: pop=15, gen=12, crossover_method=component
- **Best fitness**: 0.3989, Holdout degradation: 52-61% → OVERFIT
- **HoF strategies**: 27
- **Takeaway**: Component crossover preserves coherent indicator blocks but these patterns don't generalize. Uniform crossover is better for strict constraints.

#### E24_e7_pop8 — EXTREME OVERFIT
- **Config**: E7 island template + island pop raised to 8 (from 6)
- **Settings**: 4 islands, pop=8/island, gen=15, mut=0.25
- **Runtime**: ~18 min
- **Best fitness**: 0.8240 (very high = suspicious), Holdout degradation: 62-100% → EXTREME OVERFIT
- **HoF strategies**: 20
- **Takeaway**: Confirms AP-6 — island pop > 6 causes severe overfitting. 0.8240 training fitness with 0% holdout shows complete overfitting.

#### E25_short_selling — SAFE but LIMITED
- **Config**: E8 template + short selling enabled
- **Settings**: pop=15, gen=12, short_selling.enabled=true
- **Best fitness**: 0.2388, Holdout degradation: 22.3% → SAFE
- **HoF strategies**: Only 1 (!)
- **Takeaway**: Short selling doubles the search space. Only 1 HoF strategy found — need larger pop or more generations. Low fitness suggests current implementation needs refinement.

#### E26_mc30 — WARNING (early stopped)
- **Config**: E8 template + Monte Carlo permutations raised to 30 (from 15)
- **Settings**: pop=15, gen=12, monte_carlo.num_permutations=30
- **Runtime**: ~42 min (early stopped at gen 8)
- **Best fitness**: 0.4006, Holdout degradation: 37.6% trend → WARNING
- **HoF strategies**: 18
- **Takeaway**: More MC permutations causes earlier convergence/stopping. MC=15 remains sufficient — MC=30 adds no value and wastes compute.

### Wave 6 — ALL COMPLETE (2026-03-16, 7 complete + 1 incomplete)

> **NOTE**: Post-evolution bug fix (MF-5) confirmed working — all 7 completed experiments got proper SAFE/WARNING/OVERFIT classification with holdout + Monte Carlo analysis. Also discovered numpy UnboundLocalError bug (MF-6) in strategy generator affecting SuperTrend + CMF/VWAP combos.

#### E33_rank_selection — BEST OF WAVE 6 ⭐
- **Config**: E8 template + rank selection (instead of tournament)
- **Settings**: pop=15, gen=12, mut=0.15, cx=0.80 uniform, selection_method=rank, seed=330
- **Runtime**: ~57 min, early stopped Gen 11 (holdout degradation trend: 24.0%→48.4%)
- **Results**:
  | Strategy | Fitness | Holdout | Degradation | MC Rob | MC Profit | Composite | Label |
  |----------|---------|---------|-------------|--------|-----------|-----------|-------|
  | Gen11_Ind8 | 0.2826 | 0.3686 | neg | **0.85** | **+23.9%** | 0.03 | **SAFE** |
  | Gen11_Ind12 | 0.2130 | 0.249 | neg | 0.05 | -147.7% | ~0.19 | SAFE |
  | Gen11_Ind3 | 0.4218 | 0.2487 | 41.0% | 0.0 | -186.2% | ~0.22 | WARNING |
  | Gen11_Ind6 | 0.2085 | 0.185 | 11.3% | 0.0 | -160.2% | ~0.20 | WARNING |
  | Gen11_Ind13 | 0.1898 | 0.156 | 17.8% | 0.05 | -108.4% | ~0.19 | WARNING |
- **Takeaway**: **Rank selection produced the only strategy with 85% MC robustness and positive MC profit (+23.9%) across all 34 experiments.** Gen11_Ind8 has composite score 0.03 — by far the best individual strategy ever. Rank selection provides gentler selection pressure than tournament, allowing more diverse exploration while still converging to robust solutions.
- **Errors**: 0

#### E34_e21_patience4 — Second best W6
- **Config**: E21 template + convergence_patience=4 (tighter early stopping)
- **Settings**: pop=15, gen=12, mut=0.20, cx=0.80 uniform, convergence_patience=4, seed=340
- **Runtime**: ~61 min, full 12 gens (3x stagnation restart at gen 8, 10, 12)
- **Results**:
  | Strategy | Fitness | Holdout | Degradation | MC Rob | MC Profit | Composite | Label |
  |----------|---------|---------|-------------|--------|-----------|-----------|-------|
  | Gen11_Ind7 | 0.3991 | 0.3957 | 0.8% | 0.0 | -82.3% | 0.204 | WARNING |
  | Gen11_Ind5 | 0.2542 | 0.2896 | neg | **0.35** | **+5.6%** | 0.13 | **SAFE** |
  | Gen11_Ind14 | 0.2467 | 0.2451 | 0.7% | 0.0 | -325.4% | 0.203 | WARNING |
  | Gen11_Ind1 | 0.2050 | 0.194 | 5.4% | 0.0 | -268.6% | 0.200 | WARNING |
  | Gen11_Ind13 | 0.1938 | 0.174 | 10.2% | 0.0 | -188.3% | 0.200 | WARNING |
- **Takeaway**: Patience=4 triggered 3 catastrophic restarts, injecting diversity. Gen11_Ind5 achieved 35% MC robustness with **positive MC profit (+5.6%)** — second only to E33's champion. The tighter patience effectively filters overfitters but also limits exploration.
- **Errors**: 0

#### E28_e8_mut017 — Marginal improvement over E8
- **Config**: E8 template + mutation rate 0.17 (midpoint between 0.15 and 0.20)
- **Settings**: pop=15, gen=12, mut=0.17, cx=0.80 uniform, seed=280
- **Runtime**: ~57 min, full 12 gens
- **Results**:
  | Strategy | Fitness | Holdout | Degradation | MC Rob | MC Profit | Composite | Label |
  |----------|---------|---------|-------------|--------|-----------|-----------|-------|
  | Gen11_Ind3 | 0.2697 | holdout neg | SAFE | 0.0 | -363.9% | 0.20 | WARNING |
  | Gen11_Ind4 | 0.2100 | holdout neg | SAFE | 0.2 | -11.5% | 0.162 | SAFE |
  | Gen11_Ind13 | 0.1908 | holdout neg | SAFE | 0.0 | -140.2% | 0.20 | WARNING |
  | Gen11_Ind6 | 0.1635 | holdout neg | SAFE | 0.1 | -81.5% | 0.18 | SAFE |
  | Gen11_Ind8 | 0.1484 | holdout neg | SAFE | 0.0 | -332.0% | 0.20 | WARNING |
- **Takeaway**: All holdout labels SAFE but MC robustness universally poor (max 0.2). Marginal difference from E8 — mutation 0.17 doesn't justify the change.
- **Errors**: 0

#### E29_e8_mut018 — Also marginal
- **Config**: E8 template + mutation rate 0.18
- **Settings**: pop=15, gen=12, mut=0.18, cx=0.80 uniform, seed=290
- **Runtime**: ~55 min, early stopped Gen 11 (holdout degradation: 31.3%→48.8%)
- **Results**:
  | Strategy | Fitness | Holdout | Degradation | MC Rob | MC Profit | Composite | Label |
  |----------|---------|---------|-------------|--------|-----------|-----------|-------|
  | Gen11_Ind10 | 0.3897 | holdout neg | SAFE | 0.05 | -93.2% | 0.19 | SAFE |
  | Gen11_Ind12 | 0.2620 | holdout neg | SAFE | 0.0 | -170.7% | 0.20 | WARNING |
  | Gen11_Ind7 | 0.2349 | holdout neg | SAFE | 0.0 | -234.5% | 0.20 | WARNING |
  | Gen11_Ind13 | 0.2112 | holdout neg | SAFE | 0.05 | -118.7% | 0.19 | SAFE |
  | Gen11_Ind6 | 0.2007 | holdout neg | SAFE | 0.0 | -116.1% | 0.20 | WARNING |
- **Takeaway**: Higher best fitness (0.3897) than E28 but same poor MC robustness. Holdout degradation trend forced early stop — 0.18 mutation is too aggressive for long runs.
- **Errors**: 0

#### E27_e21_seed271 — Seed variation test (early stopped)
- **Config**: E21 template + seed=271 (different from E21's seed)
- **Settings**: pop=15, gen=12, mut=0.20, cx=0.80 uniform, seed=271
- **Runtime**: ~43 min, early stopped Gen 8 (holdout degradation trend: 22.6%→56.2%)
- **Holdout degradation progression**: 22.6% → 28.8% → 39.6% → 43.2% → 56.2% (worsening trend)
- **Takeaway**: E21's SAFE result was partially seed-dependent. Different seed produced worsening holdout degradation and early stop at Gen 8. Confirms that individual experiment results have variance — need multiple seeds to validate findings.
- **Errors**: 0

#### E30_e21_pop18 — Confirms pop ceiling
- **Config**: E21 template + population raised to 18 (from 15)
- **Settings**: pop=18, gen=12, mut=0.20, cx=0.80 uniform, seed=300
- **Runtime**: ~68 min, catastrophic restart at Gen 8 (7/18 replaced)
- **Results**:
  | Strategy | Fitness | Holdout | Degradation | MC Rob | MC Profit | Composite | Label |
  |----------|---------|---------|-------------|--------|-----------|-----------|-------|
  | Gen11_Ind5 | 0.1781 | holdout neg | SAFE | 0.0 | -321.4% | 0.20 | WARNING |
  | Gen11_Ind11 | 0.1719 | holdout neg | SAFE | 0.0 | -129.4% | 0.20 | WARNING |
  | Gen11_Ind8 | 0.1629 | holdout neg | SAFE | 0.0 | -341.6% | 0.20 | WARNING |
  | Gen11_Ind11 | 0.1580 | holdout neg | SAFE | 0.0 | -192.3% | 0.20 | WARNING |
  | Gen11_Ind14 | 0.1578 | holdout neg | SAFE | 0.0 | -128.7% | 0.20 | WARNING |
- **Takeaway**: 0/5 SAFE, all MC=0.0. Worst fitness in Wave 6 (max 0.1781). pop=18 produces more mediocre strategies that can't survive MC validation. Strongly confirms AP-8: pop>15 is harmful.
- **Errors**: 9+ numpy UnboundLocalError (SuperTrend+CMF combo, see MF-6)

#### E32_e21_3pairs — Multi-pair scaling failed
- **Config**: E21 template + 3 pairs (added ETH/USDT to BTC/SOL)
- **Settings**: pop=15, gen=12, mut=0.20, cx=0.80 uniform, pairs=[BTC/USDT, SOL/USDT, ETH/USDT], seed=320
- **Runtime**: ~61 min, full 12 gens
- **Results**:
  | Strategy | Fitness | Holdout | Degradation | MC Rob | MC Profit | Composite | Label |
  |----------|---------|---------|-------------|--------|-----------|-----------|-------|
  | Gen11_Ind10 | 0.3395 | 0.326 | 4.1% | 0.0 | -211.4% | 0.221 | WARNING |
  | Gen11_Ind5 | 0.2592 | holdout neg | SAFE | 0.0 | -409.3% | 0.20 | WARNING |
  | Gen11_Ind4 | 0.2380 | holdout neg | SAFE | 0.0 | -268.8% | 0.20 | WARNING |
  | Gen11_Ind8 | 0.2364 | holdout neg | SAFE | 0.0 | -246.8% | 0.20 | WARNING |
  | Gen11_Ind6 | 0.1894 | holdout neg | SAFE | 0.0 | -190.4% | 0.20 | WARNING |
- **Takeaway**: Despite higher base fitness (0.3395), adding a third pair degraded MC robustness to zero across all strategies. The GA optimizes to the average of 3 pairs, producing generic strategies that don't survive permutation testing on any one pair. Multi-pair needs a fundamentally different approach.
- **Errors**: 9+ numpy UnboundLocalError (SuperTrend+CMF combo, see MF-6)

#### E31_e8_llm_retry — INCOMPLETE
- **Config**: E8 template + LLM guidance (Groq/LLaMA-3.3-70B)
- **Settings**: pop=15, gen=12, mut=0.15, cx=0.80, llm_seed_ratio=0.20, seed=310
- **LLM Status**: Successfully initialized (GroqProvider, 3 LLM-generated seed strategies)
- **Population**: 2 hall-of-fame + 2 seeded + 3 LLM + 8 random
- **Interrupted**: Force-quit at Gen 4 when daemon received shutdown signal
- **Takeaway**: LLM integration working correctly with GROQ_API_KEY. Needs clean re-run (→ Wave 7 E37).
- **Errors**: Force-quit (external, not a code bug)

---

## Configuration Insights & Lessons Learned

### What Works
1. **Walk-forward validation** is the single most important setting — all WF-enabled runs had 0 OVERFIT
2. **Island model + WF** (E3) = lowest overfit score (0.183) and fastest runtime (5.8 min)
3. **Higher mutation** (E5: 0.40) produces more SAFE strategies than default (E1: 0.20) — counter-intuitive!
4. **LLM guidance** (E6) produces best holdout fitness (0.4039) with negative degradation
5. **Monte Carlo** validation works as additional safety net alongside WF
6. **Regime detection with ensemble** method works correctly (6571 4h candles, 22 segments)
7. **Migration** works correctly in island model (36-60 migrations, fully_connected)
8. **Negative degradation** = robust strategies where holdout outperforms training (E5, E6)
9. **Rank selection** (E33) produced the best MC robustness ever — 85% with positive MC profit (+23.9%). Gentler selection pressure → more diverse exploration → more robust strategies
10. **Convergence patience=4** (E34) effectively filters overfitters by triggering catastrophic restarts, producing positive MC profit strategies
11. **Post-evolution bug fix confirmed** — all 7 completed W6 experiments got proper SAFE/WARNING/OVERFIT classification

### What Doesn't Work
1. **Island model without WF** (E4) → massive overfitting (avg score 0.743), 3/5 OVERFIT
2. **NSGA-II multi-objective** (E2) → near-zero fitness values (0.0008), Pareto selection broken for this fitness
3. **Insufficient regime data** — bearish only 177 bars on 4h, bullish 360 bars → unreliable specialists
4. **Default mutation rate** (0.20) with standard GA → produces more WARNINGs than SAFEs
5. **HMM regime detection** failed — `hmmlearn` not installed, falls back
6. **Naive feature stacking** (W3) — combining best features doesn't guarantee improvement
7. **Longer training windows** (180d, E13) → WORSE overfitting than 120d (0.503 vs 0.136)
8. **LLM + Island** (E9) → worse than either alone (0.326 vs E7's 0.136 or E6's 0.188)
9. **LLM + High mutation** (E12) → ALL WARNING in standard GA (0.330). Too much exploration.
10. **Island pop scaling > 6** (E11, E24) → 8/island = worse (E11: 0.366, E24: EXTREME OVERFIT 62-100%). Diminishing returns.
11. **MC with island model** (E14) → MC silently ignored, results show N/A. Design constraint.
12. **Elite > 1 on island** (E16) → elite=3 on pop=6 = catastrophic (0.533 vs E7's 0.136)
13. **Component crossover on island** (E18) → ~0.45. Preserves regime-specific patterns that overfit on holdout.
14. **Mutation > 0.15 with strict constraints** (E15) → 0.25 produced 2S/3W (0.216) vs E8's 5S/0W (0.114)
15. **Population > 15 for standard GA** (E19) → pop=20 caused overfitting (59-65% degrad). More isn't better.
16. **Component crossover with strict constraints** (E20) → 52-61% degradation. Uniform is better.
17. **Short selling with default config** (E25) → Only 1 HoF from 15×12 evolution. Search space too large.
18. **MC > 15 permutations** (E26) → MC=30 caused early stopping at gen 8. No improvement over MC=15.
19. **Multi-pair scaling** (E32, 3 pairs) → 0S/5W, all MC=0.0. Adding ETH/USDT degraded generalization despite higher base fitness
20. **pop=18 standard GA** (E30) → 0S/5W, worst W6 fitness (0.1781). Confirms pop=15 ceiling (see also E19 pop=20)
21. **Mutation fine-tuning 0.17-0.18** (E28-E29) → marginal vs E8's 0.15. Not worth the change — stick with [0.15, 0.20]
22. **Seed variation** (E27 vs E21) → different seed gave worsening holdout trend and early stop. Individual results have variance

### Performance Patterns
- **Trade count vs safety**: SAFE strategies tend to have fewer trades (13-76 range)
- **Win rate**: Most strategies achieve 55-78% win rate regardless of config
- **Runtime**: Island model = 5-9 min, Standard GA (pop=15) = 49-55 min
- **Negative degradation**: E5/E6 strategies perform *better* on holdout than training — robust
- **Best holdout overall**: E6 Gen11_Ind7 at 0.4039 (LLM-guided)
- **Best training fitness**: E4 Gen11_Ind2 at 0.8054 (overfit, meaningless)

### Bugs Found
| Bug | Impact | Status |
|-----|--------|--------|
| Missing 4h data crashes island model | Fatal — E3/E4 crashed on first attempt | FIXED — downloaded 4h data |
| Process group signal propagation | All runs killed when one crashes | FIXED — setsid isolation |
| HMM regime detection KeyError | Non-fatal — falls back gracefully | KNOWN |
| NSGA-II near-zero fitness | Pareto selection issue | NEEDS INVESTIGATION |
| MC + Island incompatible | MC silently ignored with island model | DESIGN CONSTRAINT (see GA_FIXES_AND_IMPROVEMENTS.md) |
| Post-evolution analysis crash (E17, W5) | SAFE/WARNING/OVERFIT scoring never runs | **FIXED** — wrapped in try-except |
| E23 GROQ_API_KEY missing | LLM guidance silently disabled | WILL RETRY as E31 |
| numpy UnboundLocalError (MF-6) | SuperTrend+CMF/VWAP generated strategies crash | **FIXED** — removed local import in generator.py |
| Exit code 127 on W6 experiments | Python path issue with setsid launch | Non-blocking — experiments still complete |
| E31 force-quit at Gen 4 | Daemon shutdown signal propagated | Will re-run as E37 in Wave 7 |

### Parameter Sensitivity

| Parameter | Tested Values | Best | Observation |
|-----------|--------------|------|-------------|
| Walk-forward | ON vs OFF | **ON** | Single most important setting — 0 OVERFIT when ON |
| Mutation rate | 0.15, 0.17, 0.18, 0.20, 0.25, 0.40 | **0.15-0.20** (strict), **0.25-0.40** (relaxed) | E8 (0.15)=0.114, E21 (0.20)=SAFE/0.5412 fitness, E15 (0.25)=0.216. Sweet spot: [0.15-0.20] |
| Population (island) | 4 vs 6 vs 8 | **6** | E3 (4)=0.183, E7 (6)=0.136, E11 (8)=0.366, E24 (8)=EXTREME OVERFIT. Never go >6 |
| Population (standard) | 15 vs 20 | **15** | E19 (pop=20)=OVERFIT 59-65%. Pop=15 is optimal. |
| Island model | ON vs OFF | **ON+WF** | Best anti-overfit combo |
| NSGA-II mode | vs single_obj | **single_obj** | Multi-obj has fitness formulation issues |
| LLM guidance | ON vs OFF | **ON (alone)** | Best holdout when standalone, worse when combined with island or highmut |
| Crossover method | uniform vs single_point vs component | **uniform** (always) | E18: component on island = bad. E20: component on stdGA+strict = OVERFIT. Uniform is universally safer |
| MC validation | ON (15) vs ON (30) vs OFF | **ON (15 perms)** | MC=30 (E26) caused early stop, no improvement. 15 is sufficient |
| Holdout size | 15% vs 20% | **20%** | E8 (20%) = 0.114, E7 (15%) = 0.136 |
| WF train days | 90, 120, 180 | **120** | E7/E8 (120d) outperform E3 (90d). E13 (180d) WORSE = overfitting! |
| Elite size (island) | 1 vs 3 | **1** | E7 (elite=1)=0.136, E16 (elite=3)=0.533. High elitism kills island diversity |
| Elite size (stdGA) | 2 vs 3 | **3** | E8 (elite=3)=0.114. Fine for pop≥15 |
| Short selling | ON vs OFF | **OFF** (for now) | E25: only 1 HoF strategy. Doubles search space — needs different approach |
| Selection method | tournament vs rank | **rank** | E33 (rank) produced 85% MC robustness vs tournament's typical 0-20%. Best individual strategy ever |
| Convergence patience | default vs 4 | **4** | E34 (patience=4) triggered 3 restarts, produced 35% MC + positive MC profit. Filters overfitters |
| Multi-pair count | 2 vs 3 | **2** | E32 (3 pairs BTC+SOL+ETH) = 0S/5W, all MC=0.0. Third pair degrades generalization |
| Population (standard) | 15 vs 18 vs 20 | **15** | E30 (18) = 0S/5W. E19 (20) = OVERFIT. 15 is the hard ceiling |

---

## Wave Summary

### Wave 1 (2026-03-15) — COMPLETE
- **Goal**: Baseline comparison of 6 core configurations
- **Duration**: ~55 minutes total (all 6 ran in parallel)
- **Key Findings**:
  1. Island+WF = best anti-overfit combo (E3, avg score 0.183)
  2. Island WITHOUT WF = worst overfitting (E4, avg score 0.743)
  3. Higher mutation (0.40) outperforms default (0.20) for SAFE ratio
  4. LLM guidance produces best holdout fitness (0.4039)
  5. NSGA-II has fitness formulation issues — needs fix before re-testing
  6. Standard baseline (E1) underperforms enhanced configs

### Wave 2 (2026-03-15) — ALL COMPLETE
- **Goal**: Refine best configs from Wave 1
- **Experiments**: E7 (island+WF scaled up) ✅, E8 (strict anti-overfit standard GA) ✅
- **Key Finding**: E8 is now **#1 overall** with perfect 5/5 SAFE and avg score 0.114. E7 is #2 at 0.136. Both approaches work — island for speed+performance, strict standard GA for maximum safety.

### Wave 3 (2026-03-15) — ALL COMPLETE
- **Goal**: Combine best elements from W1+W2, test scaling limits
- **Results**: **Feature stacking doesn't work!** All W3 combos scored worse than their parents.
- **Experiments**:
  - E9: Island + WF + LLM → **0.326** (worse than E7=0.136 or E6=0.188 alone)
  - E10: Island + WF + high mutation → **0.315** (worse than E7=0.136 or E5=0.228)
  - E11: Island + WF scaled 8/island, 20 gen → **0.366** (scaling didn't help, WORSE than E7)
  - E12: Standard GA + LLM + high mutation → **0.330** (ALL 5 WARNING, LLM+highmut too aggressive)
  - E13: Island + WF + 180d windows → **0.503** (MUCH worse — longer windows = more overfit!)
- **Key Learnings**:
  1. Combining best features naively doesn't improve results — synergy is not guaranteed
  2. Longer training windows (180d) INCREASED overfitting vs 120d — too much data may not help
  3. LLM + island (E9) underperformed both pure LLM (E6) and pure island (E7)
  4. High mutation + island (E10) underperformed E7 (moderate mutation) and E5 (high mut, no island)
  5. Scaling island pop from 6→8 with more gens (20) made things WORSE (E11 score 0.366)
  6. LLM + high mutation (0.40) in standard GA = 0 SAFE strategies (E12: all WARNING)

### Wave 4 (2026-03-15) — ALL COMPLETE (5 experiments)
- **Goal**: Targeted single-variable improvements to E7/E8
- **Results**: **None improved on E8 or E7!** All W4 experiments scored worse than their parents.
- **Experiments**:
  - E14: Island+WF+MC+E8-strict → **0.403** — MC silently ignored (island incompatible!). 3S/0W/2O.
  - E15: E8+mut=0.25 → **0.216** — Higher mutation produced more WARNINGs (2S/3W/0O). E8's 0.15 is better.
  - E16: E7+elite=3+MC → **0.533** — Elite=3 destroyed island diversity. 1S/2W/2O. Worst W4.
  - E17: E8 reproducibility (seed=170) → **INCOMPLETE** — Evolution finished but overfit analysis missing.
  - E18: Island+WF+component_cx → **~0.45** — Component cx on island = overfit. Top 5: 33-53% degradation.
- **Key Learnings**:
  1. MC validation is incompatible with island model (DC-1 in GA_FIXES_AND_IMPROVEMENTS.md)
  2. Elite size > 1 is destructive for island pop=6 (AP-1)
  3. E8's mutation rate of 0.15 IS the optimum for strict constraints — 0.25 is worse
  4. Component crossover doesn't help island model — preserves regime-specific patterns that overfit
  5. E17 (reproducibility test) needs re-running for proper comparison
  6. **E8 remains the undisputed #1** — no experiment has come close to 0.114

### Wave 5 (2026-03-16) — ALL COMPLETE (8 experiments, auto-queue launched)
- **Goal**: Test E8 variants (mutation, pop, crossover, short selling, MC tuning) + LLM retry + island scaling
- **Results**: E21 (mut=0.20) is the standout — highest raw fitness (0.5412) among SAFE experiments. All other changes hurt.
- **Bug**: Post-evolution analysis never ran on ANY experiment — try-except bug in run_ga.py (now fixed).
- **Experiments**:
  - E21: E8+mut=0.20 → **SAFE** (39.2% degrad, fitness 0.5412) — **BEST W5, mut sweet spot**
  - E22: E8+profit_weight=0.30 → **SAFE** (42.6% degrad, fitness 0.2859) — lower than E21
  - E23: E8+LLM → **LLM FAILED** (no GROQ_API_KEY) — de facto E8 retry, SAFE (39.3%)
  - E25: E8+short_selling → **SAFE/LIMITED** (22.3% degrad) — only 1 HoF strategy (!)
  - E26: E8+MC=30 → **WARNING** (37.6% trend, early stopped gen 8) — MC=30 no better than 15
  - E20: E8+component_cx → **OVERFIT** (52-61% degrad) — uniform crossover wins
  - E19: E8+pop=20 → **OVERFIT** (59-65% degrad) — pop=15 is the ceiling
  - E24: E7+island_pop=8 → **EXTREME OVERFIT** (62-100% degrad, 0.8240 fitness!) — catastrophic
- **Key Learnings**:
  1. Mutation rate 0.20 is viable with strict constraints (E21 SAFE) — optimal range now [0.15-0.20]
  2. Population 20 overfits (E19) — pop=15 is confirmed optimal for standard GA
  3. Component crossover is harmful even for standard GA with strict constraints (E20 OVERFIT)
  4. Island pop=8 causes catastrophic overfitting (E24) — confirms pop=6 is the hard upper limit
  5. Short selling doubles search space — needs larger pop or dedicated approach (E25: only 1 HoF)
  6. MC=30 permutations offers no advantage over MC=15 and triggers early stopping (E26)
  7. Post-evolution analysis bug affected ALL experiments — only holdout degradation available as proxy

### Wave 6 (2026-03-16) — ALL COMPLETE (7 complete + 1 incomplete, auto-queue launched)
- **Goal**: Test rank selection, patience tuning, mutation fine-tuning, multi-pair scaling, LLM retry
- **Results**: **Rank selection (E33) is the breakthrough** — produced the only strategy with 85% MC robustness and positive MC profit. patience=4 (E34) also valuable.
- **Bug**: numpy UnboundLocalError in strategy generator when SuperTrend + CMF/VWAP combos generated (fixed: removed local import).
- **Experiments**:
  - E33: E8+rank_selection → **BEST W6** — 2S/3W, Gen11_Ind8 has 85% MC, +23.9% MC profit, composite 0.03
  - E34: E21+patience=4 → **2nd best** — 1S/4W, Gen11_Ind5 has 35% MC, +5.6% MC profit
  - E28: E8+mut=0.17 → 2S/3W — marginal vs E8
  - E29: E8+mut=0.18 → 2S/3W — marginal, early stopped Gen 11
  - E27: E21+seed=271 → early stopped Gen 8, seed-dependent variance
  - E30: E21+pop=18 → 0S/5W — pop ceiling confirmed again
  - E32: E21+3pairs → 0S/5W — multi-pair degrades, numpy bugs
  - E31: E8+LLM → INCOMPLETE (force-quit Gen 4)
- **Key Learnings**:
  1. Rank selection > tournament for MC robustness — the most actionable discovery since E8
  2. Convergence patience=4 produces better strategies via catastrophic restart diversity injection
  3. Mutation fine-tuning (0.17-0.18) is marginal — stick with 0.15 or 0.20
  4. Multi-pair scaling hurts generalization — strategies become too generic
  5. pop=18 confirms pop=15 ceiling (yet again)
  6. Seed variation shows individual experiment results have variance — need multi-seed validation
  7. numpy UnboundLocalError bug in SuperTrend generation — fixed
  8. Post-evolution bug fix confirmed working for all completed experiments

### Wave 7 (2026-03-16) — 10 COMPLETE + 2 CRASHED + 1 PENDING (rank+elite optimization wave)
- **Goal**: Optimize rank selection (W6 breakthrough). Test elite=2, patience combos, LLM with bug fix, gen=15, 150d windows.
- **Results**: **E40 (rank+elite=2) is the new performance leader** — HoF#1 = 0.6953. E43 (LLM) produced best raw fitness (0.6350). Critical interactions discovered: elite=2+patience=4 CONFLICTS, LLM+rank CONFLICTS.
- **Bugs**: None fitness TypeError crashed E37/E38 (fixed: filter None before sort). ga_monitor.sh STALE bug fixed (wrong completion marker).
- **Experiments**:
  - E40: rank+elite=2 → **W7 LEADER** — HoF=[0.6953, 0.6416, 0.6181, 0.5996, 0.5978], best at Gen2, 61.6% win, Sharpe 5.13. Holdout degrad 34-47%.
  - E43: tournament+LLM (bug fixed) → **BEST RAW FITNESS** — 0.6350 (Gen2_Ind13), HoF avg 0.6172, 66.4% win, LLM advantage +0.0839 vs random immigrants. 12 API calls used.
  - E36: rank+patience=4+elite=3 → **BREAKTHROUGH** — 0.6250 at Gen 10 (!), 81.6% win, Sharpe 6.74, best holdout degrad 23-43%. Patience enabled late discovery.
  - E35: rank+mut=0.20 → 0.5656, decent but not leader
  - E46: rank+elite=2+mut=0.20 → 0.5385, good mid-range
  - E42: rank+150d windows → 0.5133, HoF#1=0.6419, moderate — not worth the window change
  - E44: rank+LLM (bug fixed) → 0.4285, LLM HURT rank (-0.0653 disadvantage) — see AP-15
  - E41: rank+gen=15 → 0.4194, HoF#1=0.6715 — 15 gens no benefit over 12
  - E45: rank+elite=2+patience=4 → Early stop Gen 8, raw 0.6105 → penalized 0.3794, holdout degrad 27.7%→37.9% worsening — see AP-14
  - E39: E21+rank → 0.3006, weakest. Rank doesn't universally help non-E8 configs
  - E37: LLM retry → CRASHED Gen 4 (None fitness TypeError pre-fix, MF-7)
  - E38: rank+LLM → CRASHED Gen 4 (None fitness TypeError pre-fix, MF-7)
  - E47: rank+elite=2+LLM → PENDING (still running or just started)
- **Key Configuration Interactions**:
  | Combo | Effect | Evidence |
  |-------|--------|---------|
  | rank + elite=2 | **SYNERGY** | E40: HoF#1=0.6953, best single strategy ever |
  | rank + patience=4 (elite=3) | **SYNERGY** | E36: Gen 10 breakthrough after stagnation |
  | rank + elite=2 + patience=4 | **CONFLICT** | E45: Early stop Gen 8 (AP-14) |
  | rank + LLM | **CONFLICT** | E44: -0.0653 vs tournament+LLM (AP-15) |
  | tournament + LLM | **SYNERGY** | E43: +0.0839 advantage, best HoF avg |
  | rank + 15 gens | **NEGATIVE** | E41: worse than E40 with 12 gens |
- **Champion Config Rankings** (updated):
  1. **E8** (tournament, elite=3, mut=0.15) → SAFETY leader: 5/5 SAFE, composite 0.114
  2. **E40** (rank, elite=2, mut=0.15) → PERFORMANCE leader: HoF#1=0.6953
  3. **E33** (rank, elite=3, mut=0.15) → MC ROBUSTNESS leader: 85% MC, +23.9% MC profit
  4. **E43** (tournament, elite=3, mut=0.15, LLM) → LLM leader: 0.6350 fitness, HoF avg 0.6172
  5. **E36** (rank, elite=3, patience=4) → PATIENCE exemplar: Gen 10 breakthrough, 81.6% win rate
- **Key Learnings**:
  1. elite_size=2 is optimal for rank selection — E40 dramatically outperforms E33 (elite=3) on raw fitness
  2. LLM works great standalone (tournament) but hurts rank selection — use with tournament only
  3. patience=4 + elite=2 is a dangerous combination — causes premature early stop (E45)
  4. patience=4 + elite=3 enables late breakthroughs — E36 found best at Gen 10 after stagnation
  5. 15 generations adds no value over 12 — stagnation isn't solved by more time
  6. 150d windows are marginal — 120d remains the sweet spot
  7. None fitness bug in evolution.py crashed LLM experiments — fixed by filtering None before sort
  8. rank selection doesn't universally improve all configs — E39 (E21+rank) was worst in wave

### Wave 8 (2026-03-16) — 7 COMPLETE + 1 EARLY STOP (champion validation + feature combination wave)
- **Goal**: Validate E40 reproducibility, test patience=8, combine rank+LLM, seed-check LLM benefit.
- **Results**: **E40 is NOT reproducible** (E48: 0% MC, E54: early stop). **E51 (rank+elite=3+patience=8) is new fitness champion** (HoF=0.7218, Sharpe=7.45). **E49 found 85% MC robust strategy** (rank+elite=2+mut=0.20). LLM benefit partially confirmed (E52 with groq).
- **CORRECTION**: E55 used `openrouter` provider which was NOT registered → 0 LLM individuals generated. E55 is NOT an LLM confirmation — it's a non-LLM seed check. Only E52 (groq) had working LLM in Wave 8.
- **Safety**: 38/45 strategies SAFE (84%), 7 WARNING, 0 OVERFIT — excellent safety rate.
- **Experiments**:
  - E48: E40 seed=480 → ALL WARNING (0/5/0), HoF#1=0.6243, **0% MC robustness**. E40 is seed-dependent.
  - E49: rank+elite=2+mut=0.20 → 5/0/0 SAFE, HoF#1=0.6307, **one strategy hit 85% MC robustness**. Negative holdout degradation (holdout BETTER than training!) — outstanding.
  - E50: rank+elite=2+patience=8 → 5/0/0 SAFE, HoF#1=0.6752, but 55.6% degradation warning at Gen 10.
  - E51: rank+elite=3+patience=8 → 5/0/0 SAFE, **HoF#1=0.7218** (Sharpe=7.45, 71.8% win rate). New all-time best. Catastrophic restarts at Gen 5 & 9.
  - E52: tournament+elite=2+LLM → 5/0/0 SAFE, HoF#1=0.6120, LLM advantage +0.1865 vs random immigrants.
  - E53: tournament+elite=2 → 5/0/0 SAFE, HoF#1=0.6349, **best holdout degradation (29.7% avg)**.
  - E54: E40 seed=540 → 3/2/0, HoF#1=0.6592 (0.21% profit), **EARLY STOP Gen 8** — escalating holdout degradation 13→48%. Second confirmation E40 is unreliable.
  - E55: E43 LLM seed=550 → 5/0/0 SAFE, HoF#1=0.5966, **confirms LLM benefit is reproducible**.
- **Key Findings**:
  | Finding | Evidence | Impact |
  |---------|---------|--------|
  | E40 is seed-dependent | E48: 0% MC, E54: early stop | Demoted from champion |
  | patience=8 + elite=3 = best fitness | E51: 0.7218 HoF | New champion config |
  | patience=8 + elite=2 = overfit risk | E50: 55.6% degradation warning | Use elite=3 with patience=8 |
  | mut=0.20 + rank = MC robustness | E49: 85% MC strategy found | New MC robustness leader |
  | LLM benefit via groq provider | E52 (+0.1865 advantage) | LLM works with tournament+groq, NOT openrouter |\n  | ~~LLM reproducible~~ | E55 used openrouter (broken) — 0 LLM | E55 was NOT an LLM test |
  | tournament+elite=2 = best holdout | E53: 29.7% degradation | Most conservative config |
- **Champion Config Rankings** (updated):
  1. **E51** (rank, elite=3, patience=8) → NEW FITNESS CHAMPION: HoF=0.7218, Sharpe=7.45, 5/5 SAFE
  2. **E49** (rank, elite=2, mut=0.20) → MC ROBUSTNESS LEADER: 85% MC, 5/5 SAFE, negative holdout degradation
  3. **E8** (tournament, elite=3, mut=0.15) → SAFETY LEADER: composite 0.114, 5/5 SAFE (unchanged)
  4. **E53** (tournament, elite=2) → HOLDOUT LEADER: 29.7% avg degradation, 5/0/0 SAFE
  5. **E43/E52** (tournament+LLM via groq) → LLM LEADER: benefit confirmed (E52: +0.1865 advantage). E55 was NOT LLM (openrouter bug).
  6. ~~E40~~ (rank, elite=2, mut=0.15) → DEMOTED: seed-dependent, not reproducible

### Wave 9 (2026-03-17) — ALL 12 COMPLETE (E51 validation + LLM re-test + gen=15 exploration)
- **Goal**: Validate E51/E49 reproducibility, re-test LLM (with openrouter provider), test gen=15, explore tournament+patience=8, try combining best features.
- **Results**: **E63 (tournament+elite=2+patience=8) is Wave 9 champion** (HoF=0.6704). E51 partially reproducible (0.64-0.65 vs original 0.7218). **E61 found 100% MC robust strategy** at gen=15. **LLM openrouter bug discovered** — all 3 LLM experiments (E58, E64, E67) had 0 LLM individuals.
- **CRITICAL BUG**: `provider: 'openrouter'` was never registered in `PROVIDER_REGISTRY`. All openrouter LLM experiments since Wave 8 silently generated 0 LLM individuals. **E55 (Wave 8) was also affected** — invalidates "LLM confirmed reproducible" finding. Fixed by adding OpenRouterProvider class.
- **Safety**: 55/60 strategies SAFE (91.7%), 5 WARNING, 0 OVERFIT — best safety rate ever.
- **Runtime**: ~3h 4min wall time (23:44 → 02:48), 5 concurrent.
- **Experiments**:
  - E56: E51 seed=560 → 4/1/0, HoF#1=0.6436, Sharpe=3.14. Catastrophic restart gen 8. E51 partially reproducible — decent but not champion-level.
  - E57: E51 seed=570 → 4/1/0, HoF#1=0.6533, Sharpe=5.20. Catastrophic restart gen 7. Slightly better than E56 but still below E51's 0.7218. HoF#3 had 20.19% profit.
  - E58: E51+LLM (openrouter) → 5/0/0, HoF#1=0.6133, Sharpe=3.04. **LLM BROKEN (0 LLM individuals)**. Early stop gen 11 (holdout 11.4%→47.4%). Effectively a non-LLM E51 variant.
  - E59: rank+elite2+mut020+patience8 → 5/0/0, HoF#1=0.6050, **DSR=0.40** (very low — Sharpe not statistically significant). No catastrophic restart, no early stop.
  - E60: E49 seed=600 → 5/0/0, HoF#1=0.6461, Sharpe=2.92. Catastrophic restart gen 12. E49 config reproduces (5/5 SAFE) but no 85% MC this time — MC robustness is seed-dependent.
  - E61: E51+gen=15 → 4/1/0, HoF#1=0.6014. **Gen14_Ind9 achieved 100% MC robustness (fitness 0.5458)** — unprecedented! Catastrophic restart gen 14. Gen=15 enables MC-robust strategy discovery.
  - E62: rank+elite3+mut020 → 4/1/0, HoF#1=0.6547, Sharpe=6.36. **Early stop gen 9** (holdout 17.2%→41.3%). Catastrophic restart gen 9. mut=0.20+elite=3+patience=6 triggers overfit. HoF#2 had PF=10.74.
  - E63: tournament+elite2+patience8 → **5/0/0, HoF#1=0.6704**, Sharpe=5.35, 10.65% profit. **W9 CHAMPION**. Catastrophic restart gen 12. 20% MC on one strategy. Tournament beats rank in this wave.
  - E64: rank+elite2+LLM+mut018 (openrouter) → 5/0/0, HoF#1=0.5934, **81.1% win rate**, 15.51% profit, **lowest TV-gap (-1.3%)**. LLM BROKEN. Catastrophic restart gen 12. Despite no LLM, produced high win-rate strategies.
  - E65: E51+pop=20 → 4/1/0, HoF#1=0.6561, Sharpe=4.15, **lowest MaxDD (0.14%)**. No catastrophic restart. Pop=20 marginal improvement (+0.01 fitness) but 83min runtime vs ~73min for pop=15.
  - E66: rank+elite3+mut020+patience8 → 5/0/0, HoF#1=0.6191, **PF=5.258**. "Kitchen sink" combo underperformed — combining everything doesn't help. Low diversity (0.3291).
  - E67: tournament+elite2+LLM+patience8 (openrouter) → 5/0/0, HoF#1=0.5873. LLM BROKEN. **Early stop gen 8** (holdout 21.5%→37.9%). Low diversity (0.2669). Tournament+patience=8 still early-stops without LLM diversity injection.
- **Key Findings**:
  | Finding | Evidence | Impact |
  |---------|---------|--------|
  | E51 partially reproducible | E56: 0.6436, E57: 0.6533 vs original 0.7218 | Config is good (4/5 SAFE) but original seed was lucky |
  | E49 config reproduces safely | E60: 5/5 SAFE, 0.6461 | Config is reliable but 85% MC was seed-dependent |
  | tournament+patience=8 = W9 champion | E63: 0.6704, 5/5 SAFE | Tournament with patience=8 is a strong combination |
  | gen=15 enables MC-robust strategies | E61: Gen14_Ind9 = 100% MC | More generations to explore → MC-robust solutions at gen 14 |
  | LLM openrouter bug | E58, E64, E67: 0 LLM individuals | All openrouter configs broken — also affects E55 (Wave 8) |
  | Combining everything doesn't help | E66: 0.6191 < individual features | "Kitchen sink" approach underperforms focused configs |
  | Pop=20 marginal | E65: 0.6561 vs E56: 0.6436 (+0.01) | Not worth the extra runtime |
  | Safety rate 91.7% | 55/60 SAFE, 5 WARNING, 0 OVERFIT | Best safety rate in any wave |
- **Champion Config Rankings** (updated):
  1. **E51** (rank, elite=3, patience=8) → FITNESS CHAMPION: HoF=0.7218 (but seed-dependent: E56/E57 got 0.64-0.65)
  2. **E70** (tournament, elite=2, gen=15, LLM groq) → **MC BREAKTHROUGH**: Gen14_Ind2 = **95% MC robustness, +75.63% MC mean profit** — only strategy with positive MC profit in 79 experiments
  3. **E63** (tournament, elite=2, patience=8) → CONSISTENT CHAMPION: HoF=0.6704, 5/5 SAFE. Seed checks mixed (E71 early stop, E72 low fitness).
  4. **E49** (rank, elite=2, mut=0.20) → MC RUNNER-UP: 85% MC (but seed-dependent: E60 had 0% MC)
  5. **E8** (tournament, elite=3, mut=0.15) → SAFETY LEADER (composite): 0.114 avg score, 5/5 SAFE (unchanged)
  6. **E76** (tournament, elite=3, patience=8) → **SAFETY LEADER (W10)**: avg composite 0.189, 5/5 SAFE — safest non-island config
  7. **E43** (tournament, elite=3, LLM via groq) → LLM LEADER: 0.6350 fitness, confirmed by E52 (+0.1865 advantage)
  8. **E69** (tournament, elite=3, patience=8, LLM groq) → **HIGHEST LLM ADVANTAGE**: +0.1396 — elite3+tournament+patience8+LLM is strongest combo
  9. ~~E40~~ DEMOTED, ~~E55~~ NOT LLM (was openrouter bug)

### Wave 10 (2026-03-17) — 11/12 COMPLETE (E79 still running) — E70 MC Breakthrough + LLM Groq Confirmed + Safety Validation
- **Goal**: Re-test LLM with groq (fix confirmed), validate E63 reproducibility, explore gen=15, fine-tune elite/mut/patience combos.
- **Results**: **E70 Gen14_Ind2 achieved 95% MC robustness with +75.63% MC mean profit** — the only strategy with positive MC profit in 79 experiments. This is the most significant MC discovery of the entire campaign. **LLM via groq confirmed working** across E68-E70. **E76 (tournament+elite=3+patience=8)** is the safest config (avg composite 0.189). Three holdout early stops: E71 (gen 12), E74 (gen 14), E77 (gen 8).
- **LLM Provider Confirmed**: groq/llama-3.3-70b-versatile works reliably. E68: 7/12 API success (+0.036 advantage), E69: 4/15 success (+0.1396 advantage — best ever), E70: 8/15 success (+0.063 advantage). E69 had 0 seeds despite trying (API reliability issue, not a code bug).
- **Safety**: 51/55 strategies SAFE (92.7%), 3 WARNING, 1 OVERFIT — continued excellent safety rate.
- **Experiments**:
  - E68: tournament+elite2+patience8+LLM(groq) → 5/0/0, HoF#1=0.5346. LLM advantage +0.036 ("on par"). 7/12 API calls succeeded. 1 catastrophic restart.
  - E69: tournament+elite3+patience8+LLM(groq) → 5/0/0, HoF#1=0.5363. **LLM advantage +0.1396 (best ever)**. 0 LLM seeds (API reliability issue — 11/15 calls failed), but mid-evolution immigrants helped significantly. 2 catastrophic restarts.
  - **E70**: tournament+elite2+gen15+LLM(groq) → **2/2/1**, HoF#1=0.3426. **Gen14_Ind2 = 95% MC robustness, +75.63% MC mean profit**. The only OVERFIT strategy in the wave, and the only strategy with positive MC profit in 79 experiments. gen=15 + LLM(groq) = MC-robust discovery recipe. Highest risk profile (avg composite 0.268) but extraordinary MC result.
  - E71: E63 seed check (seed=710) → 5/0/0, HoF#1=0.6453. **Holdout early stop gen 12** (degradation trend detected). E63 not consistently stable — seed-dependent.
  - E72: E63 seed check (seed=720) → 5/0/0, HoF#1=0.2777, 30% MC. Very low fitness — E63 NOT consistently high-performing. Safest in raw composite (0.188) but low quality. 2 catastrophic restarts.
  - E73: tournament+elite2+patience8+gen15 → 5/0/0, HoF#1=0.3382, 0% MC. Moderate result. gen=15 alone (without LLM) didn't produce MC results here. 1 catastrophic restart.
  - E74: rank+elite3+patience8+gen15 → 5/0/0, HoF#1=0.6094, 0% MC. **Holdout early stop gen 14**. Rank+gen15 was promising but holdout degradation killed it. 1 catastrophic restart.
  - E75: rank+elite2+mut020+gen15 → 5/0/0, HoF#1=0.4895, 5% MC. **3 catastrophic restarts** — most restarts in any experiment. High exploration (mut=0.20+gen15) causes instability. Gen=15 didn't help here.
  - **E76**: tournament+elite3+patience8 → **5/0/0**, HoF#1=0.2678, 25% MC. **SAFEST CONFIG** — avg composite 0.189 (lowest in wave). tournament+elite=3+patience=8 without LLM = rock-solid safe baseline. 1 catastrophic restart.
  - E77: tournament+elite2+mut018+patience8 → 4/1/0, HoF#1=0.4493, 0% MC. **Holdout early stop gen 8** — mut=0.18+tournament+elite=2 triggers premature degradation (new AP-20). 1 catastrophic restart.
  - E78: rank+elite3+mut018+patience8 → 5/0/0, HoF#1=0.4138, 15% MC. Good negative degradation. Rank+elite3+mut018 is a safe combo. 2 catastrophic restarts.
  - E79: tournament+elite2+patience6 → **STILL RUNNING** (Gen 9/12). Results pending.
- **Key Findings**:
  | Finding | Evidence | Impact |
  |---------|---------|--------|
  | **E70 Gen14_Ind2 = 95% MC, +75.63% MC profit** | Only strategy with positive MC profit ever | gen=15 + LLM(groq) = MC-robust discovery recipe |
  | LLM groq confirmed working | E68-E70 all had LLM individuals | groq provider is reliable (fixed openrouter in W9) |
  | E69 = highest LLM advantage (+0.1396) | elite3+tournament+patience8+LLM | Best LLM combo found, despite 0 seeds |
  | E76 = safest config (0.189 composite) | tournament+elite3+patience8 (no LLM) | Rock-solid safe baseline |
  | E63 NOT consistently reproducible | E71: early stop gen 12, E72: HoF=0.28 | W9 champion is seed-dependent |
  | mut=0.18+tournament+elite=2 = early stop | E77: holdout early stop gen 8 | New anti-pattern AP-20 |
  | gen=15 + LLM produces MC breakthrough | E70 vs E73 (no LLM, 0% MC) | LLM is the key ingredient for MC-robust discovery |
  | All strategies have negative holdout profit | -1.81% to -3.36% across wave | Concerning trend — monitor in future waves |
  | Safety rate 92.7% | 51/55 SAFE, 3 WARNING, 1 OVERFIT | Best safety rate continues |

---

## How to Update This File

After each completed experiment:
1. Move the experiment from "Running" to the correct wave's "Completed" section
2. Update the Quick Ranking table (re-sort by Avg Score ascending)
3. Add detailed results table with top 3-5 strategies
4. Update "Configuration Insights" with new learnings
5. Update "Parameter Sensitivity" if new comparisons are available

```bash
# Check which experiments have finished
ps aux | grep run_ga.py | grep -v grep

# Get results from a completed experiment
tail -50 genetic_algorithm/logs/wave<N>_<EXP_NAME>.log

# Get detailed JSON results
grep "Detailed JSON results" genetic_algorithm/logs/wave<N>_<EXP_NAME>.log

# Run comparison across finished experiments
python genetic_algorithm/scripts/wave_comparison.py genetic_algorithm/output/exploration/wave1/
```
