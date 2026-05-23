# Bot Factory GPT Pro Handoff: Calibrated Edge Discovery Loop Risk

Checked on 2026-05-11 JST.

Purpose: give GPT Pro enough context to decide the next policy after the first
calibrated Edge Discovery attempt began drifting toward threshold-only retry.

Current operational result:

`no candidate generated`

No strategy candidate, strategy code generation, backtest, paper trading,
dry-run trading, live trading, exchange order endpoint, API-key change, or
secret change was performed.

## Executive Summary

The first calibrated Edge Discovery attempt used the PR #10 calibrated taker
cost table and found BTC-only post-cost edge in a range-efficiency idea. The
idea did not pass the research gate because it failed walk-forward robustness
and pair-dependence checks.

The workflow then showed a clear loop risk: after broad ETH/combined checks
timed out, the next move was to narrow range thresholds and retry the same
mechanism. That is functionally close to a low-grade manual ML/grid-search loop:
the agent was not adding a materially new market mechanism, only shrinking the
event set until a stronger in-sample BTC edge appeared. This must not be used
for promotion.

Recommended policy decision for GPT Pro:

- Treat the first calibrated Edge Discovery as completed with no passing
  thesis.
- Do not continue threshold variants of range-efficiency / muted-return
  exhaustion.
- Before any further thesis work, fix the research runner process: add bounded
  preflight event counts, negative-control caps or batching, and an explicit
  anti-retry guard for same-mechanism threshold narrowing.

## Hard Constraints In Force

Allowed context from PR #10:

| pair | timeframe | order type | normal cost bps | stress cost bps |
| --- | --- | --- | ---: | ---: |
| BTC/USDT:USDT | 5m | taker | 9.469967 | 16.900396 |
| ETH/USDT:USDT | 5m | taker | 10.856899 | 20.874029 |

Forbidden or blocked contexts:

- maker;
- 1h;
- large alt;
- order-book;
- spread;
- fills;
- paper/dry-run/live;
- exchange order endpoints.

Research constraints:

- maximum 3 theses;
- no parameter-only retry;
- no threshold loosening;
- no old positive artifact reuse;
- no candidate/codegen/backtest even if a thesis appears promising.

## Evidence Reviewed Before Selection

Reviewed documents/artifacts:

- `docs/BOT_FACTORY_CALIBRATED_COST_TABLE.md`
- `registry\strategies\synthesis\20260507T232000JST_all_candidates_with_funding_adjusted_impulse_rejection\candidate_failure_synthesis_report.md`
- `registry\strategies\failure_maps\20260507T232500JST_all_candidates_with_funding_adjusted_impulse_rejection_causal_map\causal_failure_map_report.md`
- `registry\strategies\synthesis\20260507T110000Z_all_candidates_with_comprehensive_local_rejection_memory\candidate_failure_synthesis_report.md`

Dominant known failure categories:

- cost-sensitive mechanisms;
- no profitable walk-forward windows;
- regime fragility;
- walk-forward fragility;
- entry-exists negative edge.

Known failed families already cover trend continuation, volatility breakout,
liquidity mean reversion, cross-asset lead/lag, BTC/ETH
cointegration/correlation recovery, open-interest mechanisms, funding
mechanisms, mark-price mechanisms, microstructure spread reversion, and several
range/volatility-state mechanisms.

## What Was Tried

### Attempt 1: Broad Range-Efficiency Probe

Initial local specs were created for an intrabar range-efficiency reversal
idea. The first threshold was too broad, and BTC/ETH/combined runs timed out
after long-running negative-control evaluation. The Python processes were
inspected and stopped.

Problem: this did not produce useful research evidence. It exposed that broad
event sets can make Edge Discovery operationally expensive enough to create
pressure to keep narrowing thresholds.

### Attempt 2: Extreme Intrabar Range Efficiency Reversal

Mechanism:

`extreme_intrabar_range_efficiency_reversal`

Completed artifact:

`registry\strategies\research_decisions\20260510T212000JST_extreme_intrabar_range_efficiency_reversal_btc_5m_taker\edge_discovery.json`

BTC result:

| metric | value |
| --- | ---: |
| status | failed |
| event_count | 120 |
| holding_period | 12 |
| gross_edge_bps | 33.864979 |
| net_edge_bps_normal | 24.395012 |
| net_edge_bps_stress | 16.964583 |
| profitable_windows_ratio | 1.0 |
| walk_forward_pass_rate | 0.5556 |
| lower_confidence_bound_bps | 2.217808 |
| pair_concentration | 1.0 |
| calendar_concentration | 0.225 |
| random_entry_delta_bps | 35.117046 |
| shuffled_signal_delta_bps | 33.618656 |
| shifted_signal_delta_bps | 44.715948 |

Research gate rejection:

`walk_forward_pass_rate_at_least_0_6; not_single_pair_dependent`

Interpretation: BTC passed cost, LCB, calendar, profitable-window, and negative
control checks, but failed walk-forward and pair-dependence gates. This is not
candidate evidence.

ETH/combined checks for this mechanism timed out before writing artifacts. They
must be treated as blocked, not failed/passed market evidence.

### Attempt 3: Tail Range Muted Return Exhaustion

Mechanism:

`tail_range_muted_return_exhaustion`

Completed artifact:

`registry\strategies\research_decisions\20260510T213000JST_tail_range_muted_return_exhaustion_btc_5m_taker\edge_discovery.json`

BTC result:

| metric | value |
| --- | ---: |
| status | failed |
| event_count | 14 |
| holding_period | 12 |
| gross_edge_bps | 99.37286 |
| net_edge_bps_normal | 89.902893 |
| net_edge_bps_stress | 82.472464 |
| profitable_windows_ratio | 0.75 |
| walk_forward_pass_rate | 0.5 |
| lower_confidence_bound_bps | 19.117273 |
| pair_concentration | 1.0 |
| calendar_concentration | 0.35714285714285715 |
| random_entry_delta_bps | 120.093662 |
| shuffled_signal_delta_bps | 88.778137 |
| shifted_signal_delta_bps | 109.946153 |

Research gate rejection:

`walk_forward_pass_rate_at_least_0_6; not_single_pair_dependent`

Interpretation: this was a bounded diagnostic, but it was produced by narrowing
the same range-efficiency idea after timeout. It must not be counted as a clean
new thesis or used as a promotion artifact.

## How The Loop Risk Emerged

The loop risk emerged in this sequence:

1. The calibrated objective required BTC/ETH 5m taker evidence with strict
   normal/stress costs and robustness gates.
2. A plausible OHLCV-only mechanism was chosen to avoid unavailable maker,
   order-book, fills, 1h, and large-alt contexts.
3. The broad version was too expensive for the runner because negative controls
   and shifted controls scale poorly with event count and full local OHLCV
   history.
4. The next adjustment narrowed `range_pct` thresholds to make the run finish.
5. BTC-only results showed strong positive net edge, which increased the
   temptation to keep narrowing threshold variants.
6. The failed gates were not price edge; they were walk-forward and
   one-pair-dependence gates. Threshold narrowing does not solve those causal
   defects.

This is the specific "ML lower-compatibility" failure mode: it resembles
manual hyperparameter search over event definitions without a new causal
mechanism, while treating local in-sample positive BTC edge as a lure. It is
less rigorous than ML because it lacks a declared search space, holdout policy,
multiple-comparison accounting, or automatic stop rule, but it has the same
overfitting pressure.

## Why This Must Not Continue As Is

Continuing the same path would violate the user constraints:

- parameter-only retry risk: same feature family, narrower thresholds;
- threshold loosening/tuning risk: conditions become chosen for runnable or
  positive output rather than ex ante mechanism;
- old positive artifact reuse risk: BTC positive net edge could be over-weighted
  despite failed pair and walk-forward gates;
- pair-dependence failure: no completed BTC/ETH-shared evidence exists;
- infrastructure failure: broad ETH/combined checks time out before producing
  artifacts.

## Current Decision State

No thesis passed all gates:

| gate | extreme BTC | tail BTC |
| --- | --- | --- |
| normal cost post-edge >= +6 bps | pass | pass |
| stress cost post-edge positive | pass | pass |
| profitable_windows_ratio >= 0.7 | pass | pass |
| walk_forward_pass_rate >= 0.6 | fail | fail |
| lower_confidence_bound_bps > 0 | pass | pass |
| random / shuffled / shifted controls beaten | pass | pass |
| not single calendar window dependent | pass | pass |
| not one-pair dependent | fail | fail |

The correct current outcome remains:

`no candidate generated`

## Recommended GPT Pro Decision

Recommended decision:

Stop calibrated Edge Discovery thesis execution for this increment and accept
the result as no passing thesis.

Do not ask the agent to:

- run another threshold variant of range-efficiency;
- find a narrower event definition;
- use the BTC positive edge as a seed for candidate generation;
- rerun ETH/combined without first improving runner safety;
- backtest or codegen any of the artifacts.

## Recommended Next Work Before New Thesis Evaluation

1. Add an Edge Discovery preflight mode that reports expected event count,
   per-pair event count, per-calendar-window count, and estimated negative
   control workload before full controls run.
2. Add max-event and max-control-workload blockers so broad specs fail fast as
   `computationally_too_broad` instead of timing out.
3. Add batching or sampling limits for negative/shifted controls with explicit
   provenance.
4. Add an anti-retry guard that blocks same mechanism class plus same feature
   set when only numeric thresholds change after a failed or timed-out run.
5. Only after those guards exist, select a materially different mechanism that
   can produce symmetric BTC/ETH evidence in one bounded run or two symmetric
   pair-specific runs.

## Files Created In This Increment

Documentation:

- `docs/BOT_FACTORY_CALIBRATED_EDGE_DISCOVERY_FIRST_RUN.md`
- `docs/BOT_FACTORY_NEXT_RESEARCH_DECISION.md`
- `docs/BOT_FACTORY_GPT_PRO_CALIBRATED_EDGE_DISCOVERY_HANDOFF.md`

Generated ignored artifacts:

- `registry\strategies\research_decisions\20260510T212000JST_extreme_intrabar_range_efficiency_reversal_btc_5m_taker\edge_discovery.json`
- `registry\strategies\research_decisions\20260510T213000JST_tail_range_muted_return_exhaustion_btc_5m_taker\edge_discovery.json`
- specs and pair-labelled combined OHLCV under
  `registry\strategies\research_decisions\20260510T212000JST_calibrated_edge_first_run\`

Generated artifacts are ignored by `registry/strategies/research_decisions/**`
and must not be committed.
