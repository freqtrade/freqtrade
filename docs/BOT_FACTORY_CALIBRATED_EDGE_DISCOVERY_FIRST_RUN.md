# Bot Factory Calibrated Edge Discovery First Run

Checked on 2026-05-11 JST using the PR #10 calibrated cost table.

Candidate generation result:

`no candidate generated`

No strategy candidate was generated. No strategy code generation, backtest,
paper trading, dry-run trading, live trading, exchange order endpoint, API key,
secret change, maker context, 1h context, large-alt context, order-book context,
or fills context was used.

## Inputs Reviewed

- `docs/BOT_FACTORY_CALIBRATED_COST_TABLE.md`
- `registry\strategies\synthesis\20260507T232000JST_all_candidates_with_funding_adjusted_impulse_rejection\candidate_failure_synthesis_report.md`
- `registry\strategies\failure_maps\20260507T232500JST_all_candidates_with_funding_adjusted_impulse_rejection_causal_map\causal_failure_map_report.md`
- `registry\strategies\synthesis\20260507T110000Z_all_candidates_with_comprehensive_local_rejection_memory\candidate_failure_synthesis_report.md`

Usable calibrated contexts were limited to:

| pair | timeframe | order type | normal cost bps | stress cost bps |
| --- | --- | --- | ---: | ---: |
| BTC/USDT:USDT | 5m | taker | 9.469967 | 16.900396 |
| ETH/USDT:USDT | 5m | taker | 10.856899 | 20.874029 |

Blocked contexts remained blocked: maker, 1h, large-alt, order-book, spread,
and fills.

## Failure Memory Constraint

The latest failure synthesis and causal map require the next research path to
avoid repeated failed families, parameter-only tuning, threshold loosening, and
old positive artifact reuse. The dominant failure categories were:

- cost-sensitive mechanism;
- no profitable walk-forward windows;
- regime fragility;
- walk-forward fragility;
- entry-exists negative edge.

The prior failed families already include trend continuation, volatility
breakout, liquidity mean reversion, cross-asset lead/lag, BTC/ETH
cointegration/correlation recovery, open-interest mechanisms, funding
mechanisms, mark-price mechanisms, microstructure spread reversion, and several
range/volatility-state mechanisms.

## Thesis Selection

Selected thesis count: 2, within the maximum of 3.

| thesis_id | mechanism_class | status |
| --- | --- | --- |
| `TH-EXTREME-INTRABAR-RANGE-EFFICIENCY-REVERSAL-20260510` | `extreme_intrabar_range_efficiency_reversal` | Evaluated on BTC 5m taker; rejected. |
| `TH-TAIL-RANGE-MUTED-RETURN-EXHAUSTION-20260510` | `tail_range_muted_return_exhaustion` | Diagnostic BTC run only; rejected and not pursued further because continuing would resemble parameter-only retry. |

No further threshold variants were run after the loop-risk review. ETH and
combined BTC/ETH checks for the broad extreme-range thesis timed out before
writing artifacts; the processes were stopped and the context is recorded as a
computational blocker, not a partial pass.

## Completed Edge Discovery Results

### Extreme Intrabar Range Efficiency Reversal

Command:

```powershell
.\.venv\Scripts\python.exe scripts\bot_factory_build_edge_discovery.py --ohlcv-path user_data\data\bybit\futures\BTC_USDT_USDT-5m-futures.parquet --edge-spec-json registry\strategies\research_decisions\20260510T212000JST_calibrated_edge_first_run\specs\intrabar_range_efficiency_reversal_btc_5m_taker.edge_spec.json --failure-synthesis-json registry\strategies\synthesis\20260507T232000JST_all_candidates_with_funding_adjusted_impulse_rejection\candidate_failure_synthesis.json --min-profitable-windows-ratio 0.7 --min-calendar-window-count 4 --min-profitable-calendar-windows-ratio 0.6 --min-data-span-days 180 --min-negative-control-delta-bps 1.0 --edge-discovery-id 20260510T212000JST_extreme_intrabar_range_efficiency_reversal_btc_5m_taker --reviewer-note "First calibrated Edge Discovery run using PR #10 BTC 5m taker costs; no candidate/codegen/backtest/paper/dry-run/live." --created-at 2026-05-10T21:20:00+09:00
```

Artifact:

`registry\strategies\research_decisions\20260510T212000JST_extreme_intrabar_range_efficiency_reversal_btc_5m_taker\edge_discovery.json`

Result:

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

Interpretation: this BTC-only run cleared normal cost, stress cost, profitable
window ratio, lower confidence bound, calendar concentration, and negative
controls. It still failed because the walk-forward pass rate was below 0.6 and
the evidence was BTC-only. It cannot be promoted or converted into a candidate.

### Tail Range Muted Return Exhaustion

Command:

```powershell
.\.venv\Scripts\python.exe scripts\bot_factory_build_edge_discovery.py --ohlcv-path user_data\data\bybit\futures\BTC_USDT_USDT-5m-futures.parquet --edge-spec-json registry\strategies\research_decisions\20260510T212000JST_calibrated_edge_first_run\specs\tail_range_muted_return_exhaustion_btc_5m_taker.edge_spec.json --failure-synthesis-json registry\strategies\synthesis\20260507T232000JST_all_candidates_with_funding_adjusted_impulse_rejection\candidate_failure_synthesis.json --min-profitable-windows-ratio 0.7 --min-calendar-window-count 4 --min-profitable-calendar-windows-ratio 0.6 --min-data-span-days 180 --min-negative-control-delta-bps 1.0 --edge-discovery-id 20260510T213000JST_tail_range_muted_return_exhaustion_btc_5m_taker --reviewer-note "Bounded first calibrated Edge Discovery run using PR #10 BTC 5m taker costs; no candidate/codegen/backtest/paper/dry-run/live." --created-at 2026-05-10T21:30:00+09:00
```

Artifact:

`registry\strategies\research_decisions\20260510T213000JST_tail_range_muted_return_exhaustion_btc_5m_taker\edge_discovery.json`

Result:

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

Interpretation: this bounded diagnostic also cleared normal and stress costs
but failed walk-forward and pair-dependence gates. Because it was derived after
the broad run timed out by narrowing the same range-efficiency idea, it must not
be used as a parameter-only retry path. It is recorded only as diagnostic
evidence for `no candidate generated`.

## Blocked And Stopped Runs

The broad ETH and combined BTC/ETH checks for the extreme intrabar range
efficiency thesis did not finish within the allowed local command timeout. They
were stopped after process inspection showed the Edge Discovery Python
processes were still running.

Blocked reason:

- event count and negative/shifted control evaluation were too heavy for this
  first local run;
- continuing by repeatedly changing only thresholds would violate the
  parameter-only retry and threshold-loosening constraints;
- no artifact was written for those timed-out ETH/combined runs, so they are
  blockers rather than evidence of a pass.

## Gate Decision

No thesis passed the calibrated research gate.

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

Candidate/proposal/codegen gates remained false in the generated artifacts.

Final result:

`no candidate generated`

## Generated Artifacts

Generated Edge Discovery artifacts remain local ignored artifacts under
`registry\strategies\research_decisions\...`. They are evidence for this run
but are not intended for Git.
