# Bot Factory Edge Discovery Report

## Scope

This report documents the research-first Edge Discovery gate added for the
current implementation PR. It is a gate and reporting implementation, not a
new strategy candidate.

Candidate generation result for this PR:

`no candidate generated`

No new thesis was promoted to strategy code generation during this work. That
is intentional: candidate generation is allowed only when the post-cost
research gate passes.

## Semantics Alignment

Edge Discovery horizon scoring now uses Freqtrade-compatible local event-study
semantics:

- Signals are selected from closed candles.
- Entry is evaluated from the next candle open.
- Exit is evaluated from the close of the held candle. With
  `entry_semantics=next_candle_open` and `hold_candles=1`, the event return is
  next-candle open to that same next candle close.
- Funding, mark price, open interest, long/short ratio, liquidation, and
  order-book context continue to use closed-context alignment and local
  timestamped artifacts only.
- The report records `entry_semantics=next_candle_open` in horizon samples and
  checks it in the research gate.

## Event-Level Metrics

Each Edge Discovery artifact now emits an `event_level_post_cost_edge_report`
with the required fields:

- `thesis_id`
- `mechanism_class`
- `event_count`
- `entry_signal_count`
- `gross_edge_bps`
- `cost_bps_best`
- `cost_bps_normal`
- `cost_bps_stress`
- `net_edge_bps_best`
- `net_edge_bps_normal`
- `net_edge_bps_stress`
- `profitable_windows_ratio`
- `walk_forward_pass_rate`
- `lower_confidence_bound_bps`
- `pair_concentration`
- `pair_evidence_count`
- `pair_evidence_unique_count`
- `pair_evidence_distribution`
- `calendar_concentration`
- `holding_period`
- `negative_control_random_entry_delta_bps`
- `negative_control_shuffled_signal_delta_bps`
- `negative_control_shifted_signal_delta_bps`
- `passes_research_gate`
- `rejection_reason`

`walk_forward_pass_rate` is reported from calendar-window profitability inside
the local event study. It remains a pre-codegen research robustness proxy, not a
replacement for full strategy walk-forward evaluation.

`pair_concentration` is computed only from actual event-return evidence rows
that carry `pair`, `symbol`, or `instrument` evidence. Declaring multiple
symbols in `instrument_universe` does not make the signal multi-pair. If event
rows carry no pair evidence, the gate treats the evidence as single-pair
dependent with `pair_concentration=1.0`.

## Negative Controls

The gate computes three controls per horizon:

- Random entry control with the same approximate event count and holding period.
- Shuffled signal control over the same eligible candle set.
- Shifted signal control using past and future shifts; the stronger shifted
  result is used as the leakage/alignment challenge.

A thesis is rejected when the real signal does not beat each control by the
configured minimum delta.

## Research Gate

Candidate generation requires all checks to pass:

- `net_edge_bps_normal >= 6`
- `net_edge_bps_stress > 0`
- `profitable_windows_ratio >= 0.7`
- `walk_forward_pass_rate >= 0.6`
- `lower_confidence_bound_bps > 0`
- not single-pair dependent
- not single-calendar-window dependent
- random, shuffled, and shifted controls beaten
- next-candle-open Freqtrade semantics verified

If any check fails, the artifact reports `candidate_generation_result` as
`no candidate generated`. `proposal_generation_allowed` is tied to
`candidate_generation_allowed`; a status-passed artifact with a failed
research gate cannot advance proposal or candidate generation through the
legacy proposal flag.

## Current Limitations

- `maker_fill_risk`, `no_fill_rate`, and `partial_fill_rate` are exposed in
  cost/report fields, but maker fill risk is not yet a dedicated
  fill-probability gate. This remains a follow-up TODO before promotion.
- Overlapping events, cooldown interaction, and effective sample count are
  reported only indirectly through event counts and cooldown settings. A
  follow-up diagnostic should estimate effective independent sample count and
  flag overlapping holding windows.

## Current Outcome

The implementation PR does not assert that any real market thesis passed these
gates. It only adds the gate, reports, and tests. The correct operational
outcome is:

`no candidate generated`

## Verification

Focused verification run:

```powershell
.\.venv\Scripts\python.exe -m py_compile freqtrade_ext\bot_factory\cost_model.py freqtrade_ext\bot_factory\edge_discovery.py freqtrade_ext\bot_factory\local_falsification.py scripts\bot_factory_build_edge_discovery.py
.\.venv\Scripts\python.exe -m pytest tests\test_bot_factory.py -q -k "cost_model or next_candle_open or research_gate or negative_controls"
.\.venv\Scripts\python.exe -m pytest tests\test_bot_factory.py -q -k "edge_discovery"
.\.venv\Scripts\python.exe -m pytest tests\test_bot_factory.py -q
.\.venv\Scripts\python.exe -m py_compile freqtrade_ext\bot_factory\cost_model.py freqtrade_ext\bot_factory\edge_discovery.py freqtrade_ext\bot_factory\local_falsification.py scripts\bot_factory_build_edge_discovery.py tests\test_bot_factory.py
git diff --check
```

Results: compile passed; focused cost/gate tests passed 7 tests; Edge Discovery
focused tests passed 16 tests; full `tests\test_bot_factory.py` passed and
reached `[100%]`; final compile passed; `git diff --check` passed with the
existing LF-to-CRLF working-copy warning for `docs/BOT_FACTORY_MVP_TODO.md`.
After final candidate-generation gate tightening, the combined focused selector
passed 20 tests and full `tests\test_bot_factory.py` passed again.

The requested full repository command was also attempted:

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
```

It stopped during collection because the local venv is missing
`freqtrade_client` and `optuna`.

## Safety

No backtest, paper, dry-run, live trading, order placement, API-key access,
secret change, leverage change, or exchange-facing process was started for this
report.
