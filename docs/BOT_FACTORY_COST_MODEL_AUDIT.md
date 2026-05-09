# Bot Factory Cost Model Audit

## Purpose

The research-first gate no longer treats `all_in_cost_bps=12.0` as the only
cost assumption. Edge Discovery now evaluates post-cost edge against explicit
`best`, `normal`, and `stress` scenarios before candidate generation can be
considered.

## Implementation

- Module: `freqtrade_ext/bot_factory/cost_model.py`
- Integration: `freqtrade_ext/bot_factory/edge_discovery.py`
- CLI: `scripts/bot_factory_build_edge_discovery.py`

Each scenario records:

- `scenario_name`
- `fee_bps_entry`
- `fee_bps_exit`
- `spread_bps`
- `slippage_bps_entry`
- `slippage_bps_exit`
- `adverse_selection_bps`
- `no_fill_rate`
- `partial_fill_rate`
- `exit_taker_rate`
- `stress_multiplier`
- `total_cost_bps`

The selector accepts pair, timeframe, order type, liquidity tier, and volatility
regime context. Scenario overrides can be supplied through an Edge Discovery
spec `cost_model.overrides[]`, with the matching override selected before
horizon scoring.

## Defaults

- `best`: lower fee/spread/slippage assumptions for optimistic historical
  screening.
- `normal`: initialized to the legacy `all_in_cost_bps=12.0` only when the
  top-level value is absent or unparseable. An explicit
  `all_in_cost_bps=0` is preserved for tests and does not fall back to 12.0.
- `stress`: fee/slippage/adverse-selection stack multiplied by at least 1.5x
  under the default model.

Maker-style contexts must carry no-fill, partial-fill, and adverse-selection
fields. These are reported even when they are not included in a simple bps
return calculation, so maker assumptions remain visible to reviewers.

## Gate Use

Edge Discovery still reports the legacy compatibility field
`all_in_cost_bps`, mapped to the `normal` scenario total. New reports also
include:

- `cost_bps_best`
- `cost_bps_normal`
- `cost_bps_stress`
- `net_edge_bps_best`
- `net_edge_bps_normal`
- `net_edge_bps_stress`

Candidate generation remains blocked unless the stricter research gate passes,
including `net_edge_bps_normal >= 6` and `net_edge_bps_stress > 0`.

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

It did not reach execution because collection requires optional dependencies
that are not installed in the local venv: `freqtrade_client` and `optuna`.

## Limitations

- Cost scenarios are research estimates and still require paper/live execution
  calibration before any promotion decision.
- Fill-risk fields are reported and gated for maker assumptions, but maker
  fill risk is not yet a dedicated fill-probability model or standalone
  promotion gate. No live order fill simulation or exchange-facing process is
  started.
- Passing this cost model is not evidence of live profitability.
