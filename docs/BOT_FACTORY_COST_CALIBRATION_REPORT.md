# Bot Factory Cost Calibration Runner Report

## Scope

This report documents the local-only cost calibration runner added after the
post-merge verification and cost calibration plan.

Candidate generation result:

`no candidate generated`

No strategy candidate was generated. No research thesis was explored. No
backtest, paper, dry-run, live trading, exchange order endpoint, API-key,
secret, leverage, or shorting work is permitted by this runner.

## Implementation

- Module: `freqtrade_ext/bot_factory/cost_calibration.py`
- CLI: `scripts/bot_factory_calibrate_cost_model.py`
- Default output root: `data/cost_calibration`

The runner reads local artifacts only:

- OHLCV parquet/CSV/JSON for volatility and slippage estimates.
- Optional order-book parquet/CSV/JSON for spread and maker fill proxies.
- Optional spread parquet/CSV/JSON for direct spread bps estimates.
- Optional fills parquet/CSV/JSON for no-fill, partial-fill,
  adverse-selection, exit-taker, or explicit scenario cost overrides.

## Artifact Outputs

Each run writes:

- `cost_calibration.json`
- `cost_calibration_report.md`
- `cost_table.csv`

The JSON artifact always reports:

- `candidate_generation_allowed=false`
- `proposal_generation_allowed=false`
- `strategy_codegen_allowed=false`
- `candidate_generation_result=no candidate generated`

## Required Cost Context

Each artifact carries:

- `pair`
- `timeframe`
- `order_type`
- `liquidity_tier`
- `volatility_regime`

The same context is copied onto each `best`, `normal`, and `stress` scenario.

## Gate And Blocker Semantics

The runner returns a structured blocked artifact instead of raising when local
inputs are missing, malformed, or insufficient.

Blocking cases include:

- missing `best`, `normal`, or `stress` total cost;
- missing `normal` cost;
- `stress` total cost below `normal` total cost;
- maker context without no-fill, partial-fill, adverse-selection, or
  exit-taker fields for every scenario;
- local artifact parse errors;
- required OHLCV columns, rows, or usable numeric high/low/close values
  missing.

## Safety

The runner does not call candidate generation, proposal generation, strategy
code generation, backtesting, paper trading, dry-run trading, live trading, or
exchange order endpoints. It only converts local market-structure artifacts
into cost calibration artifacts and blockers.

## Current Outcome

This increment makes cost calibration executable from local data. It does not
approve any thesis or candidate.

Operational result:

`no candidate generated`

## Verification

Focused verification run:

```powershell
.\.venv\Scripts\python.exe -m py_compile freqtrade_ext/bot_factory/cost_calibration.py scripts/bot_factory_calibrate_cost_model.py tests/test_bot_factory.py
.\.venv\Scripts\python.exe -m pytest tests/test_bot_factory.py -q -k "cost_calibration or execution_quality or cost_model"
.\.venv\Scripts\python.exe -m pytest tests/test_bot_factory.py -q
git diff --check
```

Results:

- compile passed;
- focused selector passed 11 tests and reached `[100%]`;
- full `tests/test_bot_factory.py` reached `[100%]`;
- `git diff --check` exited `0` with no whitespace errors and the existing
  LF-to-CRLF working-copy warning for `docs/BOT_FACTORY_MVP_TODO.md`.
