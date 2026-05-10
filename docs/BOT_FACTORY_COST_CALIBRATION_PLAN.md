# Bot Factory Cost Calibration Plan

## Scope

This plan defines the next non-exchange-facing calibration work required before
any new research thesis evaluation can be trusted.

Candidate generation result:

`no candidate generated`

No thesis exploration starts in this plan. No strategy candidate is generated.
No paper, dry-run, live trading, exchange order, API-key, secret, leverage, or
shorting work is permitted by this document.

## Calibration Objective

The research-first gate already evaluates `best`, `normal`, and `stress` cost
scenarios. The next step is to make those scenarios empirically defensible from
local historical artifacts before using them to judge a new thesis.

The calibration must answer:

- whether the current `best`, `normal`, and `stress` bps assumptions are
  conservative enough for the target pair, timeframe, order type, liquidity
  tier, and volatility regime;
- how maker no-fill, partial-fill, adverse selection, exit taker conversion,
  spread widening, and volatility stress change expected post-cost edge;
- which assumptions are measured from local data and which remain explicit
  reviewer overrides.

## Required Cost Scenarios

Every calibrated thesis screen must carry these three scenarios:

| Scenario | Purpose | Minimum expectation |
| --- | --- | --- |
| `best` | Optimistic historical screen | Lower fee, spread, and slippage, but still non-zero unless a local artifact justifies zero. |
| `normal` | Primary research gate | Default decision scenario for `net_edge_bps_normal`; must include realistic fee, spread, slippage, and adverse-selection assumptions. |
| `stress` | Robustness gate | Must widen spread/slippage and include volatility stress; `net_edge_bps_stress` must remain positive before candidate generation is allowed. |

The existing compatibility field `all_in_cost_bps` should continue to map to
the `normal` total cost for older report consumers.

## Required Execution Risks

Calibration must explicitly model and report the following risks.

### Maker No-Fill

Maker entries can fail to execute. A maker cost scenario must record
`no_fill_rate`, the local evidence used to estimate it, and the fallback
behavior assumed by the event study.

Required treatment:

- estimate no-fill risk from local order-book depth, next-candle tradeability
  proxies, or a clearly documented conservative fallback;
- block thesis promotion when the edge depends on assuming perfect maker fills;
- report the expected event-count reduction or missed-opportunity penalty.

### Partial-Fill

Maker entries can fill only part of the intended stake.

Required treatment:

- record `partial_fill_rate` and the assumed average filled fraction;
- show how partial fills affect realized exposure and expected bps edge;
- treat missing partial-fill evidence as a stress-scenario penalty, not as zero
  risk.

### Adverse Selection

Passive fills can be more likely immediately before unfavorable price movement.

Required treatment:

- keep `adverse_selection_bps` as a first-class cost component;
- estimate it from post-fill next-candle movement, spread pressure, order-book
  imbalance, or conservative local proxies;
- reject any thesis whose edge disappears after adverse-selection cost is
  applied in `normal` or `stress`.

### Exit Taker Conversion

Risk exits may require taker execution even if entries are maker-style.

Required treatment:

- record `exit_taker_rate`;
- model stop, timeout, liquidity failure, and signal-invalidating exits as
  potential taker exits;
- include taker-fee and exit slippage penalties in `normal` and stronger
  penalties in `stress`.

### Spread Widening

Spreads can widen during local stress, event clusters, or volatility expansion.

Required treatment:

- estimate spread bps from local order-book artifacts when available;
- otherwise derive a conservative spread proxy from OHLCV range and liquidity
  state;
- apply wider spread assumptions in `stress` and document the multiplier.

### Volatility Stress

Higher realized volatility can increase slippage, adverse selection, and
exit-taker frequency.

Required treatment:

- classify volatility regimes from local closed-candle data;
- make `stress` cost at least a high-volatility scenario rather than a fixed
  multiplier only;
- block any thesis whose post-cost edge only survives low-volatility
  assumptions.

## Data Inputs

Allowed inputs:

- local OHLCV parquet files already present in the workspace;
- local order-book, funding, mark-price, open-interest, long/short-ratio, and
  liquidation artifacts when already available and quality-checked;
- local data-quality reports;
- local Edge Discovery, local-events, and local-falsification artifacts;
- sanitized reviewer configuration with no secrets.

Forbidden inputs and actions:

- exchange order endpoints;
- paper, dry-run, canary, or live bot processes;
- API keys, secrets, private environment values, or credential-like config;
- newly downloaded private datasets committed to Git.

## Calibration Procedure

1. Inventory the current `cost_model.py` defaults and every Edge Discovery spec
   override that can affect total cost.
2. For each target pair and timeframe, compute local spread, range, volume, and
   volatility summaries from closed-candle and market-structure artifacts.
3. Build a maker-risk summary with no-fill, partial-fill, and adverse-selection
   estimates or conservative fallback values.
4. Build an exit-quality summary with expected taker-exit rate, exit slippage,
   and stress-exit assumptions.
5. Produce `best`, `normal`, and `stress` scenario records with provenance for
   every fee, spread, slippage, adverse-selection, fill-risk, and volatility
   field.
6. Run Edge Discovery only in historical local mode against those scenarios.
7. Treat missing provenance as a blocker for candidate generation.

## Acceptance Criteria Before Thesis Evaluation

Before evaluating a new research thesis:

- the scenario file format and required fields are documented;
- `best`, `normal`, and `stress` totals are reproducible from local evidence or
  explicit conservative reviewer overrides;
- maker no-fill, partial-fill, adverse selection, exit taker conversion, spread
  widening, and volatility stress are present in the report;
- every calibrated scenario is free of secrets, API keys, exchange account
  data, and private environment values;
- generated calibration outputs are kept as local artifacts and are not
  committed unless they are sanitized documentation fixtures;
- `tests/test_bot_factory.py` and `git diff --check` pass after any code or doc
  change.

## Stop Conditions

Stop and report `no candidate generated` if:

- any cost component has no local evidence and no conservative fallback;
- the thesis edge is positive only under `best`;
- `normal` cost makes `net_edge_bps_normal < 6`;
- `stress` cost makes `net_edge_bps_stress <= 0`;
- maker no-fill, partial-fill, adverse selection, exit taker conversion, spread
  widening, or volatility stress is omitted from the decision artifact;
- artifact output would require committing generated cache, backtest output, or
  private data.

## Deliverables For The Calibration Increment

The future calibration increment should produce:

- a sanitized cost scenario specification or report;
- a provenance summary for every scenario component;
- focused tests for scenario construction and blocker behavior;
- an update to `docs/BOT_FACTORY_MVP_TODO.md` with exact commands and results.

It must still report `no candidate generated` unless a later, explicitly
requested research thesis passes the calibrated research gate.
