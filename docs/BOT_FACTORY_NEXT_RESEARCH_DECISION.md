# Bot Factory Next Research Decision

Checked on 2026-05-11 JST after the first calibrated Edge Discovery run.

Decision:

`no candidate generated`

Do not proceed to strategy candidate generation, strategy codegen, backtesting,
paper trading, dry-run trading, live trading, or exchange-facing order
endpoints.

## Decision Basis

The first calibrated Edge Discovery run used only the PR #10 cost contexts:

- BTC/USDT:USDT 5m taker: normal `9.469967` bps, stress `16.900396` bps.
- ETH/USDT:USDT 5m taker: normal `10.856899` bps, stress `20.874029` bps.

The only completed artifacts were BTC 5m taker local Edge Discovery checks.
Both completed checks failed the research gate due to insufficient
walk-forward robustness and single-pair dependence. No thesis produced
BTC/ETH-shared evidence that could rule out a one-pair accident.

## What Not To Do Next

- Do not generate a strategy candidate from either completed BTC artifact.
- Do not treat the positive BTC net edge as sufficient; it failed required
  robustness gates.
- Do not continue narrowing the same range-efficiency thesis by changing only
  thresholds.
- Do not reuse this as an old positive artifact for proposal generation.
- Do not use maker, 1h, large-alt, order-book, spread, or fills contexts.
- Do not run backtests, paper, dry-run, live trading, or exchange order
  endpoints.

## Allowed Next Research Shape

The next useful work is not another threshold retry. It should first improve
research infrastructure or define a materially different mechanism.

Acceptable next research setup:

- keep BTC/ETH 5m taker calibrated costs as the only execution context;
- preflight event counts before running Edge Discovery;
- cap or batch negative-control evaluation for broad event sets before
  evaluating another thesis;
- require BTC and ETH evidence in the same run or in symmetric pair-specific
  runs before any promotion discussion;
- select a mechanism class not present in the failed-family set and not a
  narrowed variant of `extreme_intrabar_range_efficiency_reversal` or
  `tail_range_muted_return_exhaustion`.

Blocked next actions:

- `strategy_candidate_generation`
- `strategy_codegen`
- `historical_backtest`
- `paper_trading`
- `dry_run_trading`
- `live_trading`
- `exchange_order_endpoint_use`
- `parameter_only_retry`
- `threshold_loosening`
- `old_positive_artifact_reuse`

Operational result:

`no candidate generated`
