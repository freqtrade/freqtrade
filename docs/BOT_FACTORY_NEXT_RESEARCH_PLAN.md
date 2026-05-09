# Bot Factory Next Research Plan

## Candidate Generation Result

`no candidate generated`

No thesis in this implementation PR was promoted to strategy candidate
generation. This is a successful rejection state unless a thesis passes the
research gate with post-cost, negative-control, and robustness evidence.

## Required Inputs Before Any Next Thesis

The next research step must consume the latest available:

- failure synthesis
- causal failure map
- rejection memory
- validated local falsification or Edge Discovery rejection memory
- local data quality reports for any structural data family used

Past failed thesis IDs and mechanism classes must not be reused as a disguised
retry.

## Thesis Selection Limit

At most three research theses may be evaluated before a fresh synthesis update.
Preferred categories are:

- low-turnover / high-timeframe regime strategy
- no-trade filter as alpha
- liquidation / forced-flow rare events
- funding / basis as carry, not directional alpha
- execution alpha with conservative maker fill model

Each thesis must record a causal hypothesis, required local data, expected edge
source, cost exposure, falsification criteria, and the reason it is outside the
known failed mechanism classes.

## Stop Conditions

Stop before proposal or code generation when any of the following is true:

- `net_edge_bps_normal <= 0`
- gross edge does not exceed normal cost
- stress cost makes edge negative
- `profitable_windows_ratio < 0.7`
- `walk_forward_pass_rate < 0.6`
- lower confidence bound is not positive
- random, shuffled, or shifted controls are not beaten
- signal semantics do not match next-candle-open Freqtrade evaluation
- result depends on one pair or one calendar window
- the thesis repeats a failed thesis ID or mechanism class
- the path is parameter-only retry, threshold loosening, indicator variant
  farming, or FreqAI black-box retry

When a stop condition applies, report:

`no candidate generated`

## Paper Promotion Checklist

Do not mark a candidate paper-ready unless all conditions below are satisfied:

- post-cost edge is positive
- `net_edge_bps_normal >= 6`
- preferably `net_edge_bps_normal >= 12`
- stress-cost edge remains positive
- `profitable_windows_ratio >= 0.7`
- `walk_forward_pass_rate >= 0.6`
- lower confidence bound is positive
- random, shuffled, and shifted negative controls are beaten
- lookahead, recursive, and semantics alignment checks are clean
- result is not single-pair or single-month dependent
- risk overlay preserves expectancy
- paper or dry-run observation has 30 to 60 days or enough trades before any
  live path is considered

Passing a research gate does not imply live profitability. It only permits the
next local, historical, non-exchange-facing step.

## Prohibited Work

- no 33rd candidate smoke by default
- no parameter-only tuning retry
- no threshold loosening
- no indicator variant farm
- no FreqAI black-box retry
- no DCA or martingale rescue
- no leverage-based cosmetic return improvement
- no codegen from local-screen positivity alone
- no reuse of stale positive artifacts
- no generated cache, backtest output, or private dataset committed to Git
- no paper, dry-run, live trading, exchange order endpoint, API key, or secret
  work
