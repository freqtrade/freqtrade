# Bot Factory Execution Quality Audit

## Scope

Checked on 2026-05-10 JST as a post-merge documentation audit.

Candidate generation result:

`no candidate generated`

This audit does not create a strategy candidate, start thesis exploration, run
historical backtesting, start paper/dry-run/live trading, call exchange order
endpoints, use API keys, change secrets, increase leverage, or enable shorting.

## Current Execution-Quality State

The merged research-first pipeline is safer than the earlier candidate-first
flow because proposal and candidate generation are blocked unless the Edge
Discovery research gate passes. That gate is necessary but not sufficient for
execution readiness.

Confirmed strengths:

- Candidate generation remains tied to post-cost research-gate success.
- `proposal_generation_allowed` is tied to `candidate_generation_allowed` in
  Edge Discovery artifacts.
- Legacy proposal-flag bypass attempts are covered by tests.
- Event-study semantics use closed-candle signal selection with next-candle-open
  entry assumptions.
- Negative controls and robustness checks are part of the research gate.
- Current PR #7 review threads are resolved after follow-up commits.

Primary execution-quality gap:

- Cost scenarios expose maker fill risk and stress fields, but a calibrated
  execution-quality model has not yet been produced from local market-structure
  evidence.

## Risk Audit

### Order-Type Assumption Risk

Research artifacts can state maker-style assumptions, but maker execution is
not guaranteed. A thesis that only works with perfect maker fills should remain
blocked.

Audit requirement:

- every thesis screen must say whether entry and exit assumptions are maker,
  taker, or mixed;
- mixed assumptions must state when exits become taker;
- maker assumptions must carry no-fill, partial-fill, and adverse-selection
  penalties.

### Maker No-Fill Risk

No-fill risk can reduce event count and realized opportunity quality.

Audit requirement:

- report `no_fill_rate`;
- show whether the result still passes after missed maker fills are penalized;
- block the thesis if the post-cost edge depends on all maker orders filling.

### Partial-Fill Risk

Partial fills can reduce exposure without reducing all fixed decision costs.

Audit requirement:

- report `partial_fill_rate` and filled-fraction assumptions;
- adjust expected edge or effective sample contribution;
- treat missing partial-fill evidence as a blocker or stress penalty.

### Adverse-Selection Risk

Passive fills can select for worse short-term price paths.

Audit requirement:

- report `adverse_selection_bps`;
- estimate it from local post-fill movement or conservative proxies;
- require `normal` and `stress` edge to survive adverse-selection penalties.

### Exit Taker Risk

Signal failure, timeout, stop behavior, and liquidity stress can force taker
exits.

Audit requirement:

- report `exit_taker_rate`;
- include taker fee and exit slippage in cost scenarios;
- reject theses where exit taker conversion is omitted or only considered in
  narrative text.

### Spread-Widening Risk

Spreads widen in stress, especially when signals are tied to volatility or
liquidity events.

Audit requirement:

- derive spread estimates from local order-book artifacts when available;
- otherwise use conservative OHLCV/range proxies;
- require stress spread widening in the `stress` scenario.

### Volatility-Stress Risk

High volatility can raise slippage, widen spreads, worsen adverse selection,
and increase forced exits.

Audit requirement:

- classify low, normal, and high volatility regimes from local data;
- run or simulate the `stress` scenario against high-volatility assumptions;
- block any thesis whose edge only survives calm-market execution assumptions.

### Data Provenance Risk

Execution-quality artifacts are only useful if their provenance is local,
sanitized, and reproducible.

Audit requirement:

- record source artifact paths and generation commands;
- never include API keys, secrets, private environment values, exchange account
  data, or private datasets in committed reports;
- keep generated cache, backtest output, and calibration artifacts out of Git
  unless explicitly sanitized and intended as documentation fixtures.

## Prerequisites Before The Next Research Thesis

Do not evaluate a new research thesis until these prerequisites are met:

- `develop` is current and the worktree is clean except for intentional docs or
  code edits;
- PR #7 and PR #8 merged state remains present in `develop`;
- the latest failure synthesis, causal failure map, rejection memory, and
  research-first gate docs are reviewed;
- cost calibration covers `best`, `normal`, and `stress`;
- maker no-fill, partial-fill, adverse selection, exit taker conversion, spread
  widening, and volatility stress are explicit decision fields;
- local data-quality checks pass for every structural data family used;
- Edge Discovery artifacts continue to report `candidate_generation_result`;
- `proposal_generation_allowed` and `candidate_generation_allowed` cannot be
  true when `research_gate.passes_research_gate` is false;
- all new artifacts are sanitized and excluded from Git unless they are
  deliberate documentation updates;
- focused `py_compile`, `tests/test_bot_factory.py -q`, and `git diff --check`
  pass for the change set.

## Current Recommendation

The next useful increment is cost calibration and execution-quality evidence,
not thesis exploration and not candidate generation.

Until that calibration exists, the correct operational result remains:

`no candidate generated`
