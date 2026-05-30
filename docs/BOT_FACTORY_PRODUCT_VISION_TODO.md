# Bot Factory Product Vision TODO

Status: Product Vision roadmap Phases 0-6 implemented locally; paper, dry-run,
and live startup remain out of scope and require explicit future approval.

Updated: 2026-05-30 JST after Phase 5 diagnostic state discovery and Phase 6
paper observation design.

Intended path:

```text
docs/BOT_FACTORY_PRODUCT_VISION_TODO.md
```

## Goal

Freeze the product-level direction before adding more Bot Factory layers.

Bot Factory must not become an unbounded collection of strategy generators,
scorecards, gates, reports, readiness checks, and future paper/dry-run ideas.
Every future increment should be traceable to a small set of core contracts and
to one measurable North Star.

## One-Line Vision

Bot Factory is an evidence-governed, state-aware strategy selector that decides
whether to use a strategy or `no_trade` based on local as-of market state,
state-conditioned strategy evidence, cost/risk constraints, and readiness gates.

## Short Version

Bot Factory should answer:

```text
As of this local data timestamp:
  What market state are we in?
  Which strategies have strict evidence for this state and horizon profile?
  Which strategies are unsafe, stale, under-sampled, or out-of-scope?
  Should the system select a strategy or choose no_trade?
  What evidence and gate are required next?
```

The system should not be optimized to always trade. It should be optimized to
know when trading is unsupported.

## Non-Goals

Bot Factory is not:

- a universal single-strategy search engine;
- a recent-performance chaser;
- an automatic paper/dry-run/live launcher;
- an ML system that directly emits buy/sell or bot-switching commands;
- a mechanism to bypass historical, walk-forward, state-conditioned, cost,
  readiness, runtime-validation, or drift gates;
- a system that must select a strategy in every market state;
- a replacement for explicit human approval before any future paper/dry-run/live
  process.

## Safety Boundaries

Until a later explicitly approved phase, all work must remain local-only.

Do not:

- start `freqtrade trade`;
- start paper trading;
- start dry-run trading;
- start canary live or live trading;
- call exchange order endpoints;
- use API keys, secrets, private environment values, wallet/account state, or
  private exchange data;
- introduce leverage above `1.0`;
- introduce shorting;
- let selector artifacts start, stop, or switch bot processes;
- use recent paper/dry-run performance alone for adoption, readiness, or
  promotion.

## Core Product Contracts

Every core feature must belong to one of these contracts.

### 1. Market State Contract

Question:

```text
What does local data say the market state was as of timestamp T?
```

Responsibilities:

- build local-only, as-of market-state artifacts;
- support multi-horizon state, for example 5m / 15m / 1h / 4h / 1d / 1w;
- record confidence, uncertainty, stale-data status, horizon conflicts, and
  out-of-distribution indicators;
- default ambiguous, stale, conflicting, unknown, or out-of-distribution states
  to `no_trade`.

Outputs:

```text
market_state_snapshot.json
market_state_windows.jsonl
current_market_state.json
current_market_state_report.md
```

Does not answer:

```text
Which strategy should trade?
```

### 2. Strategy Evidence Contract

Question:

```text
Which strategies have evidence for which market states?
```

Responsibilities:

- convert checked local backtest / walk-forward / observation artifacts into
  observation ledgers and scorecards;
- evaluate strategies by state, horizon profile, pair/timeframe, and cost model;
- compare against no-trade, hold, incumbent, and style-specific baselines;
- preserve candidate identity and evidence lineage;
- keep proxy, relaxed-threshold, manually assembled, or incomplete artifacts
  diagnostic-only.

Outputs:

```text
observation_ledger.json
regime_fitness_scorecard.json
state_conditioned_scorecard.json
state_conditioned_scorecard_report.md
```

Does not answer:

```text
Should the current selector choose this strategy now?
```

### 3. Suitability / Matching Contract

Question:

```text
Given current local market state and strict strategy evidence, what action is allowed?
```

Responsibilities:

- build a strategy suitability matrix from strict state-conditioned scorecards;
- include `no_trade` as a first-class policy row;
- match current market identity against state, horizon profile, pair, timeframe,
  cost model, state encoder, and candidate identity;
- output `select_strategy`, `no_trade`, `shadow_only`, `watch`, `quarantine`,
  or `retire`;
- explain rejected alternatives and no-trade reasons.

Outputs:

```text
strategy_state_suitability_matrix.json
selector_matching_decision.json
selector_matching_report.md
no_trade_scorecard.json
```

Does not answer:

```text
Is this ready to start paper/dry-run/live?
```

### 4. Readiness / Runtime Governance Contract

Question:

```text
Is the selected candidate eligible for a later, tightly scoped paper-readiness path?
```

Responsibilities:

- verify local evidence completeness;
- join candidate identity across strategy source, generated metadata, historical
  metrics, walk-forward metrics, state scorecard, suitability matrix, and config
  where available;
- preserve no-startup semantics;
- gate future paper/dry-run observation design behind explicit approval;
- detect drift, quarantine unsafe candidates, and retire stale or invalidated
  evidence.

Outputs:

```text
paper_readiness.json
paper_readiness_report.md
drift_report.json
quarantine_report.json
retirement_report.json
```

Does not answer:

```text
Start a bot now.
```

### 5. Audit / Lineage Contract

Question:

```text
Can a reviewer trace every decision back to exact artifacts and versions?
```

Responsibilities:

- preserve `StrategyCandidateIdentity`;
- segment evidence by strategy, signal, risk policy, state encoder, regime
  classifier, cost model, pair/timeframe, and artifact hashes;
- fail closed on identity mismatch;
- produce reviewer-friendly Markdown reports and stable reason codes.

Outputs:

```text
candidate_identity.json
traceability.json
candidate_review_report.md
reason_codes
source_artifact_hashes
```

## Diagnostic Research Boundary

Anything that does not directly satisfy one of the five core contracts must be
treated as diagnostic research.

Examples:

- state clustering;
- analog-window search;
- ML state embeddings;
- strategy suitability ML;
- experimental leaderboards;
- future paper/dry-run observation proposals;
- dashboards;
- new strategy-family experiments.

Diagnostic research must default to:

```text
diagnostic_only = true
manual_review_only = true
selector_candidate_creation_allowed = false
paper_readiness_input_allowed = false
promotion_authorized_by_this_artifact = false
```

## North Star

The North Star is not the number of strategies generated.

The North Star is:

```text
Historical as-of selector replay can choose select_strategy or no_trade using
only information available at each decision timestamp, and can beat or justify
itself against simple baselines after cost, drawdown, exposure, opportunity cost,
and selector churn are considered.
```

Required baselines:

- always `no_trade`;
- always hold;
- best single eligible strategy;
- equal rotation;
- incumbent selector, when one exists.

Required replay metrics:

- net return after normal and stress cost;
- max drawdown;
- downside deviation;
- exposure ratio;
- turnover;
- selector churn;
- missed opportunity;
- no-trade quality;
- state coverage;
- unsupported-state rate;
- future-leakage checks;
- identity/cost/pair/timeframe mismatch checks.

## MVP Definition

Bot Factory MVP is not paper trading.

MVP is:

```text
A local historical as-of selector replay that proves whether the state-aware
selector/no_trade policy is useful compared with simple baselines, without
future leakage and without starting any trading process.
```

MVP acceptance criteria:

- [x] For each decision timestamp, only data available before that timestamp is
  used.
- [x] Market state is multi-horizon and local-as-of.
- [x] Strategy evidence is state-conditioned.
- [x] Selector can output `select_strategy` or `no_trade`.
- [x] `no_trade` records both loss avoidance and opportunity cost.
- [x] Decisions are compared with always-no-trade, hold, best-single, equal
  rotation, and incumbent baselines.
- [x] `future_data_used=true` fails validation.
- [x] Identity, pair, timeframe, cost-model, state-encoder, and strategy-version
  mismatches fail closed.
- [x] Markdown reports explain selected action, rejected alternatives, and next
  required gate.
- [x] No paper/dry-run/live/process-control action is started.

## Current State

Current capabilities:

- [x] Local-only market-state artifacts exist.
- [x] Multi-horizon market-state snapshot/current-state builders exist.
- [x] Deterministic regime/state labeling foundation exists.
- [x] Regime/state scorecard foundations exist.
- [x] Diagnostic-vs-selector-eligible boundary exists.
- [x] Strategy suitability matrix foundation exists.
- [x] Offline selector/no-trade matching foundation exists.
- [x] Paper-readiness optional inputs for state scorecard and suitability matrix
  exist.
- [x] Candidate identity and artifact lineage are first-class.
- [x] State-conditioned evidence can aggregate multiple historical/walk-forward
  windows by state scope while preserving source window lineage.
- [x] Historical as-of selector replay emits local decision JSONL and compares
  selector/no_trade decisions against simple baselines.
- [x] State-sliced strategy evaluation reports identify useful, unsafe,
  unsupported, missing, and state-crash evidence by state.
- [x] No-trade policy evaluation reports avoided drawdown, opportunity cost,
  uncertainty/OOD safety value, state-specific quality, and thresholded
  good/costly/acceptable/overused judgments.
- [x] Diagnostic state discovery emits diagnostic-only clustering, analog-window
  search, state embedding rows, suitability scoring rows, OOD/uncertainty
  calibration, deterministic-label comparison, and no-bypass gates.
- [x] Paper observation design emits future observation schema compatibility,
  evidence separation, drift, quarantine, retirement, and explicit-future-
  approval startup boundary artifacts without starting any process.

Current limitations:

- [x] State-conditioned evidence is robustly aggregated across many
  historical/walk-forward windows by state scope.
- [x] Historical as-of selector replay is implemented.
- [x] State-conditioned backtest/walk-forward reports are implemented as
  state-sliced evaluation artifacts.
- [x] No-trade / hold / incumbent baselines have state-sliced reporting and
  no-trade policy evaluation artifacts.
- [x] ML state discovery and suitability scoring are implemented as
  diagnostic-only artifacts and cannot bypass strict evidence gates.
- [x] Paper/dry-run/live startup remains out of scope and is explicitly blocked
  by Phase 6 paper observation design artifacts.

## Roadmap

### Phase 0: Product Vision Freeze

Goal:

```text
Stop uncontrolled architecture growth by defining the product vision,
contracts, North Star, and MVP.
```

TODO:

- [x] Add this document.
- [x] Link it from `docs/BOT_FACTORY_MVP_TODO.md`.
- [x] Add a contribution checklist requiring every Bot Factory PR to name its
  contract and North Star contribution.
- [x] Mark features outside the five contracts as diagnostic research by default.

Exit criteria:

- [x] Reviewers can classify every planned increment under one core contract or
  diagnostic research.
- [x] Future PRs can be rejected for not contributing to the North Star.

### Phase 1: Multi-Window State-Conditioned Evidence

Goal:

```text
Aggregate strategy evidence across many historical/walk-forward windows by
state scope without requiring all source rows to share the same state_window_id.
```

TODO:

- [x] Group scorecard evidence by:
  - strategy identity unit;
  - state_id;
  - horizon_profile_id;
  - state_encoder_version;
  - cost_model_id;
  - pair_group;
  - timeframe.
- [x] Preserve:
  - state_window_ids[];
  - decision_windows[];
  - feature_cutoff_range;
  - label_cutoff_range;
  - source_observation_count.
- [x] Fail closed when:
  - any `future_data_used=true`;
  - strategy identity differs;
  - cost model differs;
  - state encoder differs;
  - pair/timeframe is outside candidate identity.
- [x] Add tests for multi-window aggregation and mismatch failure.

Exit criteria:

- [x] Multiple historical/walk-forward windows can support one state-conditioned
  evidence row.
- [x] The evidence remains auditable and leakage-safe.

### Phase 2: Historical As-Of Selector Replay

Goal:

```text
Evaluate whether the selector/no-trade policy works when replayed through
history without future leakage.
```

TODO:

- [x] Build as-of market-state snapshots for historical decision timestamps.
- [x] Join only strategy evidence available before each timestamp.
- [x] Emit selector decisions as JSONL.
- [x] Compare against always-no-trade, hold, best-single, equal-rotation, and
  incumbent baselines.
- [x] Report net return, drawdown, exposure, turnover, missed opportunity,
  no-trade quality, and selector churn.
- [x] Add tests that leaked future state labels or future evidence are rejected.

Exit criteria:

- [x] The replay answers whether the selector adds value over simple baselines.
- [x] No paper/dry-run/live process is involved.

### Phase 3: State-Sliced Strategy Evaluation Reports

Goal:

```text
Make backtest/walk-forward reports answer where a strategy works, not only
whether it works globally.
```

TODO:

- [x] Add state-sliced backtest sections.
- [x] Add state-sliced walk-forward sections.
- [x] Add state coverage and missingness reports.
- [x] Add style-specific state gates.
- [x] Add baseline deltas per state:
  - no_trade;
  - hold;
  - incumbent;
  - style-specific baseline.
- [x] Reject positive global results that hide state-specific crashes.

Exit criteria:

- [x] A reviewer can identify useful, unsafe, unsupported, and unknown states
  for each strategy.

### Phase 4: No-Trade And Baseline Evaluation

Goal:

```text
Treat no_trade as a measurable policy, not just a safety fallback.
```

TODO:

- [x] Compute avoided drawdown.
- [x] Compute opportunity cost versus hold.
- [x] Compute opportunity cost versus best selector-eligible strategy.
- [x] Compute uncertainty/OOD safety value.
- [x] Report state-specific no-trade quality.
- [x] Add thresholds for acceptable opportunity cost by state type.

Exit criteria:

- [x] `no_trade` can be judged as good, costly, acceptable, or overused.

### Phase 5: Diagnostic State Discovery And ML

Goal:

```text
Use ML only as diagnostic support for state understanding and suitability
scoring, not as direct execution control.
```

TODO:

- [x] Build diagnostic-only state clustering.
- [x] Build analog-window search.
- [x] Build state embedding dataset.
- [x] Build suitability scoring dataset.
- [x] Add OOD / uncertainty calibration.
- [x] Compare ML diagnostics with deterministic state labels.
- [x] Keep all ML artifacts diagnostic-only until out-of-sample replay evidence
  beats deterministic baselines.

Exit criteria:

- [x] ML improves diagnosis or abstention without bypassing strict evidence
  gates.

### Phase 6: Paper Observation Design

Goal:

```text
Prepare future paper/dry-run observation without letting it override
historical/state-conditioned evidence.
```

TODO:

- [x] Define paper observation schema.
- [x] Require the same observation ledger schema.
- [x] Add drift reports.
- [x] Add quarantine and retirement triggers.
- [x] Keep startup behind explicit future approval.

Exit criteria:

- [x] Paper observation is additional evidence only, not direct promotion.

## PR Acceptance Checklist

Every future Bot Factory PR should answer:

```text
Which contract does this PR improve?
- Market State
- Strategy Evidence
- Suitability / Matching
- Readiness / Runtime Governance
- Audit / Lineage
- Diagnostic Research only

How does this PR move us toward the North Star?
What does this PR explicitly not permit?
Does this PR preserve no_trade as a first-class output?
Does this PR prevent future leakage?
Does this PR preserve candidate identity and evidence lineage?
Does this PR avoid paper/dry-run/live/process-control startup?
What baselines or reports become more informative because of this PR?
```

PRs that cannot answer these questions should remain diagnostic-only or should
not be merged into the core Bot Factory path.

## Suggested Immediate Next PR

Status: completed by the 2026-05-30 JST docs-only integration; Product Vision
Phases 0-6 have since been implemented locally.

The next PR should be documentation-only:

```text
PR: Add Bot Factory product vision / North Star TODO
```

Scope:

- add this document;
- link it from existing Bot Factory TODO docs;
- add the PR acceptance checklist;
- do not implement new logic;
- do not run backtests;
- do not start paper/dry-run/live;
- do not generate strategies.

Verification:

```text
git status --short --untracked-files=all
git diff --check
markdown/link review
docs-only diff review
```

## Open Questions

- Should the MVP baseline include buy-and-hold per pair, portfolio hold, or both?
- What decision frequency should the first as-of selector replay use?
- Which horizons are mandatory for MVP: 5m/1h/1d, or all configured horizons?
- What minimum state coverage is required before a strategy can be considered
  state-selector-eligible?
- What opportunity cost is acceptable for no-trade in uncertain states?
- When there is no incumbent, what should `incumbent` baseline mean?
- Should BTC/ETH global state become a separate context for all pair selectors?
