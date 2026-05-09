# Bot Factory Branch Decision

## Compared branches

- develop
- codex/bot-factory-candidate-factory-completion

## Summary of existing implementation on develop

`develop` contains the earlier Bot Factory skeleton, Phase 1/2/3 docs, FreqAI
checks, strategy proposal/code generation, candidate evaluation, walk-forward,
and paper-planning modules. It does not contain
`docs/BOT_FACTORY_GOAL_AUDIT.md`, and it lacks the newer research-selection,
causal failure map, failure synthesis, local event, local falsification,
Edge Discovery, signal diagnostics, and structural-data capability modules
needed by this research-first goal.

Relevant comparison evidence:

- `git show develop:docs/BOT_FACTORY_GOAL_AUDIT.md` failed because the audit
  doc is not present on `develop`.
- `git ls-tree -r --name-only develop freqtrade_ext/bot_factory scripts docs tests/test_bot_factory.py`
  showed Bot Factory baseline modules such as `candidate_evaluation.py`,
  `strategy_code.py`, `strategy_proposals.py`, and Phase runbooks, but not the
  newer research/failure-memory modules.

## Summary of existing implementation on codex/bot-factory-candidate-factory-completion

`codex/bot-factory-candidate-factory-completion` contains the current
32-candidate failure context and the infrastructure this goal must extend:
failure synthesis, causal failure maps, local event generation, local
falsification, Edge Discovery, research selection, rejection-memory handling,
signal diagnostics, structural-data quality/capability checks, and expanded
Bot Factory test coverage.

Relevant comparison evidence:

- `git show codex/bot-factory-candidate-factory-completion:docs/BOT_FACTORY_GOAL_AUDIT.md`
  succeeded and documents `candidate_count=32`, `paper_ready_count=0`,
  `walk_forward_failed_count=32`, `parameter_only_retry_allowed=false`,
  Edge Discovery/local falsification rejection memory, and safety boundaries.
- `git diff --stat develop..codex/bot-factory-candidate-factory-completion`
  reported 55 changed files with roughly 58k insertions over `develop`.
- `git diff --name-status develop..codex/bot-factory-candidate-factory-completion`
  showed the newly added Bot Factory modules and scripts required by the
  research-first path, including `candidate_failure_synthesis.py`,
  `candidate_failure_map.py`, `edge_discovery.py`, `local_events.py`,
  `local_falsification.py`, `research_selection.py`, and their CLI wrappers.

## Relevant files compared

- `docs/BOT_FACTORY_GOAL_AUDIT.md`
- `docs/BOT_FACTORY_MVP_TODO.md`
- `freqtrade_ext/bot_factory/candidate_failure_synthesis.py`
- `freqtrade_ext/bot_factory/candidate_failure_map.py`
- `freqtrade_ext/bot_factory/edge_discovery.py`
- `freqtrade_ext/bot_factory/local_events.py`
- `freqtrade_ext/bot_factory/local_falsification.py`
- `freqtrade_ext/bot_factory/research_selection.py`
- `freqtrade_ext/bot_factory/strategy_proposals.py`
- `scripts/bot_factory_build_edge_discovery.py`
- `scripts/bot_factory_build_local_events.py`
- `scripts/bot_factory_build_local_falsification.py`
- `scripts/bot_factory_select_research_thesis.py`
- `tests/test_bot_factory.py`

## Selected base branch

`codex/bot-factory-candidate-factory-completion`

## Selected PR target branch

`codex/bot-factory-candidate-factory-completion`

## Reason

This goal directly extends the prior branch's Research Lab, Edge Discovery,
failure-memory, causal-map, local-falsification, rejection-memory, and
parameter-only retry blocking infrastructure. Starting from `develop` would
require reimplementing or porting a large amount of that infrastructure before
the research-first cost, event-study, negative-control, and candidate-gate work
could be implemented. A stacked PR keeps the new research-first edge-gate
changes isolated from the already large candidate-factory completion diff.

## Risks

- The PR will depend on the prior candidate-factory branch landing or remaining
  available.
- The existing branch already has a large diff against `develop`, so reviewers
  must review this PR as a stack layer rather than a standalone change.
- Cost and fill assumptions remain research estimates until calibrated against
  real paper/live execution data; this PR must not imply paper or live
  readiness.

## What will not be included in this PR

- No generated strategy candidate unless a thesis passes all post-cost research
  gates.
- No parameter-only retry, threshold loosening, indicator variant farm, or
  FreqAI black-box retry.
- No generated backtest/cache/private-data artifacts.
- No paper, dry-run, live trading, exchange order endpoints, leverage changes,
  API key use, secret changes, or promotion workflow.
