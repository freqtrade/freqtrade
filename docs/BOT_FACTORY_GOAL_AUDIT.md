# Bot Factory Goal Audit

Checked on 2026-05-07 JST after the refreshed 32-candidate comprehensive
failure synthesis, causal map, and recent local falsification rejection memory.

## Objective Restatement

The project goal is not simple parameter optimization. The target is an
AI-assisted Bot Factory that can research a falsifiable trading theory, convert
it into safe strategy code, evaluate it on local historical artifacts, rank or
reject it, and feed failures back into future theory/code generation until a
profitable, robust, paper-ready candidate exists.

## Success Criteria

- Theory-first research intake exists and requires structured references,
  thesis identity, falsification criteria, and novelty against prior failures.
- Code generation converts approved theory artifacts into safe long-only
  rule-based, FreqAI, or hybrid ML+rule candidates without secrets, shorting,
  live data, order endpoints, or leverage above `1.0`.
- Historical evaluation, walk-forward, diagnostics, ranking, synthesis, and
  local manifests exist as source-of-truth artifacts.
- The factory can reject weak theories and stop parameter-only retries.
- At least one candidate must pass historical and walk-forward gates before
  paper readiness or promotion can be considered.
- Paper/dry-run/live startup remains blocked unless explicitly requested and
  all readiness gates are passing.

## Evidence Checklist

- Structured theory/proposal path: implemented in
  `freqtrade_ext/bot_factory/strategy_proposals.py`,
  `scripts/bot_factory_generate_strategy_proposal.py`, and proposal metadata
  under `registry/strategies/proposals/`.
- Strategy code generation: implemented in
  `freqtrade_ext/bot_factory/strategy_code.py` and
  `scripts/bot_factory_generate_strategy_code.py`; supports multiple
  hypothesis-family variants and generator modes.
- Local evaluation and registry: implemented in
  `freqtrade_ext/bot_factory/candidate_evaluation.py`,
  `freqtrade_ext/bot_factory/candidate_ranking.py`, and artifacts under
  `registry/strategies/candidates/`.
- Failure synthesis and negative knowledge base: implemented in
  `freqtrade_ext/bot_factory/candidate_failure_synthesis.py`,
  `scripts/bot_factory_synthesize_candidate_failures.py`, and latest synthesis
  `registry/strategies/synthesis/20260507T110000Z_all_candidates_with_comprehensive_local_rejection_memory/candidate_failure_synthesis.json`.
- Pre-proposal research selection gate: implemented in
  `freqtrade_ext/bot_factory/research_selection.py` and
  `scripts/bot_factory_select_research_thesis.py`. It emits local
  `research_decision.json` / `research_decision_report.md` artifacts and
  blocks repeated failed thesis IDs, repeated failed hypothesis-family tokens,
  stale research-reference mappings, missing local falsification data, missing
  causal-failure-map responses when a map is supplied, thin causal responses,
  parameter-only causal responses, mismatched map/synthesis identity, and
  unsafe dependencies before proposal generation. It now requires responses to
  the top three dominant causal failure categories plus any material category
  covering at least 70% of failed candidates. It also emits
  `research_selection_score` and blocks below-map-minimum decisions so a long
  but weak research proposal cannot advance on field presence alone. Latest
  maps also require explicit responses to every required research question,
  preventing the map's questions from being informational only. For
  high-risk `cost_sensitive_mechanism` responses, the gate now also requires
  quantified cost/edge evidence rather than qualitative fee-drag claims alone.
  It also accepts `--local-falsification-json` and, when that cost category has
  `risk_score >= 80`, blocks text-only cost claims unless the local JSON
  artifact matches the current thesis ID and shows expected edge exceeding
  all-in cost in bps with sufficient sample count. The artifact must also come
  from the Bot Factory local falsification generator
  (`factory=research_local_falsification`) and preserve historical-only safety
  scope, and its `event_source` must link back to a completed Bot Factory
  local event builder artifact whose failure-synthesis guard consumed the
  current failure synthesis and did not repeat a failed thesis ID or mechanism
  family. A crafted JSON with matching numbers, or a local falsification
  artifact sourced from an unguarded event build, is not enough. It also
  accepts `--prior-local-falsification-json` so failed or rejected
  pre-proposal falsification artifacts that match the current `thesis_id` or
  `mechanism_class` block repeated research selection before proposal/codegen.
- Local falsification evidence generation: implemented in
  `freqtrade_ext/bot_factory/local_falsification.py` and
  `scripts/bot_factory_build_local_falsification.py`. It consumes local
  closed-candle OHLCV and a local event timestamp file, optionally verifies the
  originating `local_events.json` through `--event-source-json`, then writes
  `local_falsification.json` / `local_falsification_report.md` with
  expected edge bps, all-in cost bps, net edge bps, sample count, win rate, and
  split-window profitability evidence for the research gate. It now also
  records OHLCV row count, `data_start`, `data_end`, `data_span_days`, and
  `min_data_span_days`; `--min-data-span-days` blocks short-history
  falsification artifacts before they can support a high-risk cost claim. It
  is local artifact generation only, not strategy code generation or
  backtesting.
- Local event timestamp generation: implemented in
  `freqtrade_ext/bot_factory/local_events.py` and
  `scripts/bot_factory_build_local_events.py`. It consumes local closed-candle
  OHLCV plus optional local funding-rate and mark-price files, together with a
  `factory=research_local_event_spec` JSON with explicit AND conditions over
  supported past/current features (`hour_utc`, `weekday`, `return_bps`,
  `range_pct`, `sma_distance_bps`, `volume_zscore`, `funding_rate_bps`,
  `funding_rate_delta_bps`, `mark_price_gap_bps`,
  `mark_price_gap_delta_bps`, and `mark_price_return_bps`), then writes local
  `events.csv`,
  `local_events.json`, and `local_events_report.md` artifacts for local
  falsification. The JSON records the emitted `events.csv` path so local
  falsification evidence can prove that its event timestamps came from this
  builder. Funding and mark context is joined by backward as-of merges only,
  and artifacts record auxiliary sources, per-condition diagnostics, and
  cumulative condition match counts. It can also consume the latest
  `candidate_failure_synthesis.json` and block event specs whose `thesis_id` or
  `mechanism_class` repeats a failed thesis/family unless explicitly allowed
  for diagnostics. It performs no parameter search or optimization and blocks
  unsupported future-return or unrecognized features.
- Proposal-generator enforcement of that gate: implemented in
  `freqtrade_ext/bot_factory/strategy_proposals.py` and
  `scripts/bot_factory_generate_strategy_proposal.py`. If failure synthesis
  requires a new thesis/research references, proposal acceptance now requires
  a matching approved `research_decision.json`; blocked or missing research
  decisions prevent `code_generation_eligible=true`. Proposal generation also
  verifies that the research decision consumed a causal failure map and that
  causal response coverage/quality did not report missing, weak, evidence-gap,
  or parameter-only categories. It now also recomputes the current
  dominant/material category policy from the decision's causal-map summary and
  blocks crafted decisions that omit expected material categories from the
  claimed required categories or response categories. It also blocks missing
  or below-minimum `research_selection_score` evidence, so the pre-proposal
  score cannot be bypassed by directly supplying a crafted research decision
  to proposal generation. It also blocks missing or weak required-question
  responses at proposal stage. For causal maps that include
  `causal_risk_weights`, proposal generation now also requires
  `research_selection_score_v2` weighted causal score details, preventing an
  older or crafted score payload from bypassing risk-weighted research
  selection.
- Causal failure mapping: implemented in
  `freqtrade_ext/bot_factory/candidate_failure_map.py` and
  `scripts/bot_factory_build_causal_failure_map.py`. Latest artifact
  `registry/strategies/failure_maps/20260507T110500Z_all_candidates_with_comprehensive_local_rejection_memory_causal_map/causal_failure_map.json`
  classifies all 32 candidate failures into signal sparsity, negative edge,
  walk-forward fragility, cost sensitivity, ML/rule alignment failure, and
  thesis rejection categories before the next thesis is selected. The research
  selection gate now consumes this map through `--causal-failure-map-json` and
  requires explicit `--causal-failure-response` entries for the top dominant
  categories before proposal generation can be approved. The gate also records
  response word counts, category evidence gaps, and parameter-only response
  categories. The latest map adds `minimum_research_selection_score=80`, a
  rubric for novelty, structured references, local falsification data, causal
  failure response quality, and mechanism/falsification substance, plus
  prevalence/severity-based `causal_risk_weights` for research scoring. It
  also sets `requires_research_question_responses=true`.
- Earlier directional-change research decision:
  `registry/strategies/research_decisions/20260507T054500JST_directional_change_overshoot_research_gate/research_decision.json`
  approved `TH-DIRECTIONAL-CHANGE-OVERSHOOT-20260507` for proposal generation
  only after consuming the refreshed twenty-four-family synthesis/map and
  answering all five dominant/material causal categories. This was permission
  to build an accepted proposal path, not evidence of profitability or paper
  readiness.
- Directional-change accepted proposal for that decision:
  `registry/strategies/proposals/20260506T205000Z_LongOnlyDirectionalChangeOvershootCandidate.metadata.json`
  has `status=accepted`,
  `strategy_logic_variant=directional_change_overshoot`, three research
  references, and matching 24-family synthesis/research decision evidence.
- Directional-change code/evaluation result:
  `registry/strategies/generated/LongOnlyDirectionalChangeOvershootCandidate/20260507T061500JST_directional_change_overshoot_codegen/metadata.json`
  generated an evaluation-eligible long-only strategy with static check
  `ok=true`, and signal diagnostics reported `entry_count=706`. Historical
  backtest failed with `total_return_pct=-3.2605379719999994`,
  `profit_factor=0.3464660896687346`, and
  `sortino=-93.24140769163247`. Walk-forward completed all four windows but
  failed with `pass_rate=0.0`, `profitable_windows_ratio=0.0`, and
  `total_return_pct=-3.24521717`.
- Current profitability evidence: not achieved. Latest synthesis has
  `candidate_count=32`, `paper_ready_count=0`, `zero_trade_count=7`,
  `negative_return_count=22`, `walk_forward_failed_count=32`,
  `parameter_only_retry_allowed=false`, and
  no paper-ready candidate.
- Latest research-gate discipline evidence:
  `registry/strategies/research_decisions/20260507T053000JST_material_category_gate_top3_only_block_smoke/research_decision.json`
  supplied the refreshed 24-family synthesis/map and only the top-three causal
  responses. It was blocked with
  `causal_failure_responses_cover_required_categories` because the material
  categories `no_profitable_walk_forward_windows` and
  `entry_exists_negative_edge` were missing. This confirms the next thesis
  cannot ignore broad no-profit and entry-negative-edge evidence while still
  reaching proposal generation.
- Latest no-fallback codegen discipline evidence:
  `registry/strategies/generated/LongOnlyDirectionalChangeOvershootCandidate/20260507T055500JST_directional_change_overshoot_codegen_block/metadata.json`
  deliberately blocked before support was implemented with blocker
  `strategy_logic_variant_supported`, so there was no silent fallback from
  directional-change overshoot to an existing family. The later implemented
  directional-change path then failed historical and walk-forward gates.
- Earlier post-map theory intake evidence:
  `registry/strategies/research_decisions/20260507T082500JST_range_quarticity_vol_of_vol_research_gate/research_decision.json`
  approved the distinct `range_quarticity_vol_of_vol_state` thesis for
  proposal generation with `research_selection_score=100.0`,
  `minimum_score_required=80.0`, and complete required-question responses.
  The accepted proposal was then explicitly code-generated, diagnosed,
  backtested, and walk-forward evaluated. Latest generated metadata is
  `registry/strategies/generated/LongOnlyRangeQuarticityVolOfVolCandidate/20260507T091000JST_range_quarticity_vol_of_vol_codegen/metadata.json`
  with `status=generated` and `static_check_ok=true`; diagnostics reported
  `entry_count=250`. Historical return was negative
  (`total_return_pct=-0.8648079839999998`, `trade_count=92`,
  `profit_factor=0.5266356841671438`), and walk-forward failed all four
  windows with `pass_rate=0.0`, `profitable_windows_ratio=0.0`, and
  `total_return_pct=-0.8648079839999999`.
- Latest aggregate failure evidence:
  `registry/strategies/synthesis/20260507T110000Z_all_candidates_with_comprehensive_local_rejection_memory/candidate_failure_synthesis.json`
  has `candidate_count=32`, `paper_ready_count=0`, `zero_trade_count=7`,
  `negative_return_count=22`, `walk_forward_failed_count=32`,
  `parameter_only_retry_allowed=false`,
  `local_falsification_rejection_count=10`, and
  `local_falsification_invalid_rejection_count=1`. The latest causal map is
  `registry/strategies/failure_maps/20260507T110500Z_all_candidates_with_comprehensive_local_rejection_memory_causal_map/causal_failure_map.json`
  with `candidate_count=32`, `category_count=8`,
  `requires_research_question_responses=true`,
  `minimum_research_selection_score=80`, 26 required research questions, and
  10 validated local falsification rejection contexts.
- Latest volatility-managed momentum local screen:
  `TH-VOL-MANAGED-MOMENTUM-20260507` /
  `volatility_managed_momentum_state` was screened with a fixed local event
  spec before proposal or code generation. The first one-month event builder
  completed with `event_count=547`, and the linked local falsification failed:
  `sample_count=546`, `expected_edge_bps=0.189233`,
  `all_in_cost_bps=12.0`, `net_edge_bps=-11.810767`, and
  `profitable_windows_ratio=0.0`. After BTC 5m data was expanded, the exact
  same fixed spec was re-run with the data-span gate: event generation
  produced `event_count=6965`, and local falsification failed again with
  `sample_count=6964`, `data_span_days=397.222222`,
  `expected_edge_bps=0.333213`, `all_in_cost_bps=12.0`,
  `net_edge_bps=-11.666787`, and `profitable_windows_ratio=0.0`. Event-source
  and data-span checks passed; cost-edge and window-stability checks failed.
  This is rejection evidence, not promotion evidence. The research selection
  gate can consume this kind of failed artifact through
  `--prior-local-falsification-json` to prevent retrying the same thesis ID or
  mechanism class as a fresh proposal.
- Latest futures-context local screen:
  `TH-FUNDING-MARK-DISLOCATION-RECLAIM-20260507` /
  `funding_mark_dislocation_reclaim` used the new funding-rate and mark-price
  local event features before proposal or code generation. The fixed screen
  required negative funding, improving funding, traded price below mark fair
  value, mark-gap reclaim, and short-horizon positive price confirmation. The
  event builder parsed all sources (`funding_rate` rows `1293`, `mark_price`
  rows `3191`) plus the latest failure synthesis
  (`failed_thesis_id_count=26`, `failed_family_count=26`). The new compound
  mechanism did not match prior failed families, but the artifact still
  returned `status=blocked`, `event_count=0`, and `events_generated` as the
  only blocker. Condition diagnostics showed each single condition had
  matches, but cumulative matching fell from `1025` after
  `funding_rate_bps <= -1.0` to `0` after also requiring
  `funding_rate_delta_bps >= 0.25`. This is absence/rejection evidence for the
  fixed theory, not a prompt to loosen thresholds.
- Latest approved pre-proposal research decision:
  `TH-MARK-DISCOUNT-RECLAIM-20260507` /
  `mark_discount_reclaim_continuation` used a fixed mark-discount reclaim
  screen without funding confirmation. The local event builder completed with
  `event_count=3259` under the latest failure-synthesis guard. Linked local
  falsification passed with `expected_edge_bps=16.121508`,
  `all_in_cost_bps=12.0`, `net_edge_bps=4.121508`, `sample_count=3259`,
  `data_span_days=397.222222`, and `profitable_windows_ratio=1.0`. Corrected
  research selection
  `registry/strategies/research_decisions/20260507T142500JST_mark_discount_reclaim_research_selection/research_decision.json`
  passed with `status=approved_for_proposal_generation`,
  `research_selection_score=100.0`, `proposal_generation_allowed=true`, and
  `code_generation_allowed=false`. This is not profitability or paper-readiness
  proof; it only authorizes proposal generation as the next local step.
- Latest UTC session local screen:
  `TH-UTC-SESSION-DOWNSIDE-REVERSAL-20260507` /
  `utc_session_downside_reversal_event` used the timestamp-derived
  `hour_utc`/`weekday` event features with local BTC 5m OHLCV only. The first
  one-month screen had positive post-cost edge but only 15 matched events, so
  the data was extended with historical `download-data --prepend` rather than
  loosening thresholds. The expanded parquet now has `rows=114401` from
  `2024-01-01T00:00:00+00:00` to `2025-02-01T05:20:00+00:00`, no duplicates,
  and no missing intervals. Re-running the same fixed event spec produced
  `event_count=230`, but long-history falsification failed:
  `expected_edge_bps=-0.272956`, `all_in_cost_bps=12.0`,
  `net_edge_bps=-12.272956`, `sample_count=230`, and
  `profitable_windows_ratio=0.25`. The final reissued artifact used
  `--min-data-span-days 180`, recorded `data_span_days=397.222222`, passed the
  data-span gate, and still failed cost-edge/window stability. The decision is
  reject before research approval/proposal/codegen; the short-window positive
  result was not robust.
- Latest real prior-rejection gate evidence:
  `registry/strategies/research_decisions/20260507T121000JST_volatility_managed_momentum_prior_rejection_gate/research_decision.json`
  consumed the failed local falsification artifact as both current
  `--local-falsification-json` and `--prior-local-falsification-json`.
  The decision had `research_selection_score=100.0`, but still blocked with
  `proposal_generation_allowed=false` and `code_generation_allowed=false`.
  The blockers were
  `research_thesis_not_previously_rejected_by_local_falsification` and
  `local_falsification_cost_edge_exceeds_costs`, proving the prior rejection
  memory overrides otherwise complete research-field scoring.
- Safety evidence: all recent candidate work used historical backtesting,
  local diagnostics, local manifests, local rankings, and local synthesis only.
  No paper, dry-run, live trading, exchange order endpoint, shorting, leverage
  above `1.0`, promotion, or process-control path was started.

## Gap Assessment

- The factory had become too breadth-first. The refreshed aggregate synthesis
  and causal map now force a research gate before the next proposal,
  which reduces blind family expansion but does not by itself create a
  profitable candidate.
- An initial pre-generation research-decision gate now exists and consumes the
  causal failure map with basic response quality checks, but it is still a
  rule-gated intake rather than a richer causal scoring model over
  cross-candidate failure modes.
- Proposal-stage enforcement now re-checks causal-map usage and response
  quality in supplied research decisions, closing the obvious bypass where an
  old approved decision without causal evidence could be reused. The
  range-quarticity thesis consumed the latest gated synthesis/map, then
  completed explicit code generation, diagnostics, historical backtest,
  walk-forward evaluation, ranking, synthesis, and causal-map refresh; it is a
  negative result, not promotion evidence.
- The previous explicit stop rule was validated: repeating the failed
  `liquidity_recovery_horizon` family after it entered the refreshed failure
  set is blocked by the research gate. The latest approved
  `bipower_jump_decay` thesis was outside the failed family set, but its
  generated candidate failed both historical and walk-forward gates and was
  folded into the refreshed twenty-four-family failure map. The subsequent
  `directional_change_overshoot` thesis also failed historical and
  walk-forward gates and was folded into the next refreshed failure map. The
  subsequent
  `range_quarticity_vol_of_vol_state` thesis was explicitly generated and
  evaluated, but also failed historical and walk-forward gates.
- The cross-candidate causal failure map now distinguishes signal sparsity,
  negative edge, cost sensitivity, regime fragility, ML/rule alignment failure,
  and thesis rejection, and the selection gate requires responses to its
  dominant and material categories. It now has an initial prevalence/severity
  weighting model for research selection, but it remains heuristic evidence
  triage rather than proof of a profitable mechanism.
- Profitability and paper readiness are unachieved; passing tests and many
  artifacts are infrastructure evidence only, not completion evidence.

## Completion Audit Checklist

Checked on 2026-05-07 JST against the user objective: build an AI-operated Bot
Factory that can research theories, convert approved theories to safe code, and
ultimately produce profitable candidates without degenerating into ML-style
parameter optimization.

- Requirement: theory research comes before proposal/codegen.
  Evidence: `research_selection.py`,
  `scripts/bot_factory_select_research_thesis.py`, latest causal failure map,
  structured research references, required causal responses, required research
  question responses, local falsification evidence, and prior local rejection
  evidence. Status: partially satisfied as infrastructure.
- Requirement: avoid AI-as-parameter-optimizer drift.
  Evidence: research-quality parameter-only blockers, causal response
  parameter-only blockers, proposal-stage research-decision enforcement,
  `parameter_only_retry_allowed=false` in latest synthesis, and prior local
  falsification rejection memory. Status: substantially guarded, but still
  requires vigilance in future thesis selection.
- Requirement: AI can convert approved theory to code.
  Evidence: `strategy_proposals.py`, `strategy_code.py`, proposal/codegen
  artifacts for multiple families, static checks, and generated strategy
  metadata. Status: implemented for supported variants only; unsupported
  variants correctly block rather than silently falling back.
- Requirement: generated code is evaluated on local historical artifacts.
  Evidence: candidate manifests, historical backtest artifacts, walk-forward
  artifacts, rankings, synthesis, and causal maps across 31 candidates.
  Status: implemented for historical/local evaluation paths.
- Requirement: profitable strategy candidate exists.
  Evidence inspected: latest synthesis has `paper_ready_count=0`,
  `negative_return_count=21`, `walk_forward_failed_count=31`; latest
  mark-discount reclaim candidate has `total_return_pct=-37.832122396` and
  `profit_factor=0.244072592568329`. Status: not achieved.
- Requirement: paper-ready or promotion-ready candidate exists.
  Evidence inspected: latest candidate rankings and synthesis report no
  paper-ready candidate, and paper/live promotion remains blocked. Status: not
  achieved.
- Requirement: no unsafe trading process starts without explicit permission.
  Evidence: recent artifacts record no paper, dry-run, live trading, exchange
  order endpoint use, shorting, leverage above `1.0`, promotion, or process
  control. Status: satisfied for the current work.

Completion decision: do not mark the objective complete. The factory has
stronger theory-selection, provenance, and rejection-memory controls, but the
core success criterion, a profitable locally evaluated candidate that can move
toward paper readiness, is still missing.

## Worktree Hygiene Audit

Checked on 2026-05-07 JST after the latest full Bot Factory test run,
`git diff --check`, cache cleanup, and refreshed again with
`git status --short --untracked-files=all`, `git diff --name-status`,
`git ls-files --others --exclude-standard`, and `git status --ignored` over
known generated artifact roots.

- Current non-ignored working tree: 22 tracked file changes
  (`21` modified plus `1` deleted) and 26 untracked source, documentation, or
  placeholder files. The large diff is mostly Bot Factory source, docs,
  focused tests, and CLI wrappers, not raw backtest output.
- Latest re-check after the theory-fixed generated-parameter guard increment:
  `git ls-files --others --exclude-standard` returns only the
  source, script, documentation, and `data/market_structure/.gitkeep` paths
  listed below. No `data/` runtime output or generated
  `registry/strategies/*` evidence artifact is currently leaking into the
  non-ignored Git candidate set.
- Git commit candidates if this increment is kept:
  `.gitignore`,
  `docs/BOT_FACTORY_MVP_TODO.md`,
  `docs/BOT_FACTORY_STRATEGY_GENERATION_NEXT_AGENT_PROMPT.md`,
  `docs/BOT_FACTORY_GOAL_AUDIT.md`,
  `registry/strategies/proposals/TEMPLATE.md`,
  `freqtrade_ext/bot_factory/candidate_evaluation.py`,
  `freqtrade_ext/bot_factory/candidate_failure_map.py`,
  `freqtrade_ext/bot_factory/candidate_failure_synthesis.py`,
  `freqtrade_ext/bot_factory/candidate_iteration.py`,
  `freqtrade_ext/bot_factory/candidate_ranking.py`,
  `freqtrade_ext/bot_factory/data_quality.py`,
  `freqtrade_ext/bot_factory/freqai_backtest.py`,
  `freqtrade_ext/bot_factory/freqai_checks.py`,
  `freqtrade_ext/bot_factory/freqai_prediction_diagnostics.py`,
  `freqtrade_ext/bot_factory/freqai_training.py`,
  `freqtrade_ext/bot_factory/local_events.py`,
  `freqtrade_ext/bot_factory/local_falsification.py`,
  `freqtrade_ext/bot_factory/research_selection.py`,
  `freqtrade_ext/bot_factory/signal_diagnostics.py`,
  `freqtrade_ext/bot_factory/structural_data_capabilities.py`,
  `freqtrade_ext/bot_factory/strategy_code.py`,
  `freqtrade_ext/bot_factory/strategy_proposals.py`, focused Bot Factory
  scripts, and `tests/test_bot_factory.py`.
- New source/docs files that should be added if the Bot Factory increment is
  committed:
  `docs/BOT_FACTORY_GOAL_AUDIT.md`,
  `data/market_structure/.gitkeep`,
  `freqtrade_ext/bot_factory/bybit_long_short_ratio.py`,
  `freqtrade_ext/bot_factory/bybit_open_interest.py`,
  `freqtrade_ext/bot_factory/candidate_failure_map.py`,
  `freqtrade_ext/bot_factory/candidate_failure_synthesis.py`,
  `freqtrade_ext/bot_factory/freqai_prediction_diagnostics.py`,
  `freqtrade_ext/bot_factory/local_events.py`,
  `freqtrade_ext/bot_factory/local_falsification.py`,
  `freqtrade_ext/bot_factory/research_selection.py`,
  `freqtrade_ext/bot_factory/signal_diagnostics.py`,
  `freqtrade_ext/bot_factory/structural_data_capabilities.py`,
  `scripts/bot_factory_build_causal_failure_map.py`,
  `scripts/bot_factory_build_local_events.py`,
  `scripts/bot_factory_build_local_falsification.py`,
  `scripts/bot_factory_check_funding_rate.py`,
  `scripts/bot_factory_check_long_short_ratio.py`,
  `scripts/bot_factory_check_mark_price.py`,
  `scripts/bot_factory_check_open_interest.py`,
  `scripts/bot_factory_diagnose_candidate_signals.py`,
  `scripts/bot_factory_diagnose_freqai_predictions.py`,
  `scripts/bot_factory_download_bybit_long_short_ratio.py`,
  `scripts/bot_factory_download_bybit_open_interest.py`,
  `scripts/bot_factory_report_structural_data_capabilities.py`,
  `scripts/bot_factory_select_research_thesis.py`, and
  `scripts/bot_factory_synthesize_candidate_failures.py`.
- Intended generated-artifact policy: do not add generated local artifacts
  from `data/backtests/`, `data/freqai/`, `data/freqai_training/`,
  `data/walk_forward/`, or `registry/strategies/{candidates,diagnostics,
  failure_maps,generated,proposals,research_decisions,reviews,synthesis}/`
  unless a future task explicitly asks to preserve a specific complete
  evidence artifact. `.gitignore` now keeps these artifact trees ignored while
  preserving the proposal template and `.gitkeep` directories.
- Latest ignored generated-artifact counts:
  `data/backtests=345`,
  `data/freqai=70`,
  `data/freqai_training=90`,
  `data/walk_forward=1792`,
  `registry/strategies/candidates=382`,
  `registry/strategies/diagnostics=70`,
  `registry/strategies/failure_maps=34`,
  `registry/strategies/generated=136`,
  `registry/strategies/proposals=139`,
  `registry/strategies/research_decisions=134`,
  `registry/strategies/reviews=12`, and
  `registry/strategies/synthesis=68`. These are local evidence/runtime
  artifacts, not default Git candidates.
- Cleaned local test/runtime caches outside `.venv`: latest cleanup removed
  33 targets after the evaluation-stage hyperopt-surface rejection test run,
  remaining `__pycache__` outside `.venv` is 0, and `.pytest_cache` is absent.
- Deletion candidates already handled: transient Python test/runtime caches.
  No further source/docs/script file should be deleted automatically from the
  current non-ignored set; those files are either implementation, tests,
  documentation, or the pending runbook deletion decision below.
- Decision required before commit: `docs/BOT_FACTORY_GOAL_COMMAND_RUNBOOK.md`
  is currently deleted in the working tree. Inspection of `HEAD:` shows it was
  a generic Codex `/goal` runbook covering full objective wording, first-step
  `git status`, MVP TODO reading, scope limits, hard no-trading boundaries,
  verification order, docs updates, and PR-ready checks. Those operational
  constraints are now covered by `AGENTS.md`, this audit, the current MVP TODO,
  and the next-agent prompt. Recommendation: it is reasonable to commit the
  deletion if the team no longer wants a slash-command usage guide in repo; if
  reusable `/goal` examples are still desired, restore and update it instead.
  Do not silently decide this in the same cleanup commit without owner choice.
- Residual hygiene warning: `git diff --check` passes but Git reports existing
  LF-to-CRLF working-copy warnings for `.gitignore`,
  `docs/BOT_FACTORY_MVP_TODO.md`,
  `docs/BOT_FACTORY_STRATEGY_GENERATION_NEXT_AGENT_PROMPT.md`, and
  `registry/strategies/proposals/TEMPLATE.md`.

## Proposed Review / Commit Split

This is a large but mostly coherent Bot Factory increment. If it needs to be
split for review, use logical behavior groups rather than generated-artifact
boundaries:

- Artifact hygiene and handoff docs:
  `.gitignore`, `docs/BOT_FACTORY_MVP_TODO.md`,
  `docs/BOT_FACTORY_STRATEGY_GENERATION_NEXT_AGENT_PROMPT.md`,
  `docs/BOT_FACTORY_GOAL_AUDIT.md`, and the owner decision for
  `docs/BOT_FACTORY_GOAL_COMMAND_RUNBOOK.md`.
- Failure synthesis, causal failure map, research selection, and local
  pre-proposal falsification:
  `candidate_failure_synthesis.py`, `candidate_failure_map.py`,
  `research_selection.py`, `local_events.py`, `local_falsification.py`,
  `structural_data_capabilities.py`, their CLI wrappers, funding/mark/open-
  interest data check scripts, structural capability reporting, and
  corresponding tests.
- Candidate diagnostics, evaluation, ranking, and iteration feedback:
  `signal_diagnostics.py`, `freqai_prediction_diagnostics.py`,
  `candidate_evaluation.py`, `candidate_ranking.py`,
  `candidate_iteration.py`, `scripts/bot_factory_diagnose_*`,
  `scripts/bot_factory_evaluate_candidate.py`,
  `scripts/bot_factory_run_walk_forward.py`, and corresponding tests.
- Proposal/code generation and FreqAI wrapper hardening:
  `strategy_proposals.py`, `strategy_code.py`, `freqai_backtest.py`,
  `freqai_checks.py`, `freqai_training.py`, `data_quality.py`,
  `registry/strategies/proposals/TEMPLATE.md`,
  `scripts/bot_factory_generate_strategy_proposal.py`,
  `scripts/bot_factory_download_data.py`,
  `scripts/bot_factory_run_freqai_backtest.py`,
  `scripts/bot_factory_run_freqai_training.py`, and corresponding tests.

`tests/test_bot_factory.py` spans all of these groups. If split commits are
required, split that file's patch carefully with `git add -p` or keep the whole
Bot Factory increment as one review commit after owner approval. Do not stage
ignored generated artifacts from `data/` or `registry/strategies/*` runtime
trees, and do not stage the runbook deletion until its owner decision is made.

## 2026-05-07 JST Mark Discount Reclaim Outcome

- The previously approved
  `TH-MARK-DISCOUNT-RECLAIM-20260507` research path was carried through
  accepted proposal, strategy code generation, static/data checks, historical
  backtest, signal diagnostics, candidate evaluation, ranking, synthesis, and
  causal-map refresh.
- Accepted proposal:
  `registry/strategies/proposals/20260507T055000Z_LongOnlyMarkDiscountReclaimCandidate.metadata.json`
  with `code_generation_eligible=true` and
  `strategy_logic_variant=mark_discount_reclaim_continuation`.
- Generated strategy:
  `registry/strategies/generated/LongOnlyMarkDiscountReclaimCandidate/mark_discount_reclaim_001/metadata.json`
  with static check passing.
- Historical result:
  `data/backtests/LongOnlyMarkDiscountReclaimCandidate/mark_discount_reclaim_001_20240101_20250201/metrics.json`
  failed hard: `total_return_pct=-37.832122396`,
  `profit_factor=0.244072592568329`,
  `max_drawdown_pct=37.83212239599998`, and
  `sortino=-168.6116978183358`.
- Signal diagnostics:
  `registry/strategies/diagnostics/LongOnlyMarkDiscountReclaimCandidate/mark_discount_reclaim_001/20260507T151000JST_mark_discount_reclaim_001_signal_diagnostics/signal_diagnostics.json`
  completed with `entry_signal_count=9603` and `zero_entry_signal=false`.
  This was not a sparse-signal failure; it exposed negative edge after entries
  and a mismatch between the local 6-candle event falsification and generated
  strategy behavior.
- Candidate manifest:
  `registry/strategies/candidates/LongOnlyMarkDiscountReclaimCandidate/mark_discount_reclaim_001/candidate_manifest.json`
  has `recommendation=fail` and `FAIL_COST_SENSITIVE`.
- Latest integrated failure synthesis:
  `registry/strategies/synthesis/20260507T160500JST_all_candidates_with_mark_discount_reclaim_001_edge_aware_failure_synthesis/candidate_failure_synthesis.json`
  covers 31 candidates, has `paper_ready_count=0`,
  `parameter_only_retry_allowed=false`, `requires_new_thesis_id=true`, and a
  generated-entry edge failure for `mark_discount_reclaim_001`
  (`net_edge_bps=-12.323858`, `profitable_windows_ratio=0.0`).
- Latest integrated causal failure map:
  `registry/strategies/failure_maps/20260507T161500JST_all_candidates_with_mark_discount_reclaim_001_edge_aware_causal_failure_map/causal_failure_map.json`
  covers 31 candidates, requires a research decision before proposal, and
  identifies dominant risks in regime fragility, walk-forward fragility,
  transaction-cost sensitivity, no profitable windows, negative edge after
  entries, and `generated_entry_negative_edge`.
- Source fix added after this failure: strategy proposals now preserve numeric
  `parameter_overrides`, and code generation applies known safe overrides,
  including mapping `local_falsification_hold_candles` to
  `sell_timeout_candles`. This does not make the failed candidate profitable;
  it reduces the chance that future generated code silently diverges from a
  local falsification horizon.
- Verification: `py_compile` passed for proposal/codegen/diagnostics/tests;
  focused mark-discount/mark-price tests passed 6 tests; full
  `tests/test_bot_factory.py` passed 197 tests with the existing pandas
  `PerformanceWarning` noise from `signal_diagnostics.py`.
- Cleanup: after the latest full test run, 33 `.pytest_cache` or `.venv`
  external `__pycache__` targets were removed. Generated evidence artifacts
  remain ignored and should not be staged by default.

## Direction Correction

Do not add another strategy family as the next default action. Use the research
selection gate before any new proposal or generator extension. The gate must
consume the latest refreshed 31-candidate failure synthesis, the latest causal
failure map,
explicit responses to the map's dominant categories, and proposed theory
inputs, then emit a local `research_decision.json` with:

- the concrete mechanism class under consideration,
- why it is outside the failed families,
- required data and whether local data can falsify it,
- expected edge source and transaction-cost exposure,
- local falsification JSON evidence, linked to a Bot Factory local event source
  with a passing failure-synthesis guard, when high-risk cost sensitivity
  requires quantified cost/edge proof,
- prior failed local falsification JSON evidence to prevent repeating a
  rejected thesis ID or mechanism class,
- stop conditions before code generation,
- causal-failure-map coverage and missing response categories,
- material causal-failure categories above the 70% prevalence threshold,
- response quality, category evidence gaps, and parameter-only response
  findings,
- research selection score, failed score components, and minimum-score pass
  evidence,
- explicit responses to each required research question from the causal map,
- whether proposal generation is allowed, and whether code generation remains
  blocked or deferred.

Only after that gate approves a theory should proposal generation be attempted.
For the current state, `mark_discount_reclaim_continuation`,
`bipower_jump_decay`, and
`directional_change_overshoot` have completed the accepted proposal,
strategy-code generation, signal diagnostics, historical backtest,
ranking, synthesis, and causal-map loop, and all are negative results.
`range_quarticity_vol_of_vol_state` has also completed
accepted proposal, explicit strategy-code generation, signal diagnostics,
historical backtest, walk-forward evaluation, ranking, synthesis, and
causal-map refresh, and is also a negative result. Paper readiness and
profitability remain unachieved.

## Latest Mark Fair-Value Momentum Lag Check

Checked on 2026-05-07 JST after the `mark_fair_value_momentum_lag`
proposal/codegen/diagnostics path.

- The distinct thesis
  `TH-MARK-FAIR-VALUE-MOMENTUM-LAG-20260507` passed the research-selection
  gate at
  `registry/strategies/research_decisions/20260507T164000JST_mark_fair_value_momentum_lag_research_selection/research_decision.json`
  with `status=approved_for_proposal_generation`,
  `research_selection_score=100.0`, and no blockers.
- Its pre-proposal local screen looked acceptable but is not sufficient
  profitability evidence: local events had `event_count=2274`,
  `combined_match_count_before_cooldown=13584`, and `cooldown_candles=12`;
  local falsification had `expected_edge_bps=27.99047`,
  `all_in_cost_bps=12.0`, `net_edge_bps=15.99047`,
  `sample_count=2274`, `data_span_days=397.222222`, and
  `profitable_windows_ratio=1.0`.
- Proposal/code generation completed for
  `registry/strategies/proposals/20260507T075500Z_LongOnlyMarkFairValueMomentumLagCandidate.metadata.json`
  and
  `registry/strategies/generated/LongOnlyMarkFairValueMomentumLagCandidate/mark_fair_value_momentum_lag_002/metadata.json`.
  The generated candidate was `candidate_evaluation_eligible=true` and static
  check passed, but this only proves the generator path worked.
- Full generated-entry diagnostics failed before any historical backtest
  interpretation:
  `registry/strategies/diagnostics/LongOnlyMarkFairValueMomentumLagCandidate/mark_fair_value_momentum_lag_002/20260507T170500JST_mark_fair_value_momentum_lag_002_full_signal_edge_diagnostics/signal_diagnostics.json`
  has `entry_count=2775`, `generated_entry_edge.status=fail`,
  `expected_edge_bps=0.95015`, `net_edge_bps=-11.04985`,
  `profitable_windows_ratio=0.0`, and diagnosis codes
  `GENERATED_ENTRY_EDGE_NEGATIVE_AFTER_COST` plus
  `GENERATED_ENTRY_EDGE_WINDOW_FRAGILE`.
- Follow-up correction: `local_events.py` now aligns funding/mark context to
  closed-candle availability by shifting context timestamps by
  `context_interval - base_interval` before `merge_asof`, matching generated
  strategy and diagnostics semantics. Re-running the mark fair-value event
  study produced `combined_match_count_before_cooldown=20169` and
  `event_count=2775`, exactly matching the generated diagnostics entry mask
  before and after cooldown.
- Corrected local falsification on that closed-context event set failed:
  `registry/strategies/research_decisions/20260507T173000JST_mark_fair_value_momentum_lag_closed_context_falsification/local_falsification.json`
  has `status=failed`, `expected_edge_bps=0.95015`,
  `net_edge_bps=-11.04985`, `sample_count=2775`, and
  `profitable_windows_ratio=0.0`. The earlier positive pre-proposal artifact
  is now classified as a context-timing false positive and must not support
  further proposal/codegen.
- Latest correction: local event artifacts now include
  `context_merge.semantics=closed_context_candle_availability_v1` and
  `closed_context_candle_alignment`, local falsification validates that
  event-source proof, and research selection blocks high-risk cost-edge
  artifacts whose funding/mark event source lacks current closed-context
  alignment metadata. The regenerated v2 mark fair-value artifacts are:
  `registry/strategies/research_decisions/20260507T183500JST_mark_fair_value_momentum_lag_closed_context_event_study_v2/local_events.json`
  (`event_count=2775`, `combined_match_count_before_cooldown=20169`,
  `required_contexts=["mark_price"]`, alignment true) and
  `registry/strategies/research_decisions/20260507T184000JST_mark_fair_value_momentum_lag_closed_context_falsification_v2/local_falsification.json`
  (`status=failed`, `net_edge_bps=-11.04985`,
  `profitable_windows_ratio=0.0`, event-source alignment check passing).
- Real stale-positive validation: the older positive local falsification
  artifact
  `registry/strategies/research_decisions/20260507T162500JST_mark_fair_value_momentum_lag_local_falsification/local_falsification.json`
  was run through the updated research-selection gate in
  `registry/strategies/research_decisions/20260507T185500JST_mark_fair_value_stale_context_research_selection_guard_clean/research_decision.json`.
  The decision is `status=blocked` with
  `event_source_context_alignment_valid=false`, `cost_edge_passes=false`, and
  failure reason `event_source_context_alignment_missing_or_invalid` despite
  the stale artifact's positive `net_edge_bps=15.99047`. This confirms the
  older false-positive evidence cannot authorize a proposal.
- Fresh thesis screen after the stale-evidence fix:
  `TH-LOW-RANGE-VOLUME-ABSORPTION-20260507` /
  `low_range_volume_absorption` produced a valid local event artifact
  (`event_count=78`, `combined_match_count_before_cooldown=95`,
  `context_features_used=false`) but failed local falsification in
  `registry/strategies/research_decisions/20260507T191000JST_low_range_volume_absorption_local_falsification/local_falsification.json`
  with `expected_edge_bps=-4.16165`, `net_edge_bps=-16.16165`,
  `sample_count=78`, `data_span_days=397.222222`, and
  `profitable_windows_ratio=0.0`. It must not advance to proposal/codegen.
- Completion decision remains unchanged: do not mark the goal complete, do not
  promote the candidate, and do not start paper/dry-run/live trading. The next
  step is a fresh research-selection-approved thesis that consumes the latest
  failed-falsification evidence and the refreshed failure synthesis/map. Passing
  gates still require current local falsification evidence; no profitable or
  paper-ready candidate exists yet.

## 2026-05-07 JST Local Falsification Rejection Memory

- Candidate failure synthesis now accepts failed local falsification artifacts
  as first-class rejection memory. This prevents pre-proposal negative screens
  from living only in Markdown notes.
- Refreshed synthesis:
  `registry/strategies/synthesis/20260507T192000JST_all_candidates_with_local_falsification_rejections/candidate_failure_synthesis.json`
  completed with `candidate_count=31`, `paper_ready_count=0`,
  `parameter_only_retry_allowed=false`, `requires_new_thesis_id=true`, and
  `local_falsification_rejection_count=2`.
- The two local rejection thesis IDs are
  `TH-LOW-RANGE-VOLUME-ABSORPTION-20260507` and
  `TH-MARK-FAIR-VALUE-MOMENTUM-LAG-20260507`. The rejected mechanism classes
  are `low_range_volume_absorption` and
  `mark_fair_value_momentum_lag`.
- Refreshed causal map:
  `registry/strategies/failure_maps/20260507T192500JST_all_candidates_with_local_falsification_rejections_causal_map/causal_failure_map.json`
  completed with `candidate_count=31`, `category_count=10`, and
  `requires_research_decision_before_proposal=true`. Its guidance now carries
  both local rejection thesis IDs and mechanism classes in the avoid lists.
- This is a cleanup and anti-repeat safeguard. It does not change the audit
  conclusion: there is still no profitable or paper-ready Bot Factory candidate.

## 2026-05-07 JST Thin-Book Dislocation Reversion Screen

- A new fixed pre-proposal thesis was screened:
  `TH-THIN-BOOK-DISLOCATION-REVERSION-20260507` /
  `thin_book_dislocation_reversion`. It tested whether a sharp six-candle
  downside move, wide current range, below-average volume, and negative
  48-candle SMA distance indicate a thin-book dislocation that should revert
  over six candles.
- Local event study:
  `registry/strategies/research_decisions/20260507T193500JST_thin_book_dislocation_reversion_event_study/local_events.json`
  completed with `event_count=1516` and `blocker_count=0`.
- Local falsification:
  `registry/strategies/research_decisions/20260507T194000JST_thin_book_dislocation_reversion_local_falsification/local_falsification.json`
  failed with `expected_edge_bps=1.802646`, `all_in_cost_bps=12.0`,
  `net_edge_bps=-10.197354`, `sample_count=1516`,
  `data_span_days=397.222222`, `profitable_windows_ratio=0.0`, and
  `blocker_count=2`.
- Refreshed synthesis:
  `registry/strategies/synthesis/20260507T195000JST_all_candidates_with_three_local_falsification_rejections/candidate_failure_synthesis.json`
  completed with `local_falsification_rejection_count=3`. Refreshed causal
  map:
  `registry/strategies/failure_maps/20260507T195500JST_all_candidates_with_three_local_falsification_rejections_causal_map/causal_failure_map.json`
  now includes `TH-THIN-BOOK-DISLOCATION-REVERSION-20260507` and
  `thin_book_dislocation_reversion` in its avoid guidance.
- This screen did not produce a candidate and should not be retuned. The audit
  conclusion remains unchanged: the Bot Factory is improving its rejection
  memory, but it still has no profitable or paper-ready candidate.

## 2026-05-07 JST Open-Interest Support and Data Gap

- OHLCV-only fixed screens have repeatedly failed after costs, so the factory
  now has a structural derivatives-data path for open-interest hypotheses.
- Added `check_open_interest_parquet` and
  `scripts/bot_factory_check_open_interest.py`. The checker validates a `date`
  column plus one of `open_interest`, `open`, or `close`, numeric
  non-negative values, duplicate/sorted timestamps, and expected intervals.
- Added local event builder support for `open_interest`,
  `open_interest_delta_pct`, and `open_interest_zscore`, with
  closed-context-candle availability semantics matching funding-rate and
  mark-price context. The local event CLI now accepts
  `--open-interest-path`.
- Verification passed:
  `py_compile` on the changed Python files and focused pytest
  `open_interest or local_event_builder_supports_futures_context_features or funding_rate_quality_check or mark_price_quality_check`
  passed 6 tests.
- Initial local data gap: before the downloader was added, the Bybit futures
  directory had BTC/ETH OHLCV, mark-price, funding-rate files, and leverage
  tiers, but no open-interest, liquidation, or order-book data. The new checker
  recorded this at
  `registry/strategies/checks/20260507T201000JST_open_interest_data_gap_check.json`
  with `ok=false` and a `file_exists` error for
  `user_data/data/bybit/futures/BTC_USDT_USDT-1h-open_interest.parquet`.
- Audit conclusion remains unchanged: open-interest support is useful factory
  plumbing, but no profitable or paper-ready candidate exists yet.

## 2026-05-07 JST Bybit Open-Interest Data and First Structural Screen

- Added a public Bybit V5 open-interest downloader:
  `freqtrade_ext/bot_factory/bybit_open_interest.py` and
  `scripts/bot_factory_download_bybit_open_interest.py`. It uses only the
  public market open-interest endpoint, requires no API keys, and records
  safety metadata proving no order endpoint, exchange process, leverage change,
  or shorting action.
- Downloaded BTCUSDT 1h open-interest data for `2024-01-01` to `2025-02-01`:
  `user_data/data/bybit/futures/BTC_USDT_USDT-1h-open_interest.parquet`.
  Result: `row_count=9529`, `page_count=48`, `truncated=false`,
  `api_key_used=false`, and `order_endpoint_used=false`.
- Quality check:
  `registry/strategies/checks/20260507T203000JST_open_interest_long_quality_check.json`
  passed with `ok=true`, `rows=9529`, `duplicate_timestamps=0`,
  `missing_intervals=0`, and no findings.
- First structural-data thesis:
  `TH-OPEN-INTEREST-DELEVERAGING-REBOUND-20260507` /
  `open_interest_deleveraging_rebound`. Event study
  `registry/strategies/research_decisions/20260507T204000JST_open_interest_deleveraging_rebound_event_study/local_events.json`
  completed with `event_count=203`. Local falsification
  `registry/strategies/research_decisions/20260507T204500JST_open_interest_deleveraging_rebound_local_falsification/local_falsification.json`
  failed with `expected_edge_bps=7.363403`,
  `all_in_cost_bps=12.0`, `net_edge_bps=-4.636597`,
  `sample_count=203`, `profitable_windows_ratio=0.25`, and
  `blocker_count=2`.
- Refreshed synthesis/map:
  `registry/strategies/synthesis/20260507T205000JST_all_candidates_with_open_interest_rejection/candidate_failure_synthesis.json`
  has `local_falsification_rejection_count=4`, and
  `registry/strategies/failure_maps/20260507T205500JST_all_candidates_with_open_interest_rejection_causal_map/causal_failure_map.json`
  carries the open-interest thesis/mechanism in its avoid guidance.
- Audit conclusion remains unchanged: the factory can now acquire and screen
  open-interest data, but the first structural-data screen still failed after
  costs. No profitable or paper-ready candidate exists yet.

## 2026-05-07 JST Open-Interest Impulse Continuation Screen

- A second fixed open-interest thesis was screened:
  `TH-OPEN-INTEREST-IMPULSE-CONTINUATION-20260507` /
  `open_interest_impulse_continuation`. It tested the opposite mechanism from
  deleveraging rebound: a one-hour open-interest expansion during an
  upside/high-volume range impulse should support short-horizon continuation.
- Event study:
  `registry/strategies/research_decisions/20260507T211000JST_open_interest_impulse_continuation_event_study/local_events.json`
  completed with `event_count=204` and `blocker_count=0`.
- Local falsification:
  `registry/strategies/research_decisions/20260507T211500JST_open_interest_impulse_continuation_local_falsification/local_falsification.json`
  failed with `expected_edge_bps=5.360183`,
  `all_in_cost_bps=12.0`, `net_edge_bps=-6.639817`,
  `sample_count=204`, `data_span_days=397.222222`,
  `profitable_windows_ratio=0.0`, and `blocker_count=2`.
- Refreshed synthesis/map:
  `registry/strategies/synthesis/20260507T212000JST_all_candidates_with_two_open_interest_rejections/candidate_failure_synthesis.json`
  has `local_falsification_rejection_count=5`, and
  `registry/strategies/failure_maps/20260507T212500JST_all_candidates_with_two_open_interest_rejections_causal_map/causal_failure_map.json`
  carries the impulse-continuation thesis/mechanism in its avoid guidance.
- Audit conclusion remains unchanged: two fixed open-interest mechanisms now
  fail after costs. The factory should not keep sign-flipping OI/price impulse
  variants; the next structural thesis needs materially new evidence or data.

## 2026-05-07 JST Structural Data Quality Gate

- Research selection now blocks structural-data theses unless a passing local
  data quality report is supplied. It scans thesis family, mechanism, thesis
  text, required data, and falsification plan for open-interest, liquidation,
  order-book, market-depth, and book/depth-imbalance terms.
- New CLI option:
  `scripts/bot_factory_select_research_thesis.py --local-data-quality-report-json`.
  Decision artifacts now record `local_data_quality_report_paths`; Markdown
  reports include a Local Data Quality Reports section.
- Tests prove the intended behavior: an open-interest thesis with only a local
  data path is blocked by `structural_data_quality_report_present`; the same
  thesis with a passing quality report avoids that blocker.
- Verification passed: `py_compile`, focused research-selection tests, and full
  `tests/test_bot_factory.py` all passed. Existing pandas
  `PerformanceWarning` noise from `signal_diagnostics.py` remains.
- Audit conclusion remains unchanged: this is gate hardening, not a profitable
  candidate. It reduces false progress by preventing unverified structural-data
  claims from moving toward proposal/codegen.

## 2026-05-07 JST Proposal-Stage Structural Data Quality Continuity

- Proposal generation now repeats the structural-data handoff check instead of
  trusting stale approvals. `strategy_proposals.py` detects open-interest,
  liquidation, order-book, market-depth, and book/depth-imbalance claims in
  proposal text and required data.
- Structural-data proposals now require a supplied research decision with
  recorded `local_data_quality_report_paths`, existing in-workspace report
  files, and passing `local_data_quality_reports_valid` plus
  `structural_data_quality_report_present` research-selection checks.
- Proposal metadata records `structural_data_requirement` and
  `structural_data_quality_report_gate_passed` inside
  `research_decision_constraints`, so later code generation can see whether the
  quality evidence survived the handoff.
- `scripts/bot_factory_generate_strategy_proposal.py` accepts
  `--local-data-quality-json` so quality reports can also be preserved as local
  proposal evidence.
- Verification passed: `py_compile`, focused proposal/CLI tests, and full
  `tests/test_bot_factory.py`. Existing pandas `PerformanceWarning` noise from
  `signal_diagnostics.py` remains.
- Audit conclusion remains unchanged: this prevents stale structural-data
  claims from reaching proposal/codegen, but it still does not produce a
  profitable or paper-ready candidate.

## 2026-05-07 JST Codegen Structural Data Handoff Guard

- Code generation now detects structural-data proposal metadata and blocks
  stale or incomplete handoffs. It reads `structural_data_requirement` when
  present, and otherwise scans core proposal metadata for open-interest,
  liquidation, order-book, market-depth, and book/depth-imbalance terms.
- Structural-data codegen requires a passing proposal-stage
  `research_decision_constraints` handoff with existing quality report paths
  and passing research-selection quality checks.
- Codegen also blocks structural-data proposals until a real structural-data
  code-generation variant exists. This deliberately avoids emitting OHLCV-only
  strategy code for a thesis whose actual edge depends on open-interest,
  liquidation, or order-book evidence.
- Generated metadata and `research_brief.json` preserve
  `structural_data_requirement` and `structural_data_quality_handoff` for later
  evaluation/iteration.
- Verification passed: `py_compile`, focused codegen handoff tests, and full
  `tests/test_bot_factory.py`. Existing pandas `PerformanceWarning` noise from
  `signal_diagnostics.py` remains.
- Audit conclusion remains unchanged: this is a correctness guard, not a
  profitable or paper-ready candidate.

## 2026-05-07 JST Open-Interest Crowded Short Squeeze Screen

- A third fixed open-interest thesis was screened:
  `TH-OPEN-INTEREST-CROWDED-SHORT-SQUEEZE-20260507` /
  `open_interest_crowded_short_squeeze`. It tested a different mechanism from
  deleveraging rebound and impulse continuation: open-interest expansion while
  price falls below its 24-hour SMA in muted range/volume, expecting crowded
  short positioning to rebound over 12 candles.
- Event study:
  `registry/strategies/research_decisions/20260507T223000JST_open_interest_crowded_short_squeeze_event_study/local_events.json`
  completed with `event_count=115` and `blocker_count=0`.
- Local falsification:
  `registry/strategies/research_decisions/20260507T223500JST_open_interest_crowded_short_squeeze_local_falsification/local_falsification.json`
  failed with `expected_edge_bps=5.035758`,
  `all_in_cost_bps=12.0`, `net_edge_bps=-6.964242`,
  `sample_count=115`, `data_span_days=397.222222`,
  `profitable_windows_ratio=0.0`, and `blocker_count=2`.
- Refreshed synthesis/map:
  `registry/strategies/synthesis/20260507T224000JST_all_candidates_with_three_open_interest_rejections/candidate_failure_synthesis.json`
  has `local_falsification_rejection_count=6`, and
  `registry/strategies/failure_maps/20260507T224500JST_all_candidates_with_three_open_interest_rejections_causal_map/causal_failure_map.json`
  carries the crowded-short-squeeze thesis/mechanism in its avoid guidance.
- Audit conclusion remains unchanged: three fixed open-interest mechanisms now
  fail after costs. The factory should not keep retuning OI thresholds or
  rearranging the same OI/price signs; the next structural thesis needs
  materially new data or research evidence.

## 2026-05-07 JST Structural Data Capability Report

- Added `freqtrade_ext/bot_factory/structural_data_capabilities.py` and
  `scripts/bot_factory_report_structural_data_capabilities.py` so the factory
  records which structural market-data classes are locally usable, blocked
  without new data, and forbidden for strategy code generation.
- Official Bybit docs checked on 2026-05-07 JST: open-interest is available via
  the public REST market endpoint
  `https://bybit-exchange.github.io/docs/v5/market/open-interest`;
  all-liquidation is documented as a public websocket topic
  `https://bybit-exchange.github.io/docs/v5/websocket/public/all-liquidation`;
  REST orderbook is a current market endpoint
  `https://bybit-exchange.github.io/docs/v5/market/orderbook`, and websocket
  orderbook is a live stream
  `https://bybit-exchange.github.io/docs/v5/websocket/public/orderbook`.
- Current local capability report:
  `registry/strategies/checks/20260507T225500JST_structural_data_capabilities.json`.
  It reports `local_research_usable=["open_interest"]`,
  `blocked_without_new_data=["liquidation", "order_book"]`,
  `must_not_codegen=["open_interest", "liquidation", "order_book"]`, and
  `blocker_count=3`.
- Verification passed: `py_compile` for the new module, CLI, and
  `tests/test_bot_factory.py`; focused pytest
  `-k "structural_data_capability_report"` passed 2 tests.
- Audit conclusion remains unchanged: this prevents unsupported structural-data
  claims from becoming false progress, but it is not a profitable or
  paper-ready candidate.

## 2026-05-07 JST Research-Selection Structural Capability Gate

- Connected structural data capability reports to research selection:
  `freqtrade_ext/bot_factory/research_selection.py` now accepts
  `structural_data_capability_report_paths`, records them in the decision
  thesis payload, and adds report validity plus required-class support checks.
- `scripts/bot_factory_select_research_thesis.py` now supports
  `--structural-data-capability-report-json`.
- Structural terms are classified into `open_interest`, `liquidation`, and
  `order_book`. A structural thesis is blocked when a required class is missing
  from `proposal_guidance.local_research_usable` in the capability report.
  This means liquidation/order-book theses cannot move through research
  selection with only a generic passing quality JSON while the current
  capability report marks those classes blocked without new historical data.
- Verification passed: `py_compile` for research selection, the CLI, and
  `tests/test_bot_factory.py`; focused pytest for the three structural
  research-selection tests passed.
- Audit conclusion remains unchanged: this prevents another false-progress
  path, but it is still not a profitable or paper-ready candidate. Proposal
  generation still needs a matching continuity check against stale or crafted
  research decisions.

## 2026-05-07 JST Proposal-Stage Structural Capability Continuity

- Extended `freqtrade_ext/bot_factory/strategy_proposals.py` so proposal
  generation re-checks structural capability evidence carried by a supplied
  research decision. Structural proposals now require both passing local data
  quality evidence and passing capability-report evidence.
- The proposal gate verifies
  `thesis.structural_data_capability_report_paths`, path existence inside the
  workspace, JSON parseability, `factory=structural_data_capability_report`,
  research-selection capability checks, and whether the proposal's required
  structural classes are included in `proposal_guidance.local_research_usable`.
- Proposal metadata now records capability report paths, report validity,
  usable classes, unsupported required classes, capability gate pass status,
  and required-class support status in `research_decision_constraints`.
- Verification passed: `py_compile` for `strategy_proposals.py` and
  `tests/test_bot_factory.py`; focused pytest for the four structural
  proposal/codegen continuity tests passed.
- Audit conclusion remains unchanged: this closes another stale/crafted
  evidence path, but it still does not create a profitable or paper-ready
  candidate. Structural codegen remains intentionally unsupported.

## 2026-05-07 JST Codegen Structural Capability Handoff

- Extended `freqtrade_ext/bot_factory/strategy_code.py` so strategy code
  generation recomputes `structural_data_capability_handoff` from proposal
  `research_decision_constraints`, matching the existing quality handoff.
- Added blocker check `structural_data_capability_handoff_passed`. A
  structural-data proposal must now carry capability report paths, existence
  evidence, validity evidence, capability-check evidence, required-class
  support, and no unsupported required structural class before any future
  structural codegen variant can emit strategy code.
- Codegen metadata and `research_brief` now preserve
  `structural_data_capability_handoff` beside
  `structural_data_quality_handoff`.
- Added a focused test for stale/manually crafted proposal metadata where
  quality evidence passes but capability evidence is missing. Codegen blocks it
  before the existing `structural_data_code_generation_supported` blocker.
- Verification passed: `py_compile` for `strategy_code.py` and
  `tests/test_bot_factory.py`; focused pytest
  `-k "strategy_code_generator_blocks_structural_data"` passed 3 tests; full
  `tests/test_bot_factory.py -q` passed with existing pandas
  `PerformanceWarning` warnings from `signal_diagnostics.py`.
- Audit conclusion remains unchanged: this closes another false-continuity
  path. It is still not a profitable or paper-ready candidate, and structural
  data remains intentionally blocked from strategy code generation.

## 2026-05-07 JST Worktree Hygiene Classification

- Re-checked the worktree after the structural capability/codegen guard
  increment with `git status --short --untracked-files=all`,
  `git diff --name-status`, and
  `git ls-files --others --exclude-standard`.
- Current non-ignored state is 21 tracked changes plus 22 untracked Bot
  Factory source/script/doc files. The non-ignored untracked list contains only
  `docs/BOT_FACTORY_GOAL_AUDIT.md`, Bot Factory modules under
  `freqtrade_ext/bot_factory/`, and Bot Factory scripts under `scripts/`.
- Files that should remain Git candidates: `.gitignore`, Bot Factory docs,
  `freqtrade_ext/bot_factory/*.py`, `scripts/bot_factory_*.py`,
  `tests/test_bot_factory.py`, and
  `registry/strategies/proposals/TEMPLATE.md`.
- Files that should stay ignored and not be added: `data/backtests/**`,
  `data/freqai/**`, `data/freqai_training/**`, `data/walk_forward/**`,
  `registry/strategies/checks/*.json`,
  `registry/strategies/candidates/**`,
  `registry/strategies/diagnostics/**`,
  `registry/strategies/failure_maps/**`,
  `registry/strategies/generated/**`,
  `registry/strategies/proposals/**` except `TEMPLATE.md`,
  `registry/strategies/research_decisions/**`,
  `registry/strategies/reviews/**`,
  `registry/strategies/synthesis/**`, and runtime `user_data/*` artifacts.
- Verified representative generated paths with `git check-ignore -v`:
  `data/backtests/static_check_sample.json`,
  `data/freqai/.../stdout.log`, `data/walk_forward/.../stdout.log`,
  `registry/strategies/checks/*.json`,
  `registry/strategies/proposals/*.metadata.json`,
  `registry/strategies/research_decisions/...`,
  `registry/strategies/synthesis/...`, `user_data/logs/freqtrade.log`,
  `user_data/data/bybit`, and `user_data/models/...`.
- Safe deletion candidates are only generated caches such as
  `.pytest_cache/` and `__pycache__/`. Broader ignored artifacts are retained
  by default because local artifacts are the audit trail for rejected
  hypotheses, data-quality checks, and verification runs.
- Owner decision required: `docs/BOT_FACTORY_GOAL_COMMAND_RUNBOOK.md` is
  currently deleted. Accept the deletion only if its useful content is
  superseded by `docs/BOT_FACTORY_MVP_TODO.md` and this audit; otherwise
  restore or intentionally merge it.
- Audit conclusion: the large worktree is real, but generated runtime outputs
  are not currently leaking into the Git candidate set. The remaining cleanup
  need is review/commit grouping, plus an owner decision on the deleted
  runbook.

## 2026-05-07 JST Proposal/Codegen Local Falsification Handoff

- Extended `freqtrade_ext/bot_factory/strategy_proposals.py` so proposal
  generation derives local falsification handoff status from supplied research
  decision artifacts. A high-risk `cost_sensitive_mechanism` causal map
  (`risk_score >= 80`) now requires a passing local falsification handoff
  before proposal acceptance.
- The handoff requires artifact presence, parseability, thesis match,
  positive net edge after all-in cost, valid Bot Factory factory/safety/event
  source flags, closed-context alignment, and a passing failure-synthesis
  guard. A weighted causal response without this local handoff now blocks
  proposal generation.
- Proposal metadata now records local falsification continuity fields in
  `research_decision_constraints`, including required/pass status, artifact
  counts, matching-thesis counts, passing cost-edge counts, artifact paths, and
  local-falsification blocker names.
- Extended `freqtrade_ext/bot_factory/strategy_code.py` so codegen consumes
  those proposal metadata fields and blocks stale accepted proposals with
  `local_falsification_handoff_passed` before strategy code generation.
- Added focused tests covering missing handoff at proposal time, passing
  handoff at proposal time, and stale accepted proposal metadata blocked at
  codegen.
- Verification passed: `py_compile` for `strategy_proposals.py`,
  `strategy_code.py`, and `tests/test_bot_factory.py`; focused pytest
  `-k "high_risk_decision or risk_weighted_decision_without_weighted_score or high_risk_proposal_without_local_falsification_handoff"`
  passed 4 tests; full `tests/test_bot_factory.py -q` passed with existing
  pandas `PerformanceWarning` warnings from `signal_diagnostics.py`.
- Audit conclusion remains unchanged: this closes another stale-positive
  evidence path, but it is not a profitable or paper-ready candidate.

## 2026-05-07 JST Candidate Failure Synthesis Local Rejection Validation

- Hardened `freqtrade_ext/bot_factory/candidate_failure_synthesis.py` so local
  falsification failures are counted as valid rejections only when the artifact
  is from `research_local_falsification`, has failed/rejected/blocked status,
  preserves historical-only safety scope, links to a valid Bot Factory local
  event source, proves closed-context alignment, and carries a passing
  failure-synthesis guard.
- The synthesis aggregate now records all supplied local falsification
  rejection artifacts separately from the subset counted as valid, including
  validation flags, event-source summary, invalid count, and failure reasons.
- Updated `tests/test_bot_factory.py` to keep the positive local rejection path
  fully validated and to prove that crafted/unsafe failed artifacts do not
  populate `local_falsification_failed_thesis_ids` or the valid rejection
  count.
- Verification passed: `py_compile` for
  `candidate_failure_synthesis.py` and `tests/test_bot_factory.py`; focused
  pytest `-k "candidate_failure_synthesis"` passed 3 tests; full
  `tests/test_bot_factory.py -q` passed with existing pandas
  `PerformanceWarning` warnings from `signal_diagnostics.py`; `git diff
  --check` passed with existing CRLF warnings only.
- Audit conclusion remains unchanged: the factory now remembers failed local
  screens more safely, but no profitable or paper-ready strategy exists yet.

## 2026-05-07 JST Research Selection Validated Local Rejection Memory

- Added end-to-end tests proving the safe-memory path from candidate failure
  synthesis into research selection: a fully validated local falsification
  rejection is counted by synthesis, added to failed mechanism memory, and then
  blocks research selection when the next thesis repeats that mechanism class.
- Added the paired negative test proving that a crafted/invalid failed local
  falsification artifact remains visible as invalid synthesis input but does
  not populate failed mechanism memory and does not block a new thesis through
  novelty checks.
- Verification passed: `py_compile tests/test_bot_factory.py`; focused pytest
  `-k "local_rejection_from_synthesis or candidate_failure_synthesis or prior_local_falsification_rejection"`
  passed 6 tests; full `tests/test_bot_factory.py -q` passed with existing
  pandas `PerformanceWarning` warnings from `signal_diagnostics.py`; `git diff
  --check` passed with existing CRLF warnings only.
- Audit conclusion remains unchanged: this reduces repeat-risk from failed
  pre-proposal local screens, but it is still factory control-plane work, not a
  profitable or paper-ready strategy.

## 2026-05-07 JST Research Selection Local Rejection Provenance

- Extended `freqtrade_ext/bot_factory/research_selection.py` so research
  decision artifacts distinguish validated local falsification rejection memory
  from generic failed-family memory. `novelty_assessment` now records local
  rejected thesis IDs, local rejected mechanism tokens, and exact local
  mechanism matches.
- Added explicit blocker
  `research_thesis_outside_failure_synthesis_local_rejections` and surfaced
  local rejection provenance in the `novelty_against_failure_set` score
  component and Markdown report. This makes a local-preproposal rejection
  auditable as such instead of hiding it as a generic repeated family.
- Strengthened tests to assert the provenance fields, score details, and
  explicit local-rejection blocker for validated local rejection memory while
  keeping invalid/crafted local rejection artifacts non-poisoning.
- Verification passed: `py_compile` for `research_selection.py` and
  `tests/test_bot_factory.py`; focused pytest
  `-k "local_rejection_from_synthesis or research_selection_gate_blocks_repeated_failed_family or prior_local_falsification_rejection"`
  passed 4 tests; full `tests/test_bot_factory.py -q` passed with existing
  pandas `PerformanceWarning` warnings from `signal_diagnostics.py`; `git diff
  --check` passed with existing CRLF warnings only.
- Audit conclusion remains unchanged: this improves traceability and reduces
  repeat-risk, but no profitable or paper-ready strategy exists yet.

## 2026-05-07 JST Proposal-Stage Local Rejection Novelty Continuity

- Extended `freqtrade_ext/bot_factory/strategy_proposals.py` so proposal
  generation validates the supplied research decision's novelty assessment
  instead of trusting approval flags alone. A research decision that still
  reports failed thesis/family matches now blocks proposal generation.
- Added explicit proposal blocker
  `research_decision_<n>_outside_failure_synthesis_local_rejections` when a
  research decision carries validated local falsification rejection matches.
  Proposal metadata now preserves the local rejection provenance fields inside
  `research_decision_constraints`.
- Added a crafted-approved research decision regression test proving that
  status=`approved_for_proposal_generation` and
  `proposal_generation_allowed=true` cannot bypass local rejection novelty
  memory.
- Verification passed: `py_compile` for `strategy_proposals.py` and
  `tests/test_bot_factory.py`; focused pytest
  `-k "local_rejection_novelty or unapproved_research_decision or high_risk_decision or research_decision_below_selection_score"`
  passed 5 tests after fixing an initially missing novelty extraction; full
  `tests/test_bot_factory.py -q` passed with existing pandas
  `PerformanceWarning` warnings from `signal_diagnostics.py`; `git diff
  --check` passed with existing CRLF warnings only.
- Audit conclusion remains unchanged: this closes a proposal-stage continuity
  bypass, but no profitable or paper-ready strategy exists yet.

## 2026-05-07 JST Codegen-Stage Local Rejection Novelty Continuity

- Extended `freqtrade_ext/bot_factory/strategy_code.py` so strategy code
  generation revalidates proposal metadata's `research_decision_constraints`
  before code emission. The generated metadata and `research_brief` now carry
  a `research_decision_novelty_handoff` summary.
- Added blocker `research_decision_novelty_handoff_passed` when stale or
  crafted accepted proposal metadata still records failed thesis ID matches,
  repeated failed-family matches, or validated local falsification rejection
  matches from the research decision.
- Added
  `test_strategy_code_generator_blocks_research_decision_local_rejection_novelty_handoff`
  to prove a local rejection novelty match cannot bypass proposal-stage
  blockers by reusing accepted proposal metadata.
- Verification passed: `py_compile` for `strategy_code.py` and
  `tests/test_bot_factory.py`; focused pytest
  `-k "research_decision_local_rejection_novelty_handoff or high_risk_proposal_without_local_falsification_handoff or structural_data_without_quality_handoff"`
  passed 3 tests; full `tests/test_bot_factory.py -q` passed with existing
  pandas `PerformanceWarning` warnings from `signal_diagnostics.py`; `git diff
  --check` passed with existing CRLF warnings only.
- Audit conclusion remains unchanged: this closes a codegen-stage stale
  proposal bypass, but no profitable or paper-ready strategy exists yet.

## 2026-05-07 JST Failure-Map Local Rejection Research Questions

- Extended `freqtrade_ext/bot_factory/candidate_failure_map.py` so validated
  local falsification rejection memory from candidate failure synthesis is
  promoted into causal failure map research guidance.
- `research_selection_guidance` now carries
  `validated_local_falsification_rejections`, appends local rejection prompts
  to `required_research_questions`, and explicitly blocks
  `retry_validated_local_rejection_by_parameter_tuning`.
- The causal failure map Markdown report now renders validated local
  falsification rejections with thesis ID, mechanism class, net edge, and
  profitable-window context.
- Strengthened `test_candidate_failure_map_builds_causal_categories` to prove
  the validated local rejection evidence survives into map guidance, required
  questions, blocked actions, and report text.
- Verification passed: `py_compile` for `candidate_failure_map.py` and
  `tests/test_bot_factory.py`; focused pytest
  `-k "candidate_failure_map_builds_causal_categories"` passed 1 test; full
  `tests/test_bot_factory.py -q` passed with existing pandas
  `PerformanceWarning` warnings from `signal_diagnostics.py`; `git diff
  --check` passed with existing CRLF warnings only.
- Audit conclusion remains unchanged: this improves failure-map-to-selection
  continuity, but no profitable or paper-ready strategy exists yet.

## 2026-05-07 JST Research-Selection Local Rejection Question Continuity

- Extended `freqtrade_ext/bot_factory/research_selection.py` so
  `validated_local_falsification_rejections` from causal failure map guidance
  are preserved in the research selection decision artifact and Markdown
  report.
- The report now shows validated local falsification rejection thesis IDs,
  mechanism classes, net edge, and profitable-window context in the causal
  failure map section.
- Added
  `test_research_selection_gate_requires_local_rejection_question_response`
  to prove a map-required local-rejection research question blocks selection
  when unanswered, remains visible in the decision/report, and passes only
  after a substantive indexed response is supplied.
- Verification passed: `py_compile` for `research_selection.py` and
  `tests/test_bot_factory.py`; focused pytest
  `-k "local_rejection_question_response or missing_required_research_question_responses or accepts_causal_failure_map_responses"`
  passed 3 tests; full `tests/test_bot_factory.py -q` passed with existing
  pandas `PerformanceWarning` warnings from `signal_diagnostics.py`; `git diff
  --check` passed with existing CRLF warnings only.
- Audit conclusion remains unchanged: this improves map-to-selection
  auditability and question enforcement, but no profitable or paper-ready
  strategy exists yet.

## 2026-05-07 JST Proposal-Stage Research Question Continuity

- Extended `freqtrade_ext/bot_factory/strategy_proposals.py` so proposal
  generation recomputes unanswered required research-question indexes from
  `required_research_questions` and `research_question_response_indexes`
  instead of trusting only the decision's reported missing-index list.
- Strengthened
  `research_decision_<n>_research_question_responses_complete` so its blocker
  details and `research_decision_constraints` preserve required question text,
  supplied response indexes, reported missing indexes, and recomputed missing
  indexes.
- Added
  `test_strategy_proposal_generator_recomputes_missing_research_question_responses`
  to prove a crafted approved research decision with an empty reported
  missing-index list still blocks proposal generation when a required question
  has no supplied response index.
- Verification passed: `py_compile` for `strategy_proposals.py` and
  `tests/test_bot_factory.py`; focused pytest
  `-k "recomputes_missing_research_question_responses or blocks_missing_research_question_responses or local_rejection_question_response"`
  passed 3 tests; full `tests/test_bot_factory.py -q` passed with existing
  pandas `PerformanceWarning` warnings from `signal_diagnostics.py`; `git diff
  --check` passed with existing CRLF warnings only.
- Audit conclusion remains unchanged: this closes a proposal-stage crafted
  decision bypass for required research questions, but no profitable or
  paper-ready strategy exists yet.

## 2026-05-07 JST Codegen-Stage Research Question Continuity

- Extended `freqtrade_ext/bot_factory/strategy_code.py` so strategy code
  generation revalidates accepted proposal metadata's
  `research_decision_constraints` for required research-question completion
  before emitting code.
- Added blocker `research_decision_question_handoff_passed`, which recomputes
  missing indexes from `required_research_questions` and
  `research_question_response_indexes`, then unions them with proposal-stage
  reported/computed missing indexes and weak indexes.
- Generated metadata and `research_brief` now preserve
  `research_decision_question_handoff` with required question text, supplied
  response indexes, reported missing indexes, upstream computed missing
  indexes, codegen recomputed missing indexes, and failed paths.
- Added
  `test_strategy_code_generator_blocks_research_decision_question_handoff`
  to prove crafted accepted proposal metadata cannot bypass proposal-stage
  question checks and reach strategy code emission.
- Verification passed: `py_compile` for `strategy_code.py` and
  `tests/test_bot_factory.py`; focused pytest
  `-k "research_decision_question_handoff or research_decision_local_rejection_novelty_handoff or recomputes_missing_research_question_responses"`
  passed 3 tests; full `tests/test_bot_factory.py -q` passed with existing
  pandas `PerformanceWarning` warnings from `signal_diagnostics.py`; `git diff
  --check` passed with existing CRLF warnings only.
- Audit conclusion remains unchanged: this closes a codegen-stage crafted
  proposal bypass for required research questions, but no profitable or
  paper-ready strategy exists yet.

## 2026-05-07 JST Iteration Blocked-Action Continuity

- Extended `freqtrade_ext/bot_factory/candidate_iteration.py` so iteration
  plans preserve `blocked_next_actions` from `next_candidate_input`, research
  briefs, and failure-evidence summaries.
- Added check `revision_avoids_blocked_next_actions`, which blocks revision
  text in changed assumptions, changed parameters, or changed data requirements
  when it repeats a causal failure map blocked action such as
  `retry_validated_local_rejection_by_parameter_tuning`.
- Added
  `test_candidate_iteration_blocks_causal_failure_map_blocked_next_action` to
  prove the iteration loop cannot repackage a validated local rejection retry
  as a parameter-tuning assumption while preserving otherwise valid research
  and walk-forward context.
- Verification passed: `py_compile` for `candidate_iteration.py` and
  `tests/test_bot_factory.py`; focused pytest
  `-k "candidate_iteration_blocks_causal_failure_map_blocked_next_action or candidate_iteration_plan_preserves_lineage or candidate_iteration_force_distinct or candidate_iteration_requires_research_brief"`
  passed 4 tests; full `tests/test_bot_factory.py -q` passed with existing
  pandas `PerformanceWarning` warnings from `signal_diagnostics.py`; `git diff
  --check` passed with existing CRLF warnings only.
- Audit conclusion remains unchanged: this closes an iteration-loop fallback
  into blocked parameter-tuning retries, but no profitable or paper-ready
  strategy exists yet.

## 2026-05-07 JST Evaluation-to-Iteration Blocked-Action Continuity

- Extended `freqtrade_ext/bot_factory/candidate_evaluation.py` so
  `blocked_next_actions` are preserved from proposal/generated research briefs
  and proposal `failure_synthesis_constraints` /
  `research_decision_constraints` into both candidate manifest
  `research_brief` and `next_candidate_input`.
- Added
  `test_candidate_evaluation_carries_blocked_next_actions_to_iteration` to
  prove a normal evaluation manifest passes
  `retry_validated_local_rejection_by_parameter_tuning` to the iteration loop,
  where `revision_avoids_blocked_next_actions` blocks the attempted retry.
- Verification passed: `py_compile` for `candidate_evaluation.py`,
  `candidate_iteration.py`, and `tests/test_bot_factory.py`; focused pytest
  `-k "candidate_evaluation_carries_blocked_next_actions_to_iteration or candidate_iteration_blocks_causal_failure_map_blocked_next_action or candidate_evaluation_writes_manifest_and_index"`
  passed 3 tests; full `tests/test_bot_factory.py -q` passed with existing
  pandas `PerformanceWarning` warnings from `signal_diagnostics.py`; `git diff
  --check` passed with existing CRLF warnings only.
- Audit conclusion remains unchanged: this closes a local
  evaluation-to-iteration memory drop, but no profitable or paper-ready
  strategy exists yet.

## 2026-05-07 JST Evaluation Research Handoff Preservation

- Extended `freqtrade_ext/bot_factory/candidate_evaluation.py` so candidate
  manifests preserve research handoff summaries from generated/proposal
  metadata inside the manifest `research_brief`:
  `research_decision_question_handoff`,
  `research_decision_novelty_handoff`, `local_falsification_handoff`,
  `structural_data_quality_handoff`, and
  `structural_data_capability_handoff`.
- Added
  `test_candidate_evaluation_preserves_research_handoff_summaries` to prove
  research decision question and novelty handoff summaries survive evaluation
  manifest summarization.
- Verification passed: `py_compile` for `candidate_evaluation.py` and
  `tests/test_bot_factory.py`; focused pytest
  `-k "candidate_evaluation_preserves_research_handoff_summaries or candidate_evaluation_carries_blocked_next_actions_to_iteration or candidate_evaluation_writes_manifest_and_index"`
  passed 3 tests; full `tests/test_bot_factory.py -q` passed with existing
  pandas `PerformanceWarning` warnings from `signal_diagnostics.py`; `git diff
  --check` passed with existing CRLF warnings only.
- Audit conclusion remains unchanged: this preserves handoff auditability after
  candidate evaluation, but no profitable or paper-ready strategy exists yet.

## 2026-05-07 JST Ranking Research Handoff Continuity

- Extended `freqtrade_ext/bot_factory/candidate_ranking.py` so ranked candidate
  rows preserve `blocked_next_actions`, compact `research_brief` context, and
  `research_handoff_summary` entries for research-question, novelty, local
  falsification, and structural-data handoffs.
- Extended `freqtrade_ext/bot_factory/candidate_failure_synthesis.py` so
  synthesis can recover those fields from ranking output when the original
  manifest is unavailable, aggregate blocked actions, and pass handoff
  summaries into `next_research_brief`.
- Added
  `test_candidate_ranking_preserves_research_handoff_context` and
  `test_candidate_failure_synthesis_uses_ranking_research_context_without_manifest`
  to prove ranking and downstream synthesis no longer drop research handoff or
  blocked-action memory.
- Verification passed: `py_compile` for `candidate_ranking.py`,
  `candidate_failure_synthesis.py`, and `tests/test_bot_factory.py`; focused
  pytest
  `-k "candidate_ranking_preserves_research_handoff_context or candidate_failure_synthesis_uses_ranking_research_context_without_manifest or candidate_failure_synthesis_builds_theory_first_next_brief or candidate_ranking_compares_candidates_and_gates_paper_ready"`
  passed 4 tests after correcting blocked-action ordering; full
  `tests/test_bot_factory.py -q` passed with existing pandas
  `PerformanceWarning` warnings from `signal_diagnostics.py`; `git diff
  --check` passed with existing CRLF warnings only.
- Audit conclusion remains unchanged: this closes a ranking-to-synthesis
  context drop, but no profitable or paper-ready strategy exists yet.

## 2026-05-07 JST Failure-Map Research Handoff Continuity

- Extended `freqtrade_ext/bot_factory/candidate_failure_map.py` so
  `research_selection_guidance` preserves synthesis-level
  `research_handoff_summaries` and merges upstream `blocked_next_actions`
  instead of relying only on the failure-map default blocked action list.
- Extended `freqtrade_ext/bot_factory/research_selection.py` so causal failure
  map constraints and the decision `causal_failure_map` summary preserve
  `research_handoff_summaries`.
- Added coverage in
  `test_candidate_failure_map_builds_causal_categories` and
  `test_research_selection_gate_preserves_causal_map_handoff_summaries` to
  prove handoff summaries survive through failure map and research selection.
- Verification passed: `py_compile` for `candidate_failure_map.py`,
  `research_selection.py`, and `tests/test_bot_factory.py`; focused pytest
  `-k "candidate_failure_map_builds_causal_categories or research_selection_gate_preserves_causal_map_handoff_summaries or research_selection_gate_blocks_causal_failure_map_synthesis_mismatch"`
  passed 3 tests; full `tests/test_bot_factory.py -q` passed with existing
  pandas `PerformanceWarning` warnings from `signal_diagnostics.py`; `git diff
  --check` passed with existing CRLF warnings only.
- Audit conclusion remains unchanged: this closes a failure-map to research
  selection context drop, but no profitable or paper-ready strategy exists yet.

## 2026-05-07 JST Proposal Research Handoff Continuity

- Extended `freqtrade_ext/bot_factory/research_selection.py` so decision
  `causal_failure_map` summaries preserve `blocked_next_actions` alongside
  `research_handoff_summaries`.
- Extended `freqtrade_ext/bot_factory/strategy_proposals.py` so proposal
  `research_decision_constraints` preserve causal-map blocked actions and
  handoff summaries, and proposal `research_brief` merges them with upstream
  failure-synthesis blocked actions.
- Added coverage in
  `test_strategy_proposal_generator_accepts_distinct_synthesis_thesis` and
  `test_research_selection_gate_preserves_causal_map_handoff_summaries` to
  prove the research-selection to proposal handoff no longer drops this
  context.
- Verification passed: `py_compile` for `research_selection.py`,
  `strategy_proposals.py`, and `tests/test_bot_factory.py`; focused pytest
  `-k "strategy_proposal_generator_accepts_distinct_synthesis_thesis or research_selection_gate_preserves_causal_map_handoff_summaries or strategy_proposal_generator_requires_research_decision_after_synthesis"`
  passed 3 tests; full `tests/test_bot_factory.py -q` passed with existing
  pandas `PerformanceWarning` warnings from `signal_diagnostics.py`; `git diff
  --check` passed with existing CRLF warnings only.
- Audit conclusion remains unchanged: this closes a research-selection to
  proposal context drop, but no profitable or paper-ready strategy exists yet.

## 2026-05-07 JST Codegen/Evaluation Handoff Summary Continuity

- Extended `freqtrade_ext/bot_factory/strategy_code.py` so generated metadata
  and generated `research_brief.json` preserve proposal-level
  `blocked_next_actions` and `research_handoff_summaries`.
- Extended `freqtrade_ext/bot_factory/candidate_evaluation.py` so candidate
  manifests preserve generic `research_handoff_summaries` from generated and
  proposal metadata, their nested research briefs, and research decision
  constraints.
- Added coverage in
  `test_strategy_code_generator_blocks_research_decision_question_handoff` and
  `test_candidate_evaluation_preserves_research_handoff_summaries` to prove
  generic handoff summaries and blocked actions survive codegen and candidate
  evaluation.
- Verification passed: `py_compile` for `strategy_code.py`,
  `candidate_evaluation.py`, and `tests/test_bot_factory.py`; focused pytest
  `-k "strategy_code_generator_blocks_research_decision_question_handoff or candidate_evaluation_preserves_research_handoff_summaries or candidate_evaluation_carries_blocked_next_actions_to_iteration"`
  passed 3 tests; full `tests/test_bot_factory.py -q` passed with existing
  pandas `PerformanceWarning` warnings from `signal_diagnostics.py`; `git diff
  --check` passed with existing CRLF warnings only.
- Audit conclusion remains unchanged: this closes a proposal/codegen/evaluation
  context drop, but no profitable or paper-ready strategy exists yet.

## 2026-05-07 JST Worktree Hygiene Recheck After Codegen/Evaluation

- Rechecked the working tree after the codegen/evaluation handoff summary
  increment with `git status --short --untracked-files=all`, `git diff
  --stat`, `git diff --name-status`, `git ls-files --others
  --exclude-standard`, and a Bot Factory-root-limited ignored-artifact scan.
- Current visible Git candidates remain source/doc/test/template changes:
  `.gitignore`, Bot Factory docs, modules under `freqtrade_ext/bot_factory/`,
  entrypoints under `scripts/`, `tests/test_bot_factory.py`, and
  `registry/strategies/proposals/TEMPLATE.md`.
- Current non-ignored untracked files are all Bot Factory source/script/doc
  files: `docs/BOT_FACTORY_GOAL_AUDIT.md`, 10 modules under
  `freqtrade_ext/bot_factory/`, and 11 `scripts/bot_factory_*.py` entrypoints.
  They should remain review candidates and should not be treated as generated
  disposable output.
- Current ignored artifact counts confirm the large runtime evidence trail is
  excluded from Git by `.gitignore`: `data/freqai` 58,
  `data/freqai_training` 72, `data/walk_forward` 1761,
  `registry/strategies/candidates` 382,
  `registry/strategies/diagnostics` 70,
  `registry/strategies/failure_maps` 34,
  `registry/strategies/generated` 133,
  `registry/strategies/proposals` 135,
  `registry/strategies/research_decisions` 134,
  `registry/strategies/reviews` 12, and
  `registry/strategies/synthesis` 68.
- Safe deletion remains limited to `.pytest_cache/` and non-`.venv`
  `__pycache__/` caches. Those were removed after the latest pytest run with a
  path-guarded cleanup (`removed=33`). Broader ignored JSON, CSV, Markdown,
  zip, and log artifacts were retained because they are the local evidence
  trail for rejected hypotheses, diagnostics, and verification runs.
- Owner decision remains required for the tracked deletion of
  `docs/BOT_FACTORY_GOAL_COMMAND_RUNBOOK.md`. Accept the deletion only if its
  useful content is superseded by `docs/BOT_FACTORY_MVP_TODO.md`, this audit,
  and the next-agent prompt; otherwise restore or intentionally merge it.
- Audit conclusion: the worktree is still large, but generated runtime outputs
  are not leaking into the Git candidate set. The next cleanup task is
  review/commit grouping, plus an owner decision on the deleted runbook. This
  hygiene recheck does not change the main objective status: there is still no
  profitable or paper-ready Bot Factory candidate.

## 2026-05-07 JST Funding-Neutral Impulse Drift Local Rejection

- Used the latest 31-candidate evidence before attempting another thesis:
  `registry/strategies/synthesis/20260507T224000JST_all_candidates_with_three_open_interest_rejections/candidate_failure_synthesis.json`
  and
  `registry/strategies/failure_maps/20260507T224500JST_all_candidates_with_three_open_interest_rejections_causal_map/causal_failure_map.json`.
- Built a new pre-proposal local event study for
  `TH-FUNDING-NEUTRAL-IMPULSE-DRIFT-20260507` /
  `funding_neutral_impulse_drift` using closed BTC 5m OHLCV and local 8h
  funding-rate history. The event spec tested upward 12-candle impulse with
  elevated local volume while funding was not already crowded positive.
- Event generation completed with `event_count=475` and `blocker_count=0`.
  Artifacts were written under
  `registry/strategies/research_decisions/20260507T230000JST_funding_neutral_impulse_drift_event_study/`.
- Local falsification intentionally failed and must block proposal/codegen for
  this thesis: `expected_edge_bps=-2.15646`, `all_in_cost_bps=12.0`,
  `net_edge_bps=-14.15646`, `sample_count=475`,
  `data_span_days=397.222222`, `profitable_windows_ratio=0.0`, and
  `blocker_count=2`. Failed checks were `expected_edge_exceeds_all_in_cost`
  and `profitable_windows_ratio_sufficient`.
- Refreshed rejection memory:
  `registry/strategies/synthesis/20260507T230600JST_all_candidates_with_funding_neutral_impulse_rejection/candidate_failure_synthesis.json`
  completed with `candidate_count=31`, `paper_ready_count=0`,
  `parameter_only_retry_allowed=false`, `requires_new_thesis_id=true`, and
  `local_falsification_rejection_count=7`.
- Refreshed causal map:
  `registry/strategies/failure_maps/20260507T231000JST_all_candidates_with_funding_neutral_impulse_rejection_causal_map/causal_failure_map.json`
  completed with `category_count=8` and
  `requires_research_decision_before_proposal=true`. It now includes a
  required question to avoid the validated local rejection for
  `funding_neutral_impulse_drift`.
- Audit conclusion remains unchanged: this is useful rejection evidence and
  anti-repeat memory, not a profitable or paper-ready candidate. Do not promote
  it and do not generate strategy code from this thesis.

## 2026-05-07 JST Funding-Adjusted Local Falsification Edge

- Extended local pre-proposal falsification so funding-aware theses can be
  measured against realized historical funding payments instead of price return
  alone. `freqtrade_ext/bot_factory/local_falsification.py` and
  `scripts/bot_factory_build_local_falsification.py` now accept optional
  `--funding-rate-path`.
- Funding-adjusted edge uses conservative long-payment semantics:
  `long_funding_adjustment_bps = -sum(funding_rate * 10000)` for funding
  timestamps after entry and up to exit. Artifacts separately record
  `expected_price_edge_bps`, `expected_funding_adjustment_bps`, and combined
  `expected_edge_bps`.
- Added focused regression coverage in
  `test_local_falsification_can_include_realized_long_funding_adjustment`;
  focused pytest also rechecked the existing price-only passing and failing
  local falsification tests.
- Added a funding-context guard: if a supplied `local_events.json` declares
  `funding_rate` in `context_merge.required_contexts`, local falsification now
  requires `--funding-rate-path` and otherwise blocks on
  `funding_rate_path_present_for_funding_event_source`. The focused regression
  `test_local_falsification_requires_funding_path_for_funding_event_source`
  covers this.
- Re-ran `TH-FUNDING-NEUTRAL-IMPULSE-DRIFT-20260507` with the local BTC 8h
  funding-rate parquet. It still failed:
  `expected_price_edge_bps=-2.15646`,
  `expected_funding_adjustment_bps=-0.048684`,
  `expected_edge_bps=-2.205144`, `all_in_cost_bps=12.0`,
  `net_edge_bps=-14.205144`, `sample_count=475`,
  `data_span_days=397.222222`, and `profitable_windows_ratio=0.0`.
- Refreshed rejection memory with the funding-adjusted artifact:
  `registry/strategies/synthesis/20260507T232000JST_all_candidates_with_funding_adjusted_impulse_rejection/candidate_failure_synthesis.json`
  completed with `local_falsification_rejection_count=7`, and
  `registry/strategies/failure_maps/20260507T232500JST_all_candidates_with_funding_adjusted_impulse_rejection_causal_map/causal_failure_map.json`
  completed with `requires_research_decision_before_proposal=true`.
- Verification passed: `py_compile` for local falsification implementation,
  CLI, and tests; focused pytest
  `-k "local_falsification_requires_funding_path_for_funding_event_source or local_falsification_can_include_realized_long_funding_adjustment or local_falsification_builds_cost_edge_artifact or local_falsification_blocks_cost_edge_below_cost"`
  passed 4 tests; full `tests/test_bot_factory.py -q` passed with existing
  pandas `PerformanceWarning` warnings from `signal_diagnostics.py`.
- Audit conclusion remains unchanged: the accounting is more correct, but it
  produced a stronger rejection rather than a profitable or paper-ready
  candidate.

## 2026-05-07 JST Bybit Long/Short Ratio Structural Data Input

- Added public Bybit V5 account long/short ratio support as a local structural
  market-data input for theory-first research. The downloader targets
  `GET /v5/market/account-ratio`, normalizes `buyRatio`/`sellRatio`, writes
  local parquet/CSV outputs, and records a safety scope showing no API keys,
  no account data, no order endpoints, no trading process, no leverage change,
  and no shorting.
- Added quality checks for local long/short ratio parquet files:
  `check_long_short_ratio_parquet` validates `date`,
  `long_account_ratio`/`buyRatio`, `short_account_ratio`/`sellRatio`,
  account-ratio bounds, duplicates, ordering, expected intervals, and
  consistency warnings.
- Extended structural capability reporting with `long_short_ratio`. It can
  become `local_research_usable` only when both local data and a passing
  quality report are supplied; it remains blocked for strategy codegen.
- Verification passed:
  `py_compile` for the new downloader/check scripts, data quality,
  structural capability code, and tests; focused pytest
  `-k "long_short_ratio or structural_data_capability_report_marks"` passed 6
  tests; full `tests/test_bot_factory.py -q` passed with existing pandas
  `PerformanceWarning` warnings from `signal_diagnostics.py`; `git diff
  --check` passed with CRLF normalization warnings only. Post-test cleanup
  removed 33 workspace-local `.pytest_cache`/`__pycache__` targets outside
  `.venv`, leaving 0 remaining cache targets.
- Audit conclusion remains unchanged: this improves the research input
  surface for new non-parameter theses, but it is not a profitable or
  paper-ready candidate and does not authorize proposal/codegen or any
  exchange-facing trading action.

## 2026-05-07 JST Long/Short Ratio Local Event Context

- Connected local long/short account-ratio data to the pre-proposal local
  event builder. Event specs can now reference `long_account_ratio`,
  `long_account_ratio_delta_bps`, `long_short_ratio`, and
  `long_short_ratio_zscore` after a local long/short ratio parquet/CSV is
  supplied with `--long-short-ratio-path`.
- The merge uses the existing closed-context semantics:
  `closed_context_candle_availability_v1`. Coarser long/short ratio candles
  are shifted onto base OHLCV only after the context candle is closed, matching
  the funding, mark-price, and open-interest safety model.
- Artifacts now expose `source_long_short_ratio_path`,
  `auxiliary_sources.long_short_ratio`, and
  `context_merge.contexts.long_short_ratio`, so downstream local falsification
  and review can see that the event source used local structural market data.
- Verification passed:
  `py_compile` for local events, the local-event CLI, and tests; focused
  pytest
  `-k "long_short_ratio or local_event_builder_supports_open_interest_context_features or local_event_builder_blocks_missing_required_futures_context"`
  passed 8 tests; full `tests/test_bot_factory.py -q` passed with existing
  pandas `PerformanceWarning` warnings from `signal_diagnostics.py`; `git
  diff --check` passed with CRLF normalization warnings only. Post-test
  cleanup removed 33 workspace-local `.pytest_cache`/`__pycache__` targets
  outside `.venv`, leaving 0 remaining cache targets.
- Audit conclusion remains unchanged: this removes a usability gap in the
  theory-first research loop, but it is not a profitable or paper-ready
  candidate and does not authorize generated strategy code or trading.

## 2026-05-07 JST Structural Data Safe Paths And Long/Short Sample Rejection

- Found a safety/operability issue in the new structural-data storage path:
  putting `BTC_USDT_USDT-1h-long_short_ratio.parquet` under
  `user_data/data/bybit/futures` made Freqtrade `download-data` migration
  parse `long_short_ratio` as a candle type and fail with
  `ValueError: 'long_short_ratio' is not a valid CandleType`.
- Moved generated open-interest and long/short ratio parquets to
  `data/market_structure/bybit/futures/`, added that runtime tree to
  `.gitignore`, and changed both Bybit structural downloaders to use the safe
  path by default. A tracked `data/market_structure/.gitkeep` preserves the
  directory anchor.
- Re-ran `scripts/bot_factory_download_data.py` for BTC/USDT:USDT 5m futures
  over `20260401-20260507`; it completed after the move. The quality report
  passed with `rows=246895`, no duplicate timestamps, no missing intervals,
  and no findings.
- Rechecked safe-path structural data:
  long/short ratio quality passed with `rows=865`; open interest quality
  passed with `rows=9529`; combined structural capability now marks
  `local_research_usable=["open_interest","long_short_ratio"]` while keeping
  both in `must_not_codegen`.
- Ran fixed pre-proposal thesis
  `TH-LONG-SHORT-FLUSH-REBOUND-20260507` /
  `long_short_crowding_flush_rebound` after aligning OHLCV and long/short
  periods. Local events completed with `event_count=8`, but local
  falsification failed the sample gate:
  `expected_edge_bps=49.356695`, `net_edge_bps=37.356695`,
  `sample_count=8`, `profitable_windows_ratio=1.0`, `blocker_count=1`.
- Refreshed failure memory:
  `registry/strategies/synthesis/20260507T155000JST_all_candidates_with_long_short_sample_rejection/candidate_failure_synthesis.json`
  completed with `paper_ready_count=0` and
  `local_falsification_rejection_count=8`; causal map
  `registry/strategies/failure_maps/20260507T155500JST_all_candidates_with_long_short_sample_rejection_causal_map/causal_failure_map.json`
  completed with `requires_research_decision_before_proposal=true`.
- Verification passed:
  `py_compile` for structural downloaders, local events, CLI, and tests;
  focused pytest for safe defaults and long/short context passed 5 tests; full
  `tests/test_bot_factory.py -q` passed with existing pandas
  `PerformanceWarning` warnings from `signal_diagnostics.py`; `git diff
  --check` passed with CRLF normalization warnings only. Post-test cleanup
  removed 33 workspace-local `.pytest_cache`/`__pycache__` targets outside
  `.venv`, leaving 0 remaining cache targets.
- Audit conclusion remains unchanged: this is useful plumbing and a properly
  rejected small-sample signal, not a profitable or paper-ready candidate.

## 2026-05-07 JST Extended Long/Short Ratio Falsification

- Extended Bybit BTCUSDT 1h public long/short ratio history from
  `2024-01-01` through `2026-05-07` without changing thesis thresholds.
  Download completed with `row_count=20569`, `page_count=42`,
  `truncated=false`, `api_key_used=false`, and `order_endpoint_used=false`.
- Quality check passed on the safe structural path
  `data/market_structure/bybit/futures/BTC_USDT_USDT-1h-long_short_ratio.parquet`:
  `ok=true`, `duplicate_timestamps=0`, `missing_intervals=0`, and
  `findings=[]`. Updated structural capability still marks
  `local_research_usable=["open_interest","long_short_ratio"]` and
  keeps both structural inputs in `must_not_codegen`.
- Re-ran the same fixed `TH-LONG-SHORT-FLUSH-REBOUND-20260507` /
  `long_short_crowding_flush_rebound` event spec as a diagnostic after history
  extension. This used `--allow-failed-thesis-or-family` because the prior
  small-sample rejection was already in failure memory; no thresholds were
  changed.
- Local events completed with `event_count=608`, but extended-history local
  falsification failed the real gates:
  `expected_edge_bps=1.288089`, `net_edge_bps=-10.711911`,
  `sample_count=608`, `data_span_days=857.270833`,
  `profitable_windows_ratio=0.25`, and `blocker_count=2`. The thesis is now a
  cost/stability rejection, not merely a sample-size rejection.
- Refreshed failure memory:
  `registry/strategies/synthesis/20260507T162000JST_all_candidates_with_long_short_extended_rejection/candidate_failure_synthesis.json`
  completed with `candidate_count=31`, `paper_ready_count=0`,
  `parameter_only_retry_allowed=false`, `requires_new_thesis_id=true`, and
  `local_falsification_rejection_count=7`. The refreshed map
  `registry/strategies/failure_maps/20260507T162500JST_all_candidates_with_long_short_extended_rejection_causal_map/causal_failure_map.json`
  requires a research decision before any proposal.
- Verification: no code changed during this extended-history rerun after the
  prior passing test suite. `git diff --check` passed with CRLF normalization
  warnings only. Post-script cleanup removed 2 workspace-local `__pycache__`
  targets outside `.venv`, leaving 0 remaining cache targets.
- Audit conclusion remains unchanged: longer data turned an apparent small
  positive sample into a robust rejection. No profitable or paper-ready Bot
  Factory candidate exists yet.

## 2026-05-07 JST Negative Funding Uncrowded Carry Screen

- Tested a distinct fixed pre-proposal thesis:
  `TH-NEGATIVE-FUNDING-UNCROWDED-CARRY-20260507` /
  `negative_funding_uncrowded_long_carry`. It combined local 8h funding,
  local 1h long/short account ratio, and BTC 5m OHLCV under closed-context
  candle semantics.
- Local events completed, but only with `event_count=1`. Funding-adjusted
  local falsification failed the sample gate:
  `expected_price_edge_bps=13.079717`,
  `expected_funding_adjustment_bps=0.7656`,
  `expected_edge_bps=13.845317`, `net_edge_bps=1.845317`,
  `sample_count=1`, `profitable_windows_ratio=1.0`, and `blocker_count=1`.
  This is not promotable despite the single observed positive event.
- Refreshed failure memory:
  `registry/strategies/synthesis/20260507T164000JST_all_candidates_with_negative_funding_sample_rejection/candidate_failure_synthesis.json`
  completed with `candidate_count=31`, `paper_ready_count=0`,
  `parameter_only_retry_allowed=false`, `requires_new_thesis_id=true`, and
  `local_falsification_rejection_count=8`. The refreshed map
  `registry/strategies/failure_maps/20260507T164500JST_all_candidates_with_negative_funding_sample_rejection_causal_map/causal_failure_map.json`
  requires a research decision before any proposal.
- Verification: no code changed during this fixed-thesis screen after the
  prior passing test suite. `git diff --check` passed with CRLF normalization
  warnings only. Post-script cleanup removed 2 workspace-local `__pycache__`
  targets outside `.venv`, leaving 0 remaining cache targets.
- Audit conclusion remains unchanged: this is a small-sample rejection, not a
  profitable or paper-ready candidate.

## 2026-05-07 JST Calendar-Window Failure-Memory Handoff

- Objective gap addressed: quarterly calendar-window diagnostics had been
  added to local falsification artifacts, but they were not yet preserved
  through failure synthesis, causal failure maps, and research-selection
  reports. That could let future research selection ignore calendar-regime
  instability even when local falsification recorded it.
- Implemented evidence continuity:
  `candidate_failure_synthesis.py` now carries
  `calendar_window_frequency`, `calendar_window_count`,
  `profitable_calendar_windows_ratio`, and compact
  `calendar_window_summaries` for local falsification rejections.
  `candidate_failure_map.py` carries those fields into validated local
  rejection guidance and required local-rejection questions. Research
  selection reports now render `profitable_calendar_windows_ratio`.
- Verification passed:
  `py_compile` for `candidate_failure_synthesis.py`,
  `candidate_failure_map.py`, `research_selection.py`, and
  `tests/test_bot_factory.py`; focused pytest for synthesis/map/research
  selection handoff passed 3 tests; full `tests/test_bot_factory.py -q`
  passed with existing pandas `PerformanceWarning` warnings from
  `signal_diagnostics.py`.
- Audit conclusion remains unchanged: this is stronger anti-repeat and
  calendar-regime evidence plumbing, not a profitable or paper-ready strategy.

## 2026-05-07 JST Calendar-Window Research Response Gate

- Objective gap addressed: after calendar-window rejection evidence reached
  research selection, an indexed research-question answer could still satisfy
  the generic word-count gate while ignoring the calendar evidence itself.
- Implemented gate hardening: research questions that mention
  `calendar_window`, `profitable_calendar_windows_ratio`,
  `calendar_window_summaries`, `quarterly`, or `quarter` now require the
  answer to address calendar-window evidence. Missing calendar evidence marks
  the answer weak with `calendar_window_evidence` and blocks selection.
- Verification passed:
  `py_compile` for `research_selection.py` and `tests/test_bot_factory.py`;
  focused pytest for the calendar-window response gate, local-rejection
  question gate, and causal-map handoff passed 3 tests; full
  `tests/test_bot_factory.py -q` passed with existing pandas
  `PerformanceWarning` warnings from `signal_diagnostics.py`.
- Audit conclusion remains unchanged: this closes a research-selection bypass,
  but no profitable or paper-ready strategy exists yet.

## 2026-05-07 JST Current Completion Audit Refresh

Objective restatement as concrete deliverables:

- AI-assisted Bot Factory must perform theory-first research intake, not
  AI-driven parameter optimization.
- Approved theory must be convertible into safe long-only strategy code.
- Generated candidates must be evaluated through local historical,
  walk-forward, diagnostic, ranking, synthesis, and failure-map artifacts.
- Failure memory must block repeated failed theses, parameter-only retries,
  stale evidence, and weak causal responses before proposal/codegen.
- At least one locally evaluated candidate must pass the required historical
  and walk-forward gates before paper readiness can be considered.
- No paper, dry-run, live, exchange-order, secret, shorting, leverage above
  `1.0`, or process-control path may start without explicit approval and
  passing readiness evidence.

Prompt-to-artifact checklist against current evidence:

- Requirement: theory-first research before code.
  Evidence inspected: `research_selection.py`,
  `scripts/bot_factory_select_research_thesis.py`, causal failure map
  `20260507T164500JST_all_candidates_with_negative_funding_sample_rejection_causal_map`,
  and tests around required causal/research-question responses. Status:
  infrastructure present, profitability not proven.
- Requirement: avoid parameter optimization drift.
  Evidence inspected: latest synthesis
  `20260507T164000JST_all_candidates_with_negative_funding_sample_rejection`
  has `parameter_only_retry_allowed=false` and
  `requires_new_thesis_id=true`; selection gates block parameter-only causal
  responses and now block calendar-window questions that ignore calendar
  evidence. Status: guardrails present.
- Requirement: code generation from approved theory.
  Evidence inspected: proposal/codegen modules and previously generated
  candidates. Status: supported for implemented variants, but not proof of a
  profitable strategy.
- Requirement: local historical evaluation and failure memory.
  Evidence inspected: latest synthesis JSON has `status=completed`,
  `candidate_count=31`, `paper_ready_count=0`, and
  `local_falsification_rejection_count=8`. Latest map JSON has
  `status=completed`, `candidate_count=31`, `category_count=8`,
  `requires_research_decision_before_proposal=true`,
  `requires_research_question_responses=true`,
  `minimum_research_selection_score=80`, and
  `validated_local_falsification_rejections=8`. Status: failure memory
  current and usable.
- Requirement: profitable or paper-ready candidate.
  Evidence inspected: latest synthesis reports `paper_ready_count=0`; recent
  local screens are rejected for cost/stability or sample-size reasons.
  Status: not achieved.
- Requirement: safe scope.
  Evidence inspected: latest artifacts and recorded commands remain
  historical/local-only; no paper/dry-run/live/order endpoint path is
  authorized. Status: satisfied for the current work.

Completion decision: do not mark the goal complete. The factory has stronger
research/code/evaluation/failure-memory infrastructure, but the central
success criterion remains missing: no profitable, robust, paper-ready
candidate exists.

## 2026-05-07 JST Stale Failure-Synthesis Selection Guard

- Objective gap addressed: an old failure synthesis plus matching old causal
  map could be supplied to research selection even after newer failure memory
  existed locally. That risked selecting a thesis against stale negative
  evidence.
- Implemented gate hardening: research selection now scans
  `registry/strategies/synthesis/**/candidate_failure_synthesis.json`, compares
  parseable Bot Factory synthesis `generated_at` values, and blocks a supplied
  synthesis when a newer local synthesis exists. The decision records the
  latest-path evidence in `novelty_assessment`.
- Verification passed:
  `py_compile` for `research_selection.py` and `tests/test_bot_factory.py`;
  focused pytest for stale-synthesis blocking, causal map/synthesis mismatch,
  and causal-map acceptance passed 3 tests; full `tests/test_bot_factory.py -q`
  passed with existing pandas `PerformanceWarning` warnings from
  `signal_diagnostics.py`.
- Audit conclusion remains unchanged: this prevents stale failure-memory
  bypass, but no profitable or paper-ready strategy exists yet.

## 2026-05-07 JST Stale Failure-Synthesis Proposal Guard

- Objective gap addressed: after the research-selection freshness check was
  added, an already-created old `research_decision.json` could still be paired
  with the same old failure synthesis at proposal generation. That would have
  bypassed newer local failure memory after the decision artifact was written.
- Implemented gate hardening: strategy proposal generation now recomputes
  latest failure-synthesis freshness from
  `registry/strategies/synthesis/**/candidate_failure_synthesis.json` and
  blocks stale supplied synthesis evidence. It also blocks research decisions
  whose `novelty_assessment` explicitly records
  `failure_synthesis_latest_checked=true` and
  `failure_synthesis_is_latest=false`.
- Verification passed:
  `py_compile` for `strategy_proposals.py` and `tests/test_bot_factory.py`;
  focused pytest for proposal-stage and research-selection stale synthesis
  blocking passed 2 tests; full `tests/test_bot_factory.py -q` passed with
  existing pandas `PerformanceWarning` warnings from `signal_diagnostics.py`.
- Audit conclusion remains unchanged: this closes another stale-memory bypass,
  but the main goal is still not complete because no profitable or paper-ready
  strategy exists.

## 2026-05-07 JST Crowding-Unwind Reaccumulation Research Selection

- Objective progress: a distinct thesis,
  `TH-CROWDING-UNWIND-REACCUMULATION-20260507`, now has passing
  pre-proposal local evidence. The local falsification artifact reports
  `status=passed`, `net_edge_bps=10.397636` after a 12 bps all-in cost gate,
  `sample_count=491`, `data_span_days=857.270833`, and
  `profitable_windows_ratio=0.75`.
- Research gate result: corrected research selection v3 completed with
  `status=approved_for_proposal_generation`,
  `proposal_generation_allowed=true`, `code_generation_allowed=false`,
  `research_selection_score=100.0`, and `blocker_count=0`.
- Evidence inspected: quality checks for local open interest and account-ratio
  parquet both returned `ok=true`; local event builder produced
  `event_count=491`; local falsification passed; research selection consumed
  the latest synthesis/map and answered all causal-map research questions.
- Audit conclusion remains not complete: this is the best current
  pre-proposal evidence, but no proposal, generated strategy, backtest,
  walk-forward result, ranking decision, or paper-ready candidate exists for
  it yet. The structural codegen blocker was resolved in the next increment,
  but the real thesis has not yet been carried through proposal/codegen and
  historical gates.

## 2026-05-07 JST Crowding-Unwind Local Structural Codegen

- Implemented a narrowly scoped codegen path for
  `crowding_unwind_reaccumulation`. Generated strategy code now reads only
  local historical parquet files for BTCUSDT 1h open interest and long/short
  account ratio, aligns them with closed 5m candles using the same
  closed-context availability semantics as `local_events.py`, and computes
  `open_interest_delta_pct_288`, `long_short_ratio_zscore_864`,
  `sma_distance_bps_144`, and `volume_zscore_288`.
- Structural capability reporting now marks open interest and long/short
  account ratio as `strategy_codegen_supported=true` only when their local
  parquet files exist and quality reports pass. The refreshed report
  `registry/strategies/checks/20260507T083000Z_structural_data_capabilities_codegen_oi_lsr.json`
  has `local_research_usable=["open_interest","long_short_ratio"]` and
  `must_not_codegen=["liquidation","order_book"]`.
- Proposal and research-selection structural detection now treats
  `long_short_ratio`, `long/short account ratio`, and `account ratio` as the
  `long_short_ratio` class. Unsupported structural-data variants still block
  codegen through `structural_data_code_generation_supported`.
- Verification passed: `py_compile` for the changed modules and tests, focused
  pytest for structural capability and crowding-unwind codegen passed 5 tests,
  and full `tests/test_bot_factory.py -q` passed with the existing pandas
  `PerformanceWarning` warnings from `signal_diagnostics.py`. `git diff
  --check` passed with existing LF-to-CRLF working-copy warnings only.
  Post-test cleanup removed 33 workspace-local `.pytest_cache` / `__pycache__`
  directories outside `.venv`, leaving 0 remaining cache targets.
- Audit conclusion remains unchanged: this removes a real codegen blocker but
  is not a profitable candidate. The next required step is to generate the
  actual `TH-CROWDING-UNWIND-REACCUMULATION-20260507` proposal and strategy
  artifact from the approved research decision, then run static checks,
  diagnostics/backtest/walk-forward, ranking, and readiness gates before any
  paper or promotion discussion.

## 2026-05-07 JST Crowding-Unwind Candidate Evaluation

- The real `TH-CROWDING-UNWIND-REACCUMULATION-20260507` path was carried
  through proposal generation, strategy code generation, static checks, signal
  diagnostics, historical backtest, walk-forward, candidate evaluation,
  iteration planning, and ranking.
- Evidence inspected:
  `registry/strategies/proposals/20260507T085000Z_LongOnlyCrowdingUnwindReaccumulationCandidate.metadata.json`
  was accepted; generated metadata
  `registry/strategies/generated/LongOnlyCrowdingUnwindReaccumulationCandidate/crowding_unwind_reaccumulation_001/metadata.json`
  has `status=generated`, `static_check_ok=true`, and
  `candidate_evaluation_eligible=true`.
- Signal diagnostics were encouraging but not decisive:
  `entry_count=929`, `generated_entry_edge.status=pass`,
  `net_edge_bps=13.150967`, and `profitable_windows_ratio=0.5`.
- Historical backtest rejected the candidate:
  `data/backtests/LongOnlyCrowdingUnwindReaccumulationCandidate/crowding_unwind_reaccumulation_001_20240101_20260507_dirpath/metrics.json`
  reports `total_return_pct=-1.008043973`, `trade_count=294`,
  `profit_factor=0.8187715323309974`, `sortino=-0.5357142015667924`, and
  negative expectancy.
- Trade-shape inspection shows why this should not become a parameter-tuning
  exercise: the main exit path had 280 trades and a 62.5% win rate, but
  average loss (`-0.006361`) was much larger than average win (`0.003703`).
  Timeout exits were strongly negative (`14` trades, win rate `0.1429`,
  average return `-0.004999`). The 2024H1/2024H2 windows were negative,
  2025H1 was only slightly positive, and 2025H2/2026YTD had zero trades.
  The failure is adverse payoff asymmetry plus regime fragility, not simply
  missing entries.
- OHLCV-prechecked walk-forward also rejected it:
  `data/walk_forward/LongOnlyCrowdingUnwindReaccumulationCandidate/crowding_unwind_reaccumulation_001_wf_20240101_20260507_v3_ohlcv_file/walk_forward_metrics.json`
  completed all five windows but has `recommendation=fail`, `pass_rate=0.0`,
  `profitable_windows_ratio=0.2`, `total_return_pct=-1.008043973`, and
  `max_single_window_profit_dependency=1.0`.
- Candidate registry now records the outcome:
  `registry/strategies/candidates/LongOnlyCrowdingUnwindReaccumulationCandidate/crowding_unwind_reaccumulation_001/candidate_manifest.json`
  has `recommendation=retry`, `historical_backtest=fail`, and
  `walk_forward=fail`. Ranking
  `registry/strategies/candidates/rankings/20260507T084500Z_crowding_unwind_reaccumulation_retry_ranking_gatefix/candidate_ranking.json`
  has `paper_ready_candidate_ids=[]`. Iteration
  `registry/strategies/reviews/LongOnlyCrowdingUnwindReaccumulationCandidate/crowding_unwind_reaccumulation_001/crowding_unwind_reaccumulation_001_backtest_wf_fail_iteration/iteration_plan.json`
  has `action=revise` and `evaluation_allowed_by_this_plan=false`.
- Aggregate failure memory was refreshed from all 32 local candidate manifests:
  ranking
  `registry/strategies/candidates/rankings/20260507T085500Z_all_candidates_with_crowding_unwind_rejection_ranking/candidate_ranking.json`
  has `candidate_count=32` and no paper-ready candidates; synthesis
  `registry/strategies/synthesis/20260507T090000Z_all_candidates_with_crowding_unwind_rejection/candidate_failure_synthesis.json`
  has `paper_ready_count=0`,
  `parameter_only_retry_allowed=false`, and `requires_new_thesis_id=true`;
  causal map
  `registry/strategies/failure_maps/20260507T090500Z_all_candidates_with_crowding_unwind_rejection_causal_map/causal_failure_map.json`
  has `requires_research_decision_before_proposal=true` and dominant failure
  categories in regime fragility, walk-forward fragility, cost sensitivity,
  no-profitable-windows, and entry-negative-edge.
- Pipeline quality fixes made during this evaluation: rule-based
  `scripts/bot_factory_run_backtest.py` now accepts `--ohlcv-file` and writes
  an OHLCV quality artifact before running a backtest, and
  `candidate_evaluation.py` derives a historical gate value from raw
  `metrics.json` when the metrics file lacks a `recommendation` key.
- Verification passed:
  `py_compile` for `scripts/bot_factory_run_backtest.py`,
  `candidate_evaluation.py`, and `tests/test_bot_factory.py`; focused pytest
  for candidate-evaluation gate derivation, manifest writing, and the
  crowding-unwind diagnostics variant passed 3 tests. Existing pandas
  `PerformanceWarning` warnings remain in `signal_diagnostics.py`.
- Audit conclusion remains not complete: the Bot Factory can now carry this
  structural-data thesis end to end and reject it safely, but the central
  objective still lacks a profitable, robust, paper-ready candidate. The next
  research step must be theory-level and materially different, not
  parameter-only threshold retuning of this failed candidate.

## 2026-05-07 JST Post-Deleveraging Volatility-Compression Rejection

- Tested a fixed exit/risk thesis after the crowding-unwind trade-shape
  analysis showed adverse payoff asymmetry and timeout drag. The new thesis
  `TH-POST-DELEVERAGING-VOL-COMPRESSION-20260507` /
  `post_deleveraging_volatility_compression` required OI contraction,
  long/short washout, price above 12h SMA, 30-minute nonnegative price
  stabilization, current candle `range_pct <= 0.6`, and non-withdrawn volume.
  This was a fixed local event study, not parameter search.
- Local event generation completed with `event_count=387`, no blockers, and
  closed-context structural data alignment. Artifact:
  `registry/strategies/research_decisions/20260507T091000Z_post_deleveraging_volatility_compression_events/local_events.json`.
- Local falsification rejected the thesis despite sufficient sample and span:
  `sample_count=387`, `data_span_days=857.270833`,
  `expected_edge_bps=10.587168`, `all_in_cost_bps=12.0`,
  `net_edge_bps=-1.412832`, `profitable_windows_ratio=0.5`, and
  `profitable_calendar_windows_ratio=0.6`. The failing check was
  `expected_edge_exceeds_all_in_cost`. Artifact:
  `registry/strategies/research_decisions/20260507T091500Z_post_deleveraging_volatility_compression_local_falsification/local_falsification.json`.
- Aggregate failure memory was refreshed again:
  `registry/strategies/synthesis/20260507T092000Z_all_candidates_with_post_deleveraging_vol_compression_rejection/candidate_failure_synthesis.json`
  records `local_falsification_rejection_count=1`, and causal map
  `registry/strategies/failure_maps/20260507T092500Z_all_candidates_with_post_deleveraging_vol_compression_rejection_causal_map/causal_failure_map.json`
  now requires future research to answer how it avoids this validated local
  rejection.
- Audit conclusion remains not complete: the added thesis narrowed the exit/risk
  search space and strengthened failure memory, but it did not produce a
  profitable or paper-ready candidate.

## 2026-05-07 JST Theory-Fixed Generated Parameter Guard

- Objective gap addressed: generated strategy code still declared Freqtrade
  parameters with `optimize=True`, which could let later work drift back into
  hyperopt-style parameter search even though the project objective is
  theory-first research and code generation.
- Implemented guard: `strategy_code.py` now emits generated
  `IntParameter` / `DecimalParameter` declarations with `optimize=False` and
  records `parameter_optimization_enabled=false`,
  `parameter_optimization_policy=theory_fixed_parameters_no_freqtrade_hyperopt`,
  and `safety_scope.freqtrade_hyperopt_parameter_optimization=false` in
  generated metadata.
- Safety evidence: generated-code checks now include
  `generated_code_freqtrade_hyperopt_disabled`, blocking generated code that
  contains `optimize=True` or lacks explicit `optimize=False` declarations.
- Verification: `py_compile` passed for `strategy_code.py` and
  `tests/test_bot_factory.py`; focused pytest for
  `test_strategy_code_generator_writes_long_only_strategy_and_metadata` passed
  1 test; broader `-k "strategy_code_generator"` pytest passed 42 tests.
- Handoff hygiene: the next-agent prompt now points current next-step
  instructions at the 32-candidate aggregate ranking, post-deleveraging
  synthesis, and post-deleveraging causal map. Older 31-candidate references
  remain only as historical or explicitly superseded context.
- Evaluation-stage enforcement: `candidate_evaluation.py` now rejects generated
  Strategy Code Generator artifacts whose metadata or actual generated strategy
  file exposes, omits, or fails to prove disabled Freqtrade parameter
  optimization. The focused policy/manifest/gate pytest selection passed
  3 tests, and the broader `candidate_evaluation` pytest selection passed
  13 tests.
- Audit conclusion remains not complete: this closes a real parameter-search
  drift path, but the core success criterion is still missing. There is still
  no profitable, robust, paper-ready candidate.

## 2026-05-07 JST Mark-Premium/OI Continuation Local Screen

- Tested fixed pre-proposal thesis
  `TH-MARK-PREMIUM-OI-CONTINUATION-20260507` /
  `mark_premium_open_interest_continuation` using local BTC 5m OHLCV, 4h mark
  price, 1h open interest, and 1h long/short account ratio. Conditions were
  fixed before running: expanding positive mark premium, one-hour OI expansion,
  non-overcrowded long-account ratio, six-candle nonnegative return, and
  positive volume participation. No proposal, codegen, backtest, paper, live
  trading, or parameter search was run.
- Local events completed with 33 final events and no blockers at
  `registry/strategies/research_decisions/20260507T093000Z_mark_premium_open_interest_continuation_events/local_events.json`.
  Cumulative condition survival was narrow: 117,518 rows after mark premium,
  9,032 after OI expansion, 364 after long-account ratio, 122 combined rows
  before cooldown, and 33 final events.
- Low-sample falsification passed on edge but exposed concentration:
  `sample_count=33`, `net_edge_bps=31.837718`,
  `profitable_windows_ratio=0.75`, and
  `profitable_calendar_windows_ratio=0.5`, with 32 of 33 events in `2024Q1`.
- Strict pre-proposal sample gate rejected the same fixed events:
  `registry/strategies/research_decisions/20260507T094000Z_mark_premium_open_interest_continuation_strict_sample_gate/local_falsification.json`
  has `status=failed`, `sample_count=33`, `blocker_count=1`, and blocker
  `event_sample_count_sufficient`.
- Refreshed failure memory:
  `registry/strategies/synthesis/20260507T094500Z_all_candidates_with_mark_premium_oi_sample_rejection/candidate_failure_synthesis.json`
  has `candidate_count=32`, `paper_ready_count=0`,
  `parameter_only_retry_allowed=false`, `requires_new_thesis_id=true`, and
  `local_falsification_rejection_count=2`; causal map
  `registry/strategies/failure_maps/20260507T095000Z_all_candidates_with_mark_premium_oi_sample_rejection_causal_map/causal_failure_map.json`
  requires research selection before proposal and includes the new small-sample
  local rejection.
- Verification: `git diff --check` passed with existing CRLF warnings only;
  post-script cleanup removed 2 workspace-local `__pycache__` directories
  outside `.venv`, leaving 0 remaining cache targets.
- Audit conclusion remains not complete: this was promising enough to preserve
  as evidence but not robust enough to codegen. It should not be retuned by
  thresholds; future work needs a broader theory or fresh evidence that
  survives sample-size and calendar-window stability gates.

## 2026-05-07 JST Local Falsification Calendar Stability Gates

- Objective gap addressed: local falsification already reported quarterly
  `calendar_window_summaries`, but callers could leave those metrics
  informational only. The mark-premium/OI continuation screen showed why this
  was weak: aggregate post-cost edge could pass while the sample was
  concentrated in one quarter.
- Implemented guard: `LocalFalsificationInputs` now supports
  `min_calendar_window_count` and
  `min_profitable_calendar_windows_ratio`; the artifact records both minima,
  checks `calendar_window_count_sufficient` and
  `profitable_calendar_windows_ratio_sufficient`, and the Markdown report
  renders `calendar_window_count`.
- CLI support: `scripts/bot_factory_build_local_falsification.py` exposes
  `--min-calendar-window-count` and
  `--min-profitable-calendar-windows-ratio`, defaulting to disabled behavior
  for compatibility.
- Verification: `py_compile` passed for `local_falsification.py`, the CLI, and
  `tests/test_bot_factory.py`; focused `-k "local_falsification"` pytest
  passed 11 tests; CLI `--help` showed the new options; `git diff --check`
  passed with existing CRLF warnings only. Cleanup removed 33 workspace-local
  pytest/cache directories outside `.venv`.
- Audit conclusion remains not complete: this makes pre-proposal local evidence
  harder to over-accept, but the project still has no profitable, robust,
  paper-ready candidate.

## 2026-05-07 JST Worktree Cleanup Recheck

- Rechecked after the local falsification calendar-gate increment with
  `git status --short --untracked-files=all`, `git diff --stat`,
  `git ls-files --others --exclude-standard`,
  `git ls-files --others --ignored --exclude-standard`, and a cache-target
  count outside `.venv`.
- Current non-ignored worktree remains source/docs/scripts/tests, not runtime
  cache leakage: 22 tracked file changes plus 26 untracked source/docs
  scaffolding files. The tracked diff is large at 33,114 insertions and
  2,655 deletions, so review/commit grouping remains required.
- Current ignored inventory is intentionally mostly generated runtime evidence:
  `ignored_total_count=25434`, `.venv=21603`, and
  `ignored_non_venv_count=3831`. The largest non-`.venv` groups are
  `data/walk_forward=1940`, `registry/strategies/candidates=393`,
  `data/backtests=361`, `registry/strategies/checks=252`, and
  `registry/strategies/research_decisions=193`.
- Safe deletion already performed: path-guarded cleanup removed 33
  workspace-local `.pytest_cache` / `__pycache__` directories outside `.venv`;
  a follow-up count returned `cache_target_count=0`. Broader ignored
  `data`/`registry` artifacts were retained as local evidence and should only
  be deleted after an explicit cleanup decision.
- Owner decision still required for the tracked deletion of
  `docs/BOT_FACTORY_GOAL_COMMAND_RUNBOOK.md`: keep the deletion only if the
  TODO, audit, and next-agent prompt now supersede it; otherwise restore or
  intentionally merge its useful content.

## 2026-05-07 JST Mark-Premium/OI Calendar-Gated Rejection

- Re-ran the fixed mark-premium/OI event set through the new calendar
  stability gates without threshold tuning. The artifact
  `registry/strategies/research_decisions/20260507T100000Z_mark_premium_open_interest_continuation_calendar_stability_gate/local_falsification.json`
  failed despite positive aggregate edge:
  `sample_count=33`, `net_edge_bps=31.837718`,
  `profitable_windows_ratio=0.75`, and `data_span_days=857.270833`.
- The new rejection is sharper than the prior strict sample gate:
  `calendar_window_count=2` versus required `3`, and
  `profitable_calendar_windows_ratio=0.5` versus required `0.75`. The blockers
  were exactly `calendar_window_count_sufficient` and
  `profitable_calendar_windows_ratio_sufficient`; quarter summaries were
  `2024Q1` with 32 events and positive net edge, and `2024Q4` with one event
  and negative net edge.
- Refreshed current failure memory:
  `registry/strategies/synthesis/20260507T100500Z_all_candidates_with_mark_premium_oi_calendar_rejection/candidate_failure_synthesis.json`
  has `candidate_count=32`, `paper_ready_count=0`,
  `parameter_only_retry_allowed=false`, `requires_new_thesis_id=true`, and
  `local_falsification_rejection_count=2`; causal map
  `registry/strategies/failure_maps/20260507T101000Z_all_candidates_with_mark_premium_oi_calendar_rejection_causal_map/causal_failure_map.json`
  requires research selection before proposal.
- Verification: `git diff --check` passed with existing CRLF warnings only;
  post-script cleanup removed 2 workspace-local `__pycache__` directories
  outside `.venv`, leaving 0 remaining cache targets.
- Audit conclusion remains not complete: this prevents over-accepting a
  concentrated local edge, but it is not a profitable or paper-ready Bot
  Factory candidate.

## 2026-05-07 JST Comprehensive Local Rejection Memory Refresh

- Corrected a latest-memory risk: the mark-premium/OI calendar-only synthesis
  was newest by `generated_at` but omitted later local rejection artifacts that
  the next-agent prompt already treated as required negative knowledge.
- Rebuilt current failure synthesis from the 32-candidate ranking plus the
  recent local falsification set, including closed-context
  `mark_fair_value_momentum_lag`, `low_range_volume_absorption`,
  `thin_book_dislocation_reversion`, multiple open-interest rejections,
  funding-neutral impulse, long/short extended rejection, negative-funding
  sample rejection, post-deleveraging volatility compression, and
  mark-premium/OI calendar-stability rejection.
- Current synthesis:
  `registry/strategies/synthesis/20260507T110000Z_all_candidates_with_comprehensive_local_rejection_memory/candidate_failure_synthesis.json`
  has `candidate_count=32`, `paper_ready_count=0`,
  `parameter_only_retry_allowed=false`, `requires_new_thesis_id=true`,
  `local_falsification_rejection_count=10`, and
  `local_falsification_invalid_rejection_count=1`.
- Current causal map:
  `registry/strategies/failure_maps/20260507T110500Z_all_candidates_with_comprehensive_local_rejection_memory_causal_map/causal_failure_map.json`
  has `candidate_count=32`, `category_count=8`,
  `requires_research_decision_before_proposal=true`,
  `minimum_research_selection_score=80`, 26 required research questions, and
  10 validated local falsification rejection contexts.
- Verification: `git diff --check` passed with existing CRLF warnings only;
  post-script cleanup removed 2 workspace-local `__pycache__` directories
  outside `.venv`, leaving 0 remaining cache targets.
- Audit conclusion remains not complete: the memory handoff is more coherent,
  but no profitable or paper-ready candidate exists.

## 2026-05-07 JST Research-Selection Response Template Export

- Added `freqtrade_ext/bot_factory/research_selection_template.py` and
  `scripts/bot_factory_export_research_selection_template.py` so the current
  causal map can be converted into a local JSON/Markdown checklist before any
  new thesis is selected.
- The concrete template
  `registry/strategies/research_decisions/20260507T111000Z_comprehensive_research_selection_response_template/research_selection_response_template.json`
  completed with 5 required causal failure responses, 26 required
  research-question responses, 10 validated local rejection contexts, and
  `blocker_count=0`. It now includes a
  fillable `research_selection_input_template`, a short
  `select_research_thesis_input_json_command_template`, and a full
  `select_research_thesis_command_template` PowerShell skeleton with the
  current synthesis path, causal map path, thesis metadata placeholders, local
  data/falsification/reference placeholders, and every required causal and
  research-question response placeholder.
- `scripts/bot_factory_select_research_thesis.py` now accepts
  `--research-selection-input-json` so a filled template can be loaded as
  structured thesis input. This keeps the AI research pass in JSON instead of
  forcing a long hand-assembled CLI command.
- Verification: `py_compile` passed for the new module, CLI, and
  `tests/test_bot_factory.py`; focused
  `-k "research_selection_template or research_selection_cli_loads_filled_input_json"`
  pytest passed 2 tests. The first real export found and fixed a compatibility
  bug where repository causal maps use `factory=candidate_failure_map`; after
  adding the command skeleton and JSON-input ingestion, compile, focused
  pytest, selector `--help`, and the concrete export were re-run successfully.
  `git diff --check` passed with existing CRLF warnings only. Cleanup removed
  33 workspace-local `.pytest_cache` / `__pycache__` directories outside
  `.venv`, leaving 0 remaining cache targets.
- Audit conclusion remains not complete: this reduces handoff error before
  the next theory-first research decision, but it is not a profitable or
  paper-ready candidate.

## 2026-05-07 JST Volume-Clock Liquidity Momentum Local Rejection

- Tested `TH-VOLUME-CLOCK-LIQUIDITY-MOMENTUM-20260507` /
  `volume_clock_liquidity_momentum` as a fixed, theory-backed pre-proposal
  local screen. The screen used BTC futures 5m closed candles only: UTC
  13:00-16:59, 30-minute return >= 20 bps, 288-candle volume z-score >= 1.0,
  and current candle range >= 0.2%.
- Local events completed with `event_count=686` and `blocker_count=0` at
  `registry/strategies/research_decisions/20260507T112000Z_volume_clock_liquidity_momentum_events/local_events.json`.
- Local falsification failed before research selection or proposal generation:
  `sample_count=686`, `data_span_days=857.270833`,
  `expected_edge_bps=0.255677`, `all_in_cost_bps=12.0`,
  `net_edge_bps=-11.744323`, `profitable_windows_ratio=0.25`, and
  `blocker_count=3`. Artifact:
  `registry/strategies/research_decisions/20260507T112500Z_volume_clock_liquidity_momentum_local_falsification/local_falsification.json`.
- Refreshed memory after the rejection. Current synthesis is
  `registry/strategies/synthesis/20260507T113000Z_all_candidates_with_volume_clock_liquidity_momentum_rejection/candidate_failure_synthesis.json`
  with `candidate_count=32`, `paper_ready_count=0`,
  `parameter_only_retry_allowed=false`, `requires_new_thesis_id=true`, and
  `local_falsification_rejection_count=11`. Current map is
  `registry/strategies/failure_maps/20260507T113500Z_all_candidates_with_volume_clock_liquidity_momentum_rejection_causal_map/causal_failure_map.json`;
  current template is
  `registry/strategies/research_decisions/20260507T114000Z_volume_clock_rejection_research_selection_response_template/research_selection_response_template.json`
  with 5 causal-response placeholders, 28 required research-question
  responses, and 11 validated local rejection contexts.
- Verification: `git diff --check` passed with existing CRLF warnings only;
  cleanup removed 2 workspace-local `__pycache__` directories outside `.venv`,
  leaving 0 remaining cache targets.
- Audit conclusion remains not complete: this prevented another plausible but
  unprofitable intraday volume/session thesis from reaching proposal/codegen;
  no profitable or paper-ready candidate exists.

## 2026-05-07 JST Worktree Cleanup Snapshot

- Current visible Git candidates are source/doc/test/template changes only:
  `.gitignore`, Bot Factory docs, modules under `freqtrade_ext/bot_factory/`,
  entrypoints under `scripts/`, `tests/test_bot_factory.py`,
  `registry/strategies/proposals/TEMPLATE.md`, and
  `data/market_structure/.gitkeep`.
- Current non-ignored untracked files are intended Bot Factory source/docs:
  `docs/BOT_FACTORY_GOAL_AUDIT.md`, 11 modules under
  `freqtrade_ext/bot_factory/`, 15 `scripts/bot_factory_*.py` entrypoints, and
  `data/market_structure/.gitkeep`. They should remain review candidates, not
  be treated as disposable generated output.
- Representative newly generated artifacts from the volume-clock rejection are
  ignored and should not be added to Git:
  `registry/strategies/research_decisions/20260507T112000Z_volume_clock_liquidity_momentum_events/local_events.json`,
  `registry/strategies/research_decisions/20260507T112500Z_volume_clock_liquidity_momentum_local_falsification/local_falsification.json`,
  `registry/strategies/synthesis/20260507T113000Z_all_candidates_with_volume_clock_liquidity_momentum_rejection/candidate_failure_synthesis.json`,
  `registry/strategies/failure_maps/20260507T113500Z_all_candidates_with_volume_clock_liquidity_momentum_rejection_causal_map/causal_failure_map.json`, and
  `registry/strategies/research_decisions/20260507T114000Z_volume_clock_rejection_research_selection_response_template/research_selection_response_template.json`.
  `git check-ignore -v` confirms they are covered by
  `registry/strategies/research_decisions/**`,
  `registry/strategies/synthesis/**`, and
  `registry/strategies/failure_maps/**`.
- Runtime evidence under ignored Bot Factory artifact roots can be deleted
  later if the owner wants disk cleanup, but it should not be removed
  silently because it is the local evidence trail for the failed-candidate
  memory. Workspace-local `.pytest_cache` / `__pycache__` targets outside
  `.venv` are currently cleaned to 0.
- Pending owner decision remains: `docs/BOT_FACTORY_GOAL_COMMAND_RUNBOOK.md`
  is deleted in the worktree. Do not restore or stage that deletion without an
  explicit cleanup/commit decision.

## 2026-05-07 JST Worktree Cleanup Ledger

- Rechecked the visible worktree with
  `git status --short --untracked-files=all`,
  `git diff --stat`, and `git ls-files --others --exclude-standard`.
  Current visible count is 50 paths: 21 modified tracked files, one tracked
  deletion, and 28 untracked review candidates. The tracked diff is
  `22 files changed, 33674 insertions(+), 2678 deletions(-)`.
- The 28 non-ignored untracked paths are review candidates, not disposable
  generated output: one audit doc, 11 modules under
  `freqtrade_ext/bot_factory/`, 15 `scripts/bot_factory_*.py` entrypoints, and
  `data/market_structure/.gitkeep`.
- Suggested Git review grouping:
  `.gitignore` plus `data/market_structure/.gitkeep` for artifact hygiene;
  core Bot Factory pipeline changes under `freqtrade_ext/bot_factory/` and
  matching `scripts/`; research-selection, failure-synthesis, causal-map,
  event, and local-falsification modules/CLIs; structural-data and diagnostics
  modules/CLIs; focused tests plus proposal template; docs handoff/audit/TODO
  updates.
- Ignored runtime inventory after cleanup is still intentionally local
  evidence, not Git input: `ignored_total=25461`, `.venv=21603`, and
  `ignored_non_venv=3858`. Largest non-`.venv` groups are
  `data/walk_forward=1940`, `registry/strategies/candidates=393`,
  `data/backtests=361`, `registry/strategies/checks=255`,
  `registry/strategies/research_decisions=205`,
  `registry/strategies/generated=137`,
  `registry/strategies/proposals=137`,
  `registry/strategies/synthesis=92`,
  `registry/strategies/diagnostics=74`,
  `data/freqai_training=72`, `data/freqai=58`,
  `registry/strategies/failure_maps=56`, plus smaller local user-data,
  `.vscode`, and docker ignored files.
- Safe deletion performed during cleanup: removed 28 workspace-local
  `.pytest_cache` / `__pycache__` directories outside `.venv` after verifying
  every resolved path stayed under the repository root and outside `.venv`.
  Follow-up count returned `cache_dir_count=0`.
- Do not silently delete ignored Bot Factory runtime evidence under
  `data/`, `registry/strategies/`, or `user_data/`; it is the local audit trail
  for rejected candidates and data checks. Delete it only after an explicit disk
  cleanup decision.
- Remaining owner decision is unchanged:
  `docs/BOT_FACTORY_GOAL_COMMAND_RUNBOOK.md` is deleted in the worktree. Either
  keep the deletion because the TODO/audit/next-agent prompt supersede it, or
  restore/merge any useful content before staging.

## 2026-05-07 JST ETH 5m Historical Coverage Expansion

- Reconfirmed the local OHLCV coverage before any new thesis/proposal work.
  BTC 5m futures already covered `2024-01-01T00:00:00+00:00` through
  `2026-05-07T06:30:00+00:00` with `ok=true`, `rows=246895`, no duplicate
  timestamps, and no missing intervals. ETH 5m futures covered
  `2024-01-01T00:00:00+00:00` through `2025-02-01T05:30:00+00:00` with
  `ok=true`, `rows=114403`, no duplicate timestamps, and no missing intervals.
- Expanded ETH 5m futures coverage through the historical-safe download wrapper
  only:

  ```powershell
  .\.venv\Scripts\python.exe scripts\bot_factory_download_data.py --config user_data\config.json --pairs ETH/USDT:USDT --timeframes 5m --timerange 20250201-20260507 --trading-mode futures --quality-output registry\strategies\checks\20260507T103000Z_eth_5m_ohlcv_quality_append_2025_2026.json
  ```

  The wrapper ran `freqtrade download-data`, wrote a temporary overlay with
  FreqAI disabled, and did not start paper/dry-run/live trading, order
  placement, leverage changes, shorting, or process control. Freqtrade logged
  `dry_run enabled` while downloading public historical data.
- Post-download quality report
  `registry\strategies\checks\20260507T103000Z_eth_5m_ohlcv_quality_append_2025_2026.json`
  completed with `ok=true`. ETH 5m now has `rows=246941`, start
  `2024-01-01T00:00:00+00:00`, end
  `2026-05-07T10:20:00+00:00`, `duplicate_timestamps=0`, and
  `missing_intervals=0`.
- Freqtrade also refreshed ETH futures companion public-data files:
  `user_data\data\bybit\futures\ETH_USDT_USDT-4h-mark.parquet` and
  `user_data\data\bybit\futures\ETH_USDT_USDT-8h-funding_rate.parquet`.
  `git check-ignore -v` confirms the ETH data files are ignored by
  `user_data/*`, and the quality report is ignored by
  `registry/strategies/checks/*.json`; no new Git-visible data artifacts were
  introduced.
- Verification: `git diff --check` passed with existing CRLF working-copy
  warnings only. Post-check cleanup removed 28 workspace-local
  `.pytest_cache` / `__pycache__` directories outside `.venv`, and a follow-up
  count returned `cache_dir_count=0`.
- Audit conclusion remains not complete: this removes a short-history blocker
  for future BTC/ETH local screens, but it does not create a profitable,
  robust, or paper-ready candidate and does not justify repeating failed
  BTC/ETH lead-lag, cointegration, or correlation families without a materially
  different theory and fresh local falsification.

## 2026-05-07 JST Informative OHLCV Local Event Support

- Added informative-OHLCV context support to
  `freqtrade_ext\bot_factory\local_events.py` and
  `scripts\bot_factory_build_local_events.py` so a future theory-first local
  screen can use a second closed-candle OHLCV file, such as ETH 5m, before
  proposal generation or strategy codegen.
- New event-spec features are `informative_return_bps`,
  `relative_return_bps`, `informative_range_pct`,
  `informative_sma_distance_bps`, and `informative_volume_zscore`.
  `relative_return_bps` is computed as primary OHLCV return minus informative
  OHLCV return over the same closed-candle lookback. The context is recorded as
  `informative_ohlcv` in `required_contexts`, `auxiliary_sources`, and
  `context_merge`, using the existing closed-context candle alignment
  semantics.
- CLI support: `scripts\bot_factory_build_local_events.py` now accepts
  `--informative-ohlcv-path`. The command remains a local artifact writer only:
  it does not select a thesis, generate a proposal, generate strategy code, run
  backtests, start paper/dry-run/live trading, place orders, short, increase
  leverage, or manage bot processes.
- Verification:

  ```powershell
  .\.venv\Scripts\python.exe -m py_compile freqtrade_ext\bot_factory\local_events.py scripts\bot_factory_build_local_events.py tests\test_bot_factory.py
  .\.venv\Scripts\python.exe -m pytest tests\test_bot_factory.py -q -k "informative_ohlcv or local_event_cli_maps_informative"
  .\.venv\Scripts\python.exe -m pytest tests\test_bot_factory.py -q -k "local_event"
  .\.venv\Scripts\python.exe scripts\bot_factory_build_local_events.py --help
  ```

  Results: compile passed; focused informative-OHLCV pytest passed `2 passed`;
  broader local-event pytest passed `10 passed`; CLI help shows
  `--informative-ohlcv-path`.
- Final hygiene: `git diff --check` passed with existing CRLF working-copy
  warnings only. Post-test cleanup removed 33 workspace-local
  `.pytest_cache` / `__pycache__` directories outside `.venv`, and a follow-up
  count returned `cache_dir_count=0`.
- Audit conclusion remains not complete: this is cross-asset local research
  infrastructure, not a profitable candidate. It should be used only after a
  materially distinct theory is selected against the current causal map; it
  must not be used to rerun failed BTC/ETH families as threshold tweaks.

## 2026-05-07 JST BTC/ETH Relative-Strength Continuation Local Rejection

- Tested one fixed, theory-backed, pre-proposal cross-asset screen using the
  new informative-OHLCV event support. The thesis was
  `TH-CROSS-ASSET-RELATIVE-STRENGTH-CONTINUATION-20260507` /
  `cross_asset_relative_strength_continuation`. It asked whether BTC futures
  continuation improves when BTC is already outperforming ETH over one hour
  while ETH is also positive, with BTC volume participation present.
- Research motivation was deliberately conservative: cross-sectional crypto
  return evidence and ML cross-section literature motivate relative-strength
  tests, while realistic-cost momentum literature warns that such signals can
  vanish after frictions. References recorded in the event spec:
  `doi:10.1016/j.irfa.2024.103244`, SSRN `4675565`, and
  `doi:10.1016/j.irfa.2021.101908`.
- Event spec:
  `registry\strategies\research_decisions\20260507T105000Z_cross_asset_relative_strength_continuation_event_spec\event_spec.json`.
  Fixed conditions were BTC `return_bps >= 50.0` over 12 candles, ETH
  `informative_return_bps >= 20.0` over 12 candles,
  `relative_return_bps >= 20.0` over 12 candles, BTC `volume_zscore >= 0.5`
  over 288 candles, and `cooldown_candles=36`.
- Local events completed with `event_count=733` and `blocker_count=0`:
  `registry\strategies\research_decisions\20260507T105000Z_cross_asset_relative_strength_continuation_events\local_events.json`.
  The event source used `required_contexts=["informative_ohlcv"]` and passed
  closed-context alignment checks.
- Local falsification failed before research selection, proposal generation,
  codegen, backtest, or trading:
  `registry\strategies\research_decisions\20260507T105500Z_cross_asset_relative_strength_continuation_local_falsification\local_falsification.json`.
  Results: `sample_count=733`, `data_span_days=857.270833`,
  `expected_edge_bps=-5.614326`, `all_in_cost_bps=12.0`,
  `net_edge_bps=-17.614326`, `win_rate=0.4693`,
  `profitable_windows_ratio=0.0`, `calendar_window_count=10`,
  `profitable_calendar_windows_ratio=0.0`, and `blocker_count=3`.
- Refreshed current failure memory. Current synthesis is
  `registry\strategies\synthesis\20260507T111500Z_all_candidates_with_cross_asset_relative_strength_rejection\candidate_failure_synthesis.json`
  with `candidate_count=32`, `paper_ready_count=0`,
  `parameter_only_retry_allowed=false`, `requires_new_thesis_id=true`, and
  `local_falsification_rejection_count=12`. Current map is
  `registry\strategies\failure_maps\20260507T112000Z_all_candidates_with_cross_asset_relative_strength_rejection_causal_map\causal_failure_map.json`
  with `candidate_count=32`, `category_count=8`, and
  `requires_research_decision_before_proposal=true`. Current template is
  `registry\strategies\research_decisions\20260507T112500Z_cross_asset_relative_strength_rejection_research_selection_response_template\research_selection_response_template.json`
  with 5 required causal responses, 30 required research-question responses,
  12 validated local rejection contexts, and `blocker_count=0`.
- `git check-ignore -v` confirms the event spec, local events, local
  falsification, synthesis, causal map, and template artifacts are ignored by
  the intended `registry/strategies/**` artifact rules.
- Verification: `git diff --check` passed with existing CRLF working-copy
  warnings only. Post-script cleanup removed 2 workspace-local
  `.pytest_cache` / `__pycache__` directories outside `.venv`, and a follow-up
  count returned `cache_dir_count=0`.
- Audit conclusion remains not complete: this is another local rejection, not
  a candidate. Future research must avoid cross-asset relative-strength
  continuation as a disguised BTC/ETH momentum parameter retry.

## 2026-05-07 JST Full Bot Factory Test Sweep

- Ran the full focused Bot Factory test module after the informative-OHLCV
  local-event implementation and BTC/ETH relative-strength local rejection:

  ```powershell
  .\.venv\Scripts\python.exe -m pytest tests\test_bot_factory.py -q
  ```

  Result: command exited `0` and reached `[100%]`. The only output issues were
  existing `PerformanceWarning` warnings from
  `freqtrade_ext\bot_factory\signal_diagnostics.py` about fragmented pandas
  DataFrame inserts.
- Post-test cleanup removed 33 workspace-local `.pytest_cache` / `__pycache__`
  directories outside `.venv`. Final hygiene check: `git diff --check` passed
  with existing CRLF working-copy warnings only, and a follow-up count returned
  `cache_dir_count=0`.
- Audit conclusion remains not complete: this increases confidence that the
  new local-event input surface did not regress Bot Factory tests, but passing
  tests are not a profitable or paper-ready candidate.

## 2026-05-07 JST Signal Diagnostics Warning Cleanup

- Reduced current Bot Factory verification noise from
  `freqtrade_ext\bot_factory\signal_diagnostics.py` by defragmenting the
  diagnostics DataFrame before the broad cross-asset / higher-moment /
  liquidity feature block is appended. This preserves existing feature values
  and only changes pandas block layout before additional columns are inserted.
- Verification:

  ```powershell
  .\.venv\Scripts\python.exe -m py_compile freqtrade_ext\bot_factory\signal_diagnostics.py tests\test_bot_factory.py
  .\.venv\Scripts\python.exe -m pytest tests\test_bot_factory.py -q -k "signal_diagnostics"
  .\.venv\Scripts\python.exe -m pytest tests\test_bot_factory.py -q
  ```

  Results: compile passed; focused `signal_diagnostics` pytest reached
  `[100%]` with no warning summary; full Bot Factory pytest reached `[100%]`
  with no warning summary.
- Post-test cleanup removed 33 workspace-local `.pytest_cache` / `__pycache__`
  directories outside `.venv`.
- Audit conclusion remains not complete: this improves verification clarity,
  but it is not a profitable or paper-ready candidate.

## 2026-05-07 JST Cross-Asset Relative-Strength Repeat Guard Check

- Re-ran the same BTC/ETH relative-strength local event spec against the
  refreshed synthesis
  `registry\strategies\synthesis\20260507T111500Z_all_candidates_with_cross_asset_relative_strength_rejection\candidate_failure_synthesis.json`
  as a diagnostic repeat-guard check.
- Command exited `1` as expected and wrote the blocked artifact
  `registry\strategies\research_decisions\20260507T113000Z_cross_asset_relative_strength_repeat_guard\local_events.json`
  with `status=blocked`, `event_count=0`, and `blocker_count=3`.
- The important blockers were
  `event_spec_thesis_not_in_failure_synthesis` with `matched=true` for
  `TH-CROSS-ASSET-RELATIVE-STRENGTH-CONTINUATION-20260507`, and
  `event_spec_mechanism_not_in_failure_synthesis` with `matched=true` for
  `cross_asset_relative_strength_continuation`. `events_generated` also failed
  because the guard prevented event creation.
- `git check-ignore -v` confirms the repeat-guard local event artifacts are
  ignored by `registry/strategies/research_decisions/**`.
- Verification: `git diff --check` passed with existing CRLF working-copy
  warnings only. Post-script cleanup removed 2 workspace-local
  `.pytest_cache` / `__pycache__` directories outside `.venv`, and a follow-up
  count returned `cache_dir_count=0`.
- Audit conclusion remains not complete: this confirms the anti-repeat guard
  for one rejected thesis/mechanism, but it is not a profitable candidate.

## 2026-05-07 JST Worktree Cleanup Ledger Refresh

- Rechecked the worktree after the informative-OHLCV implementation,
  BTC/ETH relative-strength rejection, repeat-guard check, and warning cleanup.
  `git status --short --untracked-files=all` still shows 50 visible paths:
  21 modified tracked files, one tracked deletion, and 28 non-ignored
  untracked review candidates.
- Current `git diff --stat` is
  `22 files changed, 34090 insertions(+), 2678 deletions(-)`. The main growth
  since the prior ledger is expected: tests, Bot Factory docs, next-agent
  prompt updates, `local_events.py`, `signal_diagnostics.py`, and related CLI
  support.
- The 28 non-ignored untracked files remain the same review-candidate
  categories: one audit doc, 11 modules under `freqtrade_ext/bot_factory/`, 15
  `scripts/bot_factory_*.py` entrypoints, and
  `data/market_structure/.gitkeep`.
- Current ignored inventory is `ignored_total=25480`, `.venv=21603`, and
  `ignored_non_venv=3877`. Newly generated cross-asset event, falsification,
  synthesis, causal-map, template, and repeat-guard artifacts are included in
  ignored runtime evidence and should not be added to Git.
- Pending owner decision is unchanged:
  `docs/BOT_FACTORY_GOAL_COMMAND_RUNBOOK.md` is deleted in the worktree. Do not
  stage that deletion or restore it without an explicit cleanup/commit
  decision.

## 2026-05-07 JST Goal Completion Audit Snapshot

- Rechecked the repository state for this handoff with:

  ```powershell
  git status --short --untracked-files=all
  git diff --stat
  ```

  Current visible worktree state is 50 paths: 21 modified tracked files, one
  tracked deletion, and 28 non-ignored untracked review candidates. Current
  diff stat is `22 files changed, 34116 insertions(+), 2678 deletions(-)`.
- Goal criterion: the Bot Factory should not merely optimize parameters. It
  should use research/theory, produce candidate strategy code, reject failed
  mechanisms with evidence, and eventually surface a robust profitable
  candidate suitable for the next paper/live-readiness phase.
- What is implemented or materially improved:
  - Research-selection, failure-synthesis, causal-map, local-event, and
    local-falsification scaffolding now exists and is covered by focused tests.
  - The current failure memory blocks parameter-only retries:
    `parameter_only_retry_allowed=false` and `requires_new_thesis_id=true` in
    the latest synthesis
    `registry\strategies\synthesis\20260507T111500Z_all_candidates_with_cross_asset_relative_strength_rejection\candidate_failure_synthesis.json`.
  - The BTC/ETH cross-asset relative-strength continuation idea was tested as
    a fixed pre-proposal local screen and rejected before proposal generation,
    strategy codegen, backtesting, or trading.
  - Re-running that rejected thesis/mechanism against the refreshed synthesis
    was blocked with `status=blocked`, proving the repeat guard for that path.
  - Full focused Bot Factory verification passes:

    ```powershell
    .\.venv\Scripts\python.exe -m pytest tests\test_bot_factory.py -q
    ```

    Result: command exited `0`, reached `[100%]`, and the latest run had no
    warning summary after the signal-diagnostics cleanup.
- What remains incomplete:
  - No paper-ready or profitable candidate exists. The latest synthesis has
    `candidate_count=32`, `paper_ready_count=0`, and all evaluated candidates
    failed gates.
  - The latest local research screen was a rejection, not a candidate:
    `expected_edge_bps=-5.614326`, `net_edge_bps=-17.614326`, and
    `profitable_calendar_windows_ratio=0.0` for the BTC/ETH relative-strength
    continuation falsification.
  - The current research-selection template still requires a new thesis that
    answers 5 causal-failure response prompts, 30 research-question responses,
    and 12 validated local rejection contexts before proposal/codegen should
    continue.
- Git cleanup position:
  - Generated local runtime evidence under `registry/strategies/**`,
    `data/backtests/**`, `data/walk_forward/**`, `data/freqai/**`,
    `user_data/data/**`, and cache directories remains intentionally ignored
    and should not be added to Git.
  - The 28 non-ignored untracked files are review candidates, not disposable
    runtime artifacts: one audit doc, 11 Bot Factory modules, 15 Bot Factory
    scripts, and `data\market_structure\.gitkeep`.
  - `docs\BOT_FACTORY_GOAL_COMMAND_RUNBOOK.md` is a pending owner decision:
    either keep the deletion as intentional cleanup or restore it, but do not
    decide implicitly.
- Audit conclusion: the larger objective is not complete and must not be
  marked complete. The next valid progress path is a materially new
  research-backed thesis that passes local falsification before proposal
  generation, strategy codegen, backtesting, or any paper/live-readiness step.

## 2026-05-07 JST Worktree Cleanup Classification

- Reconciled the current worktree against the TODO source of truth and
  next-agent prompt with:

  ```powershell
  git status --short --untracked-files=all
  git diff --stat
  git ls-files --others --exclude-standard
  git ls-files --others --ignored --exclude-standard
  Select-String -Path docs\*.md -Pattern "BOT_FACTORY_GOAL_COMMAND_RUNBOOK|GOAL_COMMAND_RUNBOOK" -CaseSensitive
  ```

- Current non-ignored untracked files are review candidates, not generated
  runtime artifacts. They should be reviewed with the source diff before a
  commit decision:
  - `docs\BOT_FACTORY_GOAL_AUDIT.md`
  - `data\market_structure\.gitkeep`
  - Bot Factory modules:
    `bybit_long_short_ratio.py`, `bybit_open_interest.py`,
    `candidate_failure_map.py`, `candidate_failure_synthesis.py`,
    `freqai_prediction_diagnostics.py`, `local_events.py`,
    `local_falsification.py`, `research_selection.py`,
    `research_selection_template.py`, `signal_diagnostics.py`, and
    `structural_data_capabilities.py`
  - Bot Factory scripts:
    `bot_factory_build_causal_failure_map.py`,
    `bot_factory_build_local_events.py`,
    `bot_factory_build_local_falsification.py`,
    `bot_factory_check_funding_rate.py`,
    `bot_factory_check_long_short_ratio.py`,
    `bot_factory_check_mark_price.py`,
    `bot_factory_check_open_interest.py`,
    `bot_factory_diagnose_candidate_signals.py`,
    `bot_factory_diagnose_freqai_predictions.py`,
    `bot_factory_download_bybit_long_short_ratio.py`,
    `bot_factory_download_bybit_open_interest.py`,
    `bot_factory_export_research_selection_template.py`,
    `bot_factory_report_structural_data_capabilities.py`,
    `bot_factory_select_research_thesis.py`, and
    `bot_factory_synthesize_candidate_failures.py`.
- Ignored inventory is intentionally local and should not be added to Git:
  `ignored_total=25480`, `.venv=21603`, `ignored_non_venv=3877`. The largest
  non-venv groups are `data\walk_forward` (1940),
  `registry\strategies` (1386), `data\backtests` (361),
  `data\freqai_training` (72), and `data\freqai` (58). These are historical
  evaluation, FreqAI, registry, and runtime evidence outputs covered by the
  updated `.gitignore` rules.
- No workspace-local `.pytest_cache` or `__pycache__` directories remain
  outside `.venv` at this checkpoint (`cache_dir_count=0`).
- The tracked deletion of `docs\BOT_FACTORY_GOAL_COMMAND_RUNBOOK.md` is not a
  safe implicit cleanup yet. References still exist in
  `docs\BOT_FACTORY_GOAL_AUDIT.md`, `docs\BOT_FACTORY_MVP_TODO.md`, and
  `docs\BOT_FACTORY_STRATEGY_GENERATION_NEXT_AGENT_PROMPT.md`. Accepting the
  deletion requires either migrating those references to the current handoff
  docs or explicitly deciding that historical references to the deleted runbook
  should remain as audit notes.
- Cleanup decision: do not delete the 28 non-ignored untracked files as
  disposable artifacts. Treat them as source/doc additions for review. Do not
  stage the ignored runtime evidence. Do not accept or revert the runbook
  deletion without an explicit owner decision.

## 2026-05-07 JST Source Diff Verification Sweep

- Verified the broad Bot Factory Python surface represented by the large
  source diff, including tracked and untracked Bot Factory modules, Bot
  Factory CLI scripts, and the focused test module:

  ```powershell
  $files = @(Get-ChildItem -Path freqtrade_ext\bot_factory -Filter *.py -File | ForEach-Object { $_.FullName }) + @(Get-ChildItem -Path scripts -Filter bot_factory_*.py -File | ForEach-Object { $_.FullName }) + @((Resolve-Path tests\test_bot_factory.py).Path); .\.venv\Scripts\python.exe -m py_compile @files
  .\.venv\Scripts\python.exe -m pytest tests\test_bot_factory.py -q -k "local_event_builder or local_falsification or candidate_failure_synthesis or research_selection_template or signal_diagnostics"
  git diff --check
  ```

  Results: py_compile exited `0`; focused pytest reached `[100%]`; and
  `git diff --check` passed with only existing LF-to-CRLF working-copy
  warnings.
- Post-verification cleanup removed generated cache directories outside
  `.venv`: 3 after py_compile and 32 after focused pytest. Final cache check
  returned `cache_dir_count=0`.
- Audit conclusion remains not complete: this raises confidence that the
  source review candidates are syntactically valid and focused core behavior
  still passes, but it does not produce a profitable or paper-ready candidate.

## 2026-05-07 JST Supported Variant Coverage Check

- Checked whether the next safe step could simply reuse an existing supported
  Strategy Code Generator variant. The answer is no.
- Evidence from the current source and latest synthesis:

  ```powershell
  Select-String -Path freqtrade_ext\bot_factory\strategy_proposals.py -Pattern "ALLOWED_STRATEGY_LOGIC_VARIANTS" -Context 0,40
  $s = Get-Content registry\strategies\synthesis\20260507T111500Z_all_candidates_with_cross_asset_relative_strength_rejection\candidate_failure_synthesis.json -Raw | ConvertFrom-Json; $summary = $s.aggregate_failure_summary; "candidate_count=$($s.candidate_count)"; "paper_ready_count=$($summary.paper_ready_count)"; "hypothesis_families_tried_count=$(@($summary.hypothesis_families_tried).Count)"; "supported_strategy_logic_variant_count=28"; "blocked_next_actions=$($summary.blocked_next_actions -join ', ')"
  ```

  Results: `candidate_count=32`, `paper_ready_count=0`,
  `hypothesis_families_tried_count=40`, and
  `supported_strategy_logic_variant_count=28`.
- Cross-asset paths were explicitly rechecked before choosing any next thesis.
  BTC/ETH lead-lag, cointegration spread, correlation recovery, and the later
  relative-strength continuation screen are all already failed or locally
  rejected. Reusing them would violate the current failure memory unless a
  materially different theory and fresh local falsification evidence are
  supplied first.
- The latest synthesis explicitly blocks:
  `parameter_only_threshold_loosen`,
  `repeat_failed_hypothesis_family_without_new_evidence`,
  `retry_validated_local_rejection_by_parameter_tuning`,
  `proposal_generation_without_approved_research_decision`, and
  `code_generation_from_blocked_or_deferred_research_decision`.
- Audit conclusion remains not complete: the generator has breadth, but the
  supported family inventory is exhausted for profitable evidence. The next
  productive step is not another existing variant smoke. It is a new,
  research-backed mechanism with a passing pre-proposal local falsification
  artifact against the latest 32-candidate failure memory.

## 2026-05-07 JST Order-Book Snapshot Quality Gate

- Added a local quality gate for timestamped order-book snapshot parquet files
  without starting any collector, websocket, paper process, trade process, or
  exchange-facing order endpoint. Bybit REST orderbook is treated as a current
  snapshot endpoint, not historical data; all-liquidation remains a public
  realtime websocket topic, not a historical REST input.
- Implementation:
  - `freqtrade_ext\bot_factory\data_quality.py` now exposes
    `check_order_book_parquet()` and
    `default_order_book_quality_output_path()`.
  - `scripts\bot_factory_check_order_book.py` checks normalized top-of-book
    snapshot parquet files with `date`, best bid, best ask, bid size, and ask
    size columns, plus optional `depth_imbalance`.
  - `freqtrade_ext\bot_factory\structural_data_capabilities.py` and
    `scripts\bot_factory_report_structural_data_capabilities.py` now accept
    order-book quality reports. A passing order-book quality report removes
    `order_book` from `blocked_without_new_data`, but `order_book` stays out
    of `local_research_usable` and remains in `must_not_codegen` until local
    event features and a supported codegen variant exist.
- Verification:

  ```powershell
  .\.venv\Scripts\python.exe -m py_compile freqtrade_ext\bot_factory\data_quality.py freqtrade_ext\bot_factory\structural_data_capabilities.py scripts\bot_factory_check_order_book.py scripts\bot_factory_report_structural_data_capabilities.py tests\test_bot_factory.py
  .\.venv\Scripts\python.exe -m pytest tests\test_bot_factory.py -q -k "order_book_quality or structural_data_capability_report"
  .\.venv\Scripts\python.exe scripts\bot_factory_check_order_book.py --help
  git diff --check
  ```

  Results: compile passed; focused pytest passed 6 tests and reached
  `[100%]`; the new CLI help rendered; `git diff --check` passed with only
  existing LF-to-CRLF working-copy warnings.
- Post-verification cleanup removed 33 workspace-local `.pytest_cache` /
  `__pycache__` directories outside `.venv`.
- Audit conclusion remains not complete: this opens a safer data-quality
  gate for future order-book/depth research, but it does not supply local
  historical order-book data, does not add order-book local-event features,
  does not permit order-book strategy codegen, and does not produce a
  profitable or paper-ready candidate.

## 2026-05-07 JST Direction Pivot to Edge Discovery

- User-level direction check: after more than a day and 32 local candidates /
  40 tried hypothesis families with no profitable or paper-ready candidate,
  continuing to generate another strategy family is no longer the right
  default action.
- Updated the source-of-truth TODO to add an explicit
  `Edge Discovery / Research Lab Pivot` before further default candidate
  generation. The pivot changes the next safe work from "make the 33rd
  strategy smoke" to "discover and falsify post-cost market effects before
  proposal/codegen."
- New required direction:
  - build edge-discovery artifacts before strategy proposals,
  - measure gross edge, all-in costs, net edge, win rate, sample count, data
    span, calendar-window stability, and concentration,
  - support local structural data inputs without treating current snapshots as
    historical unless timestamped parquet exists,
  - block proposal/codegen unless an edge artifact passes sample, span,
    post-cost, stability, and novelty gates against the latest synthesis/map,
  - ingest failed edge artifacts into failure synthesis so rejected mechanisms
    stop before proposal generation.
- Audit conclusion remains not complete: this is a direction correction, not a
  profitable candidate. The next implementation should be the edge-discovery
  artifact schema and runner, not another supported strategy variant.

## 2026-05-07 JST Edge Discovery Runner V1

- Implemented a local `research_edge_discovery` evidence layer instead of
  generating another candidate strategy:
  - `freqtrade_ext/bot_factory/edge_discovery.py`
  - `scripts/bot_factory_build_edge_discovery.py`
  - focused tests in `tests/test_bot_factory.py`
- The runner reads a `research_edge_discovery_spec` with fixed
  theory-named conditions, evaluates generated closed-candle events across
  multiple forward horizons, and writes `edge_discovery.json` plus
  `edge_discovery_report.md` under
  `registry/strategies/research_decisions/<edge_discovery_id>/`.
- The v1 artifact records OHLCV/context source paths, feature columns,
  condition diagnostics, horizon-level gross/price/funding edge,
  all-in-cost bps, net edge, win rate, sample count, rolling windows,
  quarterly calendar windows, event concentration diagnostics, blocker checks,
  and promotion guidance.
- Supported local inputs now cover primary OHLCV, informative OHLCV for
  cross-asset probes, funding rate, mark price, open interest, long/short
  ratio, and order-book quality report JSON. Current order-book REST
  snapshots are only recorded as quality reports; they are not treated as
  historical order-book features by this runner.
- Anti-parameter-search rule is enforced in v1: edge specs or conditions with
  grid/search markers such as `threshold_grid`, `parameter_grid`,
  `search_space`, `hyperopt`, `optimization`, `values`, `min/max/step`, or
  `candidates` are blocked from promotion evidence.
- Verification commands run:
  - `.\.venv\Scripts\python.exe -m py_compile freqtrade_ext\bot_factory\edge_discovery.py scripts\bot_factory_build_edge_discovery.py tests\test_bot_factory.py`
  - `.\.venv\Scripts\python.exe -m pytest tests\test_bot_factory.py -q -k "edge_discovery"`
  - `.\.venv\Scripts\python.exe -m pytest tests\test_bot_factory.py -q -k "edge_discovery or local_event_builder or local_falsification"`
  - `.\.venv\Scripts\python.exe scripts\bot_factory_build_edge_discovery.py --help`
  - `git diff --check`
- Results: compile passed; edge-discovery focused pytest passed 3 tests and
  reached `[100%]`; adjacent local event/falsification focused pytest passed
  23 tests and reached `[100%]`; the new CLI help rendered; `git diff --check`
  passed with only existing LF-to-CRLF working-copy warnings.
- Post-verification cleanup removed 33 workspace-local `.pytest_cache` /
  `__pycache__` directories outside `.venv`; follow-up cache count was `0`.
- Audit conclusion remains not complete: no profitable or paper-ready
  candidate exists. This is a necessary evidence gate before proposal/codegen,
  not a strategy, not a backtest, and not permission to start paper/live
  trading.

## 2026-05-07 JST Edge Discovery Concentration Diagnostics

- Added explicit event concentration diagnostics to
  `freqtrade_ext/bot_factory/edge_discovery.py`.
- Each `research_edge_discovery` artifact now records active day/week/month/
  quarter counts and max event share by day/week/month/quarter, and
  `edge_discovery_report.md` renders those fields in a dedicated section.
- Focused verification:
  - `.\.venv\Scripts\python.exe -m py_compile freqtrade_ext\bot_factory\edge_discovery.py tests\test_bot_factory.py`
  - `.\.venv\Scripts\python.exe -m pytest tests\test_bot_factory.py -q -k "edge_discovery"`
- Final verification also ran:
  - `.\.venv\Scripts\python.exe -m pytest tests\test_bot_factory.py -q`
  - `git diff --check`
- Results: compile passed; edge discovery focused pytest passed 8 tests and
  reached `[100%]`; full `tests/test_bot_factory.py` passed and reached
  `[100%]`; `git diff --check` passed with only existing LF-to-CRLF
  working-copy warnings.
- Post-verification cleanup removed 33 workspace-local `.pytest_cache` /
  `__pycache__` directories outside `.venv`; follow-up cache count was `0`.
- Audit conclusion remains not complete: this improves edge artifact quality,
  but it is not a profitable or paper-ready candidate.

## 2026-05-07 JST Edge Discovery Proposal / Codegen Gate

- Closed the immediate post-pivot gap where proposal/codegen could still be
  called without passing Edge Discovery evidence.
- `freqtrade_ext/bot_factory/strategy_proposals.py` now:
  - accepts `edge_discovery` / `research_edge_discovery` JSON evidence labels,
  - requires at least one passing `research_edge_discovery` artifact for the
    current thesis before proposal acceptance,
  - records a proposal-stage `edge_discovery_handoff`,
  - blocks missing, failed, mismatched-thesis, unsafe, parameter-search, or
    direct-codegen edge artifacts.
- `scripts/bot_factory_generate_strategy_proposal.py` now requires
  `--edge-discovery-json`, so the normal CLI path cannot create a proposal
  without a passing edge artifact.
- `freqtrade_ext/bot_factory/strategy_code.py` now rejects accepted/crafted
  proposal metadata that lacks a passing `edge_discovery_handoff`.
- Verification commands run:
  - `.\.venv\Scripts\python.exe -m py_compile freqtrade_ext\bot_factory\strategy_proposals.py freqtrade_ext\bot_factory\strategy_code.py scripts\bot_factory_generate_strategy_proposal.py tests\test_bot_factory.py`
  - `.\.venv\Scripts\python.exe -m pytest tests\test_bot_factory.py -q -k "edge_discovery or edge_discovery_handoff or strategy_proposal_generator_writes_safe_markdown or strategy_proposal_generator_blocks_without_passing_edge_discovery or strategy_code_generator_blocks_accepted_metadata_without_edge_discovery_handoff or strategy_code_generator_writes_long_only_strategy"`
  - `.\.venv\Scripts\python.exe -m pytest tests\test_bot_factory.py -q -k "strategy_proposal_generator or strategy_code_generator"`
  - `.\.venv\Scripts\python.exe -m pytest tests\test_bot_factory.py -q`
  - `.\.venv\Scripts\python.exe scripts\bot_factory_generate_strategy_proposal.py --help`
  - `git diff --check`
- Results: compile passed; focused handoff tests passed 7 tests; proposal /
  codegen focused regression passed; full `tests/test_bot_factory.py` passed
  and reached `[100%]`; proposal CLI help rendered with required
  `--edge-discovery-json`; `git diff --check` passed with only existing
  LF-to-CRLF working-copy warnings.
- Post-verification cleanup removed 33 workspace-local `.pytest_cache` /
  `__pycache__` directories outside `.venv`; follow-up cache count was `0`.
- Audit conclusion remains not complete: this blocks unsafe progression from
  edge evidence to strategy generation, but no profitable or paper-ready
  candidate exists.

## 2026-05-07 JST Edge Discovery Rejection Memory

- Closed the remaining post-pivot memory gap: failed or blocked
  `research_edge_discovery` artifacts can now be ingested before future
  research selection.
- `freqtrade_ext/bot_factory/candidate_failure_synthesis.py` now accepts
  `edge_discovery_paths`, classifies valid versus invalid edge rejections, and
  stores valid failed thesis IDs and mechanism classes in
  `aggregate_failure_summary`.
- `scripts/bot_factory_synthesize_candidate_failures.py` now exposes
  `--edge-discovery-json` and reports `edge_discovery_rejection_count`.
- `freqtrade_ext/bot_factory/candidate_failure_map.py` now carries validated
  edge rejections into causal-map research guidance and required questions.
- `freqtrade_ext/bot_factory/research_selection.py` now blocks a new thesis
  when its thesis ID or mechanism class matches validated edge-discovery
  rejection memory from the latest synthesis.
- Verification commands run:
  - `.\.venv\Scripts\python.exe -m py_compile freqtrade_ext\bot_factory\candidate_failure_synthesis.py freqtrade_ext\bot_factory\candidate_failure_map.py freqtrade_ext\bot_factory\research_selection.py scripts\bot_factory_synthesize_candidate_failures.py tests\test_bot_factory.py`
  - `.\.venv\Scripts\python.exe -m pytest tests\test_bot_factory.py -q -k "edge_discovery_rejection or local_rejection or research_selection_gate_blocks_validated_edge_discovery_rejection or research_selection_gate_ignores_invalid_edge_discovery_rejection"`
  - `.\.venv\Scripts\python.exe -m pytest tests\test_bot_factory.py -q -k "candidate_failure_synthesis or causal_failure_map or research_selection_gate"`
  - `.\.venv\Scripts\python.exe -m pytest tests\test_bot_factory.py -q`
  - `.\.venv\Scripts\python.exe scripts\bot_factory_synthesize_candidate_failures.py --help`
  - `git diff --check`
- Results: compile passed; focused edge/local rejection pytest passed 8 tests;
  broader synthesis/map/research-selection pytest passed 30 tests; full
  `tests/test_bot_factory.py` passed and reached `[100%]`; synthesis CLI help
  rendered with `--edge-discovery-json`; `git diff --check` passed with only
  existing LF-to-CRLF working-copy warnings.
- Post-verification cleanup removed 33 workspace-local `.pytest_cache` /
  `__pycache__` directories outside `.venv`; follow-up cache count was `0`.
- Audit conclusion remains not complete: this prevents repeated rejected edge
  mechanisms from returning to proposal/codegen, but there is still no
  profitable or paper-ready candidate.

## 2026-05-07 JST Worktree Cleanup Inventory

- Rechecked the cleanup position without deleting, restoring, staging, or
  moving files.
- Commands run:
  - `git status --short --untracked-files=all`
  - `git diff --name-status`
  - `git ls-files --others --exclude-standard`
  - `git ls-files --others --ignored --exclude-standard`
  - `git check-ignore -v data/market_structure/bybit/futures/BTC_USDT_USDT-1h-open_interest.parquet data/market_structure/bybit/futures/BTC_USDT_USDT-1h-long_short_ratio.parquet registry/strategies/research_decisions/example.json data/backtests/example.json user_data/data/bybit/futures/example.parquet`
  - `Select-String -Path docs\*.md -Pattern "BOT_FACTORY_GOAL_COMMAND_RUNBOOK" -CaseSensitive`
- Current non-ignored untracked files are source or source-like review
  candidates, not disposable runtime outputs: `docs/BOT_FACTORY_GOAL_AUDIT.md`,
  `data/market_structure/.gitkeep`, 12 new `freqtrade_ext/bot_factory/*.py`
  modules, and 17 new `scripts/bot_factory_*.py` CLIs.
- Current ignored runtime inventory is still outside Git management:
  `ignored_total=25480`, grouped as `.venv=21603`, `data=2433`,
  `registry=1386`, `user_data=56`, `.vscode=1`, and `docker=1`.
- Representative runtime paths are ignored by the intended `.gitignore` rules:
  `data/market_structure/**`, `registry/strategies/research_decisions/**`,
  `data/backtests/**`, and `user_data/*`.
- Owner decision remains required for
  `docs/BOT_FACTORY_GOAL_COMMAND_RUNBOOK.md`. It is deleted in the worktree and
  still referenced from audit/TODO/handoff docs, so accepting or reverting the
  deletion should be explicit.
- Audit conclusion remains not complete: this is hygiene and review scope
  control, not evidence of a profitable or paper-ready candidate.

## 2026-05-07 JST Edge Discovery Scope Schema

- Added explicit hypothesis-scope metadata to `research_edge_discovery`
  artifacts:
  `hypothesis_scope`, `instrument_universe`, and `market_structure_domains`.
- Supported scopes are `single_asset`, `cross_asset`, `market_neutral`,
  `funding_basis`, and `microstructure`.
- `cross_asset` and `market_neutral` specs are blocked unless
  `instrument_universe` contains at least two instruments, so a cross-asset
  claim cannot be represented as an indistinguishable single-asset edge probe.
- `edge_discovery_report.md` now renders the scope, instrument universe, and
  market-structure domains.
- Focused verification:
  - `.\.venv\Scripts\python.exe -m py_compile freqtrade_ext\bot_factory\edge_discovery.py tests\test_bot_factory.py`
  - `.\.venv\Scripts\python.exe -m pytest tests\test_bot_factory.py -q -k "edge_discovery"`
- Results: compile passed; edge discovery focused pytest passed 9 tests and
  reached `[100%]`.
- Final verification also ran
  `.\.venv\Scripts\python.exe -m pytest tests\test_bot_factory.py -q` and
  `git diff --check`; full `tests/test_bot_factory.py` passed and reached
  `[100%]`, and `git diff --check` passed with only existing LF-to-CRLF
  working-copy warnings.
- Post-verification cleanup removed 33 workspace-local `.pytest_cache` /
  `__pycache__` directories outside `.venv`; follow-up cache count was `0`.
- Audit conclusion remains not complete: this improves edge research artifact
  semantics, but it is not a profitable or paper-ready candidate.

## 2026-05-07 JST BTC/ETH Relative-Value Edge Rejection

- Ran a single fixed, non-grid `research_edge_discovery` probe after the
  user-level direction check that another generated strategy candidate would
  be the wrong default after more than a day of no profitable result.
- Thesis:
  `TH-BTC-ETH-RELATIVE-VALUE-REVERSION-20260507` /
  `btc_eth_relative_value_reversion`. The hypothesis was that after a closed
  4h ETH upside impulse, BTC underperformance versus ETH might catch up over
  short horizons. This was intentionally different from the already rejected
  BTC/ETH relative-strength continuation screen.
- Input artifact:
  `registry/strategies/research_decisions/20260507T210000JST_btc_eth_relative_value_reversion_edge/edge_spec.json`.
  It used local BTC 5m OHLCV as the primary series and local ETH 5m OHLCV as
  `informative_ohlcv`, with fixed conditions
  `informative_return_bps(48) >= 50.0` and
  `relative_return_bps(48) <= -100.0`, horizons `12`, `36`, and `72`, and
  `all_in_cost_bps=12.0`.
- Edge Discovery command:
  `.\.venv\Scripts\python.exe scripts\bot_factory_build_edge_discovery.py --ohlcv-path user_data\data\bybit\futures\BTC_USDT_USDT-5m-futures.parquet --informative-ohlcv-path user_data\data\bybit\futures\ETH_USDT_USDT-5m-futures.parquet --edge-spec-json registry\strategies\research_decisions\20260507T210000JST_btc_eth_relative_value_reversion_edge\edge_spec.json --failure-synthesis-json registry\strategies\synthesis\20260507T111500Z_all_candidates_with_cross_asset_relative_strength_rejection\candidate_failure_synthesis.json --edge-discovery-id 20260507T210000JST_btc_eth_relative_value_reversion_edge --created-at 2026-05-07T21:00:00+09:00 --reviewer-note "Fixed cross-asset relative value reversion check after broad candidate failure synthesis; no threshold grid or strategy code generation."`
- Result artifact:
  `registry/strategies/research_decisions/20260507T210000JST_btc_eth_relative_value_reversion_edge/edge_discovery.json`
  failed with `event_count=937`, `data_span_days=857.270833`,
  `passing_horizon_count=0`, `proposal_generation_allowed=false`, and
  `strategy_codegen_allowed=false`. Best horizon was `72` candles with
  `sample_count=937`, `net_edge_bps=-11.699275`,
  `profitable_windows_ratio=0.25`, and
  `profitable_calendar_windows_ratio=0.2`.
- Ingested the failed edge into rejection memory with
  `scripts/bot_factory_synthesize_candidate_failures.py --edge-discovery-json`.
  New synthesis:
  `registry/strategies/synthesis/20260507T211000JST_all_candidates_with_btc_eth_relative_value_reversion_edge_rejection/candidate_failure_synthesis.json`
  completed with `candidate_count=32`, `paper_ready_count=0`,
  `parameter_only_retry_allowed=false`, `requires_new_thesis_id=true`, and
  `edge_discovery_rejection_count=1`. The failed edge thesis ID and mechanism
  class are now recorded as
  `TH-BTC-ETH-RELATIVE-VALUE-REVERSION-20260507` and
  `btc_eth_relative_value_reversion`.
- Hygiene checks: `git diff --check` passed with only existing LF-to-CRLF
  working-copy warnings; `git check-ignore -v` confirmed the new
  `edge_spec.json`, `edge_discovery.json`, and refreshed
  `candidate_failure_synthesis.json` are ignored under the intended
  `registry/strategies/**` rules. Cache cleanup removed 2 workspace-local
  `__pycache__` directories outside `.venv`; follow-up cache count was `0`.
- Audit conclusion remains not complete: the first post-pivot fixed Edge
  Discovery probe found no post-cost edge. Repeating BTC/ETH relative-value
  reversion by threshold tuning is now a blocked path; future work needs a
  materially different market mechanism or additional local historical
  structural data before proposal generation or codegen should continue.

## 2026-05-07 JST Liquidation Quality Gate

- Added a local quality gate for user-supplied historical liquidation parquet
  files. This does not start collectors, websockets, paper/trade processes, or
  exchange-facing order endpoints. Bybit all-liquidation is still treated as a
  public realtime websocket stream, not a historical REST download.
- Implementation:
  `freqtrade_ext/bot_factory/data_quality.py` now has
  `check_liquidation_parquet()` and
  `default_liquidation_quality_output_path()`;
  `scripts/bot_factory_check_liquidation.py` exposes the CLI; and
  `freqtrade_ext/bot_factory/structural_data_capabilities.py` plus
  `scripts/bot_factory_report_structural_data_capabilities.py` accept
  `--liquidation-quality-report-json`.
- The checker accepts normalized or Bybit-style local columns: timestamp
  (`date`/`T`/`timestamp`), side (`side`/`S`), size
  (`size`/`quantity`/`qty`/`v`), and price (`price`/`p`/`bankruptcy_price`).
  It validates timestamp parsing and sort order, Buy/Sell side values,
  positive numeric size, and positive numeric price.
- Initial safety gate preserved: a passing liquidation quality report can
  remove `liquidation` from `blocked_without_new_data`, but does not permit
  codegen. The follow-up below adds local-event/Edge Discovery features, so a
  quality-checked local liquidation file may become `local_research_usable`;
  liquidation remains in `must_not_codegen`.
- Verification commands run:
  - `.\.venv\Scripts\python.exe -m py_compile freqtrade_ext\bot_factory\data_quality.py freqtrade_ext\bot_factory\structural_data_capabilities.py scripts\bot_factory_check_liquidation.py scripts\bot_factory_report_structural_data_capabilities.py tests\test_bot_factory.py`
  - `.\.venv\Scripts\python.exe scripts\bot_factory_check_liquidation.py --help`
  - `.\.venv\Scripts\python.exe -m pytest tests\test_bot_factory.py -q -k "liquidation_quality or structural_data_capability_report"`
  - `.\.venv\Scripts\python.exe -m pytest tests\test_bot_factory.py -q -k "structural_data_capability or structural_data_quality or order_book_quality or liquidation_quality"`
  - `.\.venv\Scripts\python.exe scripts\bot_factory_report_structural_data_capabilities.py --help`
  - `git diff --check`
- Results: compile passed; liquidation CLI help rendered; focused pytest
  passed 7 tests; broader structural-data pytest passed 9 tests; structural
  capability CLI help rendered with `--liquidation-quality-report-json`;
  `git diff --check` passed with only existing LF-to-CRLF working-copy
  warnings. Post-test cleanup removed 33 workspace-local `.pytest_cache` /
  `__pycache__` directories outside `.venv`; follow-up cache count was `0`.
- Audit conclusion remains not complete: this prepares a higher-value
  structural-data path for future edge research, but it does not collect
  liquidation history and does not produce a profitable or paper-ready
  candidate.

## 2026-05-07 JST Liquidation Local-Event / Edge Discovery Features

- Added closed-context liquidation features to the local event builder and Edge
  Discovery runner. This still does not start collectors, websockets,
  backtests, paper/trade processes, or exchange-facing order endpoints.
- Implementation:
  - `freqtrade_ext/bot_factory/local_events.py` now accepts
    `liquidation_path`, loads normalized or Bybit-style historical liquidation
    parquet/CSV, aggregates events into the closed base candle, and exposes
    `liquidation_count`, buy/sell/total/net notional, imbalance, and total
    notional z-score features.
  - `scripts/bot_factory_build_local_events.py`,
    `freqtrade_ext/bot_factory/edge_discovery.py`, and
    `scripts/bot_factory_build_edge_discovery.py` now accept
    `--liquidation-path`.
  - `freqtrade_ext/bot_factory/structural_data_capabilities.py` now marks
    liquidation as local-event supported. With local data and passing quality
    reports, liquidation can become `local_research_usable`; it still remains
    in `must_not_codegen` because no liquidation strategy codegen variant
    exists.
- Verification commands run:
  - `.\.venv\Scripts\python.exe -m py_compile freqtrade_ext\bot_factory\data_quality.py freqtrade_ext\bot_factory\structural_data_capabilities.py freqtrade_ext\bot_factory\local_events.py freqtrade_ext\bot_factory\edge_discovery.py scripts\bot_factory_build_local_events.py scripts\bot_factory_build_edge_discovery.py scripts\bot_factory_report_structural_data_capabilities.py tests\test_bot_factory.py`
  - `.\.venv\Scripts\python.exe scripts\bot_factory_build_local_events.py --help`
  - `.\.venv\Scripts\python.exe scripts\bot_factory_build_edge_discovery.py --help`
  - `.\.venv\Scripts\python.exe -m pytest tests\test_bot_factory.py -q -k "liquidation_context or liquidation_quality or edge_discovery_supports_liquidation or structural_data_capability_report_accepts_liquidation or local_event_cli_maps_informative_ohlcv_path or edge_discovery_cli_maps_context"`
  - `.\.venv\Scripts\python.exe -m pytest tests\test_bot_factory.py -q -k "local_event_builder_supports or local_event_cli_maps or edge_discovery or structural_data_capability or structural_data_quality or liquidation_quality"`
  - `.\.venv\Scripts\python.exe -m pytest tests\test_bot_factory.py -q`
  - `git diff --check`
- Results: compile passed; local-event and Edge Discovery CLI help rendered
  with `--liquidation-path`; focused liquidation context tests passed 7 tests;
  broader local-event/edge/structural tests passed 24 tests; full
  `tests/test_bot_factory.py` passed and reached `[100%]`; `git diff --check`
  passed with only existing LF-to-CRLF working-copy warnings. Post-test cleanup
  removed 33 workspace-local `.pytest_cache` / `__pycache__` directories
  outside `.venv`; follow-up cache count was `0`.
- Audit conclusion remains not complete: liquidation can now participate in
  fixed-theory local research when the user supplies historical data, but no
  liquidation history is present, no liquidation codegen exists, and no
  profitable or paper-ready candidate exists.

## 2026-05-07 JST Order-Book Local-Event / Edge Discovery Features

- Added closed-context order-book snapshot features to the local event builder
  and Edge Discovery runner. This still does not start collectors, websockets,
  backtests, paper/trade processes, or exchange-facing order endpoints.
- Implementation:
  - `freqtrade_ext/bot_factory/local_events.py` now accepts
    `order_book_path`, loads normalized historical top-of-book snapshot
    parquet/CSV, aggregates snapshots into the closed base candle, and exposes
    bid size, ask size, depth imbalance, spread, mid-price gap, and z-score
    features.
  - `scripts/bot_factory_build_local_events.py`,
    `freqtrade_ext/bot_factory/edge_discovery.py`, and
    `scripts/bot_factory_build_edge_discovery.py` now accept
    `--order-book-path`.
  - `freqtrade_ext/bot_factory/structural_data_capabilities.py` now marks
    order-book snapshots as local-event supported. With local data and passing
    quality reports, `order_book` can become `local_research_usable`; it still
    remains in `must_not_codegen` because no order-book strategy codegen
    variant exists.
- Verification commands run:
  - `.\.venv\Scripts\python.exe -m py_compile freqtrade_ext\bot_factory\local_events.py freqtrade_ext\bot_factory\edge_discovery.py freqtrade_ext\bot_factory\structural_data_capabilities.py scripts\bot_factory_build_local_events.py scripts\bot_factory_build_edge_discovery.py tests\test_bot_factory.py`
  - `.\.venv\Scripts\python.exe scripts\bot_factory_build_local_events.py --help`
  - `.\.venv\Scripts\python.exe scripts\bot_factory_build_edge_discovery.py --help`
  - `.\.venv\Scripts\python.exe -m pytest tests\test_bot_factory.py -q -k "order_book_context or order_book_quality or edge_discovery_supports_order_book or structural_data_capability_report_accepts_order_book or local_event_cli_maps_informative_ohlcv_path or edge_discovery_cli_maps_context"`
  - `.\.venv\Scripts\python.exe -m pytest tests\test_bot_factory.py -q -k "local_event_builder_supports or local_event_cli_maps or edge_discovery or structural_data_capability or structural_data_quality or order_book_quality"`
  - `.\.venv\Scripts\python.exe -m pytest tests\test_bot_factory.py -q`
  - `git diff --check`
- Results: compile passed; local-event and Edge Discovery CLI help rendered
  with `--order-book-path`; focused order-book context tests passed 7 tests;
  broader local-event/edge/structural tests passed 26 tests; full
  `tests/test_bot_factory.py` passed and reached `[100%]`; `git diff --check`
  passed with only existing LF-to-CRLF working-copy warnings. Post-test cleanup
  removed 33 workspace-local `.pytest_cache` / `__pycache__` directories
  outside `.venv`; follow-up cache count was `0`.
- Audit conclusion remains not complete: order-book snapshots can now
  participate in fixed-theory local research when the user supplies historical
  timestamped data, but no order-book history collector or codegen variant
  exists, and no profitable or paper-ready candidate exists.

## 2026-05-07 JST Structural Local-Event Quality Enforcement

- Closed the remaining local-research safety gap for liquidation/order-book
  context features. A parseable local parquet/CSV is no longer enough when a
  local event or Edge Discovery spec references `liquidation_*` or
  `order_book_*`; the runner also requires a supplied passing quality report
  JSON for that structural data family.
- Implementation:
  - `freqtrade_ext/bot_factory/local_events.py` accepts
    liquidation/order-book quality report paths, records report summaries, and
    blocks required structural contexts with
    `*_quality_report_passed_when_required` unless at least one supplied report
    is parseable and all supplied reports have `ok=true`.
  - `freqtrade_ext/bot_factory/edge_discovery.py` applies the same gate before
    event extraction and post-cost horizon scoring.
  - `scripts/bot_factory_build_local_events.py` and
    `scripts/bot_factory_build_edge_discovery.py` expose the corresponding
    `--liquidation-quality-report-json` and
    `--order-book-quality-report-json` arguments.
- Verification commands run:
  - `.\.venv\Scripts\python.exe -m py_compile freqtrade_ext\bot_factory\local_events.py freqtrade_ext\bot_factory\edge_discovery.py scripts\bot_factory_build_local_events.py scripts\bot_factory_build_edge_discovery.py tests\test_bot_factory.py`
  - `.\.venv\Scripts\python.exe scripts\bot_factory_build_local_events.py --help`
  - `.\.venv\Scripts\python.exe scripts\bot_factory_build_edge_discovery.py --help`
  - `.\.venv\Scripts\python.exe -m pytest tests\test_bot_factory.py -q -k "order_book_context or liquidation_context or quality_report_passed_when_required or local_event_cli_maps_informative_ohlcv_path or edge_discovery_cli_maps_context"`
  - `.\.venv\Scripts\python.exe -m pytest tests\test_bot_factory.py -q -k "local_event_builder_supports or local_event_cli_maps or edge_discovery or structural_data_capability or structural_data_quality or order_book_quality or liquidation_quality"`
  - `.\.venv\Scripts\python.exe -m pytest tests\test_bot_factory.py -q`
  - `git diff --check`
- Results: compile passed; local-event and Edge Discovery CLI help rendered
  with the new quality-report arguments; focused quality-gate tests passed
  8 tests; broader local-event/edge/structural tests passed 29 tests; full
  `tests/test_bot_factory.py` passed and reached `[100%]`; `git diff --check`
  passed with only existing LF-to-CRLF working-copy warnings. Post-test cleanup
  removed 33 workspace-local `.pytest_cache` / `__pycache__` directories
  outside `.venv`; follow-up cache count was `0`.
- Audit conclusion remains not complete: this makes the research gate more
  faithful to the documented safety boundary, but it does not supply new
  structural history, does not create a codegen variant, and does not produce a
  profitable or paper-ready candidate.

## 2026-05-07 JST Positioning Local-Event Quality Enforcement

- Closed the same local-research quality gap for positioning data. A parseable
  local parquet/CSV is no longer enough when a local event or Edge Discovery
  spec references `open_interest*` or long/short-ratio features; the runner also
  requires a supplied passing quality report JSON for that data family.
- Implementation:
  - `freqtrade_ext/bot_factory/local_events.py` accepts open-interest and
    long/short-ratio quality report paths, records report summaries, and blocks
    required positioning contexts with
    `*_quality_report_passed_when_required` unless at least one supplied report
    is parseable and all supplied reports have `ok=true`.
  - `freqtrade_ext/bot_factory/edge_discovery.py` applies the same gate before
    event extraction and post-cost horizon scoring.
  - `scripts/bot_factory_build_local_events.py` and
    `scripts/bot_factory_build_edge_discovery.py` expose
    `--open-interest-quality-report-json` and
    `--long-short-ratio-quality-report-json`.
- Verification commands run:
  - `.\.venv\Scripts\python.exe -m py_compile freqtrade_ext\bot_factory\local_events.py freqtrade_ext\bot_factory\edge_discovery.py scripts\bot_factory_build_local_events.py scripts\bot_factory_build_edge_discovery.py tests\test_bot_factory.py`
  - `.\.venv\Scripts\python.exe -m pytest tests\test_bot_factory.py -q -k "open_interest_context or long_short_ratio_context or positioning_context_without_quality_reports or local_event_cli_maps_informative_ohlcv_path or edge_discovery_cli_maps_context"`
  - `.\.venv\Scripts\python.exe scripts\bot_factory_build_local_events.py --help`
  - `.\.venv\Scripts\python.exe scripts\bot_factory_build_edge_discovery.py --help`
  - `.\.venv\Scripts\python.exe -m pytest tests\test_bot_factory.py -q -k "local_event_builder_supports or local_event_builder_blocks or edge_discovery or structural_data_capability or structural_data_quality or open_interest_quality or long_short_ratio_quality"`
  - `.\.venv\Scripts\python.exe -m pytest tests\test_bot_factory.py -q`
  - `git diff --check`
- Results: compile passed; focused positioning quality-gate tests passed
  7 tests; local-event and Edge Discovery CLI help rendered with the new
  quality-report arguments; broader local-event/edge/structural tests passed
  35 tests; full `tests/test_bot_factory.py` passed and reached `[100%]`;
  `git diff --check` passed with only existing LF-to-CRLF working-copy
  warnings. Post-test cleanup removed workspace-local `.pytest_cache` /
  `__pycache__` directories outside `.venv`; follow-up cache count was `0`.
- Audit conclusion remains not complete: this improves the pre-proposal
  evidence boundary, but no new profitable or paper-ready candidate exists.
