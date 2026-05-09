# Bot Factory Strategy Generation Next Agent Prompt

Use the following prompt for the next coding agent.

````markdown
Continue Bot Factory from the current Strategy Generation / Candidate Factory
state. The project goal is an AI-assisted candidate factory: generate multiple
rule-based, FreqAI, and hybrid ML+rule strategy candidates; evaluate them with
local historical artifacts; rank/select/reject them; and feed failures plus
reviewer findings back into later candidate iterations.

Direction correction: do not add another candidate family as the next default
action. Thirty-two local candidate manifests have failed historical,
walk-forward, or initial historical gates. The later
`mark_fair_value_momentum_lag` path also failed generated-entry edge
diagnostics before historical backtest interpretation. The local event-study
vs generated-diagnostics mismatch has now been traced to informative
funding/mark context timing: local events were merging context candles at their
open timestamps, while generated code and diagnostics use closed-candle
availability. `local_events.py` has been corrected to shift context timestamps
by `context_interval - base_interval` before `merge_asof`. Do not loosen
thresholds or keep adding variants. Local event artifacts now carry
`context_merge.semantics=closed_context_candle_availability_v1`; local
falsification and research selection reject high-risk cost-edge evidence whose
funding/mark event source lacks that closed-context alignment proof. A research
selection gate consumes the latest causal failure map; use it before any new
proposal or generator extension with `--causal-failure-map-json` and explicit
`--causal-failure-response` entries, and only proceed when it emits
`status=approved_for_proposal_generation` and a passing
`research_selection_score`.
Edge Discovery is now the default pre-proposal path. Use
`scripts/bot_factory_build_edge_discovery.py` to produce a local
`research_edge_discovery` artifact before proposal generation, and only allow
proposal/codegen after a passing edge artifact for the same thesis. Proposal
generation requires `--edge-discovery-json`, and accepted proposal metadata
must carry a passing `edge_discovery_handoff` before codegen. Failed or
blocked edge-discovery artifacts should be fed back into
`scripts/bot_factory_synthesize_candidate_failures.py --edge-discovery-json`;
valid failed edge theses/mechanisms now flow into synthesis, causal-map
guidance, and research-selection novelty blockers. Do not treat invalid
parameter-search edge artifacts as rejection evidence. Edge artifacts now
also report event concentration diagnostics by day/week/month/quarter; use
those fields to reject small or calendar-concentrated effects before any
proposal. They also carry `hypothesis_scope`, `instrument_universe`, and
`market_structure_domains`; use `single_asset`, `cross_asset`,
`market_neutral`, `funding_basis`, or `microstructure` explicitly, and provide
at least two instruments for cross-asset or market-neutral scopes.
The latest refreshed synthesis and causal map also carry failed local
falsification rejections, so a new thesis must explicitly avoid or materially
distinguish itself from `TH-MARK-FAIR-VALUE-MOMENTUM-LAG-20260507` /
`mark_fair_value_momentum_lag` and
`TH-LOW-RANGE-VOLUME-ABSORPTION-20260507` /
`low_range_volume_absorption`, plus
`TH-THIN-BOOK-DISLOCATION-REVERSION-20260507` /
`thin_book_dislocation_reversion` and
`TH-OPEN-INTEREST-DELEVERAGING-REBOUND-20260507` /
`open_interest_deleveraging_rebound`, plus
`TH-OPEN-INTEREST-IMPULSE-CONTINUATION-20260507` /
`open_interest_impulse_continuation`, plus
`TH-LONG-SHORT-FLUSH-REBOUND-20260507` /
`long_short_crowding_flush_rebound`, plus
`TH-NEGATIVE-FUNDING-UNCROWDED-CARRY-20260507` /
`negative_funding_uncrowded_long_carry`, plus
`TH-CROWDING-UNWIND-REACCUMULATION-20260507` /
`crowding_unwind_reaccumulation`, plus
`TH-POST-DELEVERAGING-VOL-COMPRESSION-20260507` /
`post_deleveraging_volatility_compression`, plus
`TH-MARK-PREMIUM-OI-CONTINUATION-20260507` /
`mark_premium_open_interest_continuation`, plus
`TH-VOLUME-CLOCK-LIQUIDITY-MOMENTUM-20260507` /
`volume_clock_liquidity_momentum`.

Latest mark fair-value momentum lag outcome note:
`TH-MARK-FAIR-VALUE-MOMENTUM-LAG-20260507` /
`mark_fair_value_momentum_lag` passed research selection and proposal/codegen
but is not viable. Research decision
`registry/strategies/research_decisions/20260507T164000JST_mark_fair_value_momentum_lag_research_selection/research_decision.json`
had `status=approved_for_proposal_generation` and
`research_selection_score=100.0`. The local pre-proposal screen looked good
(`event_count=2274`, `expected_edge_bps=27.99047`,
`net_edge_bps=15.99047`, `profitable_windows_ratio=1.0`), but that screen is
now known to be a context-timing false positive. Code generation produced
`registry/strategies/generated/LongOnlyMarkFairValueMomentumLagCandidate/mark_fair_value_momentum_lag_002/metadata.json`
with static check passing. Full generated-entry diagnostics then failed:
`registry/strategies/diagnostics/LongOnlyMarkFairValueMomentumLagCandidate/mark_fair_value_momentum_lag_002/20260507T170500JST_mark_fair_value_momentum_lag_002_full_signal_edge_diagnostics/signal_diagnostics.json`
has `entry_count=2775`, `generated_entry_edge.status=fail`,
`net_edge_bps=-11.04985`, and `profitable_windows_ratio=0.0`. After fixing
local context alignment, the re-run local event artifact
`registry/strategies/research_decisions/20260507T183500JST_mark_fair_value_momentum_lag_closed_context_event_study_v2/local_events.json`
matched generated diagnostics exactly:
`combined_match_count_before_cooldown=20169`, `cooldown_candles=12`, and
`event_count=2775`, and it records
`context_merge.semantics=closed_context_candle_availability_v1`,
`required_contexts=["mark_price"]`, and
`closed_context_candle_alignment=true`. Corrected local falsification
`registry/strategies/research_decisions/20260507T184000JST_mark_fair_value_momentum_lag_closed_context_falsification_v2/local_falsification.json`
failed with `net_edge_bps=-11.04985`, `profitable_windows_ratio=0.0`, and a
passing event-source alignment check. Do not backtest, promote, paper-start, or
recycle this thesis as a fresh candidate. Do not use the older approved
research decision or accepted proposal as current positive evidence. The older
positive local falsification artifact was also passed through the updated
research-selection gate at
`registry/strategies/research_decisions/20260507T185500JST_mark_fair_value_stale_context_research_selection_guard_clean/research_decision.json`;
it is blocked because `event_source_context_alignment_valid=false` and
`cost_edge_passes=false`.

Latest fresh local screen note:
`TH-LOW-RANGE-VOLUME-ABSORPTION-20260507` /
`low_range_volume_absorption` was screened only with local BTC 5m OHLCV before
any proposal/codegen. The event study
`registry/strategies/research_decisions/20260507T190500JST_low_range_volume_absorption_event_study/local_events.json`
completed with `event_count=78` and
`combined_match_count_before_cooldown=95`, but local falsification
`registry/strategies/research_decisions/20260507T191000JST_low_range_volume_absorption_local_falsification/local_falsification.json`
failed with `net_edge_bps=-16.16165` and
`profitable_windows_ratio=0.0`. Do not promote, propose, codegen, or retune
this thesis.

Latest thin-book dislocation screen note:
`TH-THIN-BOOK-DISLOCATION-REVERSION-20260507` /
`thin_book_dislocation_reversion` was screened as a distinct fixed local
pre-proposal thesis: sharp six-candle downside move, wide current range,
below-average volume, and negative 48-candle SMA distance. The event study
`registry/strategies/research_decisions/20260507T193500JST_thin_book_dislocation_reversion_event_study/local_events.json`
completed with `event_count=1516`, but local falsification
`registry/strategies/research_decisions/20260507T194000JST_thin_book_dislocation_reversion_local_falsification/local_falsification.json`
failed with `expected_edge_bps=1.802646`, `net_edge_bps=-10.197354`, and
`profitable_windows_ratio=0.0`. Do not propose, codegen, retune, or recycle
this thesis without materially new theory and explicit prior-rejection
handling.

Latest structural-data support note:
Open-interest support has been added to avoid repeating OHLCV-only screens
when the next theory really needs derivatives positioning data. Use
`scripts/bot_factory_check_open_interest.py` to verify local open-interest
parquet quality before any open-interest thesis, and use
`scripts/bot_factory_build_local_events.py --open-interest-path ...` when an
event spec references `open_interest`, `open_interest_delta_pct`, or
`open_interest_zscore`. The local event builder merges open-interest context
with the same closed-context-candle availability semantics as funding-rate and
mark-price context. Local events and Edge Discovery also require
`--open-interest-quality-report-json` whenever `open_interest*` features are
referenced. A public Bybit V5 open-interest downloader now exists at
`scripts/bot_factory_download_bybit_open_interest.py`; it uses no API keys and
only the public market open-interest endpoint. BTCUSDT 1h open-interest data
for `2024-01-01` to `2025-02-01` is now local at
`data/market_structure/bybit/futures/BTC_USDT_USDT-1h-open_interest.parquet` and
passed quality check
`registry/strategies/checks/20260507T152500Z_open_interest_quality_safe_path.json`
with `ok=true`, `rows=9529`, `duplicate_timestamps=0`, and
`missing_intervals=0`. Public Bybit structural market data must stay under
`data/market_structure/...`, not `user_data/data/bybit/...`; Freqtrade
`download-data` scans the latter and can fail if non-candle structural parquet
files are placed there. Liquidation and order-book files are still absent.
Research selection now requires structural-data theses to supply a passing
local data quality report via `--local-data-quality-report-json`; it blocks
open-interest, liquidation, order-book, market-depth, and book/depth-imbalance
claims without such evidence using the
`structural_data_quality_report_present` blocker. Proposal generation also
rechecks this handoff: structural-data proposals require a supplied
`research_decision.json` whose `local_data_quality_report_paths` still exist
inside the workspace and whose `local_data_quality_reports_valid` plus
`structural_data_quality_report_present` checks passed. Use
`scripts/bot_factory_generate_strategy_proposal.py --local-data-quality-json`
to preserve the same local quality report as proposal evidence. Code generation
now rechecks the proposal handoff and blocks structural-data proposals unless
the proposal metadata contains a passing quality handoff. It also blocks
structural-data codegen unless a real supported structural-data strategy
variant exists. The currently supported structural codegen variant is the
local historical `crowding_unwind_reaccumulation` path for open-interest plus
long/short account-ratio parquet. Liquidation and order-book/depth theses
still must not be rendered as OHLCV-only strategy code.

Order-book snapshot quality gate has been added for local, timestamped
top-of-book parquet files only. `freqtrade_ext/bot_factory/data_quality.py`
exposes `check_order_book_parquet()` and
`default_order_book_quality_output_path()`.
`scripts/bot_factory_check_order_book.py` validates normalized order-book
snapshots with `date`, best bid, best ask, bid size, ask size, and optional
`depth_imbalance`; `scripts/bot_factory_report_structural_data_capabilities.py`
accepts `--order-book-quality-report-json`. A passing order-book quality
report plus local timestamped snapshot data can make `order_book`
`local_research_usable`; `order_book` remains in `must_not_codegen` until a
supported codegen variant exists. Local events and Edge Discovery now accept
`--order-book-path` and can use top-of-book size, depth imbalance, spread,
mid-price gap, and z-score features. Do not treat the REST orderbook endpoint
as historical data; it is a current snapshot endpoint.

Latest long/short ratio structural screen note:
Bybit BTCUSDT 1h public long/short ratio is available at
`data/market_structure/bybit/futures/BTC_USDT_USDT-1h-long_short_ratio.parquet`
and passed quality check
`registry/strategies/checks/20260507T160000Z_long_short_ratio_quality_extended_safe_path.json`
with `ok=true`, `rows=20569`, `duplicate_timestamps=0`, and
`missing_intervals=0`. Combined structural capability report
`registry/strategies/checks/20260507T083000Z_structural_data_capabilities_codegen_oi_lsr.json`
now marks `local_research_usable=["open_interest","long_short_ratio"]` and
keeps only `liquidation` and `order_book` in `must_not_codegen`. A fixed
pre-proposal thesis
`TH-LONG-SHORT-FLUSH-REBOUND-20260507` /
`long_short_crowding_flush_rebound` was rerun with extended history and the
same fixed conditions. Local events completed with `event_count=608`, then
local falsification failed with `expected_edge_bps=1.288089`,
`net_edge_bps=-10.711911`, `sample_count=608`,
`profitable_windows_ratio=0.25`, `blocker_count=2`. Treat this as rejected
cost/stability evidence, not a candidate to promote or retune. Local events and
Edge Discovery now require `--long-short-ratio-quality-report-json` whenever
long/short-ratio features are referenced.
`TH-NEGATIVE-FUNDING-UNCROWDED-CARRY-20260507` /
`negative_funding_uncrowded_long_carry` was also screened as a separate fixed
pre-proposal thesis with funding-adjusted 8h hold logic. Local events produced
only `event_count=1`, and local falsification failed the sample gate despite
a single positive observed event: `expected_edge_bps=13.845317`,
`net_edge_bps=1.845317`, `sample_count=1`,
`profitable_windows_ratio=1.0`, `blocker_count=1`. Treat it as rejected
small-sample evidence, not a candidate to promote or retune.

Latest open-interest structural screen note:
`TH-OPEN-INTEREST-DELEVERAGING-REBOUND-20260507` /
`open_interest_deleveraging_rebound` was screened with the new BTCUSDT 1h
open-interest data plus local BTC 5m OHLCV. The event study
`registry/strategies/research_decisions/20260507T204000JST_open_interest_deleveraging_rebound_event_study/local_events.json`
completed with `event_count=203`, but local falsification
`registry/strategies/research_decisions/20260507T204500JST_open_interest_deleveraging_rebound_local_falsification/local_falsification.json`
failed with `expected_edge_bps=7.363403`, `net_edge_bps=-4.636597`, and
`profitable_windows_ratio=0.25`. Do not promote, propose, codegen, backtest,
or retune this thesis.

Latest open-interest impulse continuation note:
`TH-OPEN-INTEREST-IMPULSE-CONTINUATION-20260507` /
`open_interest_impulse_continuation` tested the opposite mechanism: one-hour
open-interest expansion during an upside/high-volume range impulse. The event
study
`registry/strategies/research_decisions/20260507T211000JST_open_interest_impulse_continuation_event_study/local_events.json`
completed with `event_count=204`, but local falsification
`registry/strategies/research_decisions/20260507T211500JST_open_interest_impulse_continuation_local_falsification/local_falsification.json`
failed with `expected_edge_bps=5.360183`, `net_edge_bps=-6.639817`, and
`profitable_windows_ratio=0.0`. Do not continue simple sign-flipped OI/price
impulse variants without materially new evidence.

Latest mark-discount outcome note:
`TH-MARK-DISCOUNT-RECLAIM-20260507` /
`mark_discount_reclaim_continuation` is no longer the next proposal step. It
was carried through proposal and code generation:
`registry/strategies/proposals/20260507T055000Z_LongOnlyMarkDiscountReclaimCandidate.metadata.json`
and
`registry/strategies/generated/LongOnlyMarkDiscountReclaimCandidate/mark_discount_reclaim_001/metadata.json`.
The checked historical backtest
`data/backtests/LongOnlyMarkDiscountReclaimCandidate/mark_discount_reclaim_001_20240101_20250201/metrics.json`
failed with `total_return_pct=-37.832122396`,
`profit_factor=0.244072592568329`,
`max_drawdown_pct=37.83212239599998`, and
`sortino=-168.6116978183358`. Signal diagnostics completed with
`entry_signal_count=9603`, so the failure was not zero-entry sparsity. Treat
`mark_discount_reclaim` as a failed family unless future work brings materially
new evidence that explicitly addresses the local-event-to-generated-strategy
mismatch.

Latest synthesis/map:
`registry/strategies/synthesis/20260507T111500Z_all_candidates_with_cross_asset_relative_strength_rejection/candidate_failure_synthesis.json`
covers 32 candidates with `paper_ready_count=0`,
`parameter_only_retry_allowed=false`, `requires_new_thesis_id=true`, and
`local_falsification_rejection_count=12`. It carries the failed
`crowding_unwind_reaccumulation` generated candidate plus the recent validated
local rejections for `post_deleveraging_volatility_compression`,
`mark_premium_open_interest_continuation`,
`mark_fair_value_momentum_lag`, `low_range_volume_absorption`,
`thin_book_dislocation_reversion`, `open_interest_deleveraging_rebound`,
`open_interest_impulse_continuation`,
`open_interest_crowded_short_squeeze`,
`funding_neutral_impulse_drift`, and
`negative_funding_uncrowded_long_carry`, plus the latest rejected
`volume_clock_liquidity_momentum` and
`cross_asset_relative_strength_continuation` screens. The mark-premium/OI screen had strong
post-cost aggregate edge but failed explicit calendar-stability gates
(`sample_count=33`, `calendar_window_count=2`,
`profitable_calendar_windows_ratio=0.5`, 32 events in `2024Q1`, one event in
`2024Q4`). The volume-clock liquidity momentum screen produced 686 events but
failed with `expected_edge_bps=0.255677`, `net_edge_bps=-11.744323`, and
`profitable_windows_ratio=0.25`; treat it as a rejected intraday
volume/session momentum mechanism, not a threshold-retuning target. The fixed
BTC/ETH relative-strength continuation screen used BTC primary OHLCV plus ETH
informative OHLCV and produced 733 events, but failed with
`expected_edge_bps=-5.614326`, `net_edge_bps=-17.614326`,
`profitable_windows_ratio=0.0`, and
`profitable_calendar_windows_ratio=0.0`; treat it as a rejected cross-asset
relative-strength continuation mechanism, not a threshold-retuning target. Do not
treat earlier passing local falsification or diagnostics as
promotion evidence because historical, walk-forward, post-cost local, sample,
closed-context, or calendar-stability gates failed. The latest aggregate
ranking is
`registry/strategies/candidates/rankings/20260507T085500Z_all_candidates_with_crowding_unwind_rejection_ranking/candidate_ranking.json`
with `paper_ready_candidate_ids=[]`.
`registry/strategies/failure_maps/20260507T112000Z_all_candidates_with_cross_asset_relative_strength_rejection_causal_map/causal_failure_map.json`
requires a research decision before proposal and flags dominant risks in
`regime_fragile_mechanism` (32 candidates), `walk_forward_fragility` (32),
`cost_sensitive_mechanism` (31), `no_profitable_walk_forward_windows` (25),
and `entry_exists_negative_edge` (22). Its causal risk weights require the
next research decision to address cost/edge, no-profitable-window,
regime-fragility, and walk-forward-fragility evidence before proposal
generation. It also requires future research to answer how it avoids
`TH-POST-DELEVERAGING-VOL-COMPRESSION-20260507` /
`post_deleveraging_volatility_compression`, which failed local falsification
with `net_edge_bps=-1.412832`, and
`TH-MARK-PREMIUM-OI-CONTINUATION-20260507` /
`mark_premium_open_interest_continuation`, which failed calendar-stability
gates despite `net_edge_bps=31.837718` because only two quarterly windows were
present and `profitable_calendar_windows_ratio=0.5`, and
`TH-VOLUME-CLOCK-LIQUIDITY-MOMENTUM-20260507` /
`volume_clock_liquidity_momentum`, which failed post-cost local falsification,
and `TH-CROSS-ASSET-RELATIVE-STRENGTH-CONTINUATION-20260507` /
`cross_asset_relative_strength_continuation`, which failed post-cost and
calendar-window local falsification. The map contains 30 required research
questions and 12 validated local rejection contexts; answer
them explicitly before any proposal. Do not use older 31-candidate,
pre-vol-compression, pre-mark-premium, sample-gate, or calendar-only
synthesis/map artifacts as current failure memory.

Latest generator/diagnostics implementation note: proposal metadata now records
numeric `parameter_overrides`, and code generation applies known, range-checked
overrides. `local_falsification_hold_candles` maps to
`sell_timeout_candles`, reducing the chance that future generated code silently
uses a timeout horizon that differs from local falsification evidence.
`signal_diagnostics.py` supports `mark_discount_reclaim_continuation`.

Latest pre-proposal screen note: `TH-UTC-SESSION-DOWNSIDE-REVERSAL-20260507`
used the local event builder's `hour_utc`/`weekday` features. The first
one-month screen showed positive post-cost local edge but only 15 events, so
BTC 5m data was extended with historical `download-data --prepend` rather than
loosening thresholds. The same fixed spec on `2024-01-01` to `2025-02-01`
produced `event_count=230` and failed long-history falsification
(`net_edge_bps=-12.272956`, `profitable_windows_ratio=0.25`). Do not promote
it, generate code, or recycle it as a fresh session-reversal idea without a new
theory and prior-falsification evidence. The final reissued artifact used
`--min-data-span-days 180`, recorded `data_span_days=397.222222`, and still
failed cost-edge/window stability.

Latest volatility-managed momentum screen note:
`TH-VOL-MANAGED-MOMENTUM-20260507` also remains rejected. Its original local
screen failed cost/edge; the exact same fixed event spec was then re-run on the
expanded `2024-01-01` to `2025-02-01` BTC 5m data with
`--min-data-span-days 180`. Event generation produced `event_count=6965`, but
local falsification still failed with `sample_count=6964`,
`data_span_days=397.222222`, `expected_edge_bps=0.333213`,
`all_in_cost_bps=12.0`, `net_edge_bps=-11.666787`, and
`profitable_windows_ratio=0.0`. Do not retry this thesis ID or
`volatility_managed_momentum_state` mechanism as a fresh candidate without a
materially new theory and explicit prior-falsification handling.

Latest futures-context screen note:
`TH-FUNDING-MARK-DISLOCATION-RECLAIM-20260507` used the new local event
builder support for funding-rate and mark-price context. The fixed screen
parsed local BTC funding and mark files plus the latest failure synthesis. The
new compound mechanism did not repeat the 26 failed families, but still
produced `event_count=0`; each single condition had matches, while cumulative
matching fell to zero after combining `funding_rate_bps <= -1.0` with
`funding_rate_delta_bps >= 0.25`. Treat this as fixed-theory absence evidence,
not a reason to loosen funding thresholds or rerun a parameter search.

Do not drift into only generating one fixed hand-written indicator template or
AI-assisted parameter tweaking. Preserve the theory-first trail: structured
research references, hypothesis identity, falsification criteria, generated
code, local evaluation artifacts, ranking reasons, and iteration inputs.

First command, required:

```powershell
git status --short --untracked-files=all
```

Read these files before changing Bot Factory code:

- `AGENTS.md` if present, plus the repository instructions already in context
- `docs/BOT_FACTORY_MVP_TODO.md`
- `docs/BOT_FACTORY_STRATEGY_GENERATION_NEXT_AGENT_PROMPT.md`
- `docs/BOT_FACTORY_PHASE2_RUNBOOK.md`
- `docs/BOT_FACTORY_PHASE2_AGENT_INSTRUCTIONS.md`
- `docs/BOT_FACTORY_PHASE3_NEXT_AGENT_PROMPT.md`
- `registry/strategies/proposals/TEMPLATE.md`
- latest accepted proposal metadata under `registry/strategies/proposals/`
- latest generated strategy metadata under `registry/strategies/generated/`
- latest ranking/synthesis artifacts listed below
- latest research decision artifacts listed below
- latest causal failure map artifacts listed below
- `docs/BOT_FACTORY_GOAL_AUDIT.md`

Current factory capabilities:

- Strategy Proposal Generator is implemented and requires structured
  theory/literature references:
  - `freqtrade_ext/bot_factory/strategy_proposals.py`
  - `scripts/bot_factory_generate_strategy_proposal.py`
  - `registry/strategies/proposals/TEMPLATE.md`
  - metadata must include `research_references` with `reference_id`, `title`,
    `source`, `published_at`, relevance rationale, and
    `motivated_thesis_ids`
  - structural-data proposals now require a supplied research decision whose
    local data quality report paths still exist and whose structural-data
    quality checks passed; `--local-data-quality-json` records the same quality
    report as proposal evidence
- Strategy Code Generator is implemented beyond the original v1 baseline:
  - `freqtrade_ext/bot_factory/strategy_code.py`
  - `scripts/bot_factory_generate_strategy_code.py`
  - supports `rule_based`, `freqai`, and `hybrid_ml`
  - structural-data proposals are blocked at codegen unless their proposal
    metadata carries a passing structural quality handoff and a supported
    structural-data code-generation variant exists
  - supports hypothesis-family variants:
    `amihud_illiquidity_premium`, `calendar_turnover_seasonality`,
    `crowding_unwind_reaccumulation`,
    `cross_asset_cointegration_spread`,
    `cross_asset_correlation_recovery`, `cross_asset_lead_lag`,
    `trend_continuation`,
    `mean_reversion_pullback`, `downside_liquidity_shock_reversal`,
    `entropy_regime_transition`, `fractal_long_memory_regime`,
    `funding_pressure_carry`,
    `directional_change_overshoot`,
    `bipower_jump_decay`,
    `intraday_session_liquidity_reclaim`,
    `liquidity_recovery_horizon`,
    `market_beta_drawdown_carry`,
    `mark_price_dislocation_reclaim`,
    `microstructure_spread_reversion`,
    `regime_state_reentry`,
    `realized_skewness_tail_shape`,
    `semivariance_asymmetry_regime`,
    `signed_volume_imbalance_accumulation`,
    `variance_ratio_regime_switch`, and
    `volatility_breakout`
  - writes per-candidate `research_brief.json` and blocks generation if
    structured references are missing, stale, irrelevant, or not mapped to the
    current `thesis_id`
- Candidate Evaluation Pipeline foundation is implemented:
  - `freqtrade_ext/bot_factory/candidate_evaluation.py`
  - `scripts/bot_factory_evaluate_candidate.py`
  - consumes local artifacts and writes candidate manifests, records, reports,
    metrics summaries, artifact paths, research brief checks, and append-only
    registry index rows
  - can record optional funding-rate quality evidence through
    `--funding-rate-quality-json` and the local
    `scripts/bot_factory_check_funding_rate.py` validator
  - `--execute-historical-chain` runs only checked historical-safe Bot Factory
    wrappers and records command logs
- Candidate Ranking / Registry foundation is implemented:
  - `freqtrade_ext/bot_factory/candidate_ranking.py`
  - `scripts/bot_factory_rank_candidates.py`
  - gates paper-ready eligibility on local historical, walk-forward, training
    artifact chains plus hypothesis-family diversity
- Candidate Iteration Loop foundation is implemented:
  - `freqtrade_ext/bot_factory/candidate_iteration.py`
  - `scripts/bot_factory_iterate_candidate.py`
  - writes bounded proposal revision inputs while blocking unsafe relaxation,
    parameter-only retries, and already-passing candidates
- Signal diagnostics are implemented:
  - `freqtrade_ext/bot_factory/signal_diagnostics.py`
  - `scripts/bot_factory_diagnose_candidate_signals.py`
  - reads generated metadata plus local historical OHLCV, optional local
    FreqAI predictions, optional local informative OHLCV, and optional local
    funding-rate parquet/csv data; it
    does not start backtesting, paper/dry-run/live trading, exchange order, or
    process-control commands
- Candidate failure synthesis is implemented:
  - `freqtrade_ext/bot_factory/candidate_failure_synthesis.py`
  - `scripts/bot_factory_synthesize_candidate_failures.py`
  - converts failed ranking evidence plus diagnostics into a local next
    theory/code-generation brief without generating code or running backtests
- Research selection gate is implemented:
  - `freqtrade_ext/bot_factory/research_selection.py`
  - `scripts/bot_factory_select_research_thesis.py`
  - consumes the latest failure synthesis plus proposed thesis inputs and
    emits local `research_decision.json` / `research_decision_report.md`
  - accepts `--causal-failure-map-json` and repeated
    `--causal-failure-response "CATEGORY=RATIONALE"` arguments; when a map is
    supplied, responses must cover the top three dominant failure categories
    plus any material category covering at least 70% of failed candidates
  - rejects thin causal responses, category responses missing expected
    evidence terms, and parameter/threshold-only tuning claims before proposal
    generation
  - computes `research_selection_score` with components for novelty,
    structured research references, local historical falsification, causal
    failure response quality, and mechanism/falsification substance; the
    latest causal map requires at least `minimum_research_selection_score=80`
    before proposal generation can pass
  - latest causal maps also require explicit
    `--research-question-response` answers for each required research question,
    so map questions are not merely displayed and ignored
  - accepts `--local-falsification-json`; when
    `cost_sensitive_mechanism` is high-risk (`risk_score >= 80`), text-only
    cost claims are not enough and the decision must include local JSON
    evidence with thesis-matched `expected_edge_bps`, `all_in_cost_bps`,
    positive net edge, and sufficient sample count. The JSON must come from
    the Bot Factory local falsification generator
    (`factory=research_local_falsification`) and preserve historical-only
    safety scope. It must also carry a valid `event_source` summary linked to a
    completed Bot Factory local event builder artifact whose failure-synthesis
    guard consumed the current synthesis and did not repeat a failed thesis or
    family, so an arbitrary `events.csv` or unguarded local event build cannot
    satisfy the gate by itself.
  - accepts `--prior-local-falsification-json`; failed or rejected
    `local_falsification.json` artifacts that match the current `thesis_id` or
    `mechanism_class` block repeated research selection before proposal/codegen
  - proposal generation rechecks high-risk local falsification continuity from
    supplied `research_decision.json` artifacts; high-risk
    `cost_sensitive_mechanism` decisions with weighted causal responses but no
    passing local falsification handoff are blocked before proposal acceptance
  - codegen consumes the proposal metadata handoff and blocks stale accepted
    proposals when `local_falsification_handoff_passed` is missing or false
  - also scans core thesis fields (`thesis_statement`, `mechanism_summary`,
    `novelty_rationale`, `edge_rationale`, `falsification_plan`, and
    `stop_conditions`) and blocks parameter-only thesis drift through
    `research_thesis_not_parameter_only`; decision artifacts expose
    `research_quality.parameter_only_field_names`
  - exposes `proposal_generation_allowed`, `code_generation_allowed`,
    blockers, deferrals, failed-family novelty matches, failed thesis ID
    matches, causal-map source matching, missing causal response categories,
    causal response quality findings, local falsification data checks, and
    historical-only safety scope
  - blocks repeated failed families, repeated failed thesis IDs, stale
    research-reference mappings, missing local falsification data, mismatched
    causal maps, missing dominant-category responses, unsafe dependencies, and
    proposal/code generation by default when the gate fails
- Local falsification evidence generation is implemented:
  - `freqtrade_ext/bot_factory/local_falsification.py`
  - `scripts/bot_factory_build_local_falsification.py`
  - consumes local closed-candle OHLCV plus a local event timestamp file and
    writes `local_falsification.json` / `local_falsification_report.md`
  - accepts `--event-source-json` to verify the originating `local_events.json`
    and confirm the thesis ID, emitted `events.csv`, source OHLCV path, status,
    factory, and historical-only safety scope
  - computes gross event forward return bps, all-in cost bps, net edge bps,
    sample count, win rate, and split-window profitability evidence for
    `--local-falsification-json`
  - records `ohlcv_row_count`, `data_start`, `data_end`, `data_span_days`, and
    `min_data_span_days`; CLI option `--min-data-span-days` blocks
    short-history false-positive evidence before it can support a research
    decision
  - writes local artifacts only; it does not generate strategy code, run
    backtests, start paper/dry-run/live trading, call exchange order endpoints,
    use shorting, use leverage above `1.0`, or manage processes
- Local event timestamp generation is implemented:
  - `freqtrade_ext/bot_factory/local_events.py`
  - `scripts/bot_factory_build_local_events.py`
  - consumes local closed-candle OHLCV plus optional local funding-rate,
    mark-price, informative-OHLCV, open-interest, and long/short-ratio files,
    together with a JSON `factory=research_local_event_spec` with explicit AND
    conditions over supported past/current closed-candle features: `hour_utc`,
    `weekday`, `return_bps`, `range_pct`, `relative_return_bps`,
    `sma_distance_bps`, `volume_zscore`, `informative_return_bps`,
    `informative_range_pct`, `informative_sma_distance_bps`,
    `informative_volume_zscore`, `funding_rate_bps`,
    `funding_rate_delta_bps`, `mark_price_gap_bps`,
    `mark_price_gap_delta_bps`, `mark_price_return_bps`, `open_interest`,
    `open_interest_delta_pct`, `open_interest_zscore`, `long_account_ratio`,
    `long_account_ratio_delta_bps`, `long_short_ratio`, and
    `long_short_ratio_zscore`
  - writes local `events.csv`, `local_events.json`, and
    `local_events_report.md` artifacts that can feed
    `bot_factory_build_local_falsification.py`; `local_events.json` records the
    emitted `events.csv` path used by `--event-source-json`
  - joins auxiliary context by backward as-of only and records
    `auxiliary_sources`, `condition_diagnostics`, and
    `cumulative_condition_match_counts` for blocked fixed-theory screens
  - accepts `--failure-synthesis-json` to block event specs whose `thesis_id`
    or `mechanism_class` repeats a failed thesis/family before proposal work
    starts; `--allow-failed-thesis-or-family` is only for explicit diagnostics
  - performs no parameter search or optimization; unsupported future-return
    or unrecognized features are blocked
  - writes local artifacts only; it does not generate strategy code, run
    backtests, start paper/dry-run/live trading, call exchange order endpoints,
    use shorting, use leverage above `1.0`, or manage processes
- Causal failure mapping is implemented:
  - `freqtrade_ext/bot_factory/candidate_failure_map.py`
  - `scripts/bot_factory_build_causal_failure_map.py`
  - consumes local `candidate_failure_synthesis.json` artifacts and emits
    `causal_failure_map.json` / `causal_failure_map_report.md`
  - classifies failures by signal sparsity, entry-with-negative-edge,
    walk-forward fragility, no profitable windows, cost sensitivity,
    regime fragility, overfit/window dependency, ML/rule alignment failure,
    training/artifact gap, and thesis rejection after entries
  - emits required research questions and blocked next actions for theory
    selection before proposal generation
  - latest research selection must answer at least
    `regime_fragile_mechanism`, `walk_forward_fragility`, and
    `cost_sensitive_mechanism`, plus material categories above the current
    70% prevalence threshold, currently `no_profitable_walk_forward_windows`
    and `entry_exists_negative_edge`
- Proposal-stage failure-synthesis novelty gating is implemented:
  - `scripts/bot_factory_generate_strategy_proposal.py` accepts
    `--failure-synthesis-json`
  - `scripts/bot_factory_generate_strategy_proposal.py` also accepts
    `--research-decision-json`
  - `freqtrade_ext/bot_factory/strategy_proposals.py` records
    `failure_synthesis_constraints` and `research_decision_constraints`
  - proposal-stage research decision checks require causal-map usage,
    complete causal responses, passing causal response quality fields, and
    current dominant/material category coverage when failure synthesis requires
    a research decision
  - proposal-stage research decision checks also require the research
    decision's causal map `source_synthesis_id` to match one of the supplied
    failure synthesis artifacts, blocking stale approvals from an older
    failure map
  - when synthesis requires a new thesis, repeated failed `thesis_id`s,
    repeated failed hypothesis-family tokens, parameter-only retries, and
    insufficient structured research references block proposal acceptance
  - when synthesis requires a new thesis/research references, missing,
    blocked, stale, thesis-mismatched, unsafe, or direct-codegen-permitting
    research decisions also block proposal acceptance
  - old research decisions without causal-map usage, with a stale causal-map
    source synthesis, or with missing, weak, evidence-gap, or parameter-only
    causal response fields also block proposal acceptance
  - research decisions now also need a passing `research_selection_score`;
    proposal generation blocks below-minimum or missing score evidence through
    `research_decision_<n>_research_selection_score_passed`
  - proposal generation also blocks research decisions that omit or weakly
    answer required causal-map research questions through
    `research_decision_<n>_research_question_responses_complete`
- Hybrid FreqAI infrastructure has been hardened:
  - generated `freqai`/`hybrid_ml` strategies call
    `self.freqai.start(dataframe, metadata, self)`
  - candidate-scoped `freqai_identifier` generation and non-secret override
    configs are implemented
  - prediction diagnostics can detect target/identifier mismatches and confirm
    `FREQAI_IDENTIFIER_MATCH`

Current failed-candidate evidence:

- `LongOnlyCryptoVolumeTrendCandidate` /
  `20260506T090000JST_volume_trend_smoke`
  - thesis family `trend_continuation`
  - real historical-safe wrapper chain completed, but recommendation is
    `retry`
  - signal diagnostics: `entry_count=45`
- `LongOnlyCryptoVolatilityBreakoutCandidate` /
  `20260506T092000JST_vol_breakout_smoke`
  - thesis family `volatility_breakout`
  - historical `trade_count=21`, `total_return_pct=-0.080348392`,
    `profit_factor=0.8432949141917558`
  - walk-forward `pass_rate=0.0`
  - signal diagnostics: `entry_count=375`
- `LongOnlyHybridMLReturnFilterCandidate` /
  `20260506T200000JST_hybrid_ml_freqai_timerange_smoke`
  - thesis family `hybrid_ml_return_filter`
  - regenerated with candidate-scoped FreqAI identifier and
    `self.freqai.start(...)`
  - prediction diagnostics: `&-future_return` present,
    `above_threshold_count=777`, `FREQAI_IDENTIFIER_MATCH`
  - signal diagnostics with merged predictions: `entry_count=0`,
    `first_zero_component=ml_filter`; ML rows do not overlap rows surviving
    the preceding rule gates
  - historical and walk-forward artifacts remain zero-trade / failed
- `LongOnlyCryptoLiquidityPullbackCandidate` /
  `20260506T101500JST_liquidity_pullback_smoke`
  - thesis family `liquidity_mean_reversion`
  - historical `trade_count=0`, `total_return_pct=0.0`
  - walk-forward `pass_rate=0.0`
  - signal diagnostics: `entry_count=0`,
    `first_zero_component=trend_filter`; the generated EMA trend filter
    eliminates rows that survived pullback and RSI recovery
- `LongOnlyDownsideLiquidityShockCandidate` /
  `20260506T204500JST_downside_liquidity_shock_smoke`
  - thesis family `downside_liquidity_shock_reversal`
  - historical `trade_count=32`,
    `total_return_pct=-0.12753034399999996`,
    `profit_factor=0.8227529290082864`
  - walk-forward `pass_rate=0.0`,
    `profitable_windows_ratio=0.5`,
    `max_single_window_profit_dependency=0.9473567258931345`
  - signal diagnostics: `entry_count=36`, no zero component
- `LongOnlyIntradaySessionLiquidityCandidate` /
  `20260506T215000JST_session_liquidity_smoke`
  - thesis `TH-CRYPTO-INTRADAY-SESSION-LIQ-001`
  - thesis family `intraday_session_liquidity`
  - strategy logic variant `intraday_session_liquidity_reclaim`
  - references:
    `doi:10.1016/j.iref.2024.103658`,
    `doi:10.1016/j.ribaf.2022.101625`, and
    `doi:10.1016/j.frl.2019.04.023`
  - signal diagnostics: `entry_count=25`, `zero_entry_signal=false`,
    rarest component `vwap_reclaim`
  - historical `trade_count=21`,
    `total_return_pct=-0.42297835199999995`,
    `profit_factor=0.4223009033119537`,
    `max_drawdown_pct=0.47976619199999965`,
    `sortino=-6.312875004031087`
  - walk-forward completed 4/4 windows but failed with `pass_rate=0.0`,
    `profitable_windows_ratio=0.0`, and
    `total_return_pct=-0.422978352`
- `LongOnlySignedVolumeImbalanceCandidate` /
  `20260506T223500JST_signed_imbalance_smoke`
  - thesis `TH-CRYPTO-SIGNED-VOLUME-IMBALANCE-001`
  - thesis family `signed_volume_imbalance`
  - strategy logic variant `signed_volume_imbalance_accumulation`
  - references:
    `doi:10.1016/j.ribaf.2025.103163`,
    `doi:10.1177/21582440211014504`, and
    `doi:10.1016/j.frl.2019.101326`
  - signal diagnostics: `entry_count=2`, `zero_entry_signal=false`,
    `diagnosis_codes=["LOW_ENTRY_SIGNALS"]`, rarest component `mid_reclaim`
  - historical `trade_count=1`, `total_return_pct=-0.010793492`,
    `profit_factor=0.0`, `sortino=-100.0`
  - walk-forward completed 4/4 windows but failed with `pass_rate=0.0`,
    `profitable_windows_ratio=0.0`, and
    `total_return_pct=-0.010793492`
- `LongOnlyEntropyRegimeCandidate` /
  `20260506T232500JST_entropy_regime_smoke`
  - thesis `TH-CRYPTO-ENTROPY-REGIME-001`
  - thesis family `entropy_regime`
  - strategy logic variant `entropy_regime_transition`
  - references:
    `doi:10.1038/s41598-018-37773-3` and
    `doi:10.1103/PhysRevLett.88.174102`
  - novelty-gated proposal was accepted against the seven-family synthesis
  - historical `trade_count=2`, `total_return_pct=0.047607614`,
    `profit_factor=3.9301317203609494`, `max_drawdown_pct=0.016247601999998553`,
    and `sortino=-100.0`; gates failed on `min_trades` and `min_sortino`
  - signal diagnostics: `entry_count=9`, `zero_entry_signal=false`; the main
    constraints were `low_directional_entropy` and `volume_filter`
  - rule-based walk-forward completed 4/4 windows but failed with
    `pass_rate=0.0`, `profitable_windows_ratio=0.25`,
    `total_return_pct=0.047607614000000006`, and
    `max_single_window_profit_dependency=1.0`
- `LongOnlyFractalMemoryCandidate` /
  `20260506T235900JST_fractal_memory_smoke`
  - thesis `TH-CRYPTO-FRACTAL-MEMORY-001`
  - thesis family `fractal_long_memory`
  - strategy logic variant `fractal_long_memory_regime`
  - references:
    `doi:10.1016/j.econlet.2017.09.013` and
    `doi:10.1016/j.physa.2017.11.025`
  - novelty-gated proposal was accepted against the eight-family synthesis
  - historical `trade_count=66`, `total_return_pct=-0.71052109`,
    `profit_factor=0.36863196318993147`,
    `max_drawdown_pct=0.8488930947586758`, and
    `sortino=-26.229503002219857`; gates failed on `min_trades`,
    `min_profit_factor`, and `min_sortino`
  - signal diagnostics: `entry_count=290`, `zero_entry_signal=false`;
    strongest bottleneck was `volume_filter`
  - rule-based walk-forward completed 4/4 windows but failed with
    `pass_rate=0.0`, `profitable_windows_ratio=0.0`,
    `total_return_pct=-0.7105210900000001`, and all four windows negative
- `LongOnlySemivarianceAsymmetryCandidate` /
  `20260506T211500JST_semivariance_asymmetry_smoke`
  - thesis `TH-CRYPTO-SEMIVARIANCE-ASYMMETRY-001`
  - thesis family `semivariance_asymmetry`
  - strategy logic variant `semivariance_asymmetry_regime`
  - references:
    `doi:10.1093/acprof:oso/9780199549498.003.0007` and
    `doi:10.1162/REST_a_00503`
  - novelty-gated proposal was accepted against the nine-family synthesis
  - historical `trade_count=159`, `total_return_pct=-1.405133739`,
    `profit_factor=0.3856342313738779`,
    `max_drawdown_pct=1.4469039549831204`, and
    `sortino=-60.80804107565447`; gates failed on `min_trades`,
    `min_profit_factor`, and `min_sortino`
  - signal diagnostics: `entry_count=361`, `zero_entry_signal=false`;
    strongest bottleneck was `volume_filter`
  - rule-based walk-forward completed 4/4 windows but failed with
    `pass_rate=0.0`, `profitable_windows_ratio=0.0`,
    `total_return_pct=-1.405133739`, and all four windows negative
- `LongOnlyFundingPressureCarryCandidate` /
  `20260506T214600JST_funding_pressure_smoke`
  - thesis `TH-CRYPTO-FUNDING-PRESSURE-CARRY-001`
  - thesis family `funding_pressure_carry`
  - strategy logic variant `funding_pressure_carry`
  - references:
    `doi:10.3386/w32936`, `doi:10.1016/j.bcra.2025.100354`, and
    `doi:10.2139/ssrn.5576424`
  - novelty-gated proposal was accepted against the ten-family synthesis;
    generated code and local funding-rate quality validation completed
  - historical `trade_count=0`, `total_return_pct=0.0`,
    `profit_factor=0.0`, `max_drawdown_pct=0.0`, and `sortino=0.0`; gates
    failed on `min_trades`, `min_profit_factor`, and `min_sortino`
  - signal diagnostics with merged local 8h funding rates:
    `entry_count=0`, `zero_entry_signal=true`,
    `first_zero_component=not_positive_crowding`, and
    funding merge `matched_row_count=8737`; strongest bottlenecks were
    `funding_pressure_releasing` and `not_positive_crowding`
  - rule-based walk-forward completed 4/4 windows but failed with
    `pass_rate=0.0`, `profitable_windows_ratio=0.0`,
    `total_return_pct=0.0`, and all four windows produced zero trades
- `LongOnlyRealizedSkewnessTailCandidate` /
  `20260506T225000JST_realized_skewness_tail_smoke`
  - thesis `TH-CRYPTO-REALIZED-SKEWNESS-TAIL-001`
  - thesis family `realized_skewness_tail`
  - strategy logic variant `realized_skewness_tail_shape`
  - references:
    `doi:10.1016/j.frl.2020.101536` and
    `doi:10.1016/j.jfineco.2015.02.009`
  - novelty-gated proposal was accepted against the eleven-family synthesis
    and generated code passed static checks
  - historical `trade_count=34`, `total_return_pct=-0.33654470599999997`,
    `profit_factor=0.29897417029856044`,
    `max_drawdown_pct=0.40137508000000255`, and
    `sortino=-13.922726698874877`; gates failed on `min_trades`,
    `min_profit_factor`, and `min_sortino`
  - signal diagnostics: `entry_count=76`, `zero_entry_signal=false`,
    no diagnosis codes; strongest bottleneck was `volume_filter`
  - rule-based walk-forward completed 4/4 windows but failed with
    `pass_rate=0.0`, `profitable_windows_ratio=0.0`,
    `total_return_pct=-0.33654470599999997`, and all four windows negative
- `LongOnlyCalendarTurnoverCandidate` /
  `20260506T231600JST_calendar_turnover_smoke`
  - thesis `TH-CRYPTO-CALENDAR-TURNOVER-001`
  - thesis family `calendar_turnover`
  - strategy logic variant `calendar_turnover_seasonality`
  - references:
    `doi:10.1016/j.frl.2019.04.023`,
    `doi:10.1016/j.frl.2018.12.004`, and
    `doi:10.24136/oc.2022.022`
  - novelty-gated proposal was accepted against the twelve-family synthesis
    and generated code passed static checks
  - historical `trade_count=28`, `total_return_pct=-0.437619035`,
    `profit_factor=0.1385374891419549`,
    `max_drawdown_pct=0.4388136099387865`, and
    `sortino=-15.799177819812716`; gates failed on `min_trades`,
    `min_profit_factor`, and `min_sortino`
  - signal diagnostics: `entry_count=52`, `zero_entry_signal=false`,
    no diagnosis codes; strongest bottleneck was `turnover_recovery`
  - rule-based walk-forward completed 4/4 windows but failed with
    `pass_rate=0.0`, `profitable_windows_ratio=0.0`,
    `total_return_pct=-0.598601605`, and all four windows negative
- `LongOnlyAmihudIlliquidityCandidate` /
  `20260506T234000JST_amihud_illiquidity_smoke`
  - thesis `TH-CRYPTO-AMIHUD-ILLIQUIDITY-001`
  - thesis family `amihud_illiquidity`
  - strategy logic variant `amihud_illiquidity_premium`
  - references:
    `doi:10.1016/S1386-4181(01)00024-6`,
    `doi:10.1177/03128962211069615`, and
    `doi:10.1016/j.econlet.2018.04.003`
  - novelty-gated proposal was accepted against the thirteen-family synthesis
    and generated code passed static checks
  - historical `trade_count=19`, `total_return_pct=-0.21688571`,
    `profit_factor=0.12808642465419795`,
    `max_drawdown_pct=0.22796556403656532`, and
    `sortino=-18.27490029434258`; gates failed on `min_trades`,
    `min_profit_factor`, and `min_sortino`
  - signal diagnostics: `entry_count=32`, `zero_entry_signal=false`,
    no diagnosis codes; strongest bottleneck was `price_impact_premium`
  - rule-based walk-forward completed 4/4 windows but failed with
    `pass_rate=0.0`, `profitable_windows_ratio=0.0`,
    `total_return_pct=-0.21688571`, and all four windows negative
- `LongOnlyCrossAssetLeadLagCandidate` /
  `20260506T235945JST_cross_asset_lead_lag_smoke`
  - thesis `TH-CROSS-ASSET-LEAD-LAG-001`
  - thesis family `btc_eth_lead_lag`
  - strategy logic variant `cross_asset_lead_lag`
  - references:
    `doi:10.1016/j.ribaf.2019.06.012` and
    `doi:10.1016/j.najef.2022.101733`
  - novelty-gated proposal was accepted against the fourteen-family synthesis
    and generated code passed static checks
  - ETH informative OHLCV was downloaded and validated for 2025-01-01 through
    2025-02-01; latest ETH quality report has `ok=true`, `rows=8996`
  - historical `trade_count=132`, `total_return_pct=-1.431901468`,
    `profit_factor=0.19664075195129493`,
    `max_drawdown_pct=1.4330841132835799`, and
    `sortino=-71.98442217231677`; gates failed on `min_trades`,
    `min_profit_factor`, and `min_sortino`
  - signal diagnostics with ETH informative OHLCV:
    `entry_count=336`, `zero_entry_signal=false`,
    `informative_ohlcv_merge.matched_row_count=8928`; strongest bottleneck
    was `volume_filter`
  - rule-based walk-forward completed 4/4 windows but failed with
    `pass_rate=0.0`, `profitable_windows_ratio=0.0`,
    `total_return_pct=-1.4319014680000002`, and all four windows negative
- `LongOnlyVarianceRatioRegimeCandidate` /
  `20260506T235951JST_variance_ratio_regime_smoke`
  - thesis `THESIS-VARIANCE-RATIO-REGIME-20260506T145951Z`
  - thesis family `variance_ratio_regime`
  - strategy logic variant `variance_ratio_regime_switch`
  - references:
    `doi:10.1093/rfs/1.1.41`,
    `doi:10.1016/j.econlet.2016.09.019`, and
    `doi:10.1016/j.econlet.2016.10.033`
  - novelty-gated proposal was accepted against the fifteen-family synthesis
    and generated code passed static checks
  - historical `trade_count=25`, `total_return_pct=-0.324340064`,
    `profit_factor=0.372702266238621`,
    `max_drawdown_pct=0.4479938240000024`, and
    `sortino=-13.886449707992519`; gates failed on `min_trades`,
    `min_profit_factor`, and `min_sortino`
  - signal diagnostics: `entry_count=52`, `zero_entry_signal=false`, no
    diagnosis codes; strongest bottleneck was `volume_filter`
  - rule-based walk-forward completed 4/4 windows but failed with
    `pass_rate=0.0`, `profitable_windows_ratio=0.25`,
    `total_return_pct=-0.32434006400000004`; only the final window was
    positive
- `LongOnlyCrossAssetCointegrationCandidate` /
  `20260506T235957JST_cross_asset_cointegration_smoke`
  - thesis `THESIS-CROSS-ASSET-COINTEGRATION-20260506T145957Z`
  - thesis family `btc_eth_cointegration`
  - strategy logic variant `cross_asset_cointegration_spread`
  - references:
    `doi:10.1108/SEF-08-2018-0264`,
    `doi:10.1007/s10203-021-00318-x`, and
    `doi:10.1186/s40854-024-00702-7`
  - novelty-gated proposal was accepted against the sixteen-family synthesis
    and generated code passed static checks
  - BTC and ETH 5m OHLCV quality checks returned `ok=true`; ETH informative
    merge matched `8928` rows in signal diagnostics
  - historical `trade_count=103`, `total_return_pct=-1.365657864`,
    `profit_factor=0.24388391371201643`,
    `max_drawdown_pct=1.3791347563258318`, and
    `sortino=-31.224341592260014`; gates failed on `min_trades`,
    `min_profit_factor`, and `min_sortino`
  - signal diagnostics: `entry_count=303`, `zero_entry_signal=false`,
    no diagnosis codes; rarest component `btc_discount_to_eth`; strongest
    bottleneck `volume_filter`
  - rule-based walk-forward completed 4/4 windows but failed with
    `pass_rate=0.0`, `profitable_windows_ratio=0.0`,
    `total_return_pct=-1.365657864`, and all four windows negative
- `LongOnlyCrossAssetCorrelationCandidate` /
  `20260506T235959JST_cross_asset_correlation_smoke`
  - thesis `THESIS-CROSS-ASSET-CORRELATION-20260506T145959Z`
  - thesis family `btc_eth_correlation_recovery`
  - strategy logic variant `cross_asset_correlation_recovery`
  - references:
    `doi:10.1016/j.qref.2021.04.002`,
    `doi:10.1016/j.irfa.2018.12.002`, and
    `doi:10.1016/j.jbef.2021.100562`
  - novelty-gated proposal was accepted against the seventeen-family
    synthesis and generated code passed static checks
  - historical `trade_count=0`, `total_return_pct=0.0`,
    `profit_factor=0.0`, `max_drawdown_pct=0.0`, and `sortino=0.0`; gates
    failed on `min_trades`, `min_profit_factor`, and `min_sortino`
  - signal diagnostics with ETH informative OHLCV:
    `entry_count=0`, `zero_entry_signal=true`,
    `first_zero_component=eth_market_support`,
    `diagnosis_codes=["ZERO_ENTRY_SIGNALS"]`,
    rarest component `correlation_breakdown`
  - rule-based walk-forward completed 4/4 windows but failed with
    `pass_rate=0.0`, `profitable_windows_ratio=0.0`,
    `total_return_pct=0.0`, and all four windows zero-trade
- `LongOnlyMarketBetaDrawdownCarryCandidate` /
  `20260507T001500JST_market_beta_drawdown_carry_smoke`
  - thesis `THESIS-MARKET-BETA-DRAWDOWN-CARRY-20260506T151500Z`
  - thesis family `market_beta_drawdown_carry`
  - strategy logic variant `market_beta_drawdown_carry`
  - references:
    `doi:10.1093/rfs/hhaa113`,
    `doi:10.1111/jofi.13119`, and
    `doi:10.1111/jofi.12513`
  - novelty-gated proposal was accepted against the eighteen-family synthesis
    and generated code passed static checks
  - historical `trade_count=32`,
    `total_return_pct=0.36291935900000016`,
    `profit_factor=1.1484070966527722`,
    `max_drawdown_pct=1.0858858426983407`, and
    `sortino=1.2844078968685824`; this is the first positive full-month real
    smoke, but gates still failed on trade count and profit factor
  - signal diagnostics: `entry_count=450`, `zero_entry_signal=false`,
    rarest component `participation_floor`, strongest bottleneck
    `participation_floor`
  - rule-based walk-forward completed 4/4 windows but failed with
    `pass_rate=0.0`, `profitable_windows_ratio=0.5`,
    `total_return_pct=0.30646605099999985`, and
    `max_single_window_profit_dependency=0.925001394501805`
- `LongOnlyRegimeStateReentryCandidate` /
  `20260507T003500JST_regime_state_reentry_smoke`
  - thesis `THESIS-REGIME-STATE-REENTRY-20260506T153500Z`
  - thesis family `regime_state_reentry`
  - strategy logic variant `regime_state_reentry`
  - references:
    `doi:10.2307/1912559`,
    `doi:10.1080/07350015.2000.10524851`, and
    `doi:10.1016/j.frl.2022.103193`
  - novelty-gated proposal was accepted against the nineteen-family synthesis
    and generated code passed static checks
  - historical `trade_count=180`,
    `total_return_pct=-2.0567343780000003`,
    `profit_factor=0.5041520570492957`,
    `max_drawdown_pct=2.0567343779999985`, and
    `sortino=-80.33297116523825`; gates failed and the strategy is negative
    after costs
  - signal diagnostics: `entry_count=676`, `zero_entry_signal=false`,
    rarest component `positive_regime_drift`, strongest bottleneck
    `participation_floor`
  - rule-based walk-forward completed 4/4 windows but failed with
    `pass_rate=0.0`, `profitable_windows_ratio=0.0`,
    `total_return_pct=-2.0766407420000004`, and all four windows negative
- `LongOnlyMarkPriceDislocationCandidate` /
  `20260507T011500JST_mark_price_dislocation_smoke`
  - thesis `THESIS-MARK-PRICE-DISLOCATION-20260506T161500Z`
  - thesis family `mark_price_dislocation_reclaim`
  - strategy logic variant `mark_price_dislocation_reclaim`
  - references:
    `bybit:mark-price-perpetual-contracts-20260324`,
    `doi:10.1016/j.econlet.2018.10.031`, and
    `doi:10.1016/j.ribaf.2019.101116`
  - novelty-gated proposal was accepted against the twenty-family synthesis
    and generated code passed static checks
  - local 4h mark-price quality check passed with `rows=997`,
    no duplicate timestamps, and no missing intervals
  - historical `trade_count=51`, `total_return_pct=-0.901716537`,
    `profit_factor=0.3544814788243831`,
    `max_drawdown_pct=0.9360849296512312`, and
    `sortino=-19.437942829690122`; exported trades were long-only with
    `is_short=false` and `leverage=1.0`
  - signal diagnostics with BTC 4h mark-price informative parquet:
    `entry_count=95`, `zero_entry_signal=false`,
    `matched_row_count=8881`, rarest/top bottleneck
    `mark_discount_pressure`
  - rule-based walk-forward completed 4/4 windows but failed with
    `pass_rate=0.0`, `profitable_windows_ratio=0.0`,
    `total_return_pct=-0.9017165370000001`; one window was zero-trade and the
    other three windows were negative
- `LongOnlyMicrostructureSpreadCandidate` /
  `20260507T013000JST_microstructure_spread_smoke`
  - thesis `THESIS-MICROSTRUCTURE-SPREAD-20260506T163000Z`
  - thesis family `microstructure_spread_reversion`
  - strategy logic variant `microstructure_spread_reversion`
  - references:
    `doi:10.1111/j.1540-6261.1984.tb03897.x`,
    `doi:10.1111/j.1540-6261.2012.01729.x`, and
    `doi:10.1016/j.jbankfin.2020.106041`
  - novelty-gated proposal was accepted against the twenty-one-family
    synthesis and generated code passed static checks
  - historical `trade_count=162`, `total_return_pct=-2.046597466`,
    `profit_factor=0.160493071193835`,
    `max_drawdown_pct=2.046597466000003`, and
    `sortino=-57.31247274157497`; exported trades were long-only with
    `is_short=false` and `leverage=1.0`
  - signal diagnostics: `entry_count=248`, `zero_entry_signal=false`,
    `diagnosis_codes=[]`, rarest/top bottleneck `spread_compressing`
  - rule-based walk-forward completed 4/4 windows but failed with
    `pass_rate=0.0`, `profitable_windows_ratio=0.0`,
    `total_return_pct=-2.046597466`, and all four windows negative
- `LongOnlyLiquidityRecoveryHorizonCandidate` /
  `liquidity_recovery_horizon_001`
  - thesis `TH-LIQUIDITY-RECOVERY-HORIZON-20260507`
  - thesis family / logic variant `liquidity_recovery_horizon`
  - approved research decision:
    `registry/strategies/research_decisions/20260507T033000JST_liquidity_recovery_horizon_research_gate/research_decision.json`
    with `status=approved_for_proposal_generation`, causal-map responses
    complete, and `research_quality.parameter_only_field_names=[]`
  - accepted proposal:
    `registry/strategies/proposals/20260506T183500Z_LongOnlyLiquidityRecoveryHorizonCandidate.metadata.json`
  - generated code:
    `registry/strategies/generated/LongOnlyLiquidityRecoveryHorizonCandidate/liquidity_recovery_horizon_001/metadata.json`
    with generated static check `ok=true`
  - historical backtest
    `data/backtests/LongOnlyLiquidityRecoveryHorizonCandidate/liquidity_recovery_horizon_001_20250101_20250103/metrics.json`
    failed gates with `trade_count=3`,
    `total_return_pct=-0.047765749999999996`, `profit_factor=0.0`,
    `win_rate=0.0`, and negative expectancy
  - signal diagnostics:
    `entry_count=9`, `zero_entry_signal=false`, rarest component
    `below_recovery_anchor`; this is low-sample negative-edge evidence, not
    zero-signal evidence
  - candidate manifest:
    `registry/strategies/candidates/LongOnlyLiquidityRecoveryHorizonCandidate/liquidity_recovery_horizon_001/candidate_manifest.json`
    has `recommendation=fail`; no walk-forward was run yet
- `LongOnlyBipowerJumpDecayCandidate` /
  `20260507T051000JST_bipower_jump_decay_full_eval`
  - thesis `TH-BIPOWER-JUMP-DECAY-20260507`
  - thesis family `realized_multipower_jump_decay`
  - strategy logic variant `bipower_jump_decay`
  - accepted proposal was generated from the approved 23-family research gate;
    generated code and static checks completed
  - signal diagnostics: `entry_count=511`, `zero_entry_signal=false`,
    `diagnosis_codes=[]`
  - historical `trade_count=159`, `total_return_pct=-2.393940186`,
    `profit_factor=0.2204711202447723`, `max_drawdown_pct=2.447073369289071`,
    and `sortino=-76.51412131065698`; gates failed
  - walk-forward completed 4/4 windows but failed with `pass_rate=0.0`,
    `profitable_windows_ratio=0.0`, and `total_return_pct=-2.368185648`
  - this is clean negative-edge evidence, not a paper-ready strategy
- `LongOnlyDirectionalChangeOvershootCandidate` /
  `20260507T064500JST_directional_change_overshoot_full_eval`
  - thesis `TH-DIRECTIONAL-CHANGE-OVERSHOOT-20260507`
  - thesis family `directional_change_overshoot`
  - strategy logic variant `directional_change_overshoot`
  - code generation and signal diagnostics completed deliberately after the
    earlier no-fallback codegen block; this is no longer awaiting codegen
  - signal diagnostics: `entry_count=706`, `zero_entry_signal=false`,
    `diagnosis_codes=[]`
  - historical `trade_count=253`, `total_return_pct=-3.2605379719999994`,
    `profit_factor=0.3464660896687346`,
    `max_drawdown_pct=3.308262864000006`, and
    `sortino=-93.24140769163247`; gates failed
  - walk-forward completed 4/4 windows but failed with `pass_rate=0.0`,
    `profitable_windows_ratio=0.0`, and `total_return_pct=-3.24521717`
  - candidate evaluation wrote `recommendation=retry`; this is negative-edge
    evidence, not a paper-ready strategy

Superseded directional-change aggregate artifacts:

- Directional-change aggregate ranking:
  `registry/strategies/candidates/rankings/20260507T065000JST_twenty_five_family_with_directional_change_overshoot_ranking/candidate_ranking.json`
  - `candidate_count=25`
  - `best_candidate_id=20260506T200000JST_hybrid_ml_freqai_timerange_smoke`
  - `paper_ready_candidate_ids=[]`
- Directional-change aggregate failure synthesis:
  `registry/strategies/synthesis/20260507T065500JST_twenty_five_family_failure_synthesis_with_directional_change_overshoot/candidate_failure_synthesis.json`
  - `candidate_count=25`
  - `paper_ready_count=0`
  - `zero_trade_count=4`
  - `negative_return_count=18`
  - `walk_forward_failed_count=25`
  - `parameter_only_retry_allowed=false`
  - `requires_new_thesis_id=true`
  - `paper_or_live_promotion_allowed=false`
  - failed families to avoid as default repeats:
    `amihud_illiquidity`, `btc_eth_cointegration`,
    `btc_eth_correlation_recovery`, `btc_eth_lead_lag`,
    `calendar_turnover`,
    `trend_continuation`, `volatility_breakout`,
    `hybrid_ml_return_filter`, `liquidity_mean_reversion`,
    `downside_liquidity_shock_reversal`, `intraday_session_liquidity`,
    `signed_volume_imbalance`, `entropy_regime`, `fractal_long_memory`,
    `semivariance_asymmetry`, `funding_pressure_carry`,
    `realized_skewness_tail`, `variance_ratio_regime`, and
    `market_beta_drawdown_carry`, `regime_state_reentry`, and
    `mark_price_dislocation_reclaim`, `microstructure_spread_reversion`, and
    `liquidity_recovery_horizon`, `realized_multipower_jump_decay`, and
    `directional_change_overshoot`
  - supersedes the twenty-four-family bipower ranking/synthesis
    as the next source of truth
  - blocked next actions include parameter-only threshold loosening,
    repeating failed hypothesis families without new evidence, paper/dry-run
    or live startup, exchange order endpoint use, and promotion from failed
    smoke artifacts
- Directional-change aggregate causal failure map:
  `registry/strategies/failure_maps/20260507T074500JST_twenty_five_family_causal_failure_map_with_question_responses/causal_failure_map.json`
  - report:
    `registry/strategies/failure_maps/20260507T074500JST_twenty_five_family_causal_failure_map_with_question_responses/causal_failure_map_report.md`
  - `status=completed`, `candidate_count=25`, `category_count=10`
  - dominant categories: `regime_fragile_mechanism=25`,
    `walk_forward_fragility=25`, `cost_sensitive_mechanism=24`,
    `no_profitable_walk_forward_windows=19`, and
    `entry_exists_negative_edge=18`
  - research selection rubric now requires
    `minimum_research_selection_score=80` across novelty, references, local
    falsification data, causal failure responses, and mechanism/falsification
    substance
  - `requires_research_question_responses=true`; each required research
    question must be answered explicitly in the research decision
  - research selection guidance requires a research decision before proposal,
    a new `thesis_id`, new references, and a substantive explanation of why
    the next mechanism should beat fee/slippage/turnover costs and survive
    expected walk-forward regimes
- Latest approved research selection decision before the 24-family refresh:
  `registry/strategies/research_decisions/20260507T043500JST_bipower_jump_decay_research_gate/research_decision.json`
  - report:
    `registry/strategies/research_decisions/20260507T043500JST_bipower_jump_decay_research_gate/research_decision_report.md`
  - `status=approved_for_proposal_generation`
  - `proposal_generation_allowed=true`
  - `code_generation_allowed=false`
  - supplied the then-latest twenty-three-family failure synthesis and causal failure map;
    causal-map match and required causal response coverage passed
  - response quality checks passed with no weak, evidence-gap, or
    parameter-only causal response categories
  - `research_quality.parameter_only_field_names=[]`
  - this supported the later `bipower_jump_decay` proposal/codegen/evaluation,
    which has now failed historical and walk-forward gates and must not be
    promoted
  - do not reuse this as approval for another proposal; any next thesis
    needs a new research decision against the latest causal failure map
- Range-quarticity research decision against the then-latest causal failure
  map:
  `registry/strategies/research_decisions/20260507T082500JST_range_quarticity_vol_of_vol_research_gate/research_decision.json`
  - `status=approved_for_proposal_generation`
  - `proposal_generation_allowed=true`
  - `code_generation_allowed=false`
  - `research_selection_score=100.0`, minimum score `80.0`
  - complete required-question responses
  - consumed by accepted proposal
    `registry/strategies/proposals/20260506T233500Z_LongOnlyRangeQuarticityVolOfVolCandidate.metadata.json`
    with `strategy_logic_variant=range_quarticity_vol_of_vol_state`
  - explicit codegen and evaluation are now complete:
    `registry/strategies/generated/LongOnlyRangeQuarticityVolOfVolCandidate/20260507T091000JST_range_quarticity_vol_of_vol_codegen/metadata.json`
    has `status=generated`, `strategy_code_generated=true`,
    `candidate_evaluation_eligible=true`, and `static_check_ok=true`
  - signal diagnostics:
    `registry/strategies/diagnostics/LongOnlyRangeQuarticityVolOfVolCandidate/20260507T091000JST_range_quarticity_vol_of_vol_codegen/20260507T091700JST_range_quarticity_signal_diagnostics/signal_diagnostics.json`
    has `entry_count=250`, `zero_entry_signal=false`
  - historical metrics:
    `data/backtests/LongOnlyRangeQuarticityVolOfVolCandidate/20260507T092000JST_range_quarticity_historical/metrics.json`
    has `trade_count=92`, `total_return_pct=-0.8648079839999998`,
    `profit_factor=0.5266356841671438`, and negative expectancy
  - walk-forward metrics:
    `data/walk_forward/LongOnlyRangeQuarticityVolOfVolCandidate/20260507T092500JST_range_quarticity_rule_walk_forward/walk_forward_metrics.json`
    has `recommendation=fail`, `completed_windows=4`, `pass_rate=0.0`,
    `profitable_windows_ratio=0.0`, `total_return_pct=-0.8648079839999999`
  - do not promote it
- Latest aggregate state after mark-discount reclaim:
  - ranking:
    `registry/strategies/candidates/rankings/20260507T153500JST_all_candidates_with_mark_discount_reclaim_001/candidate_ranking.json`
    has `candidate_count=31` and `paper_ready_candidate_ids=[]`
  - synthesis:
    `registry/strategies/synthesis/20260507T160500JST_all_candidates_with_mark_discount_reclaim_001_edge_aware_failure_synthesis/candidate_failure_synthesis.json`
    has `paper_ready_count=0`, `parameter_only_retry_allowed=false`,
    `requires_new_thesis_id=true`, and generated-entry edge failure
    `net_edge_bps=-12.323858` for `mark_discount_reclaim_001`
  - causal map:
    `registry/strategies/failure_maps/20260507T161500JST_all_candidates_with_mark_discount_reclaim_001_edge_aware_causal_failure_map/causal_failure_map.json`
    has `candidate_count=31`, `category_count=10`, and
    `requires_research_decision_before_proposal=true`
- Latest material-category research gate smoke:
  `registry/strategies/research_decisions/20260507T053000JST_material_category_gate_top3_only_block_smoke/research_decision.json`
  - report:
    `registry/strategies/research_decisions/20260507T053000JST_material_category_gate_top3_only_block_smoke/research_decision_report.md`
  - `status=blocked`
  - `proposal_generation_allowed=false`
  - `code_generation_allowed=false`
  - supplied refreshed 24-family causal failure map;
    `causal_failure_map_matches_failure_synthesis=pass`
    and `causal_failure_responses_cover_required_categories=blocked`
  - required causal response categories included `regime_fragile_mechanism`,
    `walk_forward_fragility`, `cost_sensitive_mechanism`,
    `no_profitable_walk_forward_windows`, and `entry_exists_negative_edge`
    because the latter two exceed the 70% material-category threshold;
    missing response categories were `no_profitable_walk_forward_windows` and
    `entry_exists_negative_edge`
  - blocker is `causal_failure_responses_cover_required_categories`
- Proposal-generator enforcement smoke:
  `registry/strategies/proposals/20260506T171000Z_ResearchDecisionBlockedMicrostructureRepeatCandidate.metadata.json`
  - `status=blocked`
  - `code_generation_eligible=false`
  - supplied both the latest twenty-two-family failure synthesis and the
    blocked research decision
  - records `research_decision_constraints[0].status=blocked`,
    `proposal_generation_allowed=false`, `code_generation_allowed=false`, and
    blocker `research_decision_1_approved_for_proposal_generation`
- Blocked-repeat novelty-gate smoke:
  `registry/strategies/proposals/20260506T141000Z_NoveltyGateBlockedSignedVolumeRepeatCandidate.metadata.json`
  - `status=blocked`
  - `code_generation_eligible=false`
  - blockers include `failure_synthesis_1_requires_new_thesis_id` and
    `failure_synthesis_1_requires_new_hypothesis_family`
  - records `failed_thesis_id_match=true` and
    `repeated_family_matches=["signed_volume_imbalance"]`
- Proposal-stage material-category bypass coverage:
  `tests\test_bot_factory.py::test_strategy_proposal_generator_blocks_research_decision_missing_material_causal_categories`
  verifies that a crafted research decision using the current synthesis/map
  source but claiming only top-three required categories is blocked by
  `research_decision_1_causal_required_categories_match_current_policy`.
  Expected material categories are `no_profitable_walk_forward_windows` and
  `entry_exists_negative_edge`.
- Directional-change approved research selection decision:
  `registry/strategies/research_decisions/20260507T054500JST_directional_change_overshoot_research_gate/research_decision.json`
  - `status=approved_for_proposal_generation`
  - `proposal_generation_allowed=true`
  - `code_generation_allowed=false`
  - supplied refreshed 24-family synthesis and causal failure map
  - required causal response categories included `regime_fragile_mechanism`,
    `walk_forward_fragility`, `cost_sensitive_mechanism`,
    `no_profitable_walk_forward_windows`, and `entry_exists_negative_edge`;
    all were answered with no weak, evidence-gap, or parameter-only findings
  - thesis family `directional_change_overshoot` is outside the 24 failed
    families and was approved for proposal generation only
- Directional-change accepted proposal:
  `registry/strategies/proposals/20260506T205000Z_LongOnlyDirectionalChangeOvershootCandidate.metadata.json`
  - `status=accepted`
  - `strategy_logic_variant=directional_change_overshoot`
  - `code_generation_eligible=true`
  - `research_reference_count=3` in CLI output
- Historical no-fallback codegen block:
  `registry/strategies/generated/LongOnlyDirectionalChangeOvershootCandidate/20260507T055500JST_directional_change_overshoot_codegen_block/metadata.json`
  - `status=blocked`
  - `strategy_code_generated=false`
  - `candidate_evaluation_eligible=false`
  - blocker `strategy_logic_variant_supported`; this confirms the proposal did
    not silently fall back to another family before directional-change codegen
    is intentionally implemented
- Latest generated directional-change code:
  `registry/strategies/generated/LongOnlyDirectionalChangeOvershootCandidate/20260507T061500JST_directional_change_overshoot_codegen/metadata.json`
  - `status=generated`, `strategy_code_generated=true`,
    `candidate_evaluation_eligible=true`, and `static_check_ok=true`
  - signal diagnostics artifact:
    `registry/strategies/diagnostics/LongOnlyDirectionalChangeOvershootCandidate/20260507T061500JST_directional_change_overshoot_codegen/20260507T062000JST_directional_change_overshoot_signal_diagnostics/signal_diagnostics.json`
  - historical backtest artifact:
    `data/backtests/LongOnlyDirectionalChangeOvershootCandidate/20260507T062500JST_directional_change_overshoot_historical/metrics.json`
  - walk-forward artifact:
    `data/walk_forward/LongOnlyDirectionalChangeOvershootCandidate/20260507T064500JST_directional_change_overshoot_walk_forward_rule_wrapper/walk_forward_metrics.json`
  - candidate manifest:
    `registry/strategies/candidates/LongOnlyDirectionalChangeOvershootCandidate/20260507T064500JST_directional_change_overshoot_full_eval/candidate_manifest.json`
  - result: failed historical and walk-forward gates; do not promote

Current worktree context:

- The handoff may include uncommitted Bot Factory changes and many generated
  artifacts under `registry/strategies/`, `data/backtests/`,
  `data/walk_forward/`, `data/freqai/`, and `data/freqai_training/`.
- Latest cleanup classification is recorded in
  `docs/BOT_FACTORY_GOAL_AUDIT.md` and `docs/BOT_FACTORY_MVP_TODO.md`.
  Current visible worktree state is 50 paths: 21 modified tracked files, one
  tracked deletion, and 28 non-ignored untracked review candidates. The
  untracked files are source/doc review candidates, not disposable runtime
  artifacts: one audit doc, one `data/market_structure/.gitkeep`, 11 Bot
  Factory modules, and 15 Bot Factory scripts.
- Ignored inventory is intentionally local evidence and should not be added to
  Git: `ignored_total=25480`, `.venv=21603`, and
  `ignored_non_venv=3877`. Main non-venv ignored groups are
  `data/walk_forward`, `registry/strategies`, `data/backtests`,
  `data/freqai_training`, and `data/freqai`.
- Latest source-diff verification sweep covered tracked and untracked Bot
  Factory modules, Bot Factory CLI scripts, and `tests/test_bot_factory.py`.
  `py_compile` exited `0`; focused pytest for
  `local_event_builder or local_falsification or candidate_failure_synthesis or
  research_selection_template or signal_diagnostics` reached `[100%]`; and
  `git diff --check` passed with only LF-to-CRLF warnings. Caches were cleaned
  afterward (`cache_dir_count=0`).
- Supported-variant coverage check: the current generator exposes 28 supported
  strategy logic variants, while the latest synthesis records 40 tried
  hypothesis families across 32 candidates/local screens and
  `paper_ready_count=0`. Do not assume an unused existing variant remains.
  BTC/ETH lead-lag, cointegration, correlation recovery, and relative-strength
  continuation are all failed or locally rejected. The next thesis needs a
  materially new mechanism and passing pre-proposal local falsification against
  the latest 32-candidate failure memory.
- Direction pivot: do not generate a 33rd default smoke strategy. The TODO now
  requires an `Edge Discovery / Research Lab Pivot` before further default
  proposal/codegen work. The next implementation should create local
  edge-discovery artifacts and an edge-surface runner that measures
  fee-adjusted expectancy, sample count, date span, calendar stability, and
  concentration before any proposal is written. Strategy Proposal Generator and
  Strategy Code Generator remain available for already-approved handoff tests,
  but they should stay paused as the default path until an edge-discovery
  artifact passes the latest synthesis/map gates.
- Edge Discovery runner v1 is now implemented:
  `freqtrade_ext/bot_factory/edge_discovery.py` plus
  `scripts/bot_factory_build_edge_discovery.py`. It writes
  `research_edge_discovery` artifacts under
  `registry/strategies/research_decisions/<id>/`, evaluates fixed
  theory-named conditions across multiple hold horizons, records post-cost
  horizon evidence and quarterly stability, blocks parameter-grid/search specs,
  and never authorizes direct strategy codegen. Verified with py_compile,
  focused pytest `-k "edge_discovery"` (3 passed), adjacent focused pytest
  `-k "edge_discovery or local_event_builder or local_falsification"`
  (23 passed), CLI `--help`, and `git diff --check` with only existing
  LF-to-CRLF warnings. Cache cleanup removed 33 workspace-local pytest/pycache
  directories and follow-up cache count was 0.
  Proposal/codegen hard gate is now implemented:
  `scripts/bot_factory_generate_strategy_proposal.py` requires
  `--edge-discovery-json`; proposal generation records
  `edge_discovery_handoff` and blocks missing/failed/mismatched/unsafe
  edge artifacts; codegen blocks accepted/crafted proposal metadata without a
  passing edge handoff. Verified with py_compile, focused handoff pytest
  (7 passed), proposal/codegen focused regression, full
  `tests/test_bot_factory.py` (`[100%]`), proposal CLI `--help`, and
  `git diff --check` with only existing LF-to-CRLF warnings. Cache cleanup
  removed 33 workspace-local pytest/pycache directories and follow-up cache
  count was 0.
  Remaining implementation gap: failed edge artifacts are not yet ingested into
  failure synthesis/causal maps for future novelty rejection memory.
- Preserve existing user and prior-agent changes. Do not revert unrelated
  docs, generated artifacts, tests, or the pre-existing deletion of
  `docs/BOT_FACTORY_GOAL_COMMAND_RUNBOOK.md`.
- The runbook deletion is not a safe implicit cleanup decision. References to
  `docs/BOT_FACTORY_GOAL_COMMAND_RUNBOOK.md` still exist in the audit, TODO,
  and this handoff prompt. Either migrate/accept those references
  intentionally or restore the file when the owner decides.
- Known Windows ACL warnings may appear in `git status` for local pytest/temp
  directories.

Hard safety boundaries:

- Do not start `freqtrade trade`.
- Do not start paper trading, dry-run trading, canary live, live trading, or
  any bot startup process.
- Do not stop, poll, terminate, clean up, promote, or manage any paper process.
- Do not use API keys, secrets, private environment values, exchange order
  endpoints, real order placement, leverage above `1.0`, or shorting.
- Do not promote a generated candidate to paper from one proposal, one
  generated strategy, one backtest, or any failed smoke chain.
- Keep local JSON, CSV, Markdown, and logs as the source of truth. MLflow is
  optional and must not replace local artifacts.

Handoff priority:

1. Start from the current 32-candidate ranking, synthesis, causal
   failure map, and `docs/BOT_FACTORY_GOAL_AUDIT.md`; do not start from a
   parameter-only retry.
2. Do not generate another distinct smoke candidate by default. Run
   `scripts\bot_factory_select_research_thesis.py` against the refreshed
   32-candidate synthesis and causal failure map for any proposed next thesis
   before proposal generation. Include the latest available
   `--causal-failure-map-json` and one `--causal-failure-response` for each
   required dominant/material category. Also include one
   `--research-question-response` for each required research question from the
  map. Proceed only if it returns `status=approved_for_proposal_generation`
  and `research_selection_score.passes_minimum=true`.
  If the causal map includes `causal_risk_weights`, the research decision must
  carry `research_selection_score_v2` weighted causal score details; the
  proposal generator blocks older or crafted score payloads for risk-weighted
  maps.
  If `cost_sensitive_mechanism` is high-risk (`risk_score >= 80`), the causal
  response must include quantified cost/edge evidence such as bps, basis
  points, `%`, or a numeric fee/slippage/turnover/spread reference, and must
  be backed by `--local-falsification-json` cost/edge evidence linked to a
  Bot Factory `local_events.json` through `--event-source-json`.
3. Any approved next thesis must require a new `thesis_id`, new structured
   research references, and a distinct falsifiable market mechanism outside
   the thirty-two attempted/manifested failed candidates and validated local
   rejection screens, including
   `liquidity_recovery_horizon`, `realized_multipower_jump_decay`, and
   `directional_change_overshoot`, and
   `range_quarticity_vol_of_vol_state`, and
   `mark_discount_reclaim_continuation`, and
   `crowding_unwind_reaccumulation`, and
   `post_deleveraging_volatility_compression`. It must
   also answer why it should not repeat regime fragility, walk-forward
   fragility, cost-sensitive negative edge, no-profitable walk-forward windows,
   or entry-with-negative-edge with substantive, category-specific evidence
   rather than parameter-only tuning claims. Use the latest 32-candidate
   `--failure-synthesis-json` when
   generating proposals, and include a newly approved
   `--research-decision-json`, so this is still enforced at proposal stage.
   Old research decisions without causal-map usage or passing causal response
   quality should now be rejected by the proposal generator.
4. If another real subprocess smoke is attempted, run static and OHLCV checks
   first, use only historical `freqtrade backtesting` wrappers, and record
   failed gates without treating them as completion.
5. Future labels and negative shifts may only appear in `set_freqai_targets`.
   Negative shifts remain forbidden in indicator, entry, and exit generation.
6. Continue to run static strategy scanning before any generated candidate can
   enter evaluation.

Recommended next implementation path:

- Read the current 32-candidate aggregate ranking:
  `registry/strategies/candidates/rankings/20260507T085500Z_all_candidates_with_crowding_unwind_rejection_ranking/candidate_ranking.json`.
- Read the current 32-candidate synthesis:
  `registry/strategies/synthesis/20260507T111500Z_all_candidates_with_cross_asset_relative_strength_rejection/candidate_failure_synthesis.json`.
- Read the current causal failure map:
  `registry/strategies/failure_maps/20260507T112000Z_all_candidates_with_cross_asset_relative_strength_rejection_causal_map/causal_failure_map.json`.
- Read the latest mark-discount reclaim research, proposal,
  generated/evaluated code, and manifest artifacts as the latest failed full
  loop:
  `registry/strategies/research_decisions/20260507T142500JST_mark_discount_reclaim_research_selection/research_decision.json`.
  `registry/strategies/proposals/20260507T055000Z_LongOnlyMarkDiscountReclaimCandidate.metadata.json`.
  `registry/strategies/generated/LongOnlyMarkDiscountReclaimCandidate/mark_discount_reclaim_001/metadata.json`.
  `registry/strategies/candidates/LongOnlyMarkDiscountReclaimCandidate/mark_discount_reclaim_001/candidate_manifest.json`.
- Read the range-quarticity research, proposal, generated/evaluated
  code, and manifest artifacts as historical failed-loop evidence:
  `registry/strategies/research_decisions/20260507T082500JST_range_quarticity_vol_of_vol_research_gate/research_decision.json`.
  `registry/strategies/proposals/20260506T233500Z_LongOnlyRangeQuarticityVolOfVolCandidate.metadata.json`.
  `registry/strategies/generated/LongOnlyRangeQuarticityVolOfVolCandidate/20260507T091000JST_range_quarticity_vol_of_vol_codegen/metadata.json`.
  `registry/strategies/candidates/LongOnlyRangeQuarticityVolOfVolCandidate/range_quarticity_vol_of_vol_001/candidate_manifest.json`.
- Read the directional-change research, proposal, generated/evaluated code,
  and manifest artifacts as historical failed-loop evidence:
  `registry/strategies/research_decisions/20260507T054500JST_directional_change_overshoot_research_gate/research_decision.json`.
  `registry/strategies/proposals/20260506T205000Z_LongOnlyDirectionalChangeOvershootCandidate.metadata.json`.
  `registry/strategies/generated/LongOnlyDirectionalChangeOvershootCandidate/20260507T061500JST_directional_change_overshoot_codegen/metadata.json`.
  `registry/strategies/candidates/LongOnlyDirectionalChangeOvershootCandidate/20260507T064500JST_directional_change_overshoot_full_eval/candidate_manifest.json`.
- Read the earlier approved research decision only as historical evidence:
  `registry/strategies/research_decisions/20260507T043500JST_bipower_jump_decay_research_gate/research_decision.json`.
- Optionally read the latest material-category research gate smoke as an
  enforcement smoke:
  `registry/strategies/research_decisions/20260507T053000JST_material_category_gate_top3_only_block_smoke/research_decision.json`.
- Optionally read older blocked-repeat research decisions as historical
  enforcement
  smoke:
  `registry/strategies/research_decisions/20260507T022000JST_repeat_microstructure_research_gate_block_with_causal_map/research_decision.json`.
- Read `docs/BOT_FACTORY_GOAL_AUDIT.md`; it is the explicit correction against
  breadth-first candidate enumeration.
- Use the research selection gate before adding more strategy variants, with
  causal responses for `regime_fragile_mechanism`, `walk_forward_fragility`,
  `cost_sensitive_mechanism`, `no_profitable_walk_forward_windows`, and
  `entry_exists_negative_edge` when required by the refreshed map. These
  responses must be substantive and cannot be parameter-only. The decision
  must also meet the map's `minimum_research_selection_score=80`; below-score
  decisions are blocked even when some individual fields look plausible. The
  map's required research questions must each have a substantive
  `--research-question-response`; missing or placeholder answers are blocked.
  When `cost_sensitive_mechanism` has `risk_score >= 80`, also supply
  `--local-falsification-json` evidence that matches the thesis ID and shows
  expected edge exceeding all-in cost in bps. The research gate rejects crafted
  or unsafe local falsification JSON that does not have
  `factory=research_local_falsification`, historical-only safety scope, and a
  valid Bot Factory local event source with a passing failure-synthesis guard.
  Pass failed pre-proposal screens through `--prior-local-falsification-json`
  so the same `thesis_id` or `mechanism_class` cannot be retried as a fresh
  idea.
  If no distinct thesis can pass the gate, improve causal failure synthesis or
  selection scoring instead of extending the generator.
- Directional-change, range-quarticity, and mark-discount reclaim codegen,
  signal diagnostics, historical backtest, walk-forward, candidate evaluation,
  ranking, synthesis, and causal-map refresh have already been completed and
  failed. Mark fair-value momentum lag also passed research selection and
  codegen, then failed generated-entry edge diagnostics before historical
  backtest interpretation. The local event generation mismatch has been fixed
  and now rejects that thesis pre-proposal. Research selection, proposal
  generation, and codegen now consume the local falsification handoff for
  high-risk cost-sensitive mechanisms, so stale positive local screens should
  not authorize proposal/codegen after event-builder semantics change.
  Candidate failure synthesis also validates local falsification rejections
  before counting them as failed-thesis memory; factory/status alone is no
  longer enough without historical-only safety scope, valid event source,
  closed-context alignment, and a passing failure-synthesis guard. The next
  research-selection path is covered by regression tests: validated local
  rejections block repeated mechanism classes, while invalid/crafted failed
  artifacts remain visible but do not poison novelty memory. Research selection
  decision artifacts now also mark this provenance explicitly through
  `local_falsification_failed_*` novelty fields and the
  `research_thesis_outside_failure_synthesis_local_rejections` blocker, so do
  not reinterpret a local rejection as a generic family repeat that can be
  solved by threshold tuning. Proposal generation also revalidates those
  novelty fields and blocks crafted/stale approved research decisions through
  `research_decision_<n>_novelty_assessment_passed` and
  `research_decision_<n>_outside_failure_synthesis_local_rejections`. Codegen
  now also revalidates accepted proposal metadata through
  `research_decision_novelty_handoff` and blocks stale/crafted metadata with
  `research_decision_novelty_handoff_passed` before emitting strategy code.
  Causal failure maps now also promote validated local falsification
  rejections into `validated_local_falsification_rejections`, required
  research questions, and blocked next action
  `retry_validated_local_rejection_by_parameter_tuning`, so the next research
  decision must answer that evidence rather than treating it as threshold
  tuning. Research selection decision artifacts now preserve and render those
  `validated_local_falsification_rejections`; a map-required local rejection
  research question blocks selection when unanswered and passes only after a
  substantive indexed response. The next concrete path is not another default
  family addition; use the latest failure map and validated local rejection
  evidence to improve failure synthesis or selection scoring before
  considering a genuinely new thesis.
- Use all available signal diagnostics and the fresh hybrid prediction
  diagnostics as negative evidence, especially:
  - liquidity pullback: trend filter eliminates otherwise surviving setup rows
  - hybrid ML: ML rows do not overlap rule-gate survivors despite valid fresh
    `&-future_return` predictions
  - downside shock and session liquidity: entries exist, but performance is
    negative and walk-forward fails
  - signed-volume imbalance: entries exist but are sparse; `mid_reclaim` is
    the main bottleneck and the single historical trade is negative
  - entropy regime: entries exist but are sparse; all walk-forward windows
    fail and the tiny positive return depends on one profitable window
  - fractal long memory: entries exist but all windows are negative, suggesting
    the first Hurst/path-efficiency code path is cost-sensitive and not
    robust on the local BTC 5m sample
  - semivariance asymmetry: entries exist and are frequent, but all windows are
    negative; the first good-volatility/bad-volatility implementation appears
    cost-sensitive rather than sparse
  - funding-pressure carry: local funding-rate merge worked, but entries were
    zero because `not_positive_crowding` eliminated the last surviving setup
    row; the first carry/resilience implementation is sparse on the local
    January 2025 sample
  - realized-skewness tail shape: entries exist, but all windows are negative;
    the first higher-moment tail-shape implementation appears cost-sensitive
    and volume-filter constrained
  - calendar-turnover seasonality: entries exist, but all windows are
    negative; the first weekday/weekend-turnover implementation appears
    cost-sensitive and `turnover_recovery` constrained
  - Amihud illiquidity: entries exist, but all windows are negative; the first
    price-impact premium implementation appears cost-sensitive and
    `price_impact_premium` constrained
  - BTC/ETH lead-lag: ETH informative OHLCV merged cleanly and entries exist,
    but all windows are negative; the first cross-asset spillover
    implementation appears cost-sensitive and `volume_filter` constrained
  - variance-ratio regime: entries exist, but historical return is negative
    and only one walk-forward window is profitable; the first
    random-walk-deviation implementation appears volume-filter constrained
  - BTC/ETH cointegration spread: ETH informative OHLCV merged cleanly and
    entries exist, but historical return is negative and all walk-forward
    windows are negative; the first equilibrium-spread implementation appears
    cost-sensitive and `volume_filter` constrained
  - BTC/ETH correlation recovery: ETH informative OHLCV merged cleanly, but
    the generated `eth_market_support` condition eliminates the few rows that
    survive correlation breakdown/recovery; the first dynamic-correlation
    implementation is zero-trade on the local BTC 5m sample
  - market beta drawdown carry: the first positive full-month smoke, but it
    fails walk-forward because only 2/4 windows are profitable, all windows
    fail the min-trade gate, and profit depends heavily on one window; do not
    promote it
  - regime-state reentry: entries are plentiful, but historical return is
    negative and all walk-forward windows are negative; the first
    regime-switching proxy is cost-sensitive and participation-floor
    constrained, do not promote
  - mark-price dislocation reclaim: local 4h mark-price data merged cleanly
    and entries exist, but historical return is negative and no walk-forward
    window is profitable; the first fair-value dislocation implementation is
    cost-sensitive and `mark_discount_pressure` constrained, do not promote
  - microstructure spread reversion: Roll-style and high-low spread proxies
    produced entries, but historical return and all walk-forward windows are
    negative; `spread_compressing` is the top bottleneck, do not promote
  - liquidity recovery horizon: entries exist with `entry_count=9`, but the
    initial historical run has `trade_count=3`, `profit_factor=0.0`, negative
    expectancy, and rarest signal component `below_recovery_anchor`; do not
    promote it
  - bipower jump-decay: entries exist with `entry_count=511`, but historical
    return is `-2.393940186%`, walk-forward return is `-2.368185648%`, and
    every window is negative; do not promote it
  - directional-change overshoot: entries exist with `entry_count=706`, but
    historical return is `-3.2605379719999994%`, walk-forward return is
    `-3.24521717%`, and every window is negative; do not promote it
  - range-quarticity volatility-of-volatility state: entries exist with
    `entry_count=250`, but historical return is `-0.8648079839999998%`,
    walk-forward return is `-0.8648079839999999%`, and every window is
    negative; do not promote it
  - mark-discount reclaim continuation: local 4h mark-price data merged
    cleanly and entries exist with `entry_signal_count=9603`, but historical
    return is `-37.832122396%`, `profit_factor=0.244072592568329`, and
    expectancy is negative; do not promote it
- Do not choose another thesis family without using the latest 32-candidate
  causal failure map and explicit required-question responses first. Do not
  default to loosening thresholds, removing filters, replaying failed families,
  or falling back to an older generator family.
- Proposal generation now recomputes missing required research-question
  indexes from `required_research_questions` and
  `research_question_response_indexes`. Do not trust a crafted
  `approved_for_proposal_generation` research decision merely because its
  reported `missing_research_question_response_indexes` list is empty.
- Strategy code generation also revalidates accepted proposal metadata through
  `research_decision_question_handoff_passed`; do not emit code from crafted
  proposal metadata that still has missing or weak required research-question
  responses in `research_decision_constraints`.
- Candidate iteration plans now preserve `blocked_next_actions` and block
  revision text that repeats them. Do not repackage a validated local rejection
  retry or other causal failure map blocked action as a changed assumption,
  changed parameter, or changed data requirement.
- Candidate evaluation manifests now carry `blocked_next_actions` forward from
  proposal/generated research briefs and proposal constraints into
  `next_candidate_input`; keep that handoff intact before starting iteration.
- Candidate evaluation also preserves codegen/proposal research handoff
  summaries such as `research_decision_question_handoff` and
  `research_decision_novelty_handoff` inside manifest `research_brief`; do not
  strip those fields when building ranking or iteration inputs.
- Candidate ranking now preserves `blocked_next_actions`, compact
  `research_brief` context, and `research_handoff_summary` fields on ranked
  candidate rows. Failure synthesis can use that ranking-level context even
  when original candidate manifests are unavailable, so keep those fields
  intact when changing ranking or synthesis outputs.
- Candidate failure maps now preserve synthesis-level
  `research_handoff_summaries` and merge upstream `blocked_next_actions` into
  `research_selection_guidance`. Research selection keeps those handoff
  summaries in the decision `causal_failure_map` summary; do not strip them
  before proposal generation.
- Research selection now also preserves causal-map `blocked_next_actions`, and
  proposal metadata carries causal-map blocked actions plus
  `research_handoff_summaries` into both `research_decision_constraints` and
  proposal `research_brief`. Keep those fields intact before codegen and
  evaluation.
- Generated strategy metadata and generated `research_brief.json` now preserve
  proposal-level `blocked_next_actions` and generic
  `research_handoff_summaries`; candidate evaluation manifests preserve them
  from generated/proposal metadata, nested research briefs, and constraints.
  Keep this continuity intact through ranking, failure synthesis, and
  iteration.
- Generated Freqtrade strategy parameters are theory-fixed, not a hyperopt
  surface. `strategy_code.py` emits `IntParameter` / `DecimalParameter` with
  `optimize=False` and metadata
  `parameter_optimization_policy=theory_fixed_parameters_no_freqtrade_hyperopt`.
  Do not change these back to `optimize=True` or treat candidate revision as
  threshold tuning unless the project direction is explicitly changed.
- Candidate evaluation also enforces that policy. Generated Strategy Code
  Generator artifacts now get a `generated_parameter_optimization_policy`
  check, and evaluation rejects any artifact whose metadata or actual strategy
  file exposes `optimize=True` or fails to prove the
  `theory_fixed_parameters_no_freqtrade_hyperopt` policy. Do not bypass this
  by supplying older generated metadata.
- Latest pre-proposal local screen:
  `TH-FUNDING-NEUTRAL-IMPULSE-DRIFT-20260507` /
  `funding_neutral_impulse_drift` generated 475 closed-candle events. It was
  rechecked with funding-adjusted local falsification after adding optional
  `--funding-rate-path` support and still failed with
  `expected_price_edge_bps=-2.15646`,
  `expected_funding_adjustment_bps=-0.048684`,
  `net_edge_bps=-14.205144`, and `profitable_windows_ratio=0.0`.
  The then-refreshed synthesis
  `registry/strategies/synthesis/20260507T164000JST_all_candidates_with_negative_funding_sample_rejection/candidate_failure_synthesis.json`
  has `local_falsification_rejection_count=8`, and the refreshed map
  `registry/strategies/failure_maps/20260507T164500JST_all_candidates_with_negative_funding_sample_rejection_causal_map/causal_failure_map.json`
  requires avoiding that validated rejection plus the extended-history
  long/short cost/stability rejection and negative-funding small-sample
  rejection. These artifacts are superseded by the current 32-candidate
  comprehensive local-rejection synthesis/map listed above. Do not retry
  them by loosening thresholds or repackaging them as parameter tuning.
- Local falsification now requires `--funding-rate-path` whenever the supplied
  `local_events.json` used `funding_rate` context. Do not judge a
  funding-aware event source by price returns alone.
- Local falsification calendar-window diagnostics now survive through failure
  synthesis, causal failure maps, and research-selection reports. Treat
  `profitable_calendar_windows_ratio` and `calendar_window_summaries` as
  required evidence context when answering validated local rejection questions;
  do not rely only on aggregate expected edge or equal-sample windows.
- Research-selection answers to required questions that mention
  `calendar_window`, `profitable_calendar_windows_ratio`,
  `calendar_window_summaries`, `quarterly`, or `quarter` must explicitly
  address calendar-window evidence. Generic post-cost-edge language is now a
  weak answer and blocks selection.
- Research selection now blocks stale failure synthesis inputs when a newer
  parseable Bot Factory `candidate_failure_synthesis.json` exists under
  `registry/strategies/synthesis/`. Use the latest synthesis/map pair before
  selecting a thesis; do not pair an older map with an older synthesis to
  bypass newer local falsification rejections.
- Strategy proposal generation also recomputes this latest failure-synthesis
  guard before writing proposals. Do not pass an old approved
  `research_decision.json` with its matching old synthesis after newer local
  failure memory exists; proposal generation now blocks that stale-memory
  bypass.
- Latest crowding-unwind outcome:
  `TH-CROWDING-UNWIND-REACCUMULATION-20260507` is no longer the next proposal
  step. It did pass local pre-proposal falsification
  (`net_edge_bps=10.397636`, `sample_count=491`,
  `profitable_windows_ratio=0.75`) and was carried through real proposal/codegen
  as
  `registry\strategies\generated\LongOnlyCrowdingUnwindReaccumulationCandidate\crowding_unwind_reaccumulation_001\metadata.json`.
  Generated signal diagnostics passed with `entry_count=929`,
  `net_edge_bps=13.150967`, and `profitable_windows_ratio=0.5`, but historical
  backtest failed:
  `data\backtests\LongOnlyCrowdingUnwindReaccumulationCandidate\crowding_unwind_reaccumulation_001_20240101_20260507_dirpath\metrics.json`
  has `total_return_pct=-1.008043973`,
  `profit_factor=0.8187715323309974`, and
  `sortino=-0.5357142015667924`. OHLCV-prechecked walk-forward also failed:
  `data\walk_forward\LongOnlyCrowdingUnwindReaccumulationCandidate\crowding_unwind_reaccumulation_001_wf_20240101_20260507_v3_ohlcv_file\walk_forward_metrics.json`
  has `pass_rate=0.0`, `profitable_windows_ratio=0.2`,
  `total_return_pct=-1.008043973`, and
  `max_single_window_profit_dependency=1.0`.
  Trade-shape inspection shows the failure is adverse payoff asymmetry and
  timeout drag, not zero-entry sparsity: the main exit path had 280 trades,
  win rate `0.625`, average win `0.003703`, and average loss `-0.006361`;
  `timeout_exit` had 14 trades, win rate `0.1429`, and average return
  `-0.004999`. 2024H1/2024H2 were negative, 2025H1 was only slightly
  positive, and 2025H2/2026YTD produced zero trades.
  Candidate manifest
  `registry\strategies\candidates\LongOnlyCrowdingUnwindReaccumulationCandidate\crowding_unwind_reaccumulation_001\candidate_manifest.json`
  records `recommendation=retry`, `historical_backtest=fail`, and
  `walk_forward=fail`; ranking
  `registry\strategies\candidates\rankings\20260507T084500Z_crowding_unwind_reaccumulation_retry_ranking_gatefix\candidate_ranking.json`
  has `paper_ready_candidate_ids=[]`. Do not promote, paper-start, or retry
  this by threshold loosening. Any retry must be a materially different theory
  or explicit theory-backed exit/risk mechanism with fresh local falsification
  evidence before proposal/codegen.
- The safe generated-strategy path for structural open interest plus
  long/short account ratio now exists and the current safe-path capability
  report is
  `registry\strategies\checks\20260507T084000Z_structural_data_capabilities_codegen_oi_lsr_safe_path.json`.
  It is reusable infrastructure, not positive evidence for the failed
  `crowding_unwind_reaccumulation_001` candidate. Liquidation and order-book
  remain in `must_not_codegen`.
- Public Bybit V5 long/short account ratio support has been added as another
  local structural market-data input. Use
  `scripts\bot_factory_download_bybit_long_short_ratio.py` and
  `scripts\bot_factory_check_long_short_ratio.py` to create and validate local
  evidence before using it in research selection. Structural capability reports
  now include `long_short_ratio`; strategy codegen is supported only through
  the verified local historical `crowding_unwind_reaccumulation` path.
- Local event studies can now consume a validated long/short ratio parquet/CSV
  through `scripts\bot_factory_build_local_events.py
  --long-short-ratio-path`. Supported event features are
  `long_account_ratio`, `long_account_ratio_delta_bps`, `long_short_ratio`,
  and `long_short_ratio_zscore`. Treat these as closed-context local market
  data only; outside the supported crowding-unwind variant they are for
  pre-proposal event studies and falsification, not direct codegen.
- Local event studies can also consume a second closed-candle OHLCV parquet/CSV
  through `scripts\bot_factory_build_local_events.py
  --informative-ohlcv-path`. Supported informative features are
  `informative_return_bps`, `relative_return_bps`, `informative_range_pct`,
  `informative_sma_distance_bps`, and `informative_volume_zscore`, with
  `relative_return_bps` defined as primary return minus informative return over
  the same lookback. Use this for pre-proposal cross-asset local screens only;
  it does not revive failed BTC/ETH lead-lag, cointegration, or correlation
  families without a materially different theory and fresh local falsification.
- Local falsification can now enforce calendar stability in addition to
  aggregate post-cost edge. For high-risk pre-proposal screens, prefer
  `scripts\bot_factory_build_local_falsification.py
  --min-calendar-window-count <n>
  --min-profitable-calendar-windows-ratio <ratio>` so small samples or
  one-quarter concentration do not pass merely because aggregate expected edge
  is positive.
- Use `scripts\bot_factory_export_research_selection_template.py` against the
  current causal failure map before drafting the next thesis. It exports the
  required `--causal-failure-response` and `--research-question-response`
  placeholders, validated local rejection context, blocked actions, a Markdown
  checklist, and a full `select_research_thesis_command_template` PowerShell
  skeleton with the current synthesis/map paths, thesis metadata placeholders,
  local data/falsification/reference placeholders, and every required response
  placeholder. It also exports a fillable `research_selection_input_template`
  and short `select_research_thesis_input_json_command_template`;
  `scripts\bot_factory_select_research_thesis.py` accepts
  `--research-selection-input-json` for a filled version of that JSON so future
  AI research handoffs do not need to copy a long CLI manually. The current
  template artifact is
  `registry\strategies\research_decisions\20260507T112500Z_cross_asset_relative_strength_rejection_research_selection_response_template\research_selection_response_template.json`
  with 5 required causal responses, 30 required question responses, and 12
  validated local rejection contexts.
- Local ETH 5m futures OHLCV coverage has been expanded for future cross-asset
  screens. The current quality artifact
  `registry\strategies\checks\20260507T103000Z_eth_5m_ohlcv_quality_append_2025_2026.json`
  has `ok=true`, `rows=246941`, start
  `2024-01-01T00:00:00+00:00`, end
  `2026-05-07T10:20:00+00:00`, `duplicate_timestamps=0`, and
  `missing_intervals=0`. The data files remain ignored under `user_data/*`.
  This only removes a short-history blocker; it does not revive failed
  BTC/ETH lead-lag, cointegration, or correlation families without a materially
  different theory and fresh local falsification.
- For another hybrid ML smoke, keep the candidate-scoped `freqai_identifier`
  path and `self.freqai.start(...)` validation. Choose a timerange with enough
  local pre-window OHLCV for `train_period_days=2`.
- Preserve local artifacts as the source of truth. If a subprocess is blocked
  by environment/network/public metadata access, write the blocker into the
  candidate manifest and keep the command previews, logs, and partial local
  artifacts.
- Add focused tests when changing factory behavior. Existing coverage includes
  execution blockers, wrapper failure stop behavior, CLI execution argument
  mapping, `@path` research-reference parsing, research brief preservation,
  hypothesis-diversity paper-ready blocking, ranking/failure-map research
  handoff preservation, candidate-scoped FreqAI
  identifiers, prediction diagnostics, intraday/session-liquidity logic, and
  signed-volume imbalance, semivariance asymmetry, funding-pressure,
  realized-skewness tail-shape, and market-beta drawdown-carry logic.
- Current full `tests\test_bot_factory.py -q` reaches `[100%]` without the
  prior pandas fragmented-DataFrame `PerformanceWarning` summary from
  `signal_diagnostics.py`; keep verification output clean when adding new
  diagnostic feature families.
- Latest post-pivot Edge Discovery result is also a rejection, not a
  candidate. Fixed BTC/ETH relative-value reversion
  `TH-BTC-ETH-RELATIVE-VALUE-REVERSION-20260507` /
  `btc_eth_relative_value_reversion` used local BTC 5m OHLCV plus ETH 5m
  informative OHLCV with fixed conditions
  `informative_return_bps(48) >= 50.0` and
  `relative_return_bps(48) <= -100.0`. Artifact
  `registry\strategies\research_decisions\20260507T210000JST_btc_eth_relative_value_reversion_edge\edge_discovery.json`
  has `status=failed`, `event_count=937`, `passing_horizon_count=0`, best
  `net_edge_bps=-11.699275`, `proposal_generation_allowed=false`, and
  `strategy_codegen_allowed=false`. It was ingested into
  `registry\strategies\synthesis\20260507T211000JST_all_candidates_with_btc_eth_relative_value_reversion_edge_rejection\candidate_failure_synthesis.json`
  with `edge_discovery_rejection_count=1`. Do not retry this mechanism by
  loosening thresholds.
- Liquidation data now has a local quality gate plus local event / Edge
  Discovery features, but no strategy codegen support. Use
  `scripts\bot_factory_check_liquidation.py` for user-supplied historical
  liquidation parquet with timestamp (`date`/`T`), side (`side`/`S`), size
  (`size`/`quantity`/`qty`/`v`), and price (`price`/`p`/`bankruptcy_price`)
  columns. `scripts\bot_factory_build_local_events.py` and
  `scripts\bot_factory_build_edge_discovery.py` accept `--liquidation-path` and
  can use `liquidation_count`, buy/sell/total/net notional, imbalance, and
  total-notional z-score features. `scripts\bot_factory_report_structural_data_capabilities.py`
  accepts `--liquidation-quality-report-json`; a passing quality report plus
  local data can make liquidation `local_research_usable`. The local event and
  Edge Discovery runners now also require
  `--liquidation-quality-report-json` when `liquidation_*` features are
  referenced. Liquidation remains in `must_not_codegen` until a supported
  strategy variant exists.
- Open-interest and long/short-ratio local event / Edge Discovery paths now
  also require supplied passing quality reports. Use
  `--open-interest-quality-report-json` when `open_interest*` features are
  referenced and `--long-short-ratio-quality-report-json` when long/short-ratio
  features are referenced. Do not rely on a parseable local parquet/CSV alone.
- Order-book snapshot data now also has local event / Edge Discovery features,
  but no strategy codegen support. Use `scripts\bot_factory_check_order_book.py`
  for user-supplied historical top-of-book snapshots with `date`, best bid,
  best ask, bid size, ask size, and optional `depth_imbalance`.
  `scripts\bot_factory_build_local_events.py` and
  `scripts\bot_factory_build_edge_discovery.py` accept `--order-book-path` and
  can use `order_book_bid_size`, `order_book_ask_size`,
  `order_book_depth_imbalance`, `order_book_depth_imbalance_zscore`,
  `order_book_mid_price_gap_bps`, `order_book_spread_bps`, and
  `order_book_spread_bps_zscore`. A passing quality report plus local
  timestamped snapshots can make order-book data `local_research_usable`. The
  local event and Edge Discovery runners now also require
  `--order-book-quality-report-json` when `order_book_*` features are
  referenced. Order-book remains in `must_not_codegen`.

Suggested verification after code changes:

```powershell
.\.venv\Scripts\python.exe -m py_compile `
  freqtrade_ext\bot_factory\strategy_proposals.py `
  freqtrade_ext\bot_factory\strategy_code.py `
  freqtrade_ext\bot_factory\candidate_evaluation.py `
  freqtrade_ext\bot_factory\candidate_ranking.py `
  freqtrade_ext\bot_factory\candidate_iteration.py `
  freqtrade_ext\bot_factory\data_quality.py `
  freqtrade_ext\bot_factory\signal_diagnostics.py `
  freqtrade_ext\bot_factory\candidate_failure_synthesis.py `
  freqtrade_ext\bot_factory\freqai_prediction_diagnostics.py `
  freqtrade_ext\bot_factory\bybit_long_short_ratio.py `
  freqtrade_ext\bot_factory\local_events.py `
  freqtrade_ext\bot_factory\local_falsification.py `
  freqtrade_ext\bot_factory\research_selection_template.py `
  freqtrade_ext\bot_factory\structural_data_capabilities.py `
  scripts\bot_factory_check_liquidation.py `
  scripts\bot_factory_build_local_events.py `
  scripts\bot_factory_build_edge_discovery.py `
  scripts\bot_factory_generate_strategy_proposal.py `
  scripts\bot_factory_generate_strategy_code.py `
  scripts\bot_factory_evaluate_candidate.py `
  scripts\bot_factory_check_funding_rate.py `
  scripts\bot_factory_download_bybit_long_short_ratio.py `
  scripts\bot_factory_check_long_short_ratio.py `
  scripts\bot_factory_build_local_events.py `
  scripts\bot_factory_build_local_falsification.py `
  scripts\bot_factory_export_research_selection_template.py `
  scripts\bot_factory_report_structural_data_capabilities.py `
  scripts\bot_factory_rank_candidates.py `
  scripts\bot_factory_iterate_candidate.py `
  scripts\bot_factory_diagnose_candidate_signals.py `
  scripts\bot_factory_synthesize_candidate_failures.py `
  scripts\bot_factory_diagnose_freqai_predictions.py `
  tests\test_bot_factory.py `
  registry\strategies\generated\LongOnlySemivarianceAsymmetryCandidate\20260506T211500JST_semivariance_asymmetry_smoke\LongOnlySemivarianceAsymmetryCandidate.py `
  registry\strategies\generated\LongOnlyFundingPressureCarryCandidate\20260506T214600JST_funding_pressure_smoke\LongOnlyFundingPressureCarryCandidate.py `
  registry\strategies\generated\LongOnlyRealizedSkewnessTailCandidate\20260506T225000JST_realized_skewness_tail_smoke\LongOnlyRealizedSkewnessTailCandidate.py `
  registry\strategies\generated\LongOnlyCalendarTurnoverCandidate\20260506T231600JST_calendar_turnover_smoke\LongOnlyCalendarTurnoverCandidate.py `
  registry\strategies\generated\LongOnlyAmihudIlliquidityCandidate\20260506T234000JST_amihud_illiquidity_smoke\LongOnlyAmihudIlliquidityCandidate.py `
  registry\strategies\generated\LongOnlyCrossAssetLeadLagCandidate\20260506T235945JST_cross_asset_lead_lag_smoke\LongOnlyCrossAssetLeadLagCandidate.py `
  registry\strategies\generated\LongOnlyVarianceRatioRegimeCandidate\20260506T235951JST_variance_ratio_regime_smoke\LongOnlyVarianceRatioRegimeCandidate.py `
  registry\strategies\generated\LongOnlyCrossAssetCointegrationCandidate\20260506T235957JST_cross_asset_cointegration_smoke\LongOnlyCrossAssetCointegrationCandidate.py `
  registry\strategies\generated\LongOnlyCrossAssetCorrelationCandidate\20260506T235959JST_cross_asset_correlation_smoke\LongOnlyCrossAssetCorrelationCandidate.py `
  registry\strategies\generated\LongOnlyMarketBetaDrawdownCarryCandidate\20260507T001500JST_market_beta_drawdown_carry_smoke\LongOnlyMarketBetaDrawdownCarryCandidate.py `
  registry\strategies\generated\LongOnlyRegimeStateReentryCandidate\20260507T003500JST_regime_state_reentry_smoke\LongOnlyRegimeStateReentryCandidate.py `
  registry\strategies\generated\LongOnlyMarkPriceDislocationCandidate\20260507T011500JST_mark_price_dislocation_smoke\LongOnlyMarkPriceDislocationCandidate.py

.\.venv\Scripts\python.exe -m pytest tests\test_bot_factory.py -q
.\.venv\Scripts\python.exe scripts\bot_factory_static_check.py user_data\strategies
.\.venv\Scripts\python.exe scripts\bot_factory_check_mark_price.py user_data\data\bybit\futures\BTC_USDT_USDT-4h-mark.parquet --timeframe 4h
.\.venv\Scripts\python.exe scripts\bot_factory_check_funding_rate.py user_data\data\bybit\futures\BTC_USDT_USDT-8h-funding_rate.parquet --timeframe 8h
.\.venv\Scripts\python.exe scripts\bot_factory_check_freqai_env.py
git diff --check
```

Documentation requirement:

- Update `docs/BOT_FACTORY_MVP_TODO.md` after every completed increment with
  exact commands, results, artifacts, and remaining limitations.
- Do not mark Candidate Evaluation Pipeline profitable or promotion-ready.
  Thirty-two local candidate manifests have failed gates or initial historical
  evaluation and remain rejected, retry, or fail evidence.
- Do not mark Paper trading deployment complete until an explicitly requested,
  preflight-approved paper path has been implemented, verified, and documented.
- Do not describe generated strategy code as profitable or paper-ready based on
  one proposal, one generated strategy, or one backtest.
````
