# Bot Factory Architecture Risk TODO

Status: architecture follow-up TODO for regime-aware strategy selection.

Created: 2026-05-21 JST.

Product direction: `docs/BOT_FACTORY_PRODUCT_VISION_TODO.md` is the current
product-level North Star. This document is an architecture follow-up under the
Suitability / Matching and Audit / Lineage contracts.

## Goal

Close the structural gaps between:

- real Freqtrade strategy backtests;
- strategy logic specs used by regime selector simulations;
- regime fitness scorecards;
- promotion/readiness gates;
- future runtime strategy selection.

The target architecture must make it hard to accidentally select, promote, or
paper-test a strategy whose evidence belongs to a different strategy, signal,
risk policy, cost model, or market regime.

## Non-Goals / Safety Boundaries

- Do not start `freqtrade trade`, paper trading, dry-run trading, canary live,
  live trading, or any bot process as part of these TODOs.
- Do not use API keys, secrets, exchange order endpoints, leverage above `1.0`,
  or shorting.
- Do not treat `REGIME_SCOPED_SELECTOR_ELIGIBLE` as paper/live approval.
- Do not allow raw aggregate PnL, recent observation strength, or manually
  assembled scorecards to bypass historical, walk-forward, cost robustness,
  regime fitness, and Phase 3 readiness gates.

## Current Architecture Risk Summary

The current design has the right separation of layers, but several links are
still manual or conceptual:

```text
real strategy backtest artifacts
  != automatically tied to
selector logic specs
  != automatically transformed into
regime scorecards
  != formally required by
paper readiness
```

This creates the main risk: a strategy can look good in one artifact family
while the selector or readiness layer evaluates a related but not identical
candidate.

## P0: Evidence Identity And Artifact Lineage

- [x] Define a canonical `StrategyCandidateIdentity` artifact.
  Required fields:
  - `candidate_id`
  - `strategy_id`
  - `strategy_class_name`
  - `strategy_source_path`
  - `strategy_version`
  - `signal_version`
  - `risk_policy_version`
  - `regime_classifier_version`
  - `cost_model_id`
  - `allowed_pairs`
  - `allowed_timeframes`
  - `created_at`
  - `source_artifacts`
- [x] Require the same identity object to be embedded in:
  - [x] generated strategy metadata;
  - [x] historical backtest outputs written by the checked backtest wrappers;
  - [x] walk-forward outputs;
  - [x] training manifests when applicable;
  - [x] observation ledger rows;
  - [x] regime fitness scorecards;
  - [x] selector candidates.
- [x] Add validation that rejects evidence when any identity field differs
  across artifacts unless a migration/version mapping is explicit.
- [x] Add tests proving that a backtest for strategy A cannot be consumed by a
  selector candidate for strategy B.
- [x] Add tests proving that changing `signal_version`, `risk_policy_version`,
  `regime_classifier_version`, or `cost_model_id` segments evidence.

Acceptance criteria:

- [x] A reviewer can trace a selected runtime candidate back to exact strategy
  source, metrics, trades, scorecard, and selector decision artifacts.
- [x] Mismatched strategy/logic evidence fails closed.

Implemented on 2026-05-21 JST:

- Added `freqtrade_ext/bot_factory/candidate_identity.py` with
  `strategy_candidate_identity_v1`, required-field validation, artifact
  identity comparison, and explicit migration-map support for future approved
  version remaps.
- Embedded `candidate_identity` in generated strategy metadata, checked
  backtest/FreqAI backtest wrapper outputs, FreqAI training manifests,
  observation ledger rows, regime fitness scorecards, and selector candidates.
- Embedded `candidate_identity` in checked walk-forward outputs, persisted
  `candidate_identity.json` beside `walk_forward_metrics.json`, passed the same
  `--candidate-id` into each child window wrapper, and added lineage validation
  that fails a walk-forward result when completed child metrics carry a
  different identity.
- Added candidate-identity lineage checks to observation validation, regime
  scorecard construction, runtime selector candidate evaluation, and candidate
  evaluation manifests.
- Bound `DonchianTrendBullStrategy` to
  `strong_uptrend_momentum_v1` through the same canonical identity object:
  `strong-uptrend-historical-ohlcv-candidate`.

Remaining limitation: existing historical and walk-forward artifacts produced
before this increment are not regenerated in-place; new checked wrapper outputs
carry the identity object. The deterministic converter added on 2026-05-22 JST
can now build observations, scorecards, and selector candidates from checked
local backtest artifacts.

## P0: Backtest To Observation To Scorecard Pipeline

- [x] Add a deterministic converter:

  ```text
  backtest metrics + trades + OHLCV regime labels
    -> observation ledger rows
    -> regime fitness scorecard
    -> selector candidate
  ```

- [x] Record one observation row per candidate, pair, timeframe, window, and
  market regime.
- [x] Require local baselines in the same converter:
  - `no_trade`;
  - buy-and-hold / hold baseline;
  - incumbent candidate when available.
- [x] Reject manually assembled scorecards from becoming selector inputs unless
  they explicitly record `manual_review_only=true`.
- [x] Add a CLI or helper that writes:
  - `observation_ledger.json`;
  - `regime_fitness_scorecard.json`;
  - `regime_fitness_scorecard_report.md`;
  - `selector_candidate.json`.

Acceptance criteria:

- [x] A real Freqtrade backtest can produce selector-eligibility artifacts
  without hand-building observations.
- [x] The scorecard clearly states whether the strategy beat hold/no-trade after
  normal and stress costs.

Implemented on 2026-05-22 JST:

- Added `freqtrade_ext/bot_factory/evidence_pipeline.py` and
  `scripts/bot_factory_build_regime_artifacts.py` to convert checked local
  backtest `metrics.json`, `trades.csv`, and OHLCV parquet into a deterministic
  observation ledger, regime fitness scorecard, scorecard Markdown report,
  selector candidate, traceability file, and pipeline manifest.
- The converter writes candidate observations and local `no_trade` / `hold`
  baselines per candidate, pair, timeframe, window, and classifier regime. The
  scorecard includes `baseline_comparison` rows with hold/no-trade deltas.
- `selection_candidate_from_scorecard()` now fails closed for manual or
  non-`regime_fitness_scorecard` inputs, so hand-built scorecards cannot become
  selector candidates by default.
- Added focused tests for converter output, selector lineage, baseline presence,
  manual scorecard rejection, and no-promotion safety flags.

Remaining limitation: historical artifacts produced before this increment were
not regenerated in-place. They can be converted by running the new local CLI
against checked backtest outputs.

## P0: Deterministic Market Regime Classifier

- [x] Implement a local-only regime classifier over historical candles.
  Initial labels:
  - `trend_up`
  - `trend_down`
  - `range`
  - `high_volatility`
  - `low_volatility`
  - `liquidity_stress`
  - `post_spike_reversion`
  - `mixed`
  - `unknown`
- [x] Use predeclared features:
  - rolling return;
  - realized volatility;
  - trend strength / slope;
  - range efficiency;
  - drawdown from local high;
  - volume/liquidity proxy;
  - candle gap/spread proxy when available;
  - data quality flags.
- [x] Emit `unknown` when data quality is poor or classifier confidence is low.
- [x] Persist `regime_classifier_version`.
- [x] Add churn tests so small candle perturbations do not flip labels
  excessively.
- [x] Add tests for deterministic classification on fixed OHLCV fixtures.

Acceptance criteria:

- [x] Runtime selector tests no longer need hand-authored `trend_up` labels for
  historical windows.
- [x] Strategy suitability is evaluated against reproducible regime artifacts.

Implemented on 2026-05-22 JST:

- Added `freqtrade_ext/bot_factory/market_regime.py` with a local-only
  deterministic OHLCV classifier, predeclared candle features, `unknown` output
  for weak/poor-quality windows, persisted
  `deterministic_regime_classifier_v1`, and churn reporting.
- Added deterministic fixture tests showing fixed OHLCV labels are reproducible
  and small perturbations do not exceed the configured churn threshold.

## P1: Promotion Gate Semantics Cleanup

- [x] Rename or document each gate outcome so it cannot be confused with
  paper/live approval:
  - `initial_backtest_gate.pass`
  - `eligible_for_walk_forward_review`
  - `REGIME_SCOPED_SELECTOR_ELIGIBLE`
  - `GLOBAL_SELECTOR_ELIGIBLE`
  - `paper_readiness.pass`
- [x] Add a shared gate glossary in docs.
- [x] Ensure every report includes:
  - what the gate permits;
  - what it does not permit;
  - next required gate.
- [x] Add tests or static assertions that regime scorecards always include:
  - `raw_aggregate_pnl_promotion_allowed=false`;
  - `phase3_readiness_required_after_scorecard=true`;
  - `promotion_authorized_by_this_command=false`.
- [x] Add paper-readiness checks that require regime scorecard artifacts when a
  strategy claims regime-scoped eligibility.

Acceptance criteria:

- [x] No local artifact can be interpreted as paper/live approval by name alone.
- [x] Paper readiness fails closed when required regime evidence is missing.

Implemented on 2026-05-22 JST:

- Added `freqtrade_ext/bot_factory/gate_semantics.py` and
  `docs/BOT_FACTORY_GATE_GLOSSARY.md`. Backtest, walk-forward, and regime
  scorecard reports now include gate semantics: what a gate permits, what it
  does not permit, and the next required gate.
- Regime scorecards explicitly persist
  `raw_aggregate_pnl_promotion_allowed=false`,
  `phase3_readiness_required_after_scorecard=true`, and
  `promotion_authorized_by_this_command=false`.
- Phase 3 paper readiness now accepts `--regime-scorecard` and
  `--requires-regime-scorecard`; when regime-scoped eligibility is claimed and
  the scorecard is missing or unsafe, readiness fails closed.

## P1: Strategy-Type-Specific Gates

- [x] Split gate thresholds by candidate style:
  - scalp / high-frequency;
  - intraday trend-following;
  - swing trend-following;
  - range mean-reversion;
  - defensive/no-trade policy.
- [x] Replace one global `min_trades >= 200` rule with style-aware minimums and
  sample-size requirements.
- [x] Add hold-baseline gates for trend-following candidates:
  - fail or shadow-only when return is below hold and drawdown improvement does
    not justify the opportunity cost;
  - pass only when risk-adjusted return or capital efficiency is predeclared and
    defensible.
- [x] Add no-trade opportunity-cost gates for defensive policies.
- [x] Add tests showing a low-trade trend strategy is not rejected solely by a
  scalping-style trade-count threshold.

Acceptance criteria:

- [x] `DonchianTrendBullStrategy`-style candidates are evaluated against trend
  strategy rules, not a generic scalping threshold.
- [x] A strategy that wins only by underperforming hold is not promoted without
  an explicit risk-reduction rationale.

Implemented on 2026-05-22 JST:

- Added style-aware backtest gates in
  `freqtrade_ext/bot_factory/backtest_results.py` for scalp/high-frequency,
  intraday trend, swing trend, range mean-reversion, and defensive/no-trade
  candidates.
- Trend candidates now evaluate hold-baseline delta and drawdown improvement
  instead of being rejected solely by a scalping-style trade count. Defensive
  policies evaluate no-trade opportunity cost.

## P1: Runtime Selector Stability

- [x] Add cooldown after regime change.
- [x] Add hysteresis so small regime-confidence changes do not cause rapid
  strategy switching.
- [x] Add minimum confidence threshold per regime.
- [x] Add `unknown` / low-confidence fallback to `no_trade`.
- [x] Track last selected candidate, last selected regime, and last switch
  reason in local selector state artifacts.
- [x] Add selector simulation tests for:
  - `trend_up -> range -> trend_up` churn;
  - missing features;
  - classifier version mismatch;
  - pair/timeframe mismatch;
  - process-control flags accidentally enabled.

Acceptance criteria:

- [x] Selector decisions are stable across noisy boundary conditions.
- [x] Unsafe or uncertain runtime state returns `no_trade`.

Implemented on 2026-05-22 JST:

- Added `RuntimeSelectorState`, regime-change cooldown, selector hysteresis,
  per-regime confidence thresholds, and `unknown` / low-confidence fallbacks to
  `no_trade`.
- Runtime selector decisions now persist current and next selector state,
  stable reason codes, feature-quality checks, and process-control safety flags.

## P1: Feature Quality And Data Confidence

- [x] Extend runtime checks beyond feature presence:
  - missing rate;
  - stale feature timestamp;
  - outlier count;
  - recent data gaps;
  - classifier confidence;
  - cost model freshness.
- [x] Add `feature_quality_report.json`.
- [x] Require selector candidates to declare feature quality thresholds.
- [x] Add tests that low feature quality forces `no_trade` or
  `INSUFFICIENT_EVIDENCE`.

Acceptance criteria:

- [x] A feature column existing is not enough for strategy selection.
- [x] Data problems fail closed with auditable reason codes.

Implemented on 2026-05-22 JST:

- Added `freqtrade_ext/bot_factory/feature_quality.py` to build and persist
  `feature_quality_report.json` with missing-rate, staleness, outlier, recent
  gap, classifier confidence, and cost-model freshness checks.
- Selector candidates declare feature quality thresholds. Missing or failing
  feature quality evidence blocks runtime selection with auditable reason codes.

## P2: Selector Ranking Improvements

- [x] Extend ranking beyond stress-cost PnL:
  - exposure ratio;
  - capital efficiency;
  - average holding time;
  - max drawdown;
  - downside deviation;
  - hold baseline delta;
  - no-trade opportunity cost;
  - incumbent delta;
  - pair/calendar concentration.
- [x] Predeclare selector scoring weights per regime.
- [x] Reject candidates whose advantage comes from one pair or one narrow
  calendar window.
- [x] Add reports that show selected and rejected candidates side by side.

Acceptance criteria:

- [x] Selector chooses a candidate for a clear, auditable reason beyond highest
  raw return.
- [x] The ranking does not reward capital being locked in weak strategies.

Implemented on 2026-05-22 JST:

- Extended `freqtrade_ext/bot_factory/candidate_ranking.py` with predeclared
  selector scoring weights by regime, exposure/capital efficiency/holding-time
  and drawdown-aware metrics, hold/no-trade/incumbent deltas, and pair/calendar
  concentration blockers.
- Ranking reports now show selected and rejected candidates side by side with
  reason codes and scoring weights.

## P2: Shadow Observation Compatibility

- [x] Keep current accepted observation sources limited to:
  - `backtest`;
  - `walk_forward`;
  - `local_shadow_replay`.
- [x] Continue rejecting future paper/dry-run observation source types until a
  later phase explicitly permits them.
- [x] Draft the future observation requirements for multiple dry-run/paper
  candidates, but keep startup out of scope.
- [x] Add leaderboards that separate:
  - long-term historical evidence;
  - current-regime evidence;
  - recent observation evidence;
  - data-quality confidence.

Acceptance criteria:

- [x] Future shadow observation cannot override failed historical/readiness
  evidence.
- [x] Parallel observations remain input evidence, not direct promotion.

Implemented on 2026-05-22 JST:

- `build_shadow_observation_leaderboards()` keeps accepted sources limited to
  `backtest`, `walk_forward`, and `local_shadow_replay`, continues rejecting
  future paper/dry-run sources by default, and separates long-term historical,
  current-regime, recent observation, and data-quality confidence buckets.
- The shadow leaderboard explicitly records that historical readiness override
  and direct promotion from parallel observations are not allowed.

## P2: Reporting And Review UX

- [x] Add a single candidate review report that joins:
  - identity;
  - strategy source hash/path;
  - historical metrics;
  - walk-forward metrics;
  - observation ledger summary;
  - regime scorecard;
  - hold/no-trade baseline comparison;
  - selector decision;
  - paper readiness blockers.
- [x] Make reason codes first-class and stable.
- [x] Add a compact architecture diagram to generated reports.
- [x] Add a "what changed since last candidate version" section.

Acceptance criteria:

- [x] A reviewer can decide whether a candidate should be iterated, shadowed,
  quarantined, or rejected from one report.

Implemented on 2026-05-22 JST:

- Added `freqtrade_ext/bot_factory/candidate_review.py` and
  `scripts/bot_factory_generate_candidate_review.py` to join identity,
  strategy source hash/path, historical and walk-forward metrics, observation
  ledger, scorecard, hold/no-trade baseline comparison, selector artifacts,
  paper readiness blockers, reason codes, a compact architecture diagram, and a
  "what changed since last candidate version" section.

## Recommended Implementation Order

1. Implement canonical candidate identity and validation.
2. Implement backtest-to-observation-to-scorecard conversion.
3. Implement deterministic regime classifier.
4. Connect real strategy artifacts to selector candidates.
5. Add style-aware gates and hold/no-trade baselines.
6. Add selector cooldown, hysteresis, and feature quality checks.
7. Connect regime scorecard requirements into Phase 3 paper readiness.
8. Improve ranking, reports, and future shadow-observation design.

## Current Known Examples

- `DonchianTrendBullStrategy` is now tied to `strong_uptrend_momentum_v1` by
  the canonical `StrategyCandidateIdentity`
  `strong-uptrend-historical-ohlcv-candidate`. The deterministic converter now
  exists; older historical artifacts need to be regenerated or converted with
  `scripts/bot_factory_build_regime_artifacts.py`.
- `strong_uptrend_momentum_v1`, `downtrend_defensive_rebound_v1`, and
  `range_mean_reversion_v1` are selector logic specs. Backtest-derived
  selector candidates can now be built from checked local metrics/trades/OHLCV
  artifacts, while existing synthetic examples remain local test fixtures.
- The historical uptrend replay proved selector behavior for a selected window,
  but it used close-to-close proxy evidence and relaxed calendar concentration;
  it should be superseded by converter-produced scorecards for promotion
  review.
