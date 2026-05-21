# Bot Factory Architecture Risk TODO

Status: architecture follow-up TODO for regime-aware strategy selection.

Created: 2026-05-21 JST.

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
- [ ] Require the same identity object to be embedded in:
  - [x] generated strategy metadata;
  - [x] historical backtest outputs written by the checked backtest wrappers;
  - [ ] walk-forward outputs;
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

- [ ] A reviewer can trace a selected runtime candidate back to exact strategy
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
- Added candidate-identity lineage checks to observation validation, regime
  scorecard construction, runtime selector candidate evaluation, and candidate
  evaluation manifests.
- Bound `DonchianTrendBullStrategy` to
  `strong_uptrend_momentum_v1` through the same canonical identity object:
  `strong-uptrend-historical-ohlcv-candidate`.

Remaining limitation: existing historical artifacts produced before this
increment are not regenerated in-place; new checked wrapper outputs carry the
identity object. Walk-forward output embedding remains the next P0 lineage gap.

## P0: Backtest To Observation To Scorecard Pipeline

- [ ] Add a deterministic converter:

  ```text
  backtest metrics + trades + OHLCV regime labels
    -> observation ledger rows
    -> regime fitness scorecard
    -> selector candidate
  ```

- [ ] Record one observation row per candidate, pair, timeframe, window, and
  market regime.
- [ ] Require local baselines in the same converter:
  - `no_trade`;
  - buy-and-hold / hold baseline;
  - incumbent candidate when available.
- [ ] Reject manually assembled scorecards from becoming selector inputs unless
  they explicitly record `manual_review_only=true`.
- [ ] Add a CLI or helper that writes:
  - `observation_ledger.json`;
  - `regime_fitness_scorecard.json`;
  - `regime_fitness_scorecard_report.md`;
  - `selector_candidate.json`.

Acceptance criteria:

- [ ] A real Freqtrade backtest can produce selector-eligibility artifacts
  without hand-building observations.
- [ ] The scorecard clearly states whether the strategy beat hold/no-trade after
  normal and stress costs.

## P0: Deterministic Market Regime Classifier

- [ ] Implement a local-only regime classifier over historical candles.
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
- [ ] Use predeclared features:
  - rolling return;
  - realized volatility;
  - trend strength / slope;
  - range efficiency;
  - drawdown from local high;
  - volume/liquidity proxy;
  - candle gap/spread proxy when available;
  - data quality flags.
- [ ] Emit `unknown` when data quality is poor or classifier confidence is low.
- [ ] Persist `regime_classifier_version`.
- [ ] Add churn tests so small candle perturbations do not flip labels
  excessively.
- [ ] Add tests for deterministic classification on fixed OHLCV fixtures.

Acceptance criteria:

- [ ] Runtime selector tests no longer need hand-authored `trend_up` labels for
  historical windows.
- [ ] Strategy suitability is evaluated against reproducible regime artifacts.

## P1: Promotion Gate Semantics Cleanup

- [ ] Rename or document each gate outcome so it cannot be confused with
  paper/live approval:
  - `initial_backtest_gate.pass`
  - `eligible_for_walk_forward_review`
  - `REGIME_SCOPED_SELECTOR_ELIGIBLE`
  - `GLOBAL_SELECTOR_ELIGIBLE`
  - `paper_readiness.pass`
- [ ] Add a shared gate glossary in docs.
- [ ] Ensure every report includes:
  - what the gate permits;
  - what it does not permit;
  - next required gate.
- [ ] Add tests or static assertions that regime scorecards always include:
  - `raw_aggregate_pnl_promotion_allowed=false`;
  - `phase3_readiness_required_after_scorecard=true`;
  - `promotion_authorized_by_this_command=false`.
- [ ] Add paper-readiness checks that require regime scorecard artifacts when a
  strategy claims regime-scoped eligibility.

Acceptance criteria:

- [ ] No local artifact can be interpreted as paper/live approval by name alone.
- [ ] Paper readiness fails closed when required regime evidence is missing.

## P1: Strategy-Type-Specific Gates

- [ ] Split gate thresholds by candidate style:
  - scalp / high-frequency;
  - intraday trend-following;
  - swing trend-following;
  - range mean-reversion;
  - defensive/no-trade policy.
- [ ] Replace one global `min_trades >= 200` rule with style-aware minimums and
  sample-size requirements.
- [ ] Add hold-baseline gates for trend-following candidates:
  - fail or shadow-only when return is below hold and drawdown improvement does
    not justify the opportunity cost;
  - pass only when risk-adjusted return or capital efficiency is predeclared and
    defensible.
- [ ] Add no-trade opportunity-cost gates for defensive policies.
- [ ] Add tests showing a low-trade trend strategy is not rejected solely by a
  scalping-style trade-count threshold.

Acceptance criteria:

- [ ] `DonchianTrendBullStrategy`-style candidates are evaluated against trend
  strategy rules, not a generic scalping threshold.
- [ ] A strategy that wins only by underperforming hold is not promoted without
  an explicit risk-reduction rationale.

## P1: Runtime Selector Stability

- [ ] Add cooldown after regime change.
- [ ] Add hysteresis so small regime-confidence changes do not cause rapid
  strategy switching.
- [ ] Add minimum confidence threshold per regime.
- [ ] Add `unknown` / low-confidence fallback to `no_trade`.
- [ ] Track last selected candidate, last selected regime, and last switch
  reason in local selector state artifacts.
- [ ] Add selector simulation tests for:
  - `trend_up -> range -> trend_up` churn;
  - missing features;
  - classifier version mismatch;
  - pair/timeframe mismatch;
  - process-control flags accidentally enabled.

Acceptance criteria:

- [ ] Selector decisions are stable across noisy boundary conditions.
- [ ] Unsafe or uncertain runtime state returns `no_trade`.

## P1: Feature Quality And Data Confidence

- [ ] Extend runtime checks beyond feature presence:
  - missing rate;
  - stale feature timestamp;
  - outlier count;
  - recent data gaps;
  - classifier confidence;
  - cost model freshness.
- [ ] Add `feature_quality_report.json`.
- [ ] Require selector candidates to declare feature quality thresholds.
- [ ] Add tests that low feature quality forces `no_trade` or
  `INSUFFICIENT_EVIDENCE`.

Acceptance criteria:

- [ ] A feature column existing is not enough for strategy selection.
- [ ] Data problems fail closed with auditable reason codes.

## P2: Selector Ranking Improvements

- [ ] Extend ranking beyond stress-cost PnL:
  - exposure ratio;
  - capital efficiency;
  - average holding time;
  - max drawdown;
  - downside deviation;
  - hold baseline delta;
  - no-trade opportunity cost;
  - incumbent delta;
  - pair/calendar concentration.
- [ ] Predeclare selector scoring weights per regime.
- [ ] Reject candidates whose advantage comes from one pair or one narrow
  calendar window.
- [ ] Add reports that show selected and rejected candidates side by side.

Acceptance criteria:

- [ ] Selector chooses a candidate for a clear, auditable reason beyond highest
  raw return.
- [ ] The ranking does not reward capital being locked in weak strategies.

## P2: Shadow Observation Compatibility

- [ ] Keep current accepted observation sources limited to:
  - `backtest`;
  - `walk_forward`;
  - `local_shadow_replay`.
- [ ] Continue rejecting future paper/dry-run observation source types until a
  later phase explicitly permits them.
- [ ] Draft the future observation requirements for multiple dry-run/paper
  candidates, but keep startup out of scope.
- [ ] Add leaderboards that separate:
  - long-term historical evidence;
  - current-regime evidence;
  - recent observation evidence;
  - data-quality confidence.

Acceptance criteria:

- [ ] Future shadow observation cannot override failed historical/readiness
  evidence.
- [ ] Parallel observations remain input evidence, not direct promotion.

## P2: Reporting And Review UX

- [ ] Add a single candidate review report that joins:
  - identity;
  - strategy source hash/path;
  - historical metrics;
  - walk-forward metrics;
  - observation ledger summary;
  - regime scorecard;
  - hold/no-trade baseline comparison;
  - selector decision;
  - paper readiness blockers.
- [ ] Make reason codes first-class and stable.
- [ ] Add a compact architecture diagram to generated reports.
- [ ] Add a "what changed since last candidate version" section.

Acceptance criteria:

- [ ] A reviewer can decide whether a candidate should be iterated, shadowed,
  quarantined, or rejected from one report.

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
  `strong-uptrend-historical-ohlcv-candidate`. The remaining gap is the
  deterministic converter from real backtest artifacts into observations,
  scorecards, and selector candidates.
- `strong_uptrend_momentum_v1`, `downtrend_defensive_rebound_v1`, and
  `range_mean_reversion_v1` are selector logic specs, but they are not yet
  automatically generated from or bound to real strategy backtest artifacts.
- The historical uptrend replay proved selector behavior for a selected window,
  but it used close-to-close proxy evidence and relaxed calendar concentration.
