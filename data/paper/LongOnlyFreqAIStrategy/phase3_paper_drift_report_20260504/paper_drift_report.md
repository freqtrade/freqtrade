# Paper/Backtest Drift Report

## Summary

- Strategy: LongOnlyFreqAIStrategy
- Run ID: phase3_paper_drift_report_20260504
- Status: blocked
- Historical total return pct: -0.061699
- Paper total return pct: n/a
- Return drift pct points: n/a
- Historical max drawdown pct: 0.061699
- Paper max drawdown pct: n/a
- Drawdown drift pct points: n/a
- Promotion authorized by this command: False

## Checks

- PASS: historical_metrics_within_workspace_and_present - Historical metrics path must resolve inside the workspace and exist locally.
- PASS: walk_forward_metrics_within_workspace_and_present - Walk-forward metrics path must resolve inside the workspace and exist locally.
- PASS: training_manifest_within_workspace_and_present - Training manifest path must resolve inside the workspace and exist locally.
- PASS: paper_runtime_validation_within_workspace_and_present - Paper runtime validation path must resolve inside the workspace and exist locally.
- BLOCKED: paper_metrics_within_workspace_and_present - Paper metrics path must resolve inside the workspace and exist locally.
- PASS: historical_strategy_matches - Historical metrics strategy must match the drift report candidate.
- PASS: walk_forward_source_is_phase2_completed - Walk-forward metrics must be a completed Phase 2 artifact.
- PASS: walk_forward_strategy_matches - Walk-forward metrics strategy must match the drift report candidate.
- PASS: training_source_is_phase2_freqai_training - Training manifest must be a completed Phase 2 FreqAI training artifact.
- PASS: training_strategy_matches - Training manifest strategy must match the drift report candidate.
- PASS: runtime_validation_source_is_phase3_paper_runtime_validation - Paper drift reporting must consume a Phase 3 paper runtime validation artifact.
- PASS: runtime_validation_strategy_matches - Runtime validation strategy must match the drift report candidate.
- BLOCKED: runtime_validation_passed - Paper runtime validation must pass before drift can be evaluated.
- PASS: paper_metrics_path_matches_runtime_validation - Paper metrics path must match the artifact consumed by runtime validation.
- BLOCKED: paper_metrics_source_is_local - Paper metrics must use local paper artifacts as source.
- BLOCKED: paper_metrics_strategy_matches - Paper metrics strategy must match the drift report candidate.
- BLOCKED: paper_metrics_run_id_matches_runtime_validation - Paper metrics run ID must match the runtime validation executor plan run ID.
- FAIL: walk_forward_recommendation_passed - Walk-forward recommendation must pass before drift reporting can support promotion review.
- FAIL: training_recommendation_passed - Training recommendation must pass before drift reporting can support promotion review.
- PASS: historical_return_metric_present - Historical total_return_pct must be numeric.
- PASS: historical_drawdown_metric_present - Historical max_drawdown_pct must be numeric.
- PASS: walk_forward_return_metric_present - Walk-forward summary total_return_pct must be numeric.
- PASS: walk_forward_drawdown_metric_present - Walk-forward summary max_drawdown_pct_any_window must be numeric.
- BLOCKED: paper_return_metric_present - Paper metrics total return percentage must be numeric.
- BLOCKED: paper_drawdown_metric_present - Paper metrics max drawdown percentage must be numeric.
- BLOCKED: paper_trade_count_metric_present - Paper metrics trade count must be a non-negative integer.
- FAIL: paper_trade_count_positive - Paper metrics must include at least one trade before drift can support review.
- FAIL: paper_return_not_worse_than_historical_threshold - Paper total return drift versus historical backtest must stay within threshold.
- FAIL: paper_return_not_worse_than_walk_forward_threshold - Paper total return drift versus walk-forward evidence must stay within threshold.
- FAIL: paper_drawdown_not_worse_than_historical_threshold - Paper max drawdown drift versus historical backtest must stay within threshold.
- FAIL: paper_drawdown_not_worse_than_walk_forward_threshold - Paper max drawdown drift versus walk-forward max drawdown must stay within threshold.
- PASS: runtime_validation_no_process_control_scope - Runtime validation safety scope must record no process control by the validator.
- BLOCKED: paper_metrics_safe_scope - Paper metrics safety scope must be sanitized, long-only, and local-artifact based.
- BLOCKED: paper_metrics_no_process_control_scope - Paper metrics must not record process control, polling, stop, or cleanup execution.
- PASS: drift_inputs_no_credential_values - Drift input metadata must not contain non-empty API keys, secrets, tokens, UIDs, or passwords.
- PASS: drift_inputs_no_private_env_references - Drift input metadata must not contain private environment variable references.
- PASS: reviewer_note_present - Paper/backtest drift reporting requires at least one reviewer note.

## Input Artifacts

- historical metrics: `data\freqai\LongOnlyFreqAIStrategy\phase2_safe_20250105_20250107\metrics.json`
- walk-forward metrics: `data\walk_forward\LongOnlyFreqAIStrategy\phase2_walk_forward_20250105_20250109\walk_forward_metrics.json`
- training manifest: `data\freqai_training\LongOnlyFreqAIStrategy\phase2_training_20250105_20250107\training_manifest.json`
- paper runtime validation: `data\paper\LongOnlyFreqAIStrategy\phase3_paper_runtime_validation_20260504\paper_runtime_validation.json`
- paper metrics: `data\paper\LongOnlyFreqAIStrategy\phase3_paper_startup_preflight_20260503\paper_metrics.json`

## Reviewer Notes

- Phase 3 paper/backtest drift reporting path-integrity hardening only; do not start, stop, poll, terminate, clean up, promote, or manage paper trading.

## Reporting Boundary

- Paper/backtest drift reporting is a no-process-control artifact analysis. It reads only supplied local historical, walk-forward, training, runtime validation, and paper metric JSON artifacts; it does not start, stop, poll, terminate, clean up, promote, or manage freqtrade trade, paper trading, dry-run trading, live trading, or any bot process.
- This report is not a promotion approval.
- This report does not verify process liveness outside supplied local artifacts.
- Local JSON, CSV, Markdown, and log artifacts remain the source of truth.
