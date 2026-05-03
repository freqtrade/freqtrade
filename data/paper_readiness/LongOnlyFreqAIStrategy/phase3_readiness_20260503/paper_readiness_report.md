# Paper Readiness Report

## Summary

- Strategy: LongOnlyFreqAIStrategy
- Run ID: phase3_readiness_20260503
- Readiness: fail
- Historical artifacts: `data\freqai\LongOnlyFreqAIStrategy\phase2_safe_20250105_20250107`
- Walk-forward artifacts: `data\walk_forward\LongOnlyFreqAIStrategy\phase2_walk_forward_20250105_20250109`
- Training artifacts: `data\freqai_training\LongOnlyFreqAIStrategy\phase2_training_20250105_20250107`

## Checks

- PASS: artifact_present_historical_metrics - Required candidate artifact must exist: historical_metrics.
- PASS: artifact_present_historical_report - Required candidate artifact must exist: historical_report.
- PASS: artifact_present_historical_metadata - Required candidate artifact must exist: historical_metadata.
- PASS: artifact_present_historical_trades - Required candidate artifact must exist: historical_trades.
- PASS: artifact_present_walk_forward_metrics - Required candidate artifact must exist: walk_forward_metrics.
- PASS: artifact_present_walk_forward_report - Required candidate artifact must exist: walk_forward_report.
- PASS: artifact_present_training_manifest - Required candidate artifact must exist: training_manifest.
- PASS: artifact_present_training_report - Required candidate artifact must exist: training_report.
- PASS: dry_run_true - Config must set dry_run=true for any future paper path.
- PASS: strategy_matches_candidate - Config strategy must match the readiness candidate.
- PASS: timeframe_explicit - Config must define an explicit timeframe.
- PASS: max_open_trades_explicit - Config must define a positive max_open_trades limit.
- PASS: max_open_trades_conservative - Config max_open_trades must stay within accepted simulation limits.
- PASS: stake_amount_capped - Config must define a positive numeric stake_amount cap.
- PASS: stake_amount_conservative - Config stake_amount must stay within accepted simulation limits.
- PASS: dry_run_wallet_conservative - Config must define a conservative positive dry_run_wallet.
- PASS: stake_amount_within_dry_run_wallet - Config stake_amount must not exceed dry_run_wallet.
- PASS: pair_allowlist_explicit - Config must define an explicit non-empty exchange.pair_whitelist.
- PASS: api_server_disabled - Remote API server must not be enabled for this no-startup readiness layer.
- PASS: force_entry_disabled - Config must not enable force_entry_enable for paper readiness.
- PASS: initial_state_stopped - Config initial_state must be explicitly stopped.
- PASS: cancel_open_orders_on_exit_explicit - Config must explicitly set cancel_open_orders_on_exit.
- PASS: no_credential_values - Config must not contain non-empty API keys, secrets, tokens, UIDs, or passwords.
- PASS: no_private_env_values - Config must not contain private environment variable references.
- PASS: no_leverage_above_one - Config must not set leverage above 1.0.
- PASS: no_order_endpoint_overrides - Config must not include private or order endpoint overrides.
- PASS: can_short_false - Strategy must explicitly set can_short = False.
- PASS: no_short_signals - Strategy must not reference enter_short or exit_short signals.
- PASS: leverage_hook_absent_or_capped - No leverage hook was found.
- PASS: static_strategy_check - Static strategy safety check must pass without errors.
- FAIL: historical_backtest_gate - Historical backtest gate must pass before paper readiness can pass.
- PASS: historical_trades_no_shorts - Historical exported trades must not contain short trades.
- PASS: historical_trades_no_leverage_above_one - Historical exported trades must not contain leverage above 1.0.
- FAIL: walk_forward_gate - Walk-forward recommendation must pass before paper readiness can pass.
- PASS: walk_forward_child_windows_present - Walk-forward metrics must include child window evidence.
- PASS: walk_forward_wf_01_20250105_20250107_metrics_present - Walk-forward window child artifact must exist: metrics.
- PASS: walk_forward_wf_01_20250105_20250107_trades_present - Walk-forward window child artifact must exist: trades.
- PASS: walk_forward_wf_01_20250105_20250107_freqai_metadata_present - Walk-forward window child artifact must exist: freqai_metadata.
- PASS: walk_forward_wf_01_20250105_20250107_trades_no_shorts - Walk-forward window exported trades must not contain short trades.
- PASS: walk_forward_wf_01_20250105_20250107_trades_no_leverage_above_one - Walk-forward window exported trades must not contain leverage above 1.0.
- PASS: walk_forward_wf_02_20250107_20250109_metrics_present - Walk-forward window child artifact must exist: metrics.
- PASS: walk_forward_wf_02_20250107_20250109_trades_present - Walk-forward window child artifact must exist: trades.
- PASS: walk_forward_wf_02_20250107_20250109_freqai_metadata_present - Walk-forward window child artifact must exist: freqai_metadata.
- PASS: walk_forward_wf_02_20250107_20250109_trades_no_shorts - Walk-forward window exported trades must not contain short trades.
- PASS: walk_forward_wf_02_20250107_20250109_trades_no_leverage_above_one - Walk-forward window exported trades must not contain leverage above 1.0.
- FAIL: training_factory_gate - Training factory recommendation must pass before paper readiness can pass.
- PASS: training_child_stages_present - Training manifest must include child stage evidence.
- PASS: training_freqai_backtest_child_present - Training manifest must include a freqai_backtest child stage.
- PASS: training_train_20250105_20250107_metrics_present - Training child artifact must exist: metrics.
- PASS: training_train_20250105_20250107_trades_present - Training child artifact must exist: trades.
- PASS: training_train_20250105_20250107_freqai_metadata_present - Training child artifact must exist: freqai_metadata.
- PASS: training_train_20250105_20250107_trades_no_shorts - Training child exported trades must not contain short trades.
- PASS: training_train_20250105_20250107_trades_no_leverage_above_one - Training child exported trades must not contain leverage above 1.0.
- PASS: reviewer_note_present - At least one explicit reviewer note is required before paper readiness can pass.

## Reviewer Notes

- Phase 3 no-startup paper readiness check only; do not start paper trading.

## Notes

- Paper readiness is a no-startup preflight. It does not start paper trading, dry-run trading, live trading, or any bot process.
- Failed Phase 2 gates block paper readiness.
- A future human-approved infrastructure-only smoke test is a separate path.
- Local JSON, CSV, and Markdown artifacts remain the source of truth.
