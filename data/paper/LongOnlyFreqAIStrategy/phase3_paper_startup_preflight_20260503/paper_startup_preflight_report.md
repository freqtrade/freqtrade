# Paper Startup Preflight Report

## Summary

- Strategy: LongOnlyFreqAIStrategy
- Run ID: phase3_paper_startup_preflight_20260503
- Status: blocked
- Plan status: blocked
- Startup eligible after preflight: False
- Startup executed: False

## Checks

- PASS: paper_plan_source_is_phase3_paper_run_plan - Startup preflight must consume a Phase 3 paper run plan.
- PASS: paper_plan_strategy_matches - Paper run plan strategy must match the startup preflight candidate.
- BLOCKED: paper_plan_ready - Paper run plan must be ready before startup preflight can pass.
- BLOCKED: paper_plan_has_no_blockers - Paper run plan must have no blockers.
- BLOCKED: paper_plan_future_startup_eligible - Paper run plan future startup eligibility must be true.
- PASS: paper_plan_requires_separate_user_request - Paper run plan must require a separate explicit user request.
- PASS: paper_plan_requires_stop_cleanup_first - Paper run plan must require stop and cleanup review before startup.
- PASS: paper_plan_does_not_authorize_startup - Paper run plan must not authorize startup by itself.
- BLOCKED: paper_plan_has_start_command_preview - Paper run plan must include a freqtrade trade command preview.
- BLOCKED: paper_plan_start_command_uses_freqtrade_trade - Paper run plan command preview must use freqtrade trade.
- BLOCKED: paper_plan_start_command_has_required_options - Paper run plan command preview must include one config, strategy, and strategy path.
- BLOCKED: paper_plan_start_command_config_matches_plan - Paper run plan command preview config must match a local existing plan config.
- BLOCKED: paper_plan_start_command_strategy_matches_candidate - Paper run plan command preview strategy must match the startup candidate.
- BLOCKED: paper_plan_start_command_strategy_path_matches_plan - Paper run plan command preview strategy path must match a local plan path.
- BLOCKED: confirm_paper_start_acknowledged - Startup preflight requires explicit --confirm-paper-start acknowledgement.
- BLOCKED: requested_start_command_present - Startup preflight requires the exact requested start command string.
- BLOCKED: requested_start_command_matches_plan - Requested start command must exactly match the paper run plan preview.
- PASS: reviewer_note_present - Startup preflight requires at least one reviewer note.
- PASS: stop_cleanup_artifact_within_workspace - Stop and cleanup documentation path must resolve inside the repository workspace.
- PASS: stop_cleanup_artifact_present - Stop and cleanup documentation must exist before startup preflight can pass.
- PASS: paper_run_checklist_within_workspace - Paper run checklist path must resolve inside the repository workspace.
- PASS: paper_run_checklist_present - Paper run checklist must exist before startup preflight can pass.
- PASS: paper_plan_no_startup_scope - Paper run plan must record no startup execution.
- PASS: paper_plan_no_live_or_exchange_order_scope - Paper run plan must not involve live trading or exchange order placement.
- PASS: paper_plan_no_secrets_leverage_or_shorting_scope - Paper run plan metadata must remain sanitized and long-only.
- PASS: paper_plan_local_artifacts_source_of_truth - Paper run plan must keep local artifacts as the source of truth.

## Process Metadata Design

- stdout log: `data\paper\LongOnlyFreqAIStrategy\phase3_paper_startup_preflight_20260503\logs\stdout.log`
- stderr log: `data\paper\LongOnlyFreqAIStrategy\phase3_paper_startup_preflight_20260503\logs\stderr.log`
- status snapshot: `data\paper\LongOnlyFreqAIStrategy\phase3_paper_startup_preflight_20260503\status_snapshot_template.json`
- paper metrics: `data\paper\LongOnlyFreqAIStrategy\phase3_paper_startup_preflight_20260503\paper_metrics.json`

## Reviewer Notes

- Phase 3 paper startup preflight hardening check only; do not start paper trading.

## Startup Boundary

- Paper startup preflight is a no-startup gate. It does not start freqtrade trade, paper trading, dry-run trading, live trading, or any bot process.
- This preflight records paths and templates only.
- A later explicit execution step is required before any process can start.
