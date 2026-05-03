# Paper Run Plan Checklist

## Summary

- Strategy: LongOnlyFreqAIStrategy
- Run ID: phase3_paper_plan_20260503
- Status: blocked
- Readiness: fail
- Future startup eligible: False

## Gates

- PASS: readiness_source_is_phase3_paper_readiness - Paper plans must consume a Phase 3 paper readiness report.
- PASS: readiness_strategy_matches - Readiness report strategy must match the paper plan candidate.
- BLOCKED: readiness_passed - Readiness report must be pass before any paper run can be planned.
- PASS: readiness_has_no_blockers - Readiness report must have no blockers.
- BLOCKED: readiness_has_no_failures - Readiness report must have no failed gate checks.
- PASS: readiness_no_startup_scope - Readiness evidence must be from a no-startup preflight.
- PASS: readiness_no_live_or_exchange_order_scope - Readiness evidence must not involve live trading or exchange order placement.
- PASS: readiness_metadata_sanitized - Readiness metadata must be sanitized and must not contain secrets.
- PASS: readiness_long_only_scope - Readiness scope must remain long-only with no leverage above 1.0.
- PASS: readiness_local_artifacts_source_of_truth - Readiness report must keep local artifacts as the source of truth.
- PASS: config_path_within_workspace - Paper plan config path must resolve inside the repository workspace.
- PASS: strategy_path_within_workspace - Paper plan strategy path must resolve inside the repository workspace.
- PASS: config_file_present - Paper plan requires the same dry-run config file used by readiness.
- BLOCKED: confirm_paper_acknowledged - Paper plan requires explicit --confirm-paper acknowledgement.
- PASS: reviewer_note_present - Paper plan requires at least one reviewer note.

## Reviewer Notes

- Phase 3 paper run planning hardening check only; do not start paper trading.

## Startup Boundary

- Paper run planning is a no-startup gate. It does not start paper trading, dry-run trading, live trading, freqtrade trade, or any bot process.
- This plan never authorizes startup by itself.
- A separate explicit user request is required before any future paper start.
- Stop and cleanup instructions must be reviewed before any start procedure.
