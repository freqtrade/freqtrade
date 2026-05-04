from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence


PAPER_STOP_CLEANUP_NOTICE = (
    "Paper stop and cleanup planning is a no-process-control gate. It writes "
    "future stop request and cleanup review artifacts only; it does not start, "
    "stop, poll, terminate, or manage any bot process."
)


@dataclass(frozen=True)
class PaperStopCleanupPlanInputs:
    root_dir: Path
    strategy: str
    run_id: str
    monitoring_plan_path: Path
    output_root: Path = Path("data/paper")
    reviewer_notes: Sequence[str] = field(default_factory=list)
    command: Sequence[str] = field(default_factory=list)

    @property
    def output_dir(self) -> Path:
        return self.output_root / self.strategy / self.run_id


@dataclass(frozen=True)
class PaperStopCleanupPlanCheck:
    name: str
    status: str
    severity: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_paper_monitoring_plan(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Paper monitoring plan JSON must contain an object: {path}")
    return payload


def build_paper_stop_cleanup_plan(
    inputs: PaperStopCleanupPlanInputs, monitoring_plan: dict[str, Any]
) -> dict[str, Any]:
    checks: list[PaperStopCleanupPlanCheck] = []
    monitoring = _dict_or_empty(monitoring_plan.get("monitoring"))
    planned_paths = _dict_or_empty(monitoring_plan.get("planned_paths"))
    schemas = _dict_or_empty(monitoring_plan.get("schemas"))
    safety_scope = _dict_or_empty(monitoring_plan.get("safety_scope"))

    process_metadata_path = _path_from_payload(
        planned_paths.get("process_metadata"), inputs.root_dir
    )
    status_snapshot_path = _path_from_payload(
        planned_paths.get("status_snapshot"), inputs.root_dir
    )
    stdout_path = _path_from_payload(planned_paths.get("stdout"), inputs.root_dir)
    stderr_path = _path_from_payload(planned_paths.get("stderr"), inputs.root_dir)
    paper_metrics_path = _path_from_payload(
        planned_paths.get("paper_metrics"), inputs.root_dir
    )

    checks.append(
        _check(
            "monitoring_plan_source_is_phase3_paper_monitoring_plan",
            monitoring_plan.get("phase") == "3"
            and monitoring_plan.get("factory") == "paper_monitoring_plan",
            "blocker",
            "Stop and cleanup planning must consume a Phase 3 paper monitoring plan.",
            {
                "phase": _safe_scalar(monitoring_plan.get("phase")),
                "factory": _safe_scalar(monitoring_plan.get("factory")),
            },
        )
    )
    checks.append(
        _check(
            "monitoring_plan_strategy_matches",
            monitoring_plan.get("strategy") == inputs.strategy,
            "blocker",
            "Monitoring plan strategy must match the stop and cleanup candidate.",
            {
                "monitoring_strategy": _safe_scalar(monitoring_plan.get("strategy")),
                "candidate": inputs.strategy,
            },
        )
    )
    checks.append(
        _check(
            "monitoring_plan_ready",
            monitoring_plan.get("status") == "ready",
            "blocker",
            "Monitoring plan must be ready before stop and cleanup planning can be ready.",
            {"monitoring_status": _safe_scalar(monitoring_plan.get("status"))},
        )
    )
    checks.append(
        _check(
            "monitoring_plan_has_no_blockers",
            not monitoring_plan.get("blockers"),
            "blocker",
            "Monitoring plan must have no blockers.",
            {"blocker_count": len(monitoring_plan.get("blockers") or [])},
        )
    )
    checks.append(
        _check(
            "monitoring_plan_eligible",
            monitoring.get("eligible") is True,
            "blocker",
            "Monitoring plan eligibility must be true before stop and cleanup planning can be ready.",
            {"eligible": monitoring.get("eligible")},
        )
    )
    checks.append(
        _check(
            "monitoring_plan_no_process_control",
            monitoring.get("monitoring_started") is False
            and monitoring.get("status_polling_started") is False
            and monitoring.get("process_control") is False
            and monitoring.get("process_stop_started") is False,
            "blocker",
            "Stop and cleanup planning can only consume a no-process-control monitoring plan.",
            {
                "monitoring_started": monitoring.get("monitoring_started"),
                "status_polling_started": monitoring.get("status_polling_started"),
                "process_control": monitoring.get("process_control"),
                "process_stop_started": monitoring.get("process_stop_started"),
            },
        )
    )
    checks.append(
        _check(
            "monitoring_plan_requires_runtime_artifacts",
            monitoring.get("requires_started_process_metadata") is True
            and monitoring.get("requires_status_snapshot_path") is True
            and monitoring.get("requires_stdout_stderr_logs") is True,
            "blocker",
            "Monitoring plan must require runtime metadata, status snapshots, and logs.",
            {
                "requires_started_process_metadata": monitoring.get(
                    "requires_started_process_metadata"
                ),
                "requires_status_snapshot_path": monitoring.get(
                    "requires_status_snapshot_path"
                ),
                "requires_stdout_stderr_logs": monitoring.get(
                    "requires_stdout_stderr_logs"
                ),
            },
        )
    )
    checks.append(
        _check(
            "process_metadata_path_within_workspace",
            process_metadata_path is not None
            and _path_is_within_root(process_metadata_path, inputs.root_dir),
            "blocker",
            "Process metadata path must resolve inside the repository workspace.",
            {"path": _safe_relative_path(process_metadata_path, inputs.root_dir)},
        )
    )
    checks.append(
        _check(
            "process_metadata_template_present",
            process_metadata_path is not None and process_metadata_path.is_file(),
            "blocker",
            "Process metadata template must exist before stop and cleanup planning can be ready.",
            {"path": _safe_relative_path(process_metadata_path, inputs.root_dir)},
        )
    )
    checks.append(
        _check(
            "status_snapshot_path_within_workspace",
            status_snapshot_path is not None
            and _path_is_within_root(status_snapshot_path, inputs.root_dir),
            "blocker",
            "Status snapshot path must resolve inside the repository workspace.",
            {"path": _safe_relative_path(status_snapshot_path, inputs.root_dir)},
        )
    )
    checks.append(
        _check(
            "status_snapshot_template_present",
            status_snapshot_path is not None and status_snapshot_path.is_file(),
            "blocker",
            "Status snapshot template must exist before stop and cleanup planning can be ready.",
            {"path": _safe_relative_path(status_snapshot_path, inputs.root_dir)},
        )
    )
    checks.append(
        _check(
            "stdout_log_path_within_workspace",
            stdout_path is not None and _path_is_within_root(stdout_path, inputs.root_dir),
            "blocker",
            "stdout log path must resolve inside the repository workspace.",
            {"path": _safe_relative_path(stdout_path, inputs.root_dir)},
        )
    )
    checks.append(
        _check(
            "stderr_log_path_within_workspace",
            stderr_path is not None and _path_is_within_root(stderr_path, inputs.root_dir),
            "blocker",
            "stderr log path must resolve inside the repository workspace.",
            {"path": _safe_relative_path(stderr_path, inputs.root_dir)},
        )
    )
    checks.append(
        _check(
            "paper_metrics_path_within_workspace",
            paper_metrics_path is not None
            and _path_is_within_root(paper_metrics_path, inputs.root_dir),
            "blocker",
            "Paper metrics path must resolve inside the repository workspace.",
            {"path": _safe_relative_path(paper_metrics_path, inputs.root_dir)},
        )
    )
    checks.extend(_monitoring_schema_checks(schemas))
    checks.extend(_monitoring_safety_scope_checks(safety_scope))
    checks.append(
        _check(
            "reviewer_note_present",
            bool(inputs.reviewer_notes),
            "blocker",
            "Stop and cleanup planning requires at least one reviewer note.",
            {"note_count": len(inputs.reviewer_notes)},
        )
    )

    status = "ready" if all(check.status == "pass" for check in checks) else "blocked"
    generated_at = datetime.now(UTC).isoformat()
    artifact_paths = _artifact_paths(inputs)
    stop_request_schema = build_stop_request_schema(inputs.strategy, inputs.run_id)

    return {
        "generated_at": generated_at,
        "phase": "3",
        "factory": "paper_stop_cleanup_plan",
        "status": status,
        "strategy": inputs.strategy,
        "run_id": inputs.run_id,
        "monitoring_plan_path": _safe_relative_path(
            inputs.monitoring_plan_path, inputs.root_dir
        ),
        "monitoring_summary": {
            "status": _safe_scalar(monitoring_plan.get("status")),
            "eligible": _safe_scalar(monitoring.get("eligible")),
            "monitoring_started": _safe_scalar(monitoring.get("monitoring_started")),
            "process_control": _safe_scalar(monitoring.get("process_control")),
            "monitoring_run_id": _safe_scalar(monitoring_plan.get("run_id")),
        },
        "checks": [check.to_dict() for check in checks],
        "blockers": [check.to_dict() for check in checks if check.status == "blocked"],
        "reviewer_notes": list(inputs.reviewer_notes),
        "stop_cleanup": {
            "eligible": status == "ready",
            "stop_executed": False,
            "cleanup_executed": False,
            "process_control": False,
            "process_stop_started": False,
            "status_polling_started": False,
            "requires_started_process_metadata": True,
            "requires_status_snapshot_before_stop": True,
            "requires_final_status_snapshot_after_stop": True,
            "requires_operator_review_before_future_stop": True,
            "stop_authorized_by_this_command": False,
            "cleanup_authorized_by_this_command": False,
            "deletes_source_artifacts": False,
        },
        "planned_paths": {
            "process_metadata": _safe_relative_path(
                process_metadata_path, inputs.root_dir
            ),
            "status_snapshot": _safe_relative_path(status_snapshot_path, inputs.root_dir),
            "stdout": _safe_relative_path(stdout_path, inputs.root_dir),
            "stderr": _safe_relative_path(stderr_path, inputs.root_dir),
            "paper_metrics": _safe_relative_path(paper_metrics_path, inputs.root_dir),
        },
        "schemas": {"stop_request": stop_request_schema},
        "artifact_paths": artifact_paths,
        "safety_scope": {
            "command": "paper stop and cleanup planning only",
            "bot_startup": False,
            "freqtrade_trade_executed": False,
            "paper_trading_started": False,
            "dry_run_trading_started": False,
            "live_trading": False,
            "canary_live_trading": False,
            "exchange_order_placement": False,
            "uses_api_keys_or_secrets": False,
            "leverage_above_one": False,
            "shorting": False,
            "metadata_contains_secrets": False,
            "local_artifacts_source_of_truth": True,
            "process_control": False,
            "process_stop_started": False,
            "status_polling_started": False,
        },
        "notice": PAPER_STOP_CLEANUP_NOTICE,
    }


def build_stop_request_schema(strategy: str, run_id: str) -> dict[str, Any]:
    return {
        "title": "Bot Factory Paper Stop Request",
        "type": "object",
        "strategy": strategy,
        "run_id": run_id,
        "required": [
            "generated_at",
            "strategy",
            "run_id",
            "source",
            "requested_action",
            "process_metadata_path",
            "status_snapshot_path",
            "paper_metrics_path",
            "stop_authorized_by_this_command",
            "stop_executed",
            "process_control",
            "reviewer_notes",
            "safety_scope",
        ],
        "properties": {
            "generated_at": {"type": "string"},
            "strategy": {"type": "string"},
            "run_id": {"type": "string"},
            "source": {"type": "string", "enum": ["local_paper_artifacts"]},
            "requested_action": {
                "type": "string",
                "enum": ["future_stop_review_only"],
            },
            "process_metadata_path": {"type": "string"},
            "status_snapshot_path": {"type": "string"},
            "paper_metrics_path": {"type": "string"},
            "stop_authorized_by_this_command": {"type": "boolean"},
            "stop_executed": {"type": "boolean"},
            "process_control": {"type": "boolean"},
            "reviewer_notes": {"type": "array", "items": {"type": "string"}},
            "safety_scope": {"type": "object"},
        },
        "safety_invariants": {
            "metadata_contains_secrets": False,
            "local_artifacts_source_of_truth": True,
            "live_trading": False,
            "exchange_order_placement": False,
            "process_control_by_schema": False,
            "stop_authorized_by_schema": False,
        },
    }


def write_paper_stop_cleanup_plan_artifacts(
    inputs: PaperStopCleanupPlanInputs, plan: dict[str, Any]
) -> None:
    output_dir = inputs.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "paper_stop_cleanup_plan.json").write_text(
        json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "stop_request_schema.json").write_text(
        json.dumps(plan["schemas"]["stop_request"], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "command.txt").write_text(" ".join(inputs.command), encoding="utf-8")
    write_paper_stop_cleanup_report(plan, output_dir / "paper_stop_cleanup_report.md")
    write_paper_cleanup_checklist(plan, output_dir / "cleanup_checklist.md")


def write_paper_stop_cleanup_report(plan: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    stop_cleanup = plan["stop_cleanup"]
    lines = [
        "# Paper Stop And Cleanup Plan",
        "",
        "## Summary",
        "",
        f"- Strategy: {plan['strategy']}",
        f"- Run ID: {plan['run_id']}",
        f"- Status: {plan['status']}",
        f"- Monitoring plan status: {plan['monitoring_summary']['status']}",
        f"- Stop cleanup eligible: {stop_cleanup['eligible']}",
        f"- Stop executed: {stop_cleanup['stop_executed']}",
        f"- Process control enabled: {stop_cleanup['process_control']}",
        "",
        "## Checks",
        "",
    ]
    for check in plan["checks"]:
        lines.append(f"- {check['status'].upper()}: {check['name']} - {check['message']}")

    lines.extend(
        [
            "",
            "## Planned Local Paths",
            "",
            f"- process metadata: `{plan['planned_paths']['process_metadata']}`",
            f"- status snapshot: `{plan['planned_paths']['status_snapshot']}`",
            f"- stdout log: `{plan['planned_paths']['stdout']}`",
            f"- stderr log: `{plan['planned_paths']['stderr']}`",
            f"- paper metrics: `{plan['planned_paths']['paper_metrics']}`",
            "",
            "## Reviewer Notes",
            "",
        ]
    )
    if plan["reviewer_notes"]:
        lines.extend(f"- {note}" for note in plan["reviewer_notes"])
    else:
        lines.append("- No reviewer notes were supplied.")

    lines.extend(
        [
            "",
            "## Stop And Cleanup Boundary",
            "",
            f"- {PAPER_STOP_CLEANUP_NOTICE}",
            "- This plan does not prove that a process exists or can be stopped.",
            "- A later explicit execution wrapper must validate live runtime metadata before stopping.",
            "- Local JSON, CSV, Markdown, and log artifacts remain the source of truth.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_paper_cleanup_checklist(plan: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Paper Cleanup Checklist",
        "",
        "This checklist is planning documentation only. No process was started, stopped, polled, or managed.",
        "",
        "## Before Any Future Stop",
        "",
        "- Confirm the paper stop and cleanup plan status is `ready`.",
        "- Confirm a future started process has local process metadata for the same strategy and run ID.",
        "- Confirm the latest local status snapshot targets the same process metadata.",
        "- Confirm stdout, stderr, status snapshot, and paper metrics paths are local workspace paths.",
        "- Confirm a separate explicit user request authorizes the exact future stop action.",
        "",
        "## Future Stop Review",
        "",
        "- Prefer the future wrapper's graceful stop path before any termination fallback.",
        "- Record a final local status snapshot after the process exits.",
        "- Preserve stdout, stderr, process metadata, status snapshots, and paper metrics.",
        "- Record whether stop was graceful, timed out, or required escalation.",
        "",
        "## Cleanup Boundaries",
        "",
        "- Do not delete source-of-truth JSON, CSV, Markdown, or log artifacts.",
        "- Do not write API keys, secrets, private environment values, or credentials.",
        "- Do not promote paper results to live or canary live without a later human-approved path.",
        "",
        f"- Plan status: {plan['status']}",
        f"- Strategy: {plan['strategy']}",
        f"- Run ID: {plan['run_id']}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _monitoring_schema_checks(
    schemas: dict[str, Any]
) -> list[PaperStopCleanupPlanCheck]:
    status_snapshot_schema = _dict_or_empty(schemas.get("status_snapshot"))
    paper_metrics_schema = _dict_or_empty(schemas.get("paper_metrics"))
    process_metadata_schema = _dict_or_empty(schemas.get("process_metadata"))

    return [
        _check(
            "status_snapshot_schema_has_stop_relevant_fields",
            _required_contains(
                status_snapshot_schema,
                {"status", "startup_executed", "freqtrade_trade_executed"},
            ),
            "blocker",
            "Status snapshot schema must include stop-relevant status and safety fields.",
            {"required": _safe_required(status_snapshot_schema)},
        ),
        _check(
            "paper_metrics_schema_has_trade_and_safety_fields",
            _required_contains(paper_metrics_schema, {"trade_counts", "safety_scope"}),
            "blocker",
            "Paper metrics schema must include trade counts and safety scope.",
            {"required": _safe_required(paper_metrics_schema)},
        ),
        _check(
            "process_metadata_schema_has_process_and_log_fields",
            _required_contains(
                process_metadata_schema,
                {"pid", "command", "stdout_log", "stderr_log", "status_snapshot"},
            ),
            "blocker",
            "Process metadata schema must include process identity and local log fields.",
            {"required": _safe_required(process_metadata_schema)},
        ),
    ]


def _monitoring_safety_scope_checks(
    safety_scope: dict[str, Any]
) -> list[PaperStopCleanupPlanCheck]:
    return [
        _check(
            "monitoring_plan_no_startup_scope",
            safety_scope.get("bot_startup") is False
            and safety_scope.get("freqtrade_trade_executed") is False
            and safety_scope.get("paper_trading_started") is False
            and safety_scope.get("dry_run_trading_started") is False,
            "blocker",
            "Monitoring plan must record no startup execution.",
            {
                "bot_startup": safety_scope.get("bot_startup"),
                "freqtrade_trade_executed": safety_scope.get("freqtrade_trade_executed"),
                "paper_trading_started": safety_scope.get("paper_trading_started"),
                "dry_run_trading_started": safety_scope.get("dry_run_trading_started"),
            },
        ),
        _check(
            "monitoring_plan_no_live_or_exchange_order_scope",
            safety_scope.get("live_trading") is False
            and safety_scope.get("exchange_order_placement") is False,
            "blocker",
            "Monitoring plan must not involve live trading or exchange order placement.",
            {
                "live_trading": safety_scope.get("live_trading"),
                "exchange_order_placement": safety_scope.get("exchange_order_placement"),
            },
        ),
        _check(
            "monitoring_plan_no_secrets_leverage_or_shorting_scope",
            safety_scope.get("uses_api_keys_or_secrets") is False
            and safety_scope.get("leverage_above_one") is False
            and safety_scope.get("shorting") is False
            and safety_scope.get("metadata_contains_secrets") is False,
            "blocker",
            "Monitoring plan metadata must remain sanitized and long-only.",
            {
                "uses_api_keys_or_secrets": safety_scope.get("uses_api_keys_or_secrets"),
                "leverage_above_one": safety_scope.get("leverage_above_one"),
                "shorting": safety_scope.get("shorting"),
                "metadata_contains_secrets": safety_scope.get("metadata_contains_secrets"),
            },
        ),
        _check(
            "monitoring_plan_local_artifacts_source_of_truth",
            safety_scope.get("local_artifacts_source_of_truth") is True,
            "blocker",
            "Monitoring plan must keep local artifacts as the source of truth.",
            {
                "local_artifacts_source_of_truth": safety_scope.get(
                    "local_artifacts_source_of_truth"
                )
            },
        ),
    ]


def _required_contains(schema: dict[str, Any], required: set[str]) -> bool:
    values = schema.get("required")
    if not isinstance(values, list):
        return False
    return required <= {str(value) for value in values}


def _safe_required(schema: dict[str, Any]) -> list[str]:
    values = schema.get("required")
    if not isinstance(values, list):
        return []
    return [str(value) for value in values]


def _artifact_paths(inputs: PaperStopCleanupPlanInputs) -> dict[str, str]:
    output_dir = inputs.output_dir
    return {
        "paper_stop_cleanup_plan": _safe_relative_path(
            output_dir / "paper_stop_cleanup_plan.json", inputs.root_dir
        ),
        "paper_stop_cleanup_report": _safe_relative_path(
            output_dir / "paper_stop_cleanup_report.md", inputs.root_dir
        ),
        "stop_request_schema": _safe_relative_path(
            output_dir / "stop_request_schema.json", inputs.root_dir
        ),
        "cleanup_checklist": _safe_relative_path(
            output_dir / "cleanup_checklist.md", inputs.root_dir
        ),
        "command": _safe_relative_path(output_dir / "command.txt", inputs.root_dir),
    }


def _dict_or_empty(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _path_from_payload(path_value: Any, root_dir: Path) -> Path | None:
    if not isinstance(path_value, str) or not path_value.strip():
        return None
    path = Path(path_value)
    if path.is_absolute():
        return path
    return root_dir / path


def _path_is_within_root(path: Path, root_dir: Path) -> bool:
    try:
        path.resolve().relative_to(root_dir.resolve())
        return True
    except ValueError:
        return False


def _check(
    name: str,
    passed: bool,
    severity: str,
    message: str,
    details: dict[str, Any] | None = None,
) -> PaperStopCleanupPlanCheck:
    return PaperStopCleanupPlanCheck(
        name=name,
        status="pass" if passed else "blocked",
        severity=severity,
        message=message,
        details=details or {},
    )


def _safe_scalar(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return type(value).__name__


def _safe_relative_path(path: Path | None, root_dir: Path) -> str | None:
    if path is None:
        return None
    try:
        return str(path.resolve().relative_to(root_dir.resolve()))
    except ValueError:
        return path.name
