from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence


PAPER_EXECUTION_REQUEST_NOTICE = (
    "Paper start execution request is a no-startup, no-process-control gate. "
    "It records a reviewed future start request only; it does not start "
    "freqtrade trade, paper trading, dry-run trading, live trading, stop, poll, "
    "terminate, clean up, or manage any bot process."
)


@dataclass(frozen=True)
class PaperExecutionRequestInputs:
    root_dir: Path
    strategy: str
    run_id: str
    readiness_path: Path
    plan_path: Path
    startup_preflight_path: Path
    monitoring_plan_path: Path
    stop_cleanup_plan_path: Path
    output_root: Path = Path("data/paper")
    reviewer_notes: Sequence[str] = field(default_factory=list)
    confirm_paper_execution: bool = False
    requested_start_command: str | None = None
    command: Sequence[str] = field(default_factory=list)

    @property
    def output_dir(self) -> Path:
        return self.output_root / self.strategy / self.run_id


@dataclass(frozen=True)
class PaperExecutionRequestCheck:
    name: str
    status: str
    severity: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_paper_execution_artifact(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Paper execution artifact JSON must contain an object: {path}")
    return payload


def build_paper_execution_request(
    inputs: PaperExecutionRequestInputs,
    readiness: dict[str, Any],
    paper_plan: dict[str, Any],
    startup_preflight: dict[str, Any],
    monitoring_plan: dict[str, Any],
    stop_cleanup_plan: dict[str, Any],
) -> dict[str, Any]:
    checks: list[PaperExecutionRequestCheck] = []

    readiness_status = readiness.get("readiness") or readiness.get("status")
    readiness_scope = _dict_or_empty(readiness.get("safety_scope"))
    future_startup = _dict_or_empty(paper_plan.get("future_startup"))
    plan_scope = _dict_or_empty(paper_plan.get("safety_scope"))
    startup = _dict_or_empty(startup_preflight.get("startup"))
    startup_scope = _dict_or_empty(startup_preflight.get("safety_scope"))
    process_metadata = _dict_or_empty(startup_preflight.get("process_metadata"))
    status_snapshot = _dict_or_empty(startup_preflight.get("status_snapshot"))
    log_paths = _dict_or_empty(startup_preflight.get("log_paths"))
    startup_artifacts = _dict_or_empty(startup_preflight.get("artifact_paths"))
    monitoring = _dict_or_empty(monitoring_plan.get("monitoring"))
    monitoring_scope = _dict_or_empty(monitoring_plan.get("safety_scope"))
    monitoring_planned_paths = _dict_or_empty(monitoring_plan.get("planned_paths"))
    stop_cleanup = _dict_or_empty(stop_cleanup_plan.get("stop_cleanup"))
    stop_cleanup_scope = _dict_or_empty(stop_cleanup_plan.get("safety_scope"))
    stop_cleanup_planned_paths = _dict_or_empty(stop_cleanup_plan.get("planned_paths"))

    plan_command_preview = _command_preview_from_payload(future_startup.get("command_preview"))
    startup_command_preview = _command_preview_from_payload(startup.get("command_preview"))
    expected_command = _command_string(startup_command_preview)
    requested_command = (inputs.requested_start_command or "").strip()

    process_metadata_path = _path_from_payload(
        startup_artifacts.get("process_metadata_template"), inputs.root_dir
    )
    status_snapshot_path = _path_from_payload(
        startup_artifacts.get("status_snapshot_template"), inputs.root_dir
    )
    stdout_path = _path_from_payload(log_paths.get("stdout"), inputs.root_dir)
    stderr_path = _path_from_payload(log_paths.get("stderr"), inputs.root_dir)
    paper_metrics_path = _path_from_payload(log_paths.get("paper_metrics"), inputs.root_dir)

    checks.extend(_readiness_checks(inputs, readiness, readiness_status, readiness_scope))
    checks.extend(_paper_plan_checks(inputs, paper_plan, future_startup, plan_scope))
    checks.extend(
        _startup_preflight_checks(
            inputs,
            startup_preflight,
            startup,
            startup_scope,
            process_metadata,
            status_snapshot,
        )
    )
    checks.extend(_monitoring_plan_checks(inputs, monitoring_plan, monitoring, monitoring_scope))
    checks.extend(
        _stop_cleanup_plan_checks(inputs, stop_cleanup_plan, stop_cleanup, stop_cleanup_scope)
    )
    checks.extend(
        _runtime_path_checks(
            inputs,
            process_metadata_path=process_metadata_path,
            status_snapshot_path=status_snapshot_path,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            paper_metrics_path=paper_metrics_path,
            monitoring_planned_paths=monitoring_planned_paths,
            stop_cleanup_planned_paths=stop_cleanup_planned_paths,
        )
    )

    checks.append(
        _check(
            "paper_plan_and_startup_commands_match",
            bool(plan_command_preview)
            and bool(startup_command_preview)
            and plan_command_preview == startup_command_preview,
            "blocker",
            "Paper run plan and startup preflight must agree on the exact command preview.",
            {
                "plan_command": _command_string(plan_command_preview) or None,
                "startup_command": expected_command or None,
            },
        )
    )
    checks.append(
        _check(
            "confirm_paper_execution_acknowledged",
            inputs.confirm_paper_execution,
            "blocker",
            "Paper execution request requires explicit --confirm-paper-execution acknowledgement.",
        )
    )
    checks.append(
        _check(
            "requested_start_command_present",
            bool(requested_command),
            "blocker",
            "Paper execution request requires the exact requested start command string.",
        )
    )
    checks.append(
        _check(
            "requested_start_command_matches_preflight",
            bool(requested_command) and requested_command == expected_command,
            "blocker",
            "Requested start command must exactly match the startup preflight preview.",
            {
                "requested_start_command": requested_command or None,
                "expected_start_command": expected_command or None,
            },
        )
    )
    checks.append(
        _check(
            "reviewer_note_present",
            bool(inputs.reviewer_notes),
            "blocker",
            "Paper execution request requires at least one reviewer note.",
            {"note_count": len(inputs.reviewer_notes)},
        )
    )

    status = "ready" if all(check.status == "pass" for check in checks) else "blocked"
    generated_at = datetime.now(UTC).isoformat()
    artifact_paths = _artifact_paths(inputs)
    planned_paths = {
        "process_metadata": _safe_relative_path(process_metadata_path, inputs.root_dir),
        "status_snapshot": _safe_relative_path(status_snapshot_path, inputs.root_dir),
        "stdout": _safe_relative_path(stdout_path, inputs.root_dir),
        "stderr": _safe_relative_path(stderr_path, inputs.root_dir),
        "paper_metrics": _safe_relative_path(paper_metrics_path, inputs.root_dir),
    }
    start_command_request = startup_command_preview if status == "ready" else []
    execution_manifest = {
        "strategy": inputs.strategy,
        "run_id": inputs.run_id,
        "startup_executed": False,
        "process_started": False,
        "process_control": False,
        "status_polling_started": False,
        "process_stop_started": False,
        "cleanup_executed": False,
        "startup_authorized_by_this_command": False,
        "requires_separate_process_executor": True,
        "command": start_command_request,
        "process_metadata": planned_paths["process_metadata"],
        "status_snapshot": planned_paths["status_snapshot"],
        "stdout_log": planned_paths["stdout"],
        "stderr_log": planned_paths["stderr"],
        "paper_metrics": planned_paths["paper_metrics"],
        "notice": PAPER_EXECUTION_REQUEST_NOTICE,
    }

    return {
        "generated_at": generated_at,
        "phase": "3",
        "factory": "paper_execution_request",
        "status": status,
        "strategy": inputs.strategy,
        "run_id": inputs.run_id,
        "input_paths": {
            "readiness": _safe_relative_path(inputs.readiness_path, inputs.root_dir),
            "paper_run_plan": _safe_relative_path(inputs.plan_path, inputs.root_dir),
            "paper_startup_preflight": _safe_relative_path(
                inputs.startup_preflight_path, inputs.root_dir
            ),
            "paper_monitoring_plan": _safe_relative_path(
                inputs.monitoring_plan_path, inputs.root_dir
            ),
            "paper_stop_cleanup_plan": _safe_relative_path(
                inputs.stop_cleanup_plan_path, inputs.root_dir
            ),
        },
        "summaries": {
            "readiness": _safe_scalar(readiness_status),
            "paper_run_plan": _safe_scalar(paper_plan.get("status")),
            "startup_preflight": _safe_scalar(startup_preflight.get("status")),
            "monitoring_plan": _safe_scalar(monitoring_plan.get("status")),
            "stop_cleanup_plan": _safe_scalar(stop_cleanup_plan.get("status")),
        },
        "checks": [check.to_dict() for check in checks],
        "blockers": [check.to_dict() for check in checks if check.status == "blocked"],
        "reviewer_notes": list(inputs.reviewer_notes),
        "requested_start_command": requested_command or None,
        "expected_start_command": expected_command or None,
        "execution_request": {
            "eligible": status == "ready",
            "startup_executed": False,
            "process_started": False,
            "process_control": False,
            "status_polling_started": False,
            "process_stop_started": False,
            "cleanup_executed": False,
            "startup_authorized_by_this_command": False,
            "requires_separate_process_executor": True,
            "command_preview": start_command_request,
        },
        "planned_paths": planned_paths,
        "execution_manifest": execution_manifest,
        "artifact_paths": artifact_paths,
        "safety_scope": {
            "command": "paper execution request only",
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
            "status_polling_started": False,
            "process_stop_started": False,
            "cleanup_executed": False,
        },
        "notice": PAPER_EXECUTION_REQUEST_NOTICE,
    }


def write_paper_execution_request_artifacts(
    inputs: PaperExecutionRequestInputs, request: dict[str, Any]
) -> None:
    output_dir = inputs.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "paper_execution_request.json").write_text(
        json.dumps(request, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "execution_manifest_template.json").write_text(
        json.dumps(request["execution_manifest"], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "start_command_request.txt").write_text(
        _command_string(request["execution_request"]["command_preview"]), encoding="utf-8"
    )
    (output_dir / "command.txt").write_text(" ".join(inputs.command), encoding="utf-8")
    write_paper_execution_request_report(
        request, output_dir / "paper_execution_request_report.md"
    )


def write_paper_execution_request_report(request: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    execution = request["execution_request"]
    lines = [
        "# Paper Start Execution Request",
        "",
        "## Summary",
        "",
        f"- Strategy: {request['strategy']}",
        f"- Run ID: {request['run_id']}",
        f"- Status: {request['status']}",
        f"- Readiness: {request['summaries']['readiness']}",
        f"- Paper run plan: {request['summaries']['paper_run_plan']}",
        f"- Startup preflight: {request['summaries']['startup_preflight']}",
        f"- Monitoring plan: {request['summaries']['monitoring_plan']}",
        f"- Stop cleanup plan: {request['summaries']['stop_cleanup_plan']}",
        f"- Execution request eligible: {execution['eligible']}",
        f"- Startup executed: {execution['startup_executed']}",
        f"- Process control enabled: {execution['process_control']}",
        "",
        "## Checks",
        "",
    ]
    for check in request["checks"]:
        lines.append(f"- {check['status'].upper()}: {check['name']} - {check['message']}")

    lines.extend(
        [
            "",
            "## Planned Local Paths",
            "",
            f"- process metadata: `{request['planned_paths']['process_metadata']}`",
            f"- status snapshot: `{request['planned_paths']['status_snapshot']}`",
            f"- stdout log: `{request['planned_paths']['stdout']}`",
            f"- stderr log: `{request['planned_paths']['stderr']}`",
            f"- paper metrics: `{request['planned_paths']['paper_metrics']}`",
            "",
            "## Reviewer Notes",
            "",
        ]
    )
    if request["reviewer_notes"]:
        lines.extend(f"- {note}" for note in request["reviewer_notes"])
    else:
        lines.append("- No reviewer notes were supplied.")

    lines.extend(
        [
            "",
            "## Execution Boundary",
            "",
            f"- {PAPER_EXECUTION_REQUEST_NOTICE}",
            "- This request does not prove that a process exists or can start.",
            "- A later explicit process executor would still need to start and record runtime metadata.",
            "- Local JSON, CSV, Markdown, and log artifacts remain the source of truth.",
            "",
        ]
    )
    if execution["command_preview"]:
        lines.extend(["## Start Command Request", "", "```powershell"])
        lines.append(_command_string(execution["command_preview"]))
        lines.extend(["```", ""])

    path.write_text("\n".join(lines), encoding="utf-8")


def _readiness_checks(
    inputs: PaperExecutionRequestInputs,
    readiness: dict[str, Any],
    readiness_status: Any,
    safety_scope: dict[str, Any],
) -> list[PaperExecutionRequestCheck]:
    return [
        _check(
            "readiness_source_is_phase3_paper_readiness",
            readiness.get("phase") == "3" and readiness.get("factory") == "paper_readiness",
            "blocker",
            "Execution request must consume a Phase 3 paper readiness report.",
            {
                "phase": _safe_scalar(readiness.get("phase")),
                "factory": _safe_scalar(readiness.get("factory")),
            },
        ),
        _check(
            "readiness_strategy_matches",
            readiness.get("strategy") == inputs.strategy,
            "blocker",
            "Readiness report strategy must match the execution request candidate.",
            {"readiness_strategy": _safe_scalar(readiness.get("strategy"))},
        ),
        _check(
            "readiness_passed",
            readiness_status == "pass",
            "blocker",
            "Readiness must pass before a paper execution request can be ready.",
            {"readiness": _safe_scalar(readiness_status)},
        ),
        _check(
            "readiness_has_no_blockers",
            not readiness.get("blockers"),
            "blocker",
            "Readiness report must have no blockers.",
            {"blocker_count": len(readiness.get("blockers") or [])},
        ),
        _check(
            "readiness_has_no_failures",
            not readiness.get("failures"),
            "blocker",
            "Readiness report must have no failed gate checks.",
            {"failure_count": len(readiness.get("failures") or [])},
        ),
        *_readiness_safety_scope_checks(safety_scope),
    ]


def _paper_plan_checks(
    inputs: PaperExecutionRequestInputs,
    paper_plan: dict[str, Any],
    future_startup: dict[str, Any],
    safety_scope: dict[str, Any],
) -> list[PaperExecutionRequestCheck]:
    return [
        _check(
            "paper_plan_source_is_phase3_paper_run_plan",
            paper_plan.get("phase") == "3" and paper_plan.get("factory") == "paper_run_plan",
            "blocker",
            "Execution request must consume a Phase 3 paper run plan.",
            {
                "phase": _safe_scalar(paper_plan.get("phase")),
                "factory": _safe_scalar(paper_plan.get("factory")),
            },
        ),
        _check(
            "paper_plan_strategy_matches",
            paper_plan.get("strategy") == inputs.strategy,
            "blocker",
            "Paper run plan strategy must match the execution request candidate.",
            {"plan_strategy": _safe_scalar(paper_plan.get("strategy"))},
        ),
        _check(
            "paper_plan_ready",
            paper_plan.get("status") == "ready",
            "blocker",
            "Paper run plan must be ready before a paper execution request can be ready.",
            {"plan_status": _safe_scalar(paper_plan.get("status"))},
        ),
        _check(
            "paper_plan_has_no_blockers",
            not paper_plan.get("blockers"),
            "blocker",
            "Paper run plan must have no blockers.",
            {"blocker_count": len(paper_plan.get("blockers") or [])},
        ),
        _check(
            "paper_plan_readiness_path_matches_request",
            _payload_path_matches(paper_plan.get("readiness_path"), inputs.readiness_path, inputs.root_dir),
            "blocker",
            "Paper run plan must reference the same readiness artifact.",
            {"plan_readiness_path": _safe_scalar(paper_plan.get("readiness_path"))},
        ),
        _check(
            "paper_plan_future_startup_eligible",
            future_startup.get("eligible") is True,
            "blocker",
            "Paper run plan future startup eligibility must be true.",
            {"eligible": future_startup.get("eligible")},
        ),
        _check(
            "paper_plan_has_start_command_preview",
            len(_command_preview_from_payload(future_startup.get("command_preview"))) >= 2,
            "blocker",
            "Paper run plan must include a command preview.",
        ),
        _check(
            "paper_plan_requires_separate_user_request",
            future_startup.get("requires_separate_user_request") is True,
            "blocker",
            "Paper run plan must require a separate explicit user request.",
        ),
        _check(
            "paper_plan_does_not_authorize_startup",
            future_startup.get("startup_authorized_by_this_command") is False,
            "blocker",
            "Paper run plan must not authorize startup by itself.",
        ),
        *_no_execution_safety_scope_checks("paper_plan", safety_scope),
    ]


def _startup_preflight_checks(
    inputs: PaperExecutionRequestInputs,
    startup_preflight: dict[str, Any],
    startup: dict[str, Any],
    safety_scope: dict[str, Any],
    process_metadata: dict[str, Any],
    status_snapshot: dict[str, Any],
) -> list[PaperExecutionRequestCheck]:
    return [
        _check(
            "startup_preflight_source_is_phase3_paper_startup_preflight",
            startup_preflight.get("phase") == "3"
            and startup_preflight.get("factory") == "paper_startup_preflight",
            "blocker",
            "Execution request must consume a Phase 3 paper startup preflight.",
            {
                "phase": _safe_scalar(startup_preflight.get("phase")),
                "factory": _safe_scalar(startup_preflight.get("factory")),
            },
        ),
        _check(
            "startup_preflight_strategy_matches",
            startup_preflight.get("strategy") == inputs.strategy,
            "blocker",
            "Startup preflight strategy must match the execution request candidate.",
            {"preflight_strategy": _safe_scalar(startup_preflight.get("strategy"))},
        ),
        _check(
            "startup_preflight_ready",
            startup_preflight.get("status") == "ready",
            "blocker",
            "Startup preflight must be ready before a paper execution request can be ready.",
            {"preflight_status": _safe_scalar(startup_preflight.get("status"))},
        ),
        _check(
            "startup_preflight_has_no_blockers",
            not startup_preflight.get("blockers"),
            "blocker",
            "Startup preflight must have no blockers.",
            {"blocker_count": len(startup_preflight.get("blockers") or [])},
        ),
        _check(
            "startup_preflight_plan_path_matches_request",
            _payload_path_matches(startup_preflight.get("plan_path"), inputs.plan_path, inputs.root_dir),
            "blocker",
            "Startup preflight must reference the same paper run plan artifact.",
            {"preflight_plan_path": _safe_scalar(startup_preflight.get("plan_path"))},
        ),
        _check(
            "startup_preflight_startup_eligible",
            startup.get("eligible") is True,
            "blocker",
            "Startup preflight eligibility must be true.",
            {"eligible": startup.get("eligible")},
        ),
        _check(
            "startup_preflight_did_not_execute_startup",
            startup.get("startup_executed") is False
            and process_metadata.get("startup_executed") is False
            and process_metadata.get("process_started") is False
            and status_snapshot.get("startup_executed") is False,
            "blocker",
            "Execution request can only consume a no-startup preflight.",
            {
                "startup_executed": startup.get("startup_executed"),
                "process_started": process_metadata.get("process_started"),
                "status_snapshot_startup_executed": status_snapshot.get("startup_executed"),
            },
        ),
        _check(
            "startup_preflight_does_not_authorize_startup",
            startup.get("startup_authorized_by_this_command") is False,
            "blocker",
            "Startup preflight must not authorize startup by itself.",
        ),
        _check(
            "startup_preflight_requires_separate_execution",
            startup.get("requires_separate_execution_after_preflight") is True,
            "blocker",
            "Startup preflight must require a separate execution step.",
        ),
        *_no_execution_safety_scope_checks("startup_preflight", safety_scope),
    ]


def _monitoring_plan_checks(
    inputs: PaperExecutionRequestInputs,
    monitoring_plan: dict[str, Any],
    monitoring: dict[str, Any],
    safety_scope: dict[str, Any],
) -> list[PaperExecutionRequestCheck]:
    return [
        _check(
            "monitoring_plan_source_is_phase3_paper_monitoring_plan",
            monitoring_plan.get("phase") == "3"
            and monitoring_plan.get("factory") == "paper_monitoring_plan",
            "blocker",
            "Execution request must consume a Phase 3 paper monitoring plan.",
            {
                "phase": _safe_scalar(monitoring_plan.get("phase")),
                "factory": _safe_scalar(monitoring_plan.get("factory")),
            },
        ),
        _check(
            "monitoring_plan_strategy_matches",
            monitoring_plan.get("strategy") == inputs.strategy,
            "blocker",
            "Monitoring plan strategy must match the execution request candidate.",
            {"monitoring_strategy": _safe_scalar(monitoring_plan.get("strategy"))},
        ),
        _check(
            "monitoring_plan_ready",
            monitoring_plan.get("status") == "ready",
            "blocker",
            "Monitoring plan must be ready before a paper execution request can be ready.",
            {"monitoring_status": _safe_scalar(monitoring_plan.get("status"))},
        ),
        _check(
            "monitoring_plan_has_no_blockers",
            not monitoring_plan.get("blockers"),
            "blocker",
            "Monitoring plan must have no blockers.",
            {"blocker_count": len(monitoring_plan.get("blockers") or [])},
        ),
        _check(
            "monitoring_plan_startup_preflight_path_matches_request",
            _payload_path_matches(
                monitoring_plan.get("startup_preflight_path"),
                inputs.startup_preflight_path,
                inputs.root_dir,
            ),
            "blocker",
            "Monitoring plan must reference the same startup preflight artifact.",
            {
                "monitoring_startup_preflight_path": _safe_scalar(
                    monitoring_plan.get("startup_preflight_path")
                )
            },
        ),
        _check(
            "monitoring_plan_eligible",
            monitoring.get("eligible") is True,
            "blocker",
            "Monitoring plan eligibility must be true.",
            {"eligible": monitoring.get("eligible")},
        ),
        _check(
            "monitoring_plan_no_process_control",
            monitoring.get("monitoring_started") is False
            and monitoring.get("status_polling_started") is False
            and monitoring.get("process_control") is False
            and monitoring.get("process_stop_started") is False,
            "blocker",
            "Monitoring plan must remain no-process-control.",
            {
                "monitoring_started": monitoring.get("monitoring_started"),
                "status_polling_started": monitoring.get("status_polling_started"),
                "process_control": monitoring.get("process_control"),
                "process_stop_started": monitoring.get("process_stop_started"),
            },
        ),
        *_no_execution_safety_scope_checks("monitoring_plan", safety_scope),
    ]


def _stop_cleanup_plan_checks(
    inputs: PaperExecutionRequestInputs,
    stop_cleanup_plan: dict[str, Any],
    stop_cleanup: dict[str, Any],
    safety_scope: dict[str, Any],
) -> list[PaperExecutionRequestCheck]:
    return [
        _check(
            "stop_cleanup_plan_source_is_phase3_paper_stop_cleanup_plan",
            stop_cleanup_plan.get("phase") == "3"
            and stop_cleanup_plan.get("factory") == "paper_stop_cleanup_plan",
            "blocker",
            "Execution request must consume a Phase 3 paper stop and cleanup plan.",
            {
                "phase": _safe_scalar(stop_cleanup_plan.get("phase")),
                "factory": _safe_scalar(stop_cleanup_plan.get("factory")),
            },
        ),
        _check(
            "stop_cleanup_plan_strategy_matches",
            stop_cleanup_plan.get("strategy") == inputs.strategy,
            "blocker",
            "Stop and cleanup plan strategy must match the execution request candidate.",
            {"stop_cleanup_strategy": _safe_scalar(stop_cleanup_plan.get("strategy"))},
        ),
        _check(
            "stop_cleanup_plan_ready",
            stop_cleanup_plan.get("status") == "ready",
            "blocker",
            "Stop and cleanup plan must be ready before a paper execution request can be ready.",
            {"stop_cleanup_status": _safe_scalar(stop_cleanup_plan.get("status"))},
        ),
        _check(
            "stop_cleanup_plan_has_no_blockers",
            not stop_cleanup_plan.get("blockers"),
            "blocker",
            "Stop and cleanup plan must have no blockers.",
            {"blocker_count": len(stop_cleanup_plan.get("blockers") or [])},
        ),
        _check(
            "stop_cleanup_plan_monitoring_path_matches_request",
            _payload_path_matches(
                stop_cleanup_plan.get("monitoring_plan_path"),
                inputs.monitoring_plan_path,
                inputs.root_dir,
            ),
            "blocker",
            "Stop and cleanup plan must reference the same monitoring plan artifact.",
            {
                "stop_cleanup_monitoring_path": _safe_scalar(
                    stop_cleanup_plan.get("monitoring_plan_path")
                )
            },
        ),
        _check(
            "stop_cleanup_plan_eligible",
            stop_cleanup.get("eligible") is True,
            "blocker",
            "Stop and cleanup plan eligibility must be true.",
            {"eligible": stop_cleanup.get("eligible")},
        ),
        _check(
            "stop_cleanup_plan_no_process_control",
            stop_cleanup.get("stop_executed") is False
            and stop_cleanup.get("cleanup_executed") is False
            and stop_cleanup.get("process_control") is False
            and stop_cleanup.get("process_stop_started") is False
            and stop_cleanup.get("status_polling_started") is False,
            "blocker",
            "Stop and cleanup plan must remain no-process-control.",
            {
                "stop_executed": stop_cleanup.get("stop_executed"),
                "cleanup_executed": stop_cleanup.get("cleanup_executed"),
                "process_control": stop_cleanup.get("process_control"),
                "process_stop_started": stop_cleanup.get("process_stop_started"),
                "status_polling_started": stop_cleanup.get("status_polling_started"),
            },
        ),
        _check(
            "stop_cleanup_plan_preserves_review_guardrails",
            stop_cleanup.get("requires_operator_review_before_future_stop") is True
            and stop_cleanup.get("stop_authorized_by_this_command") is False
            and stop_cleanup.get("cleanup_authorized_by_this_command") is False
            and stop_cleanup.get("deletes_source_artifacts") is False,
            "blocker",
            "Stop and cleanup plan must preserve review and artifact-retention guardrails.",
            {
                "requires_operator_review_before_future_stop": stop_cleanup.get(
                    "requires_operator_review_before_future_stop"
                ),
                "stop_authorized_by_this_command": stop_cleanup.get(
                    "stop_authorized_by_this_command"
                ),
                "cleanup_authorized_by_this_command": stop_cleanup.get(
                    "cleanup_authorized_by_this_command"
                ),
                "deletes_source_artifacts": stop_cleanup.get("deletes_source_artifacts"),
            },
        ),
        *_stop_cleanup_safety_scope_checks(safety_scope),
    ]


def _runtime_path_checks(
    inputs: PaperExecutionRequestInputs,
    *,
    process_metadata_path: Path | None,
    status_snapshot_path: Path | None,
    stdout_path: Path | None,
    stderr_path: Path | None,
    paper_metrics_path: Path | None,
    monitoring_planned_paths: dict[str, Any],
    stop_cleanup_planned_paths: dict[str, Any],
) -> list[PaperExecutionRequestCheck]:
    path_checks = [
        _local_existing_path_check(
            "process_metadata_template",
            process_metadata_path,
            inputs.root_dir,
            "Process metadata template",
        ),
        _local_existing_path_check(
            "status_snapshot_template",
            status_snapshot_path,
            inputs.root_dir,
            "Status snapshot template",
        ),
        _local_path_check("stdout_log", stdout_path, inputs.root_dir, "stdout log"),
        _local_path_check("stderr_log", stderr_path, inputs.root_dir, "stderr log"),
        _local_path_check(
            "paper_metrics", paper_metrics_path, inputs.root_dir, "paper metrics"
        ),
    ]
    path_checks.extend(
        [
            _check(
                "monitoring_paths_match_startup_preflight",
                _payload_path_matches(
                    monitoring_planned_paths.get("process_metadata"),
                    process_metadata_path,
                    inputs.root_dir,
                )
                and _payload_path_matches(
                    monitoring_planned_paths.get("status_snapshot"),
                    status_snapshot_path,
                    inputs.root_dir,
                )
                and _payload_path_matches(
                    monitoring_planned_paths.get("stdout"), stdout_path, inputs.root_dir
                )
                and _payload_path_matches(
                    monitoring_planned_paths.get("stderr"), stderr_path, inputs.root_dir
                )
                and _payload_path_matches(
                    monitoring_planned_paths.get("paper_metrics"),
                    paper_metrics_path,
                    inputs.root_dir,
                ),
                "blocker",
                "Monitoring plan runtime paths must match startup preflight paths.",
            ),
            _check(
                "stop_cleanup_paths_match_monitoring_plan",
                _payload_path_matches(
                    stop_cleanup_planned_paths.get("process_metadata"),
                    process_metadata_path,
                    inputs.root_dir,
                )
                and _payload_path_matches(
                    stop_cleanup_planned_paths.get("status_snapshot"),
                    status_snapshot_path,
                    inputs.root_dir,
                )
                and _payload_path_matches(
                    stop_cleanup_planned_paths.get("stdout"), stdout_path, inputs.root_dir
                )
                and _payload_path_matches(
                    stop_cleanup_planned_paths.get("stderr"), stderr_path, inputs.root_dir
                )
                and _payload_path_matches(
                    stop_cleanup_planned_paths.get("paper_metrics"),
                    paper_metrics_path,
                    inputs.root_dir,
                ),
                "blocker",
                "Stop and cleanup plan runtime paths must match monitoring/startup paths.",
            ),
        ]
    )
    return path_checks


def _readiness_safety_scope_checks(
    safety_scope: dict[str, Any]
) -> list[PaperExecutionRequestCheck]:
    return [
        _check(
            "readiness_no_startup_scope",
            safety_scope.get("bot_startup") is False
            and safety_scope.get("freqtrade_trade") is False
            and safety_scope.get("paper_trading_started") is False
            and safety_scope.get("dry_run_trading_started") is False,
            "blocker",
            "Readiness report must remain no-startup.",
        ),
        _check(
            "readiness_no_live_or_exchange_order_scope",
            safety_scope.get("live_trading") is False
            and safety_scope.get("exchange_order_placement") is False,
            "blocker",
            "Readiness report must not involve live trading or exchange order placement.",
        ),
        _check(
            "readiness_no_secrets_leverage_or_shorting_scope",
            safety_scope.get("uses_api_keys_or_secrets") is False
            and safety_scope.get("leverage_above_one") is False
            and safety_scope.get("shorting") is False
            and safety_scope.get("metadata_contains_secrets") is False,
            "blocker",
            "Readiness report metadata must remain sanitized and long-only.",
        ),
        _check(
            "readiness_local_artifacts_source_of_truth",
            safety_scope.get("local_artifacts_source_of_truth") is True,
            "blocker",
            "Readiness report must keep local artifacts as the source of truth.",
        ),
    ]


def _no_execution_safety_scope_checks(
    prefix: str, safety_scope: dict[str, Any]
) -> list[PaperExecutionRequestCheck]:
    return [
        _check(
            f"{prefix}_no_startup_scope",
            safety_scope.get("bot_startup") is False
            and safety_scope.get("freqtrade_trade_executed") is False
            and safety_scope.get("paper_trading_started") is False
            and safety_scope.get("dry_run_trading_started") is False,
            "blocker",
            f"{prefix} must record no startup execution.",
        ),
        _check(
            f"{prefix}_no_live_or_exchange_order_scope",
            safety_scope.get("live_trading") is False
            and safety_scope.get("exchange_order_placement") is False,
            "blocker",
            f"{prefix} must not involve live trading or exchange order placement.",
        ),
        _check(
            f"{prefix}_no_secrets_leverage_or_shorting_scope",
            safety_scope.get("uses_api_keys_or_secrets") is False
            and safety_scope.get("leverage_above_one") is False
            and safety_scope.get("shorting") is False
            and safety_scope.get("metadata_contains_secrets") is False,
            "blocker",
            f"{prefix} metadata must remain sanitized and long-only.",
        ),
        _check(
            f"{prefix}_local_artifacts_source_of_truth",
            safety_scope.get("local_artifacts_source_of_truth") is True,
            "blocker",
            f"{prefix} must keep local artifacts as the source of truth.",
        ),
    ]


def _stop_cleanup_safety_scope_checks(
    safety_scope: dict[str, Any]
) -> list[PaperExecutionRequestCheck]:
    checks = _no_execution_safety_scope_checks("stop_cleanup_plan", safety_scope)
    checks.extend(
        [
            _check(
                "stop_cleanup_plan_safety_no_process_control",
                safety_scope.get("process_control") is False
                and safety_scope.get("process_stop_started") is False
                and safety_scope.get("status_polling_started") is False,
                "blocker",
                "Stop and cleanup safety scope must record no process control.",
            )
        ]
    )
    return checks


def _local_existing_path_check(
    name: str, path: Path | None, root_dir: Path, label: str
) -> PaperExecutionRequestCheck:
    return _check(
        f"{name}_within_workspace_and_present",
        path is not None and _path_is_within_root(path, root_dir) and path.is_file(),
        "blocker",
        f"{label} path must resolve inside the workspace and exist locally.",
        {"path": _safe_relative_path(path, root_dir)},
    )


def _local_path_check(
    name: str, path: Path | None, root_dir: Path, label: str
) -> PaperExecutionRequestCheck:
    return _check(
        f"{name}_within_workspace",
        path is not None and _path_is_within_root(path, root_dir),
        "blocker",
        f"{label} path must resolve inside the workspace.",
        {"path": _safe_relative_path(path, root_dir)},
    )


def _artifact_paths(inputs: PaperExecutionRequestInputs) -> dict[str, str]:
    output_dir = inputs.output_dir
    return {
        "paper_execution_request": _safe_relative_path(
            output_dir / "paper_execution_request.json", inputs.root_dir
        ),
        "paper_execution_request_report": _safe_relative_path(
            output_dir / "paper_execution_request_report.md", inputs.root_dir
        ),
        "execution_manifest_template": _safe_relative_path(
            output_dir / "execution_manifest_template.json", inputs.root_dir
        ),
        "start_command_request": _safe_relative_path(
            output_dir / "start_command_request.txt", inputs.root_dir
        ),
        "command": _safe_relative_path(output_dir / "command.txt", inputs.root_dir),
    }


def _dict_or_empty(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _command_preview_from_payload(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(token) for token in value if str(token)]


def _command_string(command: Sequence[Any]) -> str:
    return " ".join(str(token) for token in command).strip()


def _path_from_payload(path_value: Any, root_dir: Path) -> Path | None:
    if isinstance(path_value, Path):
        return path_value
    if not isinstance(path_value, str) or not path_value.strip():
        return None
    path = Path(path_value)
    if path.is_absolute():
        return path
    return root_dir / path


def _payload_path_matches(payload_value: Any, expected: Path | None, root_dir: Path) -> bool:
    payload_path = _path_from_payload(payload_value, root_dir)
    if payload_path is None or expected is None:
        return False
    return payload_path.resolve() == expected.resolve()


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
) -> PaperExecutionRequestCheck:
    return PaperExecutionRequestCheck(
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
