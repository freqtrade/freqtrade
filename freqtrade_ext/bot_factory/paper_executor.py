from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence


PAPER_PROCESS_EXECUTOR_NOTICE = (
    "Paper process executor planning is a no-startup, no-process-control gate. "
    "It records a reviewed executor manifest draft only; it does not start "
    "freqtrade trade, paper trading, dry-run trading, live trading, stop, poll, "
    "terminate, clean up, or manage any bot process."
)


@dataclass(frozen=True)
class PaperProcessExecutorPlanInputs:
    root_dir: Path
    strategy: str
    run_id: str
    execution_request_path: Path
    output_root: Path = Path("data/paper")
    reviewer_notes: Sequence[str] = field(default_factory=list)
    confirm_process_executor_plan: bool = False
    requested_start_command: str | None = None
    command: Sequence[str] = field(default_factory=list)

    @property
    def output_dir(self) -> Path:
        return self.output_root / self.strategy / self.run_id


@dataclass(frozen=True)
class PaperProcessExecutorPlanCheck:
    name: str
    status: str
    severity: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_paper_process_executor_artifact(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Paper process executor artifact JSON must contain an object: {path}")
    return payload


def build_paper_process_executor_plan(
    inputs: PaperProcessExecutorPlanInputs, execution_request: dict[str, Any]
) -> dict[str, Any]:
    checks: list[PaperProcessExecutorPlanCheck] = []
    request = _dict_or_empty(execution_request.get("execution_request"))
    request_scope = _dict_or_empty(execution_request.get("safety_scope"))
    request_manifest = _dict_or_empty(execution_request.get("execution_manifest"))
    request_artifacts = _dict_or_empty(execution_request.get("artifact_paths"))
    planned_paths_payload = _dict_or_empty(execution_request.get("planned_paths"))

    command_preview = _command_preview_from_payload(request.get("command_preview"))
    expected_command = _command_string(command_preview)
    requested_command = (inputs.requested_start_command or "").strip()
    request_expected_command = _safe_string(execution_request.get("expected_start_command"))
    request_requested_command = _safe_string(execution_request.get("requested_start_command"))
    manifest_command = _command_preview_from_payload(request_manifest.get("command"))

    process_metadata_path = _path_from_payload(
        planned_paths_payload.get("process_metadata"), inputs.root_dir
    )
    status_snapshot_path = _path_from_payload(
        planned_paths_payload.get("status_snapshot"), inputs.root_dir
    )
    stdout_path = _path_from_payload(planned_paths_payload.get("stdout"), inputs.root_dir)
    stderr_path = _path_from_payload(planned_paths_payload.get("stderr"), inputs.root_dir)
    paper_metrics_path = _path_from_payload(
        planned_paths_payload.get("paper_metrics"), inputs.root_dir
    )
    execution_manifest_template_path = _path_from_payload(
        request_artifacts.get("execution_manifest_template"), inputs.root_dir
    )
    start_command_request_path = _path_from_payload(
        request_artifacts.get("start_command_request"), inputs.root_dir
    )

    checks.extend(_execution_request_checks(inputs, execution_request, request))
    checks.extend(
        _command_checks(
            inputs=inputs,
            command_preview=command_preview,
            manifest_command=manifest_command,
            expected_command=expected_command,
            request_expected_command=request_expected_command,
            request_requested_command=request_requested_command,
            requested_command=requested_command,
        )
    )
    checks.extend(
        _runtime_path_checks(
            inputs,
            process_metadata_path=process_metadata_path,
            status_snapshot_path=status_snapshot_path,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            paper_metrics_path=paper_metrics_path,
            execution_manifest_template_path=execution_manifest_template_path,
            start_command_request_path=start_command_request_path,
        )
    )
    checks.extend(_manifest_checks(request_manifest, command_preview, planned_paths_payload))
    checks.extend(_execution_request_safety_scope_checks(request_scope))
    checks.append(
        _check(
            "confirm_process_executor_plan_acknowledged",
            inputs.confirm_process_executor_plan,
            "blocker",
            "Paper process executor planning requires explicit --confirm-process-executor-plan acknowledgement.",
        )
    )
    checks.append(
        _check(
            "reviewer_note_present",
            bool(inputs.reviewer_notes),
            "blocker",
            "Paper process executor planning requires at least one reviewer note.",
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
    executor_command = command_preview if status == "ready" else []
    executor_manifest = {
        "strategy": inputs.strategy,
        "run_id": inputs.run_id,
        "source_execution_request": _safe_relative_path(
            inputs.execution_request_path, inputs.root_dir
        ),
        "startup_executed": False,
        "process_started": False,
        "process_control": False,
        "status_polling_started": False,
        "process_stop_started": False,
        "cleanup_executed": False,
        "start_authorized_by_this_command": False,
        "requires_explicit_user_start_after_plan": True,
        "requires_ready_stop_cleanup_plan_before_start": True,
        "command": executor_command,
        "process_metadata": planned_paths["process_metadata"],
        "status_snapshot": planned_paths["status_snapshot"],
        "stdout_log": planned_paths["stdout"],
        "stderr_log": planned_paths["stderr"],
        "paper_metrics": planned_paths["paper_metrics"],
        "notice": PAPER_PROCESS_EXECUTOR_NOTICE,
    }

    return {
        "generated_at": generated_at,
        "phase": "3",
        "factory": "paper_process_executor_plan",
        "status": status,
        "strategy": inputs.strategy,
        "run_id": inputs.run_id,
        "execution_request_path": _safe_relative_path(
            inputs.execution_request_path, inputs.root_dir
        ),
        "execution_request_summary": {
            "status": _safe_scalar(execution_request.get("status")),
            "eligible": _safe_scalar(request.get("eligible")),
            "execution_request_run_id": _safe_scalar(execution_request.get("run_id")),
            "requires_separate_process_executor": _safe_scalar(
                request.get("requires_separate_process_executor")
            ),
        },
        "checks": [check.to_dict() for check in checks],
        "blockers": [check.to_dict() for check in checks if check.status == "blocked"],
        "reviewer_notes": list(inputs.reviewer_notes),
        "requested_start_command": requested_command or None,
        "expected_start_command": expected_command or None,
        "executor_plan": {
            "eligible": status == "ready",
            "startup_executed": False,
            "process_started": False,
            "process_control": False,
            "status_polling_started": False,
            "process_stop_started": False,
            "cleanup_executed": False,
            "start_authorized_by_this_command": False,
            "requires_explicit_user_start_after_plan": True,
            "requires_ready_stop_cleanup_plan_before_start": True,
            "command_preview": executor_command,
        },
        "planned_paths": planned_paths,
        "executor_manifest": executor_manifest,
        "artifact_paths": artifact_paths,
        "safety_scope": {
            "command": "paper process executor planning only",
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
        "notice": PAPER_PROCESS_EXECUTOR_NOTICE,
    }


def write_paper_process_executor_plan_artifacts(
    inputs: PaperProcessExecutorPlanInputs, plan: dict[str, Any]
) -> None:
    output_dir = inputs.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "paper_process_executor_plan.json").write_text(
        json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "process_executor_manifest.json").write_text(
        json.dumps(plan["executor_manifest"], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "start_command_review.txt").write_text(
        _command_string(plan["executor_plan"]["command_preview"]), encoding="utf-8"
    )
    (output_dir / "command.txt").write_text(" ".join(inputs.command), encoding="utf-8")
    write_paper_process_executor_report(
        plan, output_dir / "paper_process_executor_report.md"
    )
    write_operator_start_checklist(plan, output_dir / "operator_start_checklist.md")


def write_paper_process_executor_report(plan: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    executor = plan["executor_plan"]
    lines = [
        "# Paper Process Executor Plan",
        "",
        "## Summary",
        "",
        f"- Strategy: {plan['strategy']}",
        f"- Run ID: {plan['run_id']}",
        f"- Status: {plan['status']}",
        f"- Execution request status: {plan['execution_request_summary']['status']}",
        f"- Execution request eligible: {plan['execution_request_summary']['eligible']}",
        f"- Executor plan eligible: {executor['eligible']}",
        f"- Startup executed: {executor['startup_executed']}",
        f"- Process started: {executor['process_started']}",
        f"- Process control enabled: {executor['process_control']}",
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
            "## Executor Boundary",
            "",
            f"- {PAPER_PROCESS_EXECUTOR_NOTICE}",
            "- This plan does not prove that a process exists or can start.",
            "- A later explicit process executor would still need to start and record runtime metadata.",
            "- Local JSON, CSV, Markdown, and log artifacts remain the source of truth.",
            "",
        ]
    )
    if executor["command_preview"]:
        lines.extend(["## Reviewed Start Command", "", "```powershell"])
        lines.append(_command_string(executor["command_preview"]))
        lines.extend(["```", ""])

    path.write_text("\n".join(lines), encoding="utf-8")


def write_operator_start_checklist(plan: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Paper Operator Start Checklist",
        "",
        "This checklist is planning documentation only. No process was started, stopped, polled, or managed.",
        "",
        "## Required Before Any Future Start",
        "",
        "- Confirm the paper process executor plan status is `ready`.",
        "- Confirm the source paper execution request status is `ready` and still references the same strategy.",
        "- Confirm the reviewed start command exactly matches the execution request command.",
        "- Confirm process metadata, status snapshot, stdout, stderr, and paper metrics paths are local workspace paths.",
        "- Confirm stop and cleanup artifacts have been reviewed before startup.",
        "- Confirm a separate explicit user request authorizes the exact future start action.",
        "",
        "## Startup Boundaries",
        "",
        "- Do not use API keys, secrets, private environment values, or credential-bearing configs.",
        "- Do not start live trading, canary live trading, exchange order placement, leverage above 1.0, or shorting.",
        "- Preserve stdout, stderr, process metadata, status snapshots, paper metrics, and source-of-truth artifacts.",
        "",
        f"- Plan status: {plan['status']}",
        f"- Strategy: {plan['strategy']}",
        f"- Run ID: {plan['run_id']}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _execution_request_checks(
    inputs: PaperProcessExecutorPlanInputs,
    execution_request: dict[str, Any],
    request: dict[str, Any],
) -> list[PaperProcessExecutorPlanCheck]:
    return [
        _check(
            "execution_request_source_is_phase3_paper_execution_request",
            execution_request.get("phase") == "3"
            and execution_request.get("factory") == "paper_execution_request",
            "blocker",
            "Process executor planning must consume a Phase 3 paper execution request.",
            {
                "phase": _safe_scalar(execution_request.get("phase")),
                "factory": _safe_scalar(execution_request.get("factory")),
            },
        ),
        _check(
            "execution_request_strategy_matches",
            execution_request.get("strategy") == inputs.strategy,
            "blocker",
            "Execution request strategy must match the process executor candidate.",
            {"execution_request_strategy": _safe_scalar(execution_request.get("strategy"))},
        ),
        _check(
            "execution_request_ready",
            execution_request.get("status") == "ready",
            "blocker",
            "Execution request must be ready before process executor planning can be ready.",
            {"execution_request_status": _safe_scalar(execution_request.get("status"))},
        ),
        _check(
            "execution_request_has_no_blockers",
            not execution_request.get("blockers"),
            "blocker",
            "Execution request must have no blockers.",
            {"blocker_count": len(execution_request.get("blockers") or [])},
        ),
        _check(
            "execution_request_eligible",
            request.get("eligible") is True,
            "blocker",
            "Execution request eligibility must be true.",
            {"eligible": request.get("eligible")},
        ),
        _check(
            "execution_request_requires_separate_process_executor",
            request.get("requires_separate_process_executor") is True,
            "blocker",
            "Execution request must require a separate process executor.",
        ),
        _check(
            "execution_request_did_not_start_or_manage_process",
            request.get("startup_executed") is False
            and request.get("process_started") is False
            and request.get("process_control") is False
            and request.get("status_polling_started") is False
            and request.get("process_stop_started") is False
            and request.get("cleanup_executed") is False,
            "blocker",
            "Execution request must remain no-startup and no-process-control.",
            {
                "startup_executed": request.get("startup_executed"),
                "process_started": request.get("process_started"),
                "process_control": request.get("process_control"),
                "status_polling_started": request.get("status_polling_started"),
                "process_stop_started": request.get("process_stop_started"),
                "cleanup_executed": request.get("cleanup_executed"),
            },
        ),
        _check(
            "execution_request_does_not_authorize_startup",
            request.get("startup_authorized_by_this_command") is False,
            "blocker",
            "Execution request must not authorize startup by itself.",
        ),
    ]


def _command_checks(
    *,
    inputs: PaperProcessExecutorPlanInputs,
    command_preview: list[str],
    manifest_command: list[str],
    expected_command: str,
    request_expected_command: str | None,
    request_requested_command: str | None,
    requested_command: str,
) -> list[PaperProcessExecutorPlanCheck]:
    config_values = _option_values(command_preview, "--config")
    strategy_values = _option_values(command_preview, "--strategy")
    strategy_path_values = _option_values(command_preview, "--strategy-path")
    return [
        _check(
            "execution_request_has_start_command",
            len(command_preview) >= 2,
            "blocker",
            "Execution request must include a start command preview.",
            {"command_token_count": len(command_preview)},
        ),
        _check(
            "execution_manifest_command_matches_request",
            bool(command_preview) and manifest_command == command_preview,
            "blocker",
            "Execution manifest command must match the execution request command preview.",
        ),
        _check(
            "execution_request_start_command_uses_freqtrade_trade",
            len(command_preview) >= 2
            and _is_freqtrade_binary(command_preview[0])
            and command_preview[1] == "trade",
            "blocker",
            "Execution request command preview must use freqtrade trade.",
            {
                "executable": command_preview[0] if command_preview else None,
                "subcommand": command_preview[1] if len(command_preview) > 1 else None,
            },
        ),
        _check(
            "execution_request_start_command_has_required_options",
            len(config_values) == 1
            and len(strategy_values) == 1
            and len(strategy_path_values) == 1,
            "blocker",
            "Execution request command preview must include one config, strategy, and strategy path.",
            {
                "config_count": len(config_values),
                "strategy_count": len(strategy_values),
                "strategy_path_count": len(strategy_path_values),
            },
        ),
        _check(
            "execution_request_start_command_strategy_matches_candidate",
            len(strategy_values) == 1 and strategy_values[0] == inputs.strategy,
            "blocker",
            "Execution request command preview strategy must match the process executor candidate.",
            {
                "command_strategy": strategy_values[0] if len(strategy_values) == 1 else None,
                "candidate": inputs.strategy,
            },
        ),
        _check(
            "execution_request_expected_command_matches_preview",
            bool(expected_command)
            and request_expected_command == expected_command
            and request_requested_command == expected_command,
            "blocker",
            "Execution request expected/requested command strings must match the command preview.",
            {
                "expected_start_command": request_expected_command,
                "requested_start_command": request_requested_command,
                "command_preview": expected_command or None,
            },
        ),
        _check(
            "requested_start_command_present",
            bool(requested_command),
            "blocker",
            "Process executor planning requires the exact requested start command string.",
        ),
        _check(
            "requested_start_command_matches_execution_request",
            bool(requested_command) and requested_command == expected_command,
            "blocker",
            "Requested start command must exactly match the execution request command preview.",
            {
                "requested_start_command": requested_command or None,
                "expected_start_command": expected_command or None,
            },
        ),
    ]


def _runtime_path_checks(
    inputs: PaperProcessExecutorPlanInputs,
    *,
    process_metadata_path: Path | None,
    status_snapshot_path: Path | None,
    stdout_path: Path | None,
    stderr_path: Path | None,
    paper_metrics_path: Path | None,
    execution_manifest_template_path: Path | None,
    start_command_request_path: Path | None,
) -> list[PaperProcessExecutorPlanCheck]:
    return [
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
        _local_existing_path_check(
            "execution_manifest_template",
            execution_manifest_template_path,
            inputs.root_dir,
            "Execution manifest template",
        ),
        _local_existing_path_check(
            "start_command_request",
            start_command_request_path,
            inputs.root_dir,
            "Start command request",
        ),
    ]


def _manifest_checks(
    manifest: dict[str, Any],
    command_preview: list[str],
    planned_paths: dict[str, Any],
) -> list[PaperProcessExecutorPlanCheck]:
    return [
        _check(
            "execution_manifest_requires_separate_process_executor",
            manifest.get("requires_separate_process_executor") is True,
            "blocker",
            "Execution manifest must require a separate process executor.",
        ),
        _check(
            "execution_manifest_no_startup_or_process_control",
            manifest.get("startup_executed") is False
            and manifest.get("process_started") is False
            and manifest.get("process_control") is False
            and manifest.get("status_polling_started") is False
            and manifest.get("process_stop_started") is False
            and manifest.get("cleanup_executed") is False
            and manifest.get("startup_authorized_by_this_command") is False,
            "blocker",
            "Execution manifest template must remain no-startup and no-process-control.",
        ),
        _check(
            "execution_manifest_paths_match_request",
            manifest.get("process_metadata") == planned_paths.get("process_metadata")
            and manifest.get("status_snapshot") == planned_paths.get("status_snapshot")
            and manifest.get("stdout_log") == planned_paths.get("stdout")
            and manifest.get("stderr_log") == planned_paths.get("stderr")
            and manifest.get("paper_metrics") == planned_paths.get("paper_metrics"),
            "blocker",
            "Execution manifest paths must match the execution request planned paths.",
        ),
        _check(
            "execution_manifest_has_command_preview",
            bool(command_preview) and bool(_command_preview_from_payload(manifest.get("command"))),
            "blocker",
            "Execution manifest must include the reviewed command preview.",
        ),
    ]


def _execution_request_safety_scope_checks(
    safety_scope: dict[str, Any]
) -> list[PaperProcessExecutorPlanCheck]:
    return [
        _check(
            "execution_request_no_startup_scope",
            safety_scope.get("bot_startup") is False
            and safety_scope.get("freqtrade_trade_executed") is False
            and safety_scope.get("paper_trading_started") is False
            and safety_scope.get("dry_run_trading_started") is False,
            "blocker",
            "Execution request must record no startup execution.",
        ),
        _check(
            "execution_request_no_live_or_exchange_order_scope",
            safety_scope.get("live_trading") is False
            and safety_scope.get("exchange_order_placement") is False,
            "blocker",
            "Execution request must not involve live trading or exchange order placement.",
        ),
        _check(
            "execution_request_no_secrets_leverage_or_shorting_scope",
            safety_scope.get("uses_api_keys_or_secrets") is False
            and safety_scope.get("leverage_above_one") is False
            and safety_scope.get("shorting") is False
            and safety_scope.get("metadata_contains_secrets") is False,
            "blocker",
            "Execution request metadata must remain sanitized and long-only.",
        ),
        _check(
            "execution_request_local_artifacts_source_of_truth",
            safety_scope.get("local_artifacts_source_of_truth") is True,
            "blocker",
            "Execution request must keep local artifacts as the source of truth.",
        ),
        _check(
            "execution_request_no_process_control_scope",
            safety_scope.get("process_control") is False
            and safety_scope.get("status_polling_started") is False
            and safety_scope.get("process_stop_started") is False
            and safety_scope.get("cleanup_executed") is False,
            "blocker",
            "Execution request safety scope must record no process control.",
        ),
    ]


def _local_existing_path_check(
    name: str, path: Path | None, root_dir: Path, label: str
) -> PaperProcessExecutorPlanCheck:
    return _check(
        f"{name}_within_workspace_and_present",
        path is not None and _path_is_within_root(path, root_dir) and path.is_file(),
        "blocker",
        f"{label} path must resolve inside the workspace and exist locally.",
        {"path": _safe_relative_path(path, root_dir)},
    )


def _local_path_check(
    name: str, path: Path | None, root_dir: Path, label: str
) -> PaperProcessExecutorPlanCheck:
    return _check(
        f"{name}_within_workspace",
        path is not None and _path_is_within_root(path, root_dir),
        "blocker",
        f"{label} path must resolve inside the workspace.",
        {"path": _safe_relative_path(path, root_dir)},
    )


def _artifact_paths(inputs: PaperProcessExecutorPlanInputs) -> dict[str, str]:
    output_dir = inputs.output_dir
    return {
        "paper_process_executor_plan": _safe_relative_path(
            output_dir / "paper_process_executor_plan.json", inputs.root_dir
        ),
        "paper_process_executor_report": _safe_relative_path(
            output_dir / "paper_process_executor_report.md", inputs.root_dir
        ),
        "process_executor_manifest": _safe_relative_path(
            output_dir / "process_executor_manifest.json", inputs.root_dir
        ),
        "operator_start_checklist": _safe_relative_path(
            output_dir / "operator_start_checklist.md", inputs.root_dir
        ),
        "start_command_review": _safe_relative_path(
            output_dir / "start_command_review.txt", inputs.root_dir
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


def _option_values(command: Sequence[Any], option: str) -> list[str]:
    values: list[str] = []
    tokens = [str(token) for token in command]
    for index, token in enumerate(tokens):
        if token == option and index + 1 < len(tokens):
            values.append(tokens[index + 1])
    return values


def _is_freqtrade_binary(value: Any) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    return Path(value).name.lower() in {"freqtrade", "freqtrade.exe"}


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
) -> PaperProcessExecutorPlanCheck:
    return PaperProcessExecutorPlanCheck(
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


def _safe_string(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip()


def _safe_relative_path(path: Path | None, root_dir: Path) -> str | None:
    if path is None:
        return None
    try:
        return str(path.resolve().relative_to(root_dir.resolve()))
    except ValueError:
        return path.name
