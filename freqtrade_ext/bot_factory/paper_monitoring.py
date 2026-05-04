from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence


PAPER_MONITORING_NOTICE = (
    "Paper monitoring planning is a no-startup, no-process-control gate. It writes "
    "status and metrics artifact schemas only; it does not start, stop, poll, or "
    "manage any bot process."
)

STATUS_VALUES = ["not_started", "starting", "running", "stopping", "stopped", "failed"]


@dataclass(frozen=True)
class PaperMonitoringPlanInputs:
    root_dir: Path
    strategy: str
    run_id: str
    startup_preflight_path: Path
    output_root: Path = Path("data/paper")
    reviewer_notes: Sequence[str] = field(default_factory=list)
    command: Sequence[str] = field(default_factory=list)

    @property
    def output_dir(self) -> Path:
        return self.output_root / self.strategy / self.run_id


@dataclass(frozen=True)
class PaperMonitoringPlanCheck:
    name: str
    status: str
    severity: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_paper_startup_preflight(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Paper startup preflight JSON must contain an object: {path}")
    return payload


def build_paper_monitoring_plan(
    inputs: PaperMonitoringPlanInputs, startup_preflight: dict[str, Any]
) -> dict[str, Any]:
    checks: list[PaperMonitoringPlanCheck] = []
    startup = _dict_or_empty(startup_preflight.get("startup"))
    process_metadata = _dict_or_empty(startup_preflight.get("process_metadata"))
    status_snapshot = _dict_or_empty(startup_preflight.get("status_snapshot"))
    log_paths = _dict_or_empty(startup_preflight.get("log_paths"))
    preflight_artifacts = _dict_or_empty(startup_preflight.get("artifact_paths"))
    safety_scope = _dict_or_empty(startup_preflight.get("safety_scope"))

    process_metadata_path = _path_from_payload(
        preflight_artifacts.get("process_metadata_template"), inputs.root_dir
    )
    status_snapshot_path = _path_from_payload(
        preflight_artifacts.get("status_snapshot_template"), inputs.root_dir
    )
    stdout_path = _path_from_payload(log_paths.get("stdout"), inputs.root_dir)
    stderr_path = _path_from_payload(log_paths.get("stderr"), inputs.root_dir)
    paper_metrics_path = _path_from_payload(log_paths.get("paper_metrics"), inputs.root_dir)

    checks.append(
        _check(
            "startup_preflight_source_is_phase3_paper_startup_preflight",
            startup_preflight.get("phase") == "3"
            and startup_preflight.get("factory") == "paper_startup_preflight",
            "blocker",
            "Monitoring schemas must consume a Phase 3 paper startup preflight.",
            {
                "phase": _safe_scalar(startup_preflight.get("phase")),
                "factory": _safe_scalar(startup_preflight.get("factory")),
            },
        )
    )
    checks.append(
        _check(
            "startup_preflight_strategy_matches",
            startup_preflight.get("strategy") == inputs.strategy,
            "blocker",
            "Startup preflight strategy must match the monitoring candidate.",
            {
                "preflight_strategy": _safe_scalar(startup_preflight.get("strategy")),
                "candidate": inputs.strategy,
            },
        )
    )
    checks.append(
        _check(
            "startup_preflight_ready",
            startup_preflight.get("status") == "ready",
            "blocker",
            "Startup preflight must be ready before monitoring artifacts can be ready.",
            {"preflight_status": _safe_scalar(startup_preflight.get("status"))},
        )
    )
    checks.append(
        _check(
            "startup_preflight_has_no_blockers",
            not startup_preflight.get("blockers"),
            "blocker",
            "Startup preflight must have no blockers.",
            {"blocker_count": len(startup_preflight.get("blockers") or [])},
        )
    )
    checks.append(
        _check(
            "startup_preflight_startup_eligible",
            startup.get("eligible") is True,
            "blocker",
            "Startup preflight startup eligibility must be true.",
            {"eligible": startup.get("eligible")},
        )
    )
    checks.append(
        _check(
            "startup_preflight_did_not_execute_startup",
            startup.get("startup_executed") is False
            and process_metadata.get("startup_executed") is False
            and process_metadata.get("process_started") is False,
            "blocker",
            "Monitoring planning can only consume a no-startup preflight.",
            {
                "startup_executed": startup.get("startup_executed"),
                "process_metadata_startup_executed": process_metadata.get(
                    "startup_executed"
                ),
                "process_started": process_metadata.get("process_started"),
            },
        )
    )
    checks.append(
        _check(
            "startup_preflight_does_not_authorize_startup",
            startup.get("startup_authorized_by_this_command") is False,
            "blocker",
            "Startup preflight must not authorize startup by itself.",
            {
                "startup_authorized_by_this_command": startup.get(
                    "startup_authorized_by_this_command"
                )
            },
        )
    )
    checks.append(
        _check(
            "startup_preflight_requires_separate_execution",
            startup.get("requires_separate_execution_after_preflight") is True,
            "blocker",
            "Startup preflight must require a separate execution step.",
            {
                "requires_separate_execution_after_preflight": startup.get(
                    "requires_separate_execution_after_preflight"
                )
            },
        )
    )
    checks.append(
        _check(
            "process_metadata_template_within_workspace",
            process_metadata_path is not None
            and _path_is_within_root(process_metadata_path, inputs.root_dir),
            "blocker",
            "Process metadata template path must resolve inside the repository workspace.",
            {"path": _safe_relative_path(process_metadata_path, inputs.root_dir)},
        )
    )
    checks.append(
        _check(
            "process_metadata_template_present",
            process_metadata_path is not None and process_metadata_path.is_file(),
            "blocker",
            "Process metadata template must exist before monitoring can be ready.",
            {"path": _safe_relative_path(process_metadata_path, inputs.root_dir)},
        )
    )
    checks.append(
        _check(
            "status_snapshot_template_within_workspace",
            status_snapshot_path is not None
            and _path_is_within_root(status_snapshot_path, inputs.root_dir),
            "blocker",
            "Status snapshot template path must resolve inside the repository workspace.",
            {"path": _safe_relative_path(status_snapshot_path, inputs.root_dir)},
        )
    )
    checks.append(
        _check(
            "status_snapshot_template_present",
            status_snapshot_path is not None and status_snapshot_path.is_file(),
            "blocker",
            "Status snapshot template must exist before monitoring can be ready.",
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
    checks.extend(_startup_preflight_safety_scope_checks(safety_scope))
    checks.append(
        _check(
            "status_snapshot_template_records_no_startup",
            status_snapshot.get("startup_executed") is False
            and status_snapshot.get("bot_startup") is False
            and status_snapshot.get("freqtrade_trade_executed") is False
            and status_snapshot.get("paper_trading_started") is False,
            "blocker",
            "Status snapshot template must record no startup execution.",
            {
                "startup_executed": status_snapshot.get("startup_executed"),
                "bot_startup": status_snapshot.get("bot_startup"),
                "freqtrade_trade_executed": status_snapshot.get(
                    "freqtrade_trade_executed"
                ),
                "paper_trading_started": status_snapshot.get("paper_trading_started"),
            },
        )
    )
    checks.append(
        _check(
            "reviewer_note_present",
            bool(inputs.reviewer_notes),
            "blocker",
            "Monitoring schema planning requires at least one reviewer note.",
            {"note_count": len(inputs.reviewer_notes)},
        )
    )

    status = "ready" if all(check.status == "pass" for check in checks) else "blocked"
    generated_at = datetime.now(UTC).isoformat()
    artifact_paths = _artifact_paths(inputs)
    status_snapshot_schema = build_status_snapshot_schema(inputs.strategy, inputs.run_id)
    paper_metrics_schema = build_paper_metrics_schema(inputs.strategy, inputs.run_id)
    process_metadata_schema = build_process_metadata_schema(inputs.strategy, inputs.run_id)

    return {
        "generated_at": generated_at,
        "phase": "3",
        "factory": "paper_monitoring_plan",
        "status": status,
        "strategy": inputs.strategy,
        "run_id": inputs.run_id,
        "startup_preflight_path": _safe_relative_path(
            inputs.startup_preflight_path, inputs.root_dir
        ),
        "startup_preflight_summary": {
            "status": _safe_scalar(startup_preflight.get("status")),
            "startup_eligible": _safe_scalar(startup.get("eligible")),
            "startup_executed": _safe_scalar(startup.get("startup_executed")),
            "preflight_run_id": _safe_scalar(startup_preflight.get("run_id")),
        },
        "checks": [check.to_dict() for check in checks],
        "blockers": [check.to_dict() for check in checks if check.status == "blocked"],
        "reviewer_notes": list(inputs.reviewer_notes),
        "monitoring": {
            "eligible": status == "ready",
            "monitoring_started": False,
            "status_polling_started": False,
            "process_control": False,
            "process_stop_started": False,
            "requires_started_process_metadata": True,
            "requires_status_snapshot_path": True,
            "requires_stdout_stderr_logs": True,
            "requires_separate_execution_after_start": True,
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
        "schemas": {
            "status_snapshot": status_snapshot_schema,
            "paper_metrics": paper_metrics_schema,
            "process_metadata": process_metadata_schema,
        },
        "artifact_paths": artifact_paths,
        "safety_scope": {
            "command": "paper monitoring schema planning only",
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
        },
        "notice": PAPER_MONITORING_NOTICE,
    }


def build_status_snapshot_schema(strategy: str, run_id: str) -> dict[str, Any]:
    return {
        "title": "Bot Factory Paper Status Snapshot",
        "type": "object",
        "strategy": strategy,
        "run_id": run_id,
        "required": [
            "generated_at",
            "strategy",
            "run_id",
            "status",
            "startup_executed",
            "bot_startup",
            "freqtrade_trade_executed",
            "paper_trading_started",
            "dry_run_trading_started",
            "live_trading",
            "exchange_order_placement",
            "message",
        ],
        "properties": {
            "generated_at": {"type": "string"},
            "strategy": {"type": "string"},
            "run_id": {"type": "string"},
            "status": {"type": "string", "enum": STATUS_VALUES},
            "pid": {"type": ["integer", "null"]},
            "startup_executed": {"type": "boolean"},
            "bot_startup": {"type": "boolean"},
            "freqtrade_trade_executed": {"type": "boolean"},
            "paper_trading_started": {"type": "boolean"},
            "dry_run_trading_started": {"type": "boolean"},
            "live_trading": {"type": "boolean"},
            "exchange_order_placement": {"type": "boolean"},
            "open_trade_count": {"type": ["integer", "null"]},
            "closed_trade_count": {"type": ["integer", "null"]},
            "last_heartbeat_at": {"type": ["string", "null"]},
            "message": {"type": "string"},
        },
        "safety_invariants": {
            "metadata_contains_secrets": False,
            "live_trading": False,
            "exchange_order_placement": False,
            "leverage_above_one": False,
            "shorting": False,
        },
    }


def build_paper_metrics_schema(strategy: str, run_id: str) -> dict[str, Any]:
    return {
        "title": "Bot Factory Paper Metrics Snapshot",
        "type": "object",
        "strategy": strategy,
        "run_id": run_id,
        "required": [
            "generated_at",
            "strategy",
            "run_id",
            "source",
            "status",
            "trade_counts",
            "profit",
            "risk",
            "safety_scope",
        ],
        "properties": {
            "generated_at": {"type": "string"},
            "strategy": {"type": "string"},
            "run_id": {"type": "string"},
            "source": {"type": "string", "enum": ["local_paper_artifacts"]},
            "status": {"type": "string", "enum": STATUS_VALUES},
            "trade_counts": {
                "type": "object",
                "required": ["open", "closed", "total"],
                "properties": {
                    "open": {"type": "integer"},
                    "closed": {"type": "integer"},
                    "total": {"type": "integer"},
                },
            },
            "profit": {
                "type": "object",
                "properties": {
                    "realized": {"type": ["number", "null"]},
                    "unrealized": {"type": ["number", "null"]},
                    "currency": {"type": ["string", "null"]},
                },
            },
            "risk": {
                "type": "object",
                "properties": {
                    "max_drawdown_pct": {"type": ["number", "null"]},
                    "max_open_trades": {"type": ["integer", "null"]},
                },
            },
            "safety_scope": {"type": "object"},
        },
        "safety_invariants": {
            "metadata_contains_secrets": False,
            "local_artifacts_source_of_truth": True,
            "live_trading": False,
            "exchange_order_placement": False,
            "leverage_above_one": False,
            "shorting": False,
        },
    }


def build_process_metadata_schema(strategy: str, run_id: str) -> dict[str, Any]:
    return {
        "title": "Bot Factory Paper Process Metadata",
        "type": "object",
        "strategy": strategy,
        "run_id": run_id,
        "required": [
            "strategy",
            "run_id",
            "process_started",
            "startup_executed",
            "pid",
            "started_at",
            "ended_at",
            "command",
            "stdout_log",
            "stderr_log",
            "status_snapshot",
            "paper_metrics",
            "notice",
        ],
        "properties": {
            "strategy": {"type": "string"},
            "run_id": {"type": "string"},
            "process_started": {"type": "boolean"},
            "startup_executed": {"type": "boolean"},
            "pid": {"type": ["integer", "null"]},
            "started_at": {"type": ["string", "null"]},
            "ended_at": {"type": ["string", "null"]},
            "command": {"type": "array", "items": {"type": "string"}},
            "stdout_log": {"type": "string"},
            "stderr_log": {"type": "string"},
            "status_snapshot": {"type": "string"},
            "paper_metrics": {"type": "string"},
            "notice": {"type": "string"},
        },
        "safety_invariants": {
            "metadata_contains_secrets": False,
            "stdout_stderr_are_local_paths": True,
            "status_snapshot_is_local_path": True,
            "paper_metrics_is_local_path": True,
        },
    }


def write_paper_monitoring_plan_artifacts(
    inputs: PaperMonitoringPlanInputs, plan: dict[str, Any]
) -> None:
    output_dir = inputs.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "paper_monitoring_plan.json").write_text(
        json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "status_snapshot_schema.json").write_text(
        json.dumps(plan["schemas"]["status_snapshot"], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "paper_metrics_schema.json").write_text(
        json.dumps(plan["schemas"]["paper_metrics"], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "process_metadata_schema.json").write_text(
        json.dumps(plan["schemas"]["process_metadata"], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "command.txt").write_text(" ".join(inputs.command), encoding="utf-8")
    write_paper_monitoring_report(plan, output_dir / "paper_monitoring_report.md")


def write_paper_monitoring_report(plan: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    monitoring = plan["monitoring"]
    lines = [
        "# Paper Monitoring Schema Plan",
        "",
        "## Summary",
        "",
        f"- Strategy: {plan['strategy']}",
        f"- Run ID: {plan['run_id']}",
        f"- Status: {plan['status']}",
        f"- Startup preflight status: {plan['startup_preflight_summary']['status']}",
        f"- Monitoring eligible: {monitoring['eligible']}",
        f"- Monitoring started: {monitoring['monitoring_started']}",
        f"- Process control enabled: {monitoring['process_control']}",
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
            "## Monitoring Boundary",
            "",
            f"- {PAPER_MONITORING_NOTICE}",
            "- These schemas do not prove that a paper process exists or is healthy.",
            "- A later explicit execution and monitoring path must validate runtime data.",
            "- Local JSON, CSV, Markdown, and log artifacts remain the source of truth.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _startup_preflight_safety_scope_checks(
    safety_scope: dict[str, Any]
) -> list[PaperMonitoringPlanCheck]:
    return [
        _check(
            "startup_preflight_no_startup_scope",
            safety_scope.get("bot_startup") is False
            and safety_scope.get("freqtrade_trade_executed") is False
            and safety_scope.get("paper_trading_started") is False
            and safety_scope.get("dry_run_trading_started") is False,
            "blocker",
            "Startup preflight must record no startup execution.",
            {
                "bot_startup": safety_scope.get("bot_startup"),
                "freqtrade_trade_executed": safety_scope.get("freqtrade_trade_executed"),
                "paper_trading_started": safety_scope.get("paper_trading_started"),
                "dry_run_trading_started": safety_scope.get("dry_run_trading_started"),
            },
        ),
        _check(
            "startup_preflight_no_live_or_exchange_order_scope",
            safety_scope.get("live_trading") is False
            and safety_scope.get("exchange_order_placement") is False,
            "blocker",
            "Startup preflight must not involve live trading or exchange order placement.",
            {
                "live_trading": safety_scope.get("live_trading"),
                "exchange_order_placement": safety_scope.get("exchange_order_placement"),
            },
        ),
        _check(
            "startup_preflight_no_secrets_leverage_or_shorting_scope",
            safety_scope.get("uses_api_keys_or_secrets") is False
            and safety_scope.get("leverage_above_one") is False
            and safety_scope.get("shorting") is False
            and safety_scope.get("metadata_contains_secrets") is False,
            "blocker",
            "Startup preflight metadata must remain sanitized and long-only.",
            {
                "uses_api_keys_or_secrets": safety_scope.get("uses_api_keys_or_secrets"),
                "leverage_above_one": safety_scope.get("leverage_above_one"),
                "shorting": safety_scope.get("shorting"),
                "metadata_contains_secrets": safety_scope.get("metadata_contains_secrets"),
            },
        ),
        _check(
            "startup_preflight_local_artifacts_source_of_truth",
            safety_scope.get("local_artifacts_source_of_truth") is True,
            "blocker",
            "Startup preflight must keep local artifacts as the source of truth.",
            {
                "local_artifacts_source_of_truth": safety_scope.get(
                    "local_artifacts_source_of_truth"
                )
            },
        ),
    ]


def _artifact_paths(inputs: PaperMonitoringPlanInputs) -> dict[str, str]:
    output_dir = inputs.output_dir
    return {
        "paper_monitoring_plan": _safe_relative_path(
            output_dir / "paper_monitoring_plan.json", inputs.root_dir
        ),
        "paper_monitoring_report": _safe_relative_path(
            output_dir / "paper_monitoring_report.md", inputs.root_dir
        ),
        "status_snapshot_schema": _safe_relative_path(
            output_dir / "status_snapshot_schema.json", inputs.root_dir
        ),
        "paper_metrics_schema": _safe_relative_path(
            output_dir / "paper_metrics_schema.json", inputs.root_dir
        ),
        "process_metadata_schema": _safe_relative_path(
            output_dir / "process_metadata_schema.json", inputs.root_dir
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
) -> PaperMonitoringPlanCheck:
    return PaperMonitoringPlanCheck(
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
