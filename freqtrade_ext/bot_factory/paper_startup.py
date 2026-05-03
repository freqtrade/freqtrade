from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence


PAPER_STARTUP_PREFLIGHT_NOTICE = (
    "Paper startup preflight is a no-startup gate. It does not start "
    "freqtrade trade, paper trading, dry-run trading, live trading, or any bot process."
)


@dataclass(frozen=True)
class PaperStartupPreflightInputs:
    root_dir: Path
    strategy: str
    run_id: str
    plan_path: Path
    output_root: Path = Path("data/paper")
    reviewer_notes: Sequence[str] = field(default_factory=list)
    confirm_paper_start: bool = False
    requested_start_command: str | None = None
    command: Sequence[str] = field(default_factory=list)

    @property
    def output_dir(self) -> Path:
        return self.output_root / self.strategy / self.run_id


@dataclass(frozen=True)
class PaperStartupPreflightCheck:
    name: str
    status: str
    severity: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_paper_run_plan(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Paper run plan JSON must contain an object: {path}")
    return payload


def build_paper_startup_preflight(
    inputs: PaperStartupPreflightInputs, plan: dict[str, Any]
) -> dict[str, Any]:
    checks: list[PaperStartupPreflightCheck] = []
    future_startup = plan.get("future_startup", {})
    if not isinstance(future_startup, dict):
        future_startup = {}
    safety_scope = plan.get("safety_scope", {})
    if not isinstance(safety_scope, dict):
        safety_scope = {}

    command_preview = _command_preview_from_plan(future_startup)
    expected_command = _command_string(command_preview)
    requested_command = (inputs.requested_start_command or "").strip()
    config_values = _option_values(command_preview, "--config")
    strategy_values = _option_values(command_preview, "--strategy")
    strategy_path_values = _option_values(command_preview, "--strategy-path")
    command_config_path = (
        _path_from_payload(config_values[0], inputs.root_dir)
        if len(config_values) == 1
        else None
    )
    command_strategy_path = (
        _path_from_payload(strategy_path_values[0], inputs.root_dir)
        if len(strategy_path_values) == 1
        else None
    )
    plan_config_path = _path_from_payload(plan.get("config_path"), inputs.root_dir)
    plan_strategy_path = _path_from_payload(plan.get("strategy_path"), inputs.root_dir)
    artifacts = plan.get("artifact_paths", {})
    if not isinstance(artifacts, dict):
        artifacts = {}

    checklist_path = _path_from_payload(artifacts.get("paper_run_checklist"), inputs.root_dir)
    stop_cleanup_path = _path_from_payload(artifacts.get("stop_cleanup"), inputs.root_dir)

    checks.append(
        _check(
            "paper_plan_source_is_phase3_paper_run_plan",
            plan.get("phase") == "3" and plan.get("factory") == "paper_run_plan",
            "blocker",
            "Startup preflight must consume a Phase 3 paper run plan.",
            {
                "phase": _safe_scalar(plan.get("phase")),
                "factory": _safe_scalar(plan.get("factory")),
            },
        )
    )
    checks.append(
        _check(
            "paper_plan_strategy_matches",
            plan.get("strategy") == inputs.strategy,
            "blocker",
            "Paper run plan strategy must match the startup preflight candidate.",
            {
                "plan_strategy": _safe_scalar(plan.get("strategy")),
                "candidate": inputs.strategy,
            },
        )
    )
    checks.append(
        _check(
            "paper_plan_ready",
            plan.get("status") == "ready",
            "blocker",
            "Paper run plan must be ready before startup preflight can pass.",
            {"plan_status": _safe_scalar(plan.get("status"))},
        )
    )
    checks.append(
        _check(
            "paper_plan_has_no_blockers",
            not plan.get("blockers"),
            "blocker",
            "Paper run plan must have no blockers.",
            {"blocker_count": len(plan.get("blockers") or [])},
        )
    )
    checks.append(
        _check(
            "paper_plan_future_startup_eligible",
            future_startup.get("eligible") is True,
            "blocker",
            "Paper run plan future startup eligibility must be true.",
            {"eligible": future_startup.get("eligible")},
        )
    )
    checks.append(
        _check(
            "paper_plan_requires_separate_user_request",
            future_startup.get("requires_separate_user_request") is True,
            "blocker",
            "Paper run plan must require a separate explicit user request.",
            {
                "requires_separate_user_request": future_startup.get(
                    "requires_separate_user_request"
                )
            },
        )
    )
    checks.append(
        _check(
            "paper_plan_requires_stop_cleanup_first",
            future_startup.get("requires_stop_cleanup_first") is True,
            "blocker",
            "Paper run plan must require stop and cleanup review before startup.",
            {"requires_stop_cleanup_first": future_startup.get("requires_stop_cleanup_first")},
        )
    )
    checks.append(
        _check(
            "paper_plan_does_not_authorize_startup",
            future_startup.get("startup_authorized_by_this_command") is False,
            "blocker",
            "Paper run plan must not authorize startup by itself.",
            {
                "startup_authorized_by_this_command": future_startup.get(
                    "startup_authorized_by_this_command"
                )
            },
        )
    )
    checks.append(
        _check(
            "paper_plan_has_start_command_preview",
            len(command_preview) >= 2,
            "blocker",
            "Paper run plan must include a freqtrade trade command preview.",
            {"command_token_count": len(command_preview)},
        )
    )
    checks.append(
        _check(
            "paper_plan_start_command_uses_freqtrade_trade",
            len(command_preview) >= 2
            and _is_freqtrade_binary(command_preview[0])
            and command_preview[1] == "trade",
            "blocker",
            "Paper run plan command preview must use freqtrade trade.",
            {
                "executable": command_preview[0] if command_preview else None,
                "subcommand": command_preview[1] if len(command_preview) > 1 else None,
            },
        )
    )
    checks.append(
        _check(
            "paper_plan_start_command_has_required_options",
            len(config_values) == 1
            and len(strategy_values) == 1
            and len(strategy_path_values) == 1,
            "blocker",
            "Paper run plan command preview must include one config, strategy, and strategy path.",
            {
                "config_count": len(config_values),
                "strategy_count": len(strategy_values),
                "strategy_path_count": len(strategy_path_values),
            },
        )
    )
    checks.append(
        _check(
            "paper_plan_start_command_config_matches_plan",
            command_config_path is not None
            and plan_config_path is not None
            and _paths_equal(command_config_path, plan_config_path)
            and _path_is_within_root(command_config_path, inputs.root_dir)
            and command_config_path.is_file(),
            "blocker",
            "Paper run plan command preview config must match a local existing plan config.",
            {
                "command_config_path": _safe_relative_path(command_config_path, inputs.root_dir),
                "plan_config_path": _safe_relative_path(plan_config_path, inputs.root_dir),
            },
        )
    )
    checks.append(
        _check(
            "paper_plan_start_command_strategy_matches_candidate",
            len(strategy_values) == 1 and strategy_values[0] == inputs.strategy,
            "blocker",
            "Paper run plan command preview strategy must match the startup candidate.",
            {
                "command_strategy": strategy_values[0] if len(strategy_values) == 1 else None,
                "candidate": inputs.strategy,
            },
        )
    )
    checks.append(
        _check(
            "paper_plan_start_command_strategy_path_matches_plan",
            command_strategy_path is not None
            and plan_strategy_path is not None
            and _paths_equal(command_strategy_path, plan_strategy_path)
            and _path_is_within_root(command_strategy_path, inputs.root_dir)
            and command_strategy_path.exists(),
            "blocker",
            "Paper run plan command preview strategy path must match a local plan path.",
            {
                "command_strategy_path": _safe_relative_path(
                    command_strategy_path, inputs.root_dir
                ),
                "plan_strategy_path": _safe_relative_path(plan_strategy_path, inputs.root_dir),
            },
        )
    )
    checks.append(
        _check(
            "confirm_paper_start_acknowledged",
            inputs.confirm_paper_start,
            "blocker",
            "Startup preflight requires explicit --confirm-paper-start acknowledgement.",
        )
    )
    checks.append(
        _check(
            "requested_start_command_present",
            bool(requested_command),
            "blocker",
            "Startup preflight requires the exact requested start command string.",
        )
    )
    checks.append(
        _check(
            "requested_start_command_matches_plan",
            bool(requested_command) and requested_command == expected_command,
            "blocker",
            "Requested start command must exactly match the paper run plan preview.",
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
            "Startup preflight requires at least one reviewer note.",
            {"note_count": len(inputs.reviewer_notes)},
        )
    )
    checks.append(
        _check(
            "stop_cleanup_artifact_within_workspace",
            stop_cleanup_path is not None
            and _path_is_within_root(stop_cleanup_path, inputs.root_dir),
            "blocker",
            "Stop and cleanup documentation path must resolve inside the repository workspace.",
            {
                "path": _safe_relative_path(stop_cleanup_path, inputs.root_dir)
                if stop_cleanup_path is not None
                else None
            },
        )
    )
    checks.append(
        _check(
            "stop_cleanup_artifact_present",
            stop_cleanup_path is not None and stop_cleanup_path.is_file(),
            "blocker",
            "Stop and cleanup documentation must exist before startup preflight can pass.",
            {
                "path": _safe_relative_path(stop_cleanup_path, inputs.root_dir)
                if stop_cleanup_path is not None
                else None
            },
        )
    )
    checks.append(
        _check(
            "paper_run_checklist_within_workspace",
            checklist_path is not None and _path_is_within_root(checklist_path, inputs.root_dir),
            "blocker",
            "Paper run checklist path must resolve inside the repository workspace.",
            {
                "path": _safe_relative_path(checklist_path, inputs.root_dir)
                if checklist_path is not None
                else None
            },
        )
    )
    checks.append(
        _check(
            "paper_run_checklist_present",
            checklist_path is not None and checklist_path.is_file(),
            "blocker",
            "Paper run checklist must exist before startup preflight can pass.",
            {
                "path": _safe_relative_path(checklist_path, inputs.root_dir)
                if checklist_path is not None
                else None
            },
        )
    )
    checks.extend(_plan_safety_scope_checks(safety_scope))

    status = "ready" if all(check.status == "pass" for check in checks) else "blocked"
    startup_command = command_preview if status == "ready" else []
    generated_at = datetime.now(UTC).isoformat()
    artifact_paths = _artifact_paths(inputs)
    log_paths = {
        "stdout": _safe_relative_path(inputs.output_dir / "logs" / "stdout.log", inputs.root_dir),
        "stderr": _safe_relative_path(inputs.output_dir / "logs" / "stderr.log", inputs.root_dir),
        "status_snapshot": artifact_paths["status_snapshot_template"],
        "paper_metrics": _safe_relative_path(
            inputs.output_dir / "paper_metrics.json", inputs.root_dir
        ),
    }

    process_metadata = {
        "strategy": inputs.strategy,
        "run_id": inputs.run_id,
        "process_started": False,
        "startup_executed": False,
        "pid": None,
        "started_at": None,
        "ended_at": None,
        "command": startup_command,
        "stdout_log": log_paths["stdout"],
        "stderr_log": log_paths["stderr"],
        "status_snapshot": log_paths["status_snapshot"],
        "paper_metrics": log_paths["paper_metrics"],
        "notice": PAPER_STARTUP_PREFLIGHT_NOTICE,
    }
    status_snapshot = {
        "generated_at": generated_at,
        "strategy": inputs.strategy,
        "run_id": inputs.run_id,
        "status": "not_started",
        "startup_executed": False,
        "bot_startup": False,
        "freqtrade_trade_executed": False,
        "paper_trading_started": False,
        "dry_run_trading_started": False,
        "live_trading": False,
        "exchange_order_placement": False,
        "message": "Template only; no paper process was started.",
    }

    return {
        "generated_at": generated_at,
        "phase": "3",
        "factory": "paper_startup_preflight",
        "status": status,
        "strategy": inputs.strategy,
        "run_id": inputs.run_id,
        "plan_path": _safe_relative_path(inputs.plan_path, inputs.root_dir),
        "plan_summary": {
            "status": _safe_scalar(plan.get("status")),
            "plan_run_id": _safe_scalar(plan.get("run_id")),
            "readiness": _safe_scalar(
                plan.get("readiness_summary", {}).get("readiness")
                if isinstance(plan.get("readiness_summary"), dict)
                else None
            ),
        },
        "checks": [check.to_dict() for check in checks],
        "blockers": [check.to_dict() for check in checks if check.status == "blocked"],
        "reviewer_notes": list(inputs.reviewer_notes),
        "requested_start_command": requested_command or None,
        "expected_start_command": expected_command or None,
        "startup": {
            "eligible": status == "ready",
            "startup_executed": False,
            "startup_authorized_by_this_command": False,
            "requires_separate_execution_after_preflight": True,
            "command_preview": startup_command,
        },
        "process_metadata": process_metadata,
        "status_snapshot": status_snapshot,
        "log_paths": log_paths,
        "artifact_paths": artifact_paths,
        "safety_scope": {
            "command": "paper startup preflight only",
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
        "notice": PAPER_STARTUP_PREFLIGHT_NOTICE,
    }


def write_paper_startup_preflight_artifacts(
    inputs: PaperStartupPreflightInputs, preflight: dict[str, Any]
) -> None:
    output_dir = inputs.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "paper_startup_preflight.json").write_text(
        json.dumps(preflight, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "process_metadata_template.json").write_text(
        json.dumps(preflight["process_metadata"], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "status_snapshot_template.json").write_text(
        json.dumps(preflight["status_snapshot"], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "start_command_preview.txt").write_text(
        _command_string(preflight["startup"]["command_preview"]), encoding="utf-8"
    )
    (output_dir / "command.txt").write_text(" ".join(inputs.command), encoding="utf-8")
    write_paper_startup_preflight_report(
        preflight, output_dir / "paper_startup_preflight_report.md"
    )


def write_paper_startup_preflight_report(preflight: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    startup = preflight["startup"]
    lines = [
        "# Paper Startup Preflight Report",
        "",
        "## Summary",
        "",
        f"- Strategy: {preflight['strategy']}",
        f"- Run ID: {preflight['run_id']}",
        f"- Status: {preflight['status']}",
        f"- Plan status: {preflight['plan_summary']['status']}",
        f"- Startup eligible after preflight: {startup['eligible']}",
        f"- Startup executed: {startup['startup_executed']}",
        "",
        "## Checks",
        "",
    ]
    for check in preflight["checks"]:
        lines.append(f"- {check['status'].upper()}: {check['name']} - {check['message']}")

    lines.extend(["", "## Process Metadata Design", ""])
    lines.extend(
        [
            f"- stdout log: `{preflight['log_paths']['stdout']}`",
            f"- stderr log: `{preflight['log_paths']['stderr']}`",
            f"- status snapshot: `{preflight['log_paths']['status_snapshot']}`",
            f"- paper metrics: `{preflight['log_paths']['paper_metrics']}`",
        ]
    )

    lines.extend(["", "## Reviewer Notes", ""])
    if preflight["reviewer_notes"]:
        lines.extend(f"- {note}" for note in preflight["reviewer_notes"])
    else:
        lines.append("- No reviewer notes were supplied.")

    lines.extend(
        [
            "",
            "## Startup Boundary",
            "",
            f"- {PAPER_STARTUP_PREFLIGHT_NOTICE}",
            "- This preflight records paths and templates only.",
            "- A later explicit execution step is required before any process can start.",
            "",
        ]
    )
    if startup["command_preview"]:
        lines.extend(["## Command Preview", "", "```powershell"])
        lines.append(_command_string(startup["command_preview"]))
        lines.extend(["```", ""])

    path.write_text("\n".join(lines), encoding="utf-8")


def _plan_safety_scope_checks(
    safety_scope: dict[str, Any]
) -> list[PaperStartupPreflightCheck]:
    return [
        _check(
            "paper_plan_no_startup_scope",
            safety_scope.get("bot_startup") is False
            and safety_scope.get("freqtrade_trade_executed") is False
            and safety_scope.get("paper_trading_started") is False
            and safety_scope.get("dry_run_trading_started") is False,
            "blocker",
            "Paper run plan must record no startup execution.",
            {
                "bot_startup": safety_scope.get("bot_startup"),
                "freqtrade_trade_executed": safety_scope.get("freqtrade_trade_executed"),
                "paper_trading_started": safety_scope.get("paper_trading_started"),
                "dry_run_trading_started": safety_scope.get("dry_run_trading_started"),
            },
        ),
        _check(
            "paper_plan_no_live_or_exchange_order_scope",
            safety_scope.get("live_trading") is False
            and safety_scope.get("exchange_order_placement") is False,
            "blocker",
            "Paper run plan must not involve live trading or exchange order placement.",
            {
                "live_trading": safety_scope.get("live_trading"),
                "exchange_order_placement": safety_scope.get("exchange_order_placement"),
            },
        ),
        _check(
            "paper_plan_no_secrets_leverage_or_shorting_scope",
            safety_scope.get("uses_api_keys_or_secrets") is False
            and safety_scope.get("leverage_above_one") is False
            and safety_scope.get("shorting") is False
            and safety_scope.get("metadata_contains_secrets") is False,
            "blocker",
            "Paper run plan metadata must remain sanitized and long-only.",
            {
                "uses_api_keys_or_secrets": safety_scope.get("uses_api_keys_or_secrets"),
                "leverage_above_one": safety_scope.get("leverage_above_one"),
                "shorting": safety_scope.get("shorting"),
                "metadata_contains_secrets": safety_scope.get("metadata_contains_secrets"),
            },
        ),
        _check(
            "paper_plan_local_artifacts_source_of_truth",
            safety_scope.get("local_artifacts_source_of_truth") is True,
            "blocker",
            "Paper run plan must keep local artifacts as the source of truth.",
            {
                "local_artifacts_source_of_truth": safety_scope.get(
                    "local_artifacts_source_of_truth"
                )
            },
        ),
    ]


def _artifact_paths(inputs: PaperStartupPreflightInputs) -> dict[str, str]:
    output_dir = inputs.output_dir
    return {
        "paper_startup_preflight": _safe_relative_path(
            output_dir / "paper_startup_preflight.json", inputs.root_dir
        ),
        "paper_startup_preflight_report": _safe_relative_path(
            output_dir / "paper_startup_preflight_report.md", inputs.root_dir
        ),
        "process_metadata_template": _safe_relative_path(
            output_dir / "process_metadata_template.json", inputs.root_dir
        ),
        "status_snapshot_template": _safe_relative_path(
            output_dir / "status_snapshot_template.json", inputs.root_dir
        ),
        "start_command_preview": _safe_relative_path(
            output_dir / "start_command_preview.txt", inputs.root_dir
        ),
        "command": _safe_relative_path(output_dir / "command.txt", inputs.root_dir),
    }


def _command_preview_from_plan(future_startup: dict[str, Any]) -> list[str]:
    preview = future_startup.get("command_preview")
    if not isinstance(preview, list):
        return []
    return [str(token) for token in preview if str(token)]


def _command_string(command: Sequence[Any]) -> str:
    return " ".join(str(token) for token in command).strip()


def _path_from_payload(path_value: Any, root_dir: Path) -> Path | None:
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


def _paths_equal(left: Path, right: Path) -> bool:
    return left.resolve() == right.resolve()


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
) -> PaperStartupPreflightCheck:
    return PaperStartupPreflightCheck(
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
