from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence


PAPER_PLAN_NOTICE = (
    "Paper run planning is a no-startup gate. It does not start paper trading, "
    "dry-run trading, live trading, freqtrade trade, or any bot process."
)


@dataclass(frozen=True)
class PaperRunPlanInputs:
    root_dir: Path
    strategy: str
    run_id: str
    readiness_path: Path
    config_path: Path | None = None
    strategy_path: Path | None = None
    output_root: Path = Path("data/paper")
    reviewer_notes: Sequence[str] = field(default_factory=list)
    confirm_paper: bool = False
    command: Sequence[str] = field(default_factory=list)
    freqtrade_binary: str = "freqtrade"

    @property
    def output_dir(self) -> Path:
        return self.output_root / self.strategy / self.run_id


@dataclass(frozen=True)
class PaperRunPlanCheck:
    name: str
    status: str
    severity: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_paper_readiness(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Readiness JSON must contain an object: {path}")
    return payload


def build_paper_run_plan(
    inputs: PaperRunPlanInputs, readiness: dict[str, Any]
) -> dict[str, Any]:
    checks: list[PaperRunPlanCheck] = []
    readiness_status = readiness.get("readiness") or readiness.get("status")
    readiness_scope = readiness.get("safety_scope", {})
    if not isinstance(readiness_scope, dict):
        readiness_scope = {}

    config_path = _resolved_config_path(inputs, readiness)
    strategy_path = _resolve_workspace_path(
        inputs.strategy_path or Path("user_data/strategies"), inputs.root_dir
    )

    checks.append(
        _check(
            "readiness_source_is_phase3_paper_readiness",
            readiness.get("phase") == "3" and readiness.get("factory") == "paper_readiness",
            "blocker",
            "Paper plans must consume a Phase 3 paper readiness report.",
            {
                "phase": _safe_scalar(readiness.get("phase")),
                "factory": _safe_scalar(readiness.get("factory")),
            },
        )
    )
    checks.append(
        _check(
            "readiness_strategy_matches",
            readiness.get("strategy") == inputs.strategy,
            "blocker",
            "Readiness report strategy must match the paper plan candidate.",
            {
                "readiness_strategy": _safe_scalar(readiness.get("strategy")),
                "candidate": inputs.strategy,
            },
        )
    )
    checks.append(
        _check(
            "readiness_passed",
            readiness_status == "pass",
            "blocker",
            "Readiness report must be pass before any paper run can be planned.",
            {"readiness": _safe_scalar(readiness_status)},
        )
    )
    checks.append(
        _check(
            "readiness_has_no_blockers",
            not readiness.get("blockers"),
            "blocker",
            "Readiness report must have no blockers.",
            {"blocker_count": len(readiness.get("blockers") or [])},
        )
    )
    checks.append(
        _check(
            "readiness_has_no_failures",
            not readiness.get("failures"),
            "blocker",
            "Readiness report must have no failed gate checks.",
            {"failure_count": len(readiness.get("failures") or [])},
        )
    )
    checks.append(
        _check(
            "readiness_no_startup_scope",
            readiness_scope.get("bot_startup") is False
            and readiness_scope.get("freqtrade_trade") is False
            and readiness_scope.get("paper_trading_started") is False
            and readiness_scope.get("dry_run_trading_started") is False,
            "blocker",
            "Readiness evidence must be from a no-startup preflight.",
            {
                "bot_startup": readiness_scope.get("bot_startup"),
                "freqtrade_trade": readiness_scope.get("freqtrade_trade"),
                "paper_trading_started": readiness_scope.get("paper_trading_started"),
                "dry_run_trading_started": readiness_scope.get("dry_run_trading_started"),
            },
        )
    )
    checks.append(
        _check(
            "readiness_no_live_or_exchange_order_scope",
            readiness_scope.get("live_trading") is False
            and readiness_scope.get("exchange_order_placement") is False,
            "blocker",
            "Readiness evidence must not involve live trading or exchange order placement.",
            {
                "live_trading": readiness_scope.get("live_trading"),
                "exchange_order_placement": readiness_scope.get("exchange_order_placement"),
            },
        )
    )
    checks.append(
        _check(
            "readiness_metadata_sanitized",
            readiness_scope.get("uses_api_keys_or_secrets") is False
            and readiness_scope.get("metadata_contains_secrets") is False,
            "blocker",
            "Readiness metadata must be sanitized and must not contain secrets.",
            {
                "uses_api_keys_or_secrets": readiness_scope.get(
                    "uses_api_keys_or_secrets"
                ),
                "metadata_contains_secrets": readiness_scope.get(
                    "metadata_contains_secrets"
                ),
            },
        )
    )
    checks.append(
        _check(
            "readiness_long_only_scope",
            readiness_scope.get("leverage_above_one") is False
            and readiness_scope.get("shorting") is False,
            "blocker",
            "Readiness scope must remain long-only with no leverage above 1.0.",
            {
                "leverage_above_one": readiness_scope.get("leverage_above_one"),
                "shorting": readiness_scope.get("shorting"),
            },
        )
    )
    checks.append(
        _check(
            "readiness_local_artifacts_source_of_truth",
            readiness_scope.get("local_artifacts_source_of_truth") is True,
            "blocker",
            "Readiness report must keep local artifacts as the source of truth.",
            {
                "local_artifacts_source_of_truth": readiness_scope.get(
                    "local_artifacts_source_of_truth"
                )
            },
        )
    )
    checks.append(
        _check(
            "config_path_within_workspace",
            config_path is not None and _path_is_within_root(config_path, inputs.root_dir),
            "blocker",
            "Paper plan config path must resolve inside the repository workspace.",
            {
                "config_path": _safe_relative_path(config_path, inputs.root_dir)
                if config_path is not None
                else None
            },
        )
    )
    checks.append(
        _check(
            "strategy_path_within_workspace",
            _path_is_within_root(strategy_path, inputs.root_dir),
            "blocker",
            "Paper plan strategy path must resolve inside the repository workspace.",
            {"strategy_path": _safe_relative_path(strategy_path, inputs.root_dir)},
        )
    )
    checks.append(
        _check(
            "config_file_present",
            config_path is not None and config_path.is_file(),
            "blocker",
            "Paper plan requires the same dry-run config file used by readiness.",
            {
                "config_path": _safe_relative_path(config_path, inputs.root_dir)
                if config_path is not None
                else None
            },
        )
    )
    checks.append(
        _check(
            "confirm_paper_acknowledged",
            inputs.confirm_paper,
            "blocker",
            "Paper plan requires explicit --confirm-paper acknowledgement.",
        )
    )
    checks.append(
        _check(
            "reviewer_note_present",
            bool(inputs.reviewer_notes),
            "blocker",
            "Paper plan requires at least one reviewer note.",
            {"note_count": len(inputs.reviewer_notes)},
        )
    )

    status = "ready" if all(check.status == "pass" for check in checks) else "blocked"
    command_preview = (
        _paper_start_command_preview(
            inputs=inputs,
            config_path=config_path,
            strategy_path=strategy_path,
        )
        if status == "ready" and config_path is not None
        else []
    )

    plan = {
        "generated_at": datetime.now(UTC).isoformat(),
        "phase": "3",
        "factory": "paper_run_plan",
        "status": status,
        "strategy": inputs.strategy,
        "run_id": inputs.run_id,
        "readiness_path": _safe_relative_path(inputs.readiness_path, inputs.root_dir),
        "readiness_summary": {
            "readiness": _safe_scalar(readiness_status),
            "readiness_run_id": _safe_scalar(readiness.get("run_id")),
            "readiness_generated_at": _safe_scalar(readiness.get("generated_at")),
        },
        "config_path": _safe_relative_path(config_path, inputs.root_dir)
        if config_path is not None
        else None,
        "strategy_path": _safe_relative_path(strategy_path, inputs.root_dir),
        "checks": [check.to_dict() for check in checks],
        "blockers": [check.to_dict() for check in checks if check.status == "blocked"],
        "reviewer_notes": list(inputs.reviewer_notes),
        "future_startup": {
            "eligible": status == "ready",
            "requires_separate_user_request": True,
            "requires_stop_cleanup_first": True,
            "startup_authorized_by_this_command": False,
            "command_preview": command_preview,
        },
        "artifact_paths": {
            "paper_run_plan": _safe_relative_path(
                inputs.output_dir / "paper_run_plan.json", inputs.root_dir
            ),
            "paper_run_checklist": _safe_relative_path(
                inputs.output_dir / "paper_run_checklist.md", inputs.root_dir
            ),
            "stop_cleanup": _safe_relative_path(
                inputs.output_dir / "stop_cleanup.md", inputs.root_dir
            ),
            "command": _safe_relative_path(inputs.output_dir / "command.txt", inputs.root_dir),
        },
        "safety_scope": {
            "command": "paper run planning only",
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
        "notice": PAPER_PLAN_NOTICE,
    }
    return plan


def write_paper_run_plan_artifacts(inputs: PaperRunPlanInputs, plan: dict[str, Any]) -> None:
    output_dir = inputs.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "paper_run_plan.json").write_text(
        json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "command.txt").write_text(" ".join(inputs.command), encoding="utf-8")
    write_paper_run_checklist(plan, output_dir / "paper_run_checklist.md")
    write_stop_cleanup(plan, output_dir / "stop_cleanup.md")


def write_paper_run_checklist(plan: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    future_startup = plan["future_startup"]
    lines = [
        "# Paper Run Plan Checklist",
        "",
        "## Summary",
        "",
        f"- Strategy: {plan['strategy']}",
        f"- Run ID: {plan['run_id']}",
        f"- Status: {plan['status']}",
        f"- Readiness: {plan['readiness_summary']['readiness']}",
        f"- Future startup eligible: {future_startup['eligible']}",
        "",
        "## Gates",
        "",
    ]
    for check in plan["checks"]:
        lines.append(
            f"- {check['status'].upper()}: {check['name']} - {check['message']}"
        )

    lines.extend(["", "## Reviewer Notes", ""])
    if plan["reviewer_notes"]:
        lines.extend(f"- {note}" for note in plan["reviewer_notes"])
    else:
        lines.append("- No reviewer notes were supplied.")

    lines.extend(
        [
            "",
            "## Startup Boundary",
            "",
            f"- {PAPER_PLAN_NOTICE}",
            "- This plan never authorizes startup by itself.",
            "- A separate explicit user request is required before any future paper start.",
            "- Stop and cleanup instructions must be reviewed before any start procedure.",
            "",
        ]
    )
    if future_startup["command_preview"]:
        lines.extend(["## Command Preview", "", "```powershell"])
        lines.append(" ".join(future_startup["command_preview"]))
        lines.extend(["```", ""])

    path.write_text("\n".join(lines), encoding="utf-8")


def write_stop_cleanup(plan: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Paper Run Stop And Cleanup",
        "",
        "This is planning documentation only. No bot process was started.",
        "",
        "## Required Before Any Future Start",
        "",
        "- Confirm the paper run plan status is `ready`.",
        "- Confirm readiness remains `pass` and references the intended strategy/config.",
        "- Confirm the config remains `dry_run=true` and contains no credentials.",
        "- Confirm a separate explicit user request authorizes the exact future start command.",
        "",
        "## Stop Procedure For A Future Started Paper Process",
        "",
        "- Use the future wrapper's recorded process metadata to identify the process.",
        "- Request a graceful stop through the wrapper before terminating a process.",
        "- Confirm no paper process remains running before collecting final artifacts.",
        "- Preserve stdout, stderr, status snapshots, metrics, and sanitized metadata.",
        "",
        "## Cleanup Boundaries",
        "",
        "- Do not delete local source-of-truth JSON, CSV, Markdown, or log artifacts.",
        "- Do not write API keys, secrets, private environment values, or credentials.",
        "- Do not promote paper results to live or canary live without a later human-approved path.",
        "",
        f"- Plan status: {plan['status']}",
        f"- Strategy: {plan['strategy']}",
        f"- Run ID: {plan['run_id']}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _paper_start_command_preview(
    *,
    inputs: PaperRunPlanInputs,
    config_path: Path,
    strategy_path: Path,
) -> list[str]:
    return [
        inputs.freqtrade_binary,
        "trade",
        "--config",
        _safe_relative_path(config_path, inputs.root_dir),
        "--strategy",
        inputs.strategy,
        "--strategy-path",
        _safe_relative_path(strategy_path, inputs.root_dir),
    ]


def _resolved_config_path(
    inputs: PaperRunPlanInputs, readiness: dict[str, Any]
) -> Path | None:
    if inputs.config_path is not None:
        path = inputs.config_path
    else:
        config_value = readiness.get("config_path")
        if not isinstance(config_value, str) or not config_value.strip():
            return None
        path = Path(config_value)
    if path.is_absolute():
        return path
    return inputs.root_dir / path


def _resolve_workspace_path(path: Path, root_dir: Path) -> Path:
    if path.is_absolute():
        return path
    return root_dir / path


def _check(
    name: str,
    passed: bool,
    severity: str,
    message: str,
    details: dict[str, Any] | None = None,
) -> PaperRunPlanCheck:
    return PaperRunPlanCheck(
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


def _safe_relative_path(path: Path, root_dir: Path) -> str:
    try:
        return str(path.resolve().relative_to(root_dir.resolve()))
    except ValueError:
        return path.name


def _path_is_within_root(path: Path, root_dir: Path) -> bool:
    try:
        path.resolve().relative_to(root_dir.resolve())
        return True
    except ValueError:
        return False
