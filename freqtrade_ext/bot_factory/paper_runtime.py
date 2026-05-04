from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence


PAPER_RUNTIME_VALIDATION_NOTICE = (
    "Paper runtime validation is a no-process-control artifact gate. It reads "
    "only supplied local JSON and log artifacts; it does not start, stop, poll, "
    "terminate, clean up, or manage freqtrade trade, paper trading, dry-run "
    "trading, live trading, or any bot process."
)

STATUS_VALUES = {"not_started", "starting", "running", "stopping", "stopped", "failed"}
_CREDENTIAL_KEY_RE = re.compile(
    r"(?i)(^key$|api[_-]?key|secret|password|passwd|token|uid|jwt|credential|chat_id)"
)
_PRIVATE_ENV_RE = re.compile(r"\$\{[^}]*?(KEY|SECRET|TOKEN|PASSWORD|PASSWD|UID|JWT)[^}]*?\}", re.I)
_SHORT_KEYS = {"shorting", "is_short", "enter_short", "exit_short", "allow_short"}
_PROCESS_CONTROL_KEYS = {
    "process_control",
    "status_polling_started",
    "process_stop_started",
    "cleanup_executed",
}


@dataclass(frozen=True)
class PaperRuntimeValidationInputs:
    root_dir: Path
    strategy: str
    run_id: str
    process_executor_plan_path: Path
    process_metadata_path: Path
    status_snapshot_path: Path
    stdout_path: Path
    stderr_path: Path
    paper_metrics_path: Path
    output_root: Path = Path("data/paper")
    reviewer_notes: Sequence[str] = field(default_factory=list)
    command: Sequence[str] = field(default_factory=list)

    @property
    def output_dir(self) -> Path:
        return self.output_root / self.strategy / self.run_id


@dataclass(frozen=True)
class PaperRuntimeValidationCheck:
    name: str
    status: str
    severity: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_paper_runtime_artifact(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Paper runtime artifact JSON must contain an object: {path}")
    return payload


def build_paper_runtime_validation(
    inputs: PaperRuntimeValidationInputs,
    process_executor_plan: dict[str, Any],
    process_metadata: dict[str, Any],
    status_snapshot: dict[str, Any],
    paper_metrics: dict[str, Any],
) -> dict[str, Any]:
    checks: list[PaperRuntimeValidationCheck] = []
    executor_plan = _dict_or_empty(process_executor_plan.get("executor_plan"))
    executor_manifest = _dict_or_empty(process_executor_plan.get("executor_manifest"))
    planned_paths = _dict_or_empty(process_executor_plan.get("planned_paths"))
    plan_scope = _dict_or_empty(process_executor_plan.get("safety_scope"))
    metrics_scope = _dict_or_empty(paper_metrics.get("safety_scope"))

    command_preview = _command_preview_from_payload(executor_plan.get("command_preview"))

    checks.extend(
        _process_executor_plan_checks(
            inputs,
            process_executor_plan,
            executor_plan,
            executor_manifest,
            plan_scope,
            command_preview,
        )
    )
    checks.extend(_runtime_path_checks(inputs, planned_paths, executor_manifest))
    checks.extend(
        _runtime_schema_checks(
            process_metadata,
            status_snapshot,
            paper_metrics,
        )
    )
    checks.extend(
        _runtime_consistency_checks(
            inputs,
            process_executor_plan,
            process_metadata,
            status_snapshot,
            paper_metrics,
            command_preview,
        )
    )
    checks.extend(
        _runtime_safety_checks(
            process_metadata,
            status_snapshot,
            paper_metrics,
            metrics_scope,
        )
    )
    checks.append(
        _check(
            "reviewer_note_present",
            bool(inputs.reviewer_notes),
            "blocker",
            "Paper runtime validation requires at least one reviewer note.",
            {"note_count": len(inputs.reviewer_notes)},
        )
    )

    status = "pass" if all(check.status == "pass" for check in checks) else "blocked"
    generated_at = datetime.now(UTC).isoformat()
    artifact_paths = _artifact_paths(inputs)

    return {
        "generated_at": generated_at,
        "phase": "3",
        "factory": "paper_runtime_validation",
        "status": status,
        "strategy": inputs.strategy,
        "run_id": inputs.run_id,
        "input_paths": {
            "paper_process_executor_plan": _safe_relative_path(
                inputs.process_executor_plan_path, inputs.root_dir
            ),
            "process_metadata": _safe_relative_path(
                inputs.process_metadata_path, inputs.root_dir
            ),
            "status_snapshot": _safe_relative_path(
                inputs.status_snapshot_path, inputs.root_dir
            ),
            "stdout": _safe_relative_path(inputs.stdout_path, inputs.root_dir),
            "stderr": _safe_relative_path(inputs.stderr_path, inputs.root_dir),
            "paper_metrics": _safe_relative_path(
                inputs.paper_metrics_path, inputs.root_dir
            ),
        },
        "summaries": {
            "process_executor_plan_status": _safe_scalar(process_executor_plan.get("status")),
            "process_executor_plan_run_id": _safe_scalar(
                process_executor_plan.get("run_id")
            ),
            "process_started": _safe_scalar(process_metadata.get("process_started")),
            "startup_executed": _safe_scalar(process_metadata.get("startup_executed")),
            "runtime_status": _safe_scalar(status_snapshot.get("status")),
            "paper_metrics_status": _safe_scalar(paper_metrics.get("status")),
        },
        "checks": [check.to_dict() for check in checks],
        "blockers": [check.to_dict() for check in checks if check.status == "blocked"],
        "reviewer_notes": list(inputs.reviewer_notes),
        "runtime_validation": {
            "valid": status == "pass",
            "status_snapshot": _safe_scalar(status_snapshot.get("status")),
            "process_started": _safe_scalar(process_metadata.get("process_started")),
            "startup_executed": _safe_scalar(process_metadata.get("startup_executed")),
            "process_control": False,
            "status_polling_started": False,
            "process_stop_started": False,
            "cleanup_executed": False,
            "bot_startup_performed_by_validator": False,
            "polling_performed_by_validator": False,
            "stop_performed_by_validator": False,
        },
        "runtime_artifacts": {
            "process_metadata": _safe_runtime_artifact_summary(process_metadata),
            "status_snapshot": _safe_runtime_artifact_summary(status_snapshot),
            "paper_metrics": _safe_runtime_artifact_summary(paper_metrics),
            "stdout": _safe_log_summary(inputs.stdout_path, inputs.root_dir),
            "stderr": _safe_log_summary(inputs.stderr_path, inputs.root_dir),
        },
        "artifact_paths": artifact_paths,
        "safety_scope": {
            "command": "paper runtime artifact validation only",
            "bot_startup_by_validator": False,
            "freqtrade_trade_executed_by_validator": False,
            "paper_trading_started_by_validator": False,
            "dry_run_trading_started_by_validator": False,
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
        "notice": PAPER_RUNTIME_VALIDATION_NOTICE,
    }


def write_paper_runtime_validation_artifacts(
    inputs: PaperRuntimeValidationInputs, validation: dict[str, Any]
) -> None:
    output_dir = inputs.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "paper_runtime_validation.json").write_text(
        json.dumps(validation, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "runtime_artifacts_manifest.json").write_text(
        json.dumps(validation["runtime_artifacts"], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "command.txt").write_text(" ".join(inputs.command), encoding="utf-8")
    write_paper_runtime_validation_report(
        validation, output_dir / "paper_runtime_validation_report.md"
    )


def write_paper_runtime_validation_report(
    validation: dict[str, Any], path: Path
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    runtime = validation["runtime_validation"]
    lines = [
        "# Paper Runtime Artifact Validation",
        "",
        "## Summary",
        "",
        f"- Strategy: {validation['strategy']}",
        f"- Run ID: {validation['run_id']}",
        f"- Status: {validation['status']}",
        f"- Process executor plan status: {validation['summaries']['process_executor_plan_status']}",
        f"- Runtime status snapshot: {runtime['status_snapshot']}",
        f"- Process started: {runtime['process_started']}",
        f"- Startup executed: {runtime['startup_executed']}",
        f"- Process control by validator: {runtime['process_control']}",
        "",
        "## Checks",
        "",
    ]
    for check in validation["checks"]:
        lines.append(f"- {check['status'].upper()}: {check['name']} - {check['message']}")

    lines.extend(
        [
            "",
            "## Runtime Artifacts",
            "",
            f"- process metadata: `{validation['input_paths']['process_metadata']}`",
            f"- status snapshot: `{validation['input_paths']['status_snapshot']}`",
            f"- stdout log: `{validation['input_paths']['stdout']}`",
            f"- stderr log: `{validation['input_paths']['stderr']}`",
            f"- paper metrics: `{validation['input_paths']['paper_metrics']}`",
            "",
            "## Reviewer Notes",
            "",
        ]
    )
    if validation["reviewer_notes"]:
        lines.extend(f"- {note}" for note in validation["reviewer_notes"])
    else:
        lines.append("- No reviewer notes were supplied.")

    lines.extend(
        [
            "",
            "## Validation Boundary",
            "",
            f"- {PAPER_RUNTIME_VALIDATION_NOTICE}",
            "- This validation does not prove that a process was started by Bot Factory.",
            "- This validation does not poll a process or verify liveness outside the supplied local artifacts.",
            "- Local JSON, CSV, Markdown, and log artifacts remain the source of truth.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _process_executor_plan_checks(
    inputs: PaperRuntimeValidationInputs,
    process_executor_plan: dict[str, Any],
    executor_plan: dict[str, Any],
    executor_manifest: dict[str, Any],
    safety_scope: dict[str, Any],
    command_preview: list[str],
) -> list[PaperRuntimeValidationCheck]:
    return [
        _check(
            "process_executor_plan_source_is_phase3_paper_process_executor_plan",
            process_executor_plan.get("phase") == "3"
            and process_executor_plan.get("factory") == "paper_process_executor_plan",
            "blocker",
            "Runtime validation must consume a Phase 3 paper process executor plan.",
            {
                "phase": _safe_scalar(process_executor_plan.get("phase")),
                "factory": _safe_scalar(process_executor_plan.get("factory")),
            },
        ),
        _check(
            "process_executor_plan_strategy_matches",
            process_executor_plan.get("strategy") == inputs.strategy,
            "blocker",
            "Process executor plan strategy must match the runtime validation candidate.",
            {"plan_strategy": _safe_scalar(process_executor_plan.get("strategy"))},
        ),
        _check(
            "process_executor_plan_ready",
            process_executor_plan.get("status") == "ready",
            "blocker",
            "Process executor plan must be ready before runtime artifacts can pass validation.",
            {"plan_status": _safe_scalar(process_executor_plan.get("status"))},
        ),
        _check(
            "process_executor_plan_has_no_blockers",
            not process_executor_plan.get("blockers"),
            "blocker",
            "Process executor plan must have no blockers.",
            {"blocker_count": len(process_executor_plan.get("blockers") or [])},
        ),
        _check(
            "process_executor_plan_eligible",
            executor_plan.get("eligible") is True,
            "blocker",
            "Process executor plan eligibility must be true.",
            {"eligible": executor_plan.get("eligible")},
        ),
        _check(
            "process_executor_plan_records_no_plan_side_process_control",
            executor_plan.get("startup_executed") is False
            and executor_plan.get("process_started") is False
            and executor_plan.get("process_control") is False
            and executor_plan.get("status_polling_started") is False
            and executor_plan.get("process_stop_started") is False
            and executor_plan.get("cleanup_executed") is False
            and executor_plan.get("start_authorized_by_this_command") is False,
            "blocker",
            "Process executor plan must remain a no-startup/no-process-control plan.",
        ),
        _check(
            "process_executor_plan_requires_explicit_user_start_after_plan",
            executor_plan.get("requires_explicit_user_start_after_plan") is True
            and executor_manifest.get("requires_explicit_user_start_after_plan") is True,
            "blocker",
            "Process executor plan must require a separate explicit user start after planning.",
        ),
        _check(
            "process_executor_plan_has_command_preview",
            len(command_preview) >= 2,
            "blocker",
            "Process executor plan must include the reviewed command preview.",
            {"command_token_count": len(command_preview)},
        ),
        _check(
            "process_executor_plan_no_process_control_scope",
            safety_scope.get("process_control") is False
            and safety_scope.get("status_polling_started") is False
            and safety_scope.get("process_stop_started") is False
            and safety_scope.get("cleanup_executed") is False,
            "blocker",
            "Process executor plan safety scope must record no process control.",
        ),
        _check(
            "process_executor_plan_safe_scope",
            safety_scope.get("live_trading") is False
            and safety_scope.get("exchange_order_placement") is False
            and safety_scope.get("uses_api_keys_or_secrets") is False
            and safety_scope.get("metadata_contains_secrets") is False
            and safety_scope.get("leverage_above_one") is False
            and safety_scope.get("shorting") is False
            and safety_scope.get("local_artifacts_source_of_truth") is True,
            "blocker",
            "Process executor plan safety scope must remain sanitized, long-only, and local-artifact based.",
        ),
    ]


def _runtime_path_checks(
    inputs: PaperRuntimeValidationInputs,
    planned_paths: dict[str, Any],
    executor_manifest: dict[str, Any],
) -> list[PaperRuntimeValidationCheck]:
    return [
        _local_existing_path_check(
            "process_metadata",
            inputs.process_metadata_path,
            inputs.root_dir,
            "Process metadata",
        ),
        _local_existing_path_check(
            "status_snapshot",
            inputs.status_snapshot_path,
            inputs.root_dir,
            "Status snapshot",
        ),
        _local_existing_path_check(
            "stdout_log", inputs.stdout_path, inputs.root_dir, "stdout log"
        ),
        _local_existing_path_check(
            "stderr_log", inputs.stderr_path, inputs.root_dir, "stderr log"
        ),
        _local_existing_path_check(
            "paper_metrics",
            inputs.paper_metrics_path,
            inputs.root_dir,
            "Paper metrics",
        ),
        _path_match_check(
            "process_metadata_path_matches_executor_plan",
            planned_paths.get("process_metadata"),
            inputs.process_metadata_path,
            inputs.root_dir,
            "Process metadata path must match the process executor plan.",
        ),
        _path_match_check(
            "status_snapshot_path_matches_executor_plan",
            planned_paths.get("status_snapshot"),
            inputs.status_snapshot_path,
            inputs.root_dir,
            "Status snapshot path must match the process executor plan.",
        ),
        _path_match_check(
            "stdout_log_path_matches_executor_plan",
            planned_paths.get("stdout"),
            inputs.stdout_path,
            inputs.root_dir,
            "stdout log path must match the process executor plan.",
        ),
        _path_match_check(
            "stderr_log_path_matches_executor_plan",
            planned_paths.get("stderr"),
            inputs.stderr_path,
            inputs.root_dir,
            "stderr log path must match the process executor plan.",
        ),
        _path_match_check(
            "paper_metrics_path_matches_executor_plan",
            planned_paths.get("paper_metrics"),
            inputs.paper_metrics_path,
            inputs.root_dir,
            "Paper metrics path must match the process executor plan.",
        ),
        _path_match_check(
            "process_metadata_path_matches_executor_manifest",
            executor_manifest.get("process_metadata"),
            inputs.process_metadata_path,
            inputs.root_dir,
            "Process metadata path must match the executor manifest.",
        ),
        _path_match_check(
            "status_snapshot_path_matches_executor_manifest",
            executor_manifest.get("status_snapshot"),
            inputs.status_snapshot_path,
            inputs.root_dir,
            "Status snapshot path must match the executor manifest.",
        ),
        _path_match_check(
            "stdout_log_path_matches_executor_manifest",
            executor_manifest.get("stdout_log"),
            inputs.stdout_path,
            inputs.root_dir,
            "stdout log path must match the executor manifest.",
        ),
        _path_match_check(
            "stderr_log_path_matches_executor_manifest",
            executor_manifest.get("stderr_log"),
            inputs.stderr_path,
            inputs.root_dir,
            "stderr log path must match the executor manifest.",
        ),
        _path_match_check(
            "paper_metrics_path_matches_executor_manifest",
            executor_manifest.get("paper_metrics"),
            inputs.paper_metrics_path,
            inputs.root_dir,
            "Paper metrics path must match the executor manifest.",
        ),
    ]


def _runtime_schema_checks(
    process_metadata: dict[str, Any],
    status_snapshot: dict[str, Any],
    paper_metrics: dict[str, Any],
) -> list[PaperRuntimeValidationCheck]:
    trade_counts = _dict_or_empty(paper_metrics.get("trade_counts"))
    return [
        _required_fields_check(
            "process_metadata_required_fields_present",
            process_metadata,
            {
                "strategy",
                "run_id",
                "process_started",
                "startup_executed",
                "pid",
                "command",
                "stdout_log",
                "stderr_log",
                "status_snapshot",
                "paper_metrics",
                "notice",
            },
            "Process metadata must include required runtime fields.",
        ),
        _required_fields_check(
            "status_snapshot_required_fields_present",
            status_snapshot,
            {
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
            },
            "Status snapshot must include required runtime status fields.",
        ),
        _required_fields_check(
            "paper_metrics_required_fields_present",
            paper_metrics,
            {
                "generated_at",
                "strategy",
                "run_id",
                "source",
                "status",
                "trade_counts",
                "profit",
                "risk",
                "safety_scope",
            },
            "Paper metrics must include required runtime metric fields.",
        ),
        _check(
            "status_snapshot_status_known",
            status_snapshot.get("status") in STATUS_VALUES,
            "blocker",
            "Status snapshot must use a known local status value.",
            {"status": _safe_scalar(status_snapshot.get("status"))},
        ),
        _check(
            "paper_metrics_source_is_local",
            paper_metrics.get("source") == "local_paper_artifacts",
            "blocker",
            "Paper metrics must use local paper artifacts as source.",
            {"source": _safe_scalar(paper_metrics.get("source"))},
        ),
        _check(
            "paper_metrics_status_matches_snapshot_status",
            paper_metrics.get("status") == status_snapshot.get("status"),
            "blocker",
            "Paper metrics status must match the status snapshot.",
            {
                "paper_metrics_status": _safe_scalar(paper_metrics.get("status")),
                "status_snapshot": _safe_scalar(status_snapshot.get("status")),
            },
        ),
        _check(
            "paper_metrics_trade_counts_are_consistent",
            _non_negative_int(trade_counts.get("open"))
            and _non_negative_int(trade_counts.get("closed"))
            and _non_negative_int(trade_counts.get("total"))
            and trade_counts.get("open") + trade_counts.get("closed")
            == trade_counts.get("total"),
            "blocker",
            "Paper metrics trade counts must be non-negative and internally consistent.",
            {
                "open": _safe_scalar(trade_counts.get("open")),
                "closed": _safe_scalar(trade_counts.get("closed")),
                "total": _safe_scalar(trade_counts.get("total")),
            },
        ),
    ]


def _runtime_consistency_checks(
    inputs: PaperRuntimeValidationInputs,
    process_executor_plan: dict[str, Any],
    process_metadata: dict[str, Any],
    status_snapshot: dict[str, Any],
    paper_metrics: dict[str, Any],
    command_preview: list[str],
) -> list[PaperRuntimeValidationCheck]:
    plan_run_id = process_executor_plan.get("run_id")
    metadata_command = _command_preview_from_payload(process_metadata.get("command"))
    return [
        _check(
            "runtime_strategy_matches_candidate",
            process_metadata.get("strategy") == inputs.strategy
            and status_snapshot.get("strategy") == inputs.strategy
            and paper_metrics.get("strategy") == inputs.strategy,
            "blocker",
            "Runtime artifacts must all reference the same strategy candidate.",
            {
                "process_metadata_strategy": _safe_scalar(process_metadata.get("strategy")),
                "status_snapshot_strategy": _safe_scalar(status_snapshot.get("strategy")),
                "paper_metrics_strategy": _safe_scalar(paper_metrics.get("strategy")),
            },
        ),
        _check(
            "runtime_run_id_matches_process_executor_plan",
            process_metadata.get("run_id") == plan_run_id
            and status_snapshot.get("run_id") == plan_run_id
            and paper_metrics.get("run_id") == plan_run_id,
            "blocker",
            "Runtime artifact run IDs must match the process executor plan run ID.",
            {
                "plan_run_id": _safe_scalar(plan_run_id),
                "process_metadata_run_id": _safe_scalar(process_metadata.get("run_id")),
                "status_snapshot_run_id": _safe_scalar(status_snapshot.get("run_id")),
                "paper_metrics_run_id": _safe_scalar(paper_metrics.get("run_id")),
            },
        ),
        _path_match_check(
            "process_metadata_status_snapshot_path_matches_input",
            process_metadata.get("status_snapshot"),
            inputs.status_snapshot_path,
            inputs.root_dir,
            "Process metadata status snapshot path must match the supplied status snapshot.",
        ),
        _path_match_check(
            "process_metadata_paper_metrics_path_matches_input",
            process_metadata.get("paper_metrics"),
            inputs.paper_metrics_path,
            inputs.root_dir,
            "Process metadata paper metrics path must match the supplied paper metrics.",
        ),
        _path_match_check(
            "process_metadata_stdout_path_matches_input",
            process_metadata.get("stdout_log"),
            inputs.stdout_path,
            inputs.root_dir,
            "Process metadata stdout path must match the supplied stdout log.",
        ),
        _path_match_check(
            "process_metadata_stderr_path_matches_input",
            process_metadata.get("stderr_log"),
            inputs.stderr_path,
            inputs.root_dir,
            "Process metadata stderr path must match the supplied stderr log.",
        ),
        _check(
            "process_metadata_command_matches_executor_plan",
            bool(command_preview) and metadata_command == command_preview,
            "blocker",
            "Process metadata command must match the reviewed process executor plan command.",
            {
                "metadata_command": _command_string(metadata_command) or None,
                "executor_plan_command": _command_string(command_preview) or None,
            },
        ),
    ]


def _runtime_safety_checks(
    process_metadata: dict[str, Any],
    status_snapshot: dict[str, Any],
    paper_metrics: dict[str, Any],
    metrics_scope: dict[str, Any],
) -> list[PaperRuntimeValidationCheck]:
    payloads = {
        "process_metadata": process_metadata,
        "status_snapshot": status_snapshot,
        "paper_metrics": paper_metrics,
    }
    credential_findings = _credential_findings(payloads)
    private_env_findings = _private_env_findings(payloads)
    leverage_findings = _leverage_findings(payloads)
    short_findings = _short_findings(payloads)
    process_control_findings = _truthy_key_findings(payloads, _PROCESS_CONTROL_KEYS)
    return [
        _check(
            "runtime_no_live_or_exchange_order_scope",
            status_snapshot.get("live_trading") is False
            and status_snapshot.get("exchange_order_placement") is False
            and metrics_scope.get("live_trading") is False
            and metrics_scope.get("exchange_order_placement") is False
            and metrics_scope.get("canary_live_trading", False) is False,
            "blocker",
            "Runtime artifacts must not record live/canary trading or exchange order placement.",
            {
                "status_live_trading": status_snapshot.get("live_trading"),
                "status_exchange_order_placement": status_snapshot.get(
                    "exchange_order_placement"
                ),
                "metrics_live_trading": metrics_scope.get("live_trading"),
                "metrics_exchange_order_placement": metrics_scope.get(
                    "exchange_order_placement"
                ),
            },
        ),
        _check(
            "runtime_metadata_no_credential_values",
            not credential_findings,
            "blocker",
            "Runtime metadata must not contain non-empty API keys, secrets, tokens, UIDs, or passwords.",
            {"credential_key_paths": [finding["path"] for finding in credential_findings]},
        ),
        _check(
            "runtime_metadata_no_private_env_references",
            not private_env_findings,
            "blocker",
            "Runtime metadata must not contain private environment variable references.",
            {"env_reference_paths": private_env_findings},
        ),
        _check(
            "runtime_no_leverage_above_one",
            metrics_scope.get("leverage_above_one") is False and not leverage_findings,
            "blocker",
            "Runtime artifacts must not record leverage above 1.0.",
            {"leverage_paths": [finding["path"] for finding in leverage_findings]},
        ),
        _check(
            "runtime_no_shorting",
            metrics_scope.get("shorting") is False and not short_findings,
            "blocker",
            "Runtime artifacts must not record shorting.",
            {"short_paths": [finding["path"] for finding in short_findings]},
        ),
        _check(
            "runtime_metadata_sanitized_scope",
            metrics_scope.get("metadata_contains_secrets") is False
            and metrics_scope.get("uses_api_keys_or_secrets", False) is False,
            "blocker",
            "Paper metrics safety scope must record sanitized metadata.",
            {
                "metadata_contains_secrets": metrics_scope.get("metadata_contains_secrets"),
                "uses_api_keys_or_secrets": metrics_scope.get(
                    "uses_api_keys_or_secrets"
                ),
            },
        ),
        _check(
            "runtime_local_artifacts_source_of_truth",
            metrics_scope.get("local_artifacts_source_of_truth") is True,
            "blocker",
            "Paper metrics must keep local artifacts as the source of truth.",
        ),
        _check(
            "runtime_no_process_control_scope",
            not process_control_findings,
            "blocker",
            "Runtime validation artifacts must not record process control, polling, stop, or cleanup execution.",
            {"process_control_paths": [finding["path"] for finding in process_control_findings]},
        ),
    ]


def _local_existing_path_check(
    name: str, path: Path, root_dir: Path, label: str
) -> PaperRuntimeValidationCheck:
    return _check(
        f"{name}_within_workspace_and_present",
        _path_is_within_root(path, root_dir) and path.is_file(),
        "blocker",
        f"{label} path must resolve inside the workspace and exist locally.",
        {"path": _safe_relative_path(path, root_dir)},
    )


def _path_match_check(
    name: str,
    payload_path: Any,
    expected_path: Path,
    root_dir: Path,
    message: str,
) -> PaperRuntimeValidationCheck:
    payload_resolved = _path_from_payload(payload_path, root_dir)
    return _check(
        name,
        payload_resolved is not None and _same_resolved_path(payload_resolved, expected_path),
        "blocker",
        message,
        {
            "payload_path": _safe_relative_path(payload_resolved, root_dir),
            "expected_path": _safe_relative_path(expected_path, root_dir),
        },
    )


def _required_fields_check(
    name: str,
    payload: dict[str, Any],
    required: set[str],
    message: str,
) -> PaperRuntimeValidationCheck:
    missing = sorted(required - set(payload.keys()))
    return _check(name, not missing, "blocker", message, {"missing_fields": missing})


def _artifact_paths(inputs: PaperRuntimeValidationInputs) -> dict[str, str]:
    output_dir = inputs.output_dir
    return {
        "paper_runtime_validation": _safe_relative_path(
            output_dir / "paper_runtime_validation.json", inputs.root_dir
        ),
        "paper_runtime_validation_report": _safe_relative_path(
            output_dir / "paper_runtime_validation_report.md", inputs.root_dir
        ),
        "runtime_artifacts_manifest": _safe_relative_path(
            output_dir / "runtime_artifacts_manifest.json", inputs.root_dir
        ),
        "command": _safe_relative_path(output_dir / "command.txt", inputs.root_dir),
    }


def _safe_runtime_artifact_summary(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "strategy": _safe_scalar(payload.get("strategy")),
        "run_id": _safe_scalar(payload.get("run_id")),
        "status": _safe_scalar(payload.get("status")),
        "process_started": _safe_scalar(payload.get("process_started")),
        "startup_executed": _safe_scalar(payload.get("startup_executed")),
    }


def _safe_log_summary(path: Path, root_dir: Path) -> dict[str, Any]:
    return {
        "path": _safe_relative_path(path, root_dir),
        "exists": path.is_file(),
        "size_bytes": path.stat().st_size if path.is_file() else None,
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


def _path_is_within_root(path: Path, root_dir: Path) -> bool:
    try:
        path.resolve().relative_to(root_dir.resolve())
        return True
    except ValueError:
        return False


def _same_resolved_path(left: Path, right: Path) -> bool:
    return left.resolve() == right.resolve()


def _check(
    name: str,
    passed: bool,
    severity: str,
    message: str,
    details: dict[str, Any] | None = None,
) -> PaperRuntimeValidationCheck:
    return PaperRuntimeValidationCheck(
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


def _non_negative_int(value: Any) -> bool:
    return isinstance(value, int) and value >= 0


def _credential_findings(payload: Any, prefix: str = "") -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if _CREDENTIAL_KEY_RE.search(str(key)) and _has_credential_value(value):
                findings.append({"path": path})
            findings.extend(_credential_findings(value, path))
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            findings.extend(_credential_findings(value, f"{prefix}[{index}]"))
    return findings


def _private_env_findings(payload: Any, prefix: str = "") -> list[str]:
    findings: list[str] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            findings.extend(_private_env_findings(value, path))
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            findings.extend(_private_env_findings(value, f"{prefix}[{index}]"))
    elif isinstance(payload, str) and _PRIVATE_ENV_RE.search(payload):
        findings.append(prefix)
    return findings


def _leverage_findings(payload: Any, prefix: str = "") -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if "leverage" in str(key).lower() and _numeric_above_one(value):
                findings.append({"path": path})
            findings.extend(_leverage_findings(value, path))
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            findings.extend(_leverage_findings(value, f"{prefix}[{index}]"))
    return findings


def _short_findings(payload: Any, prefix: str = "") -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if str(key).lower() in _SHORT_KEYS and _truthy(value):
                findings.append({"path": path})
            findings.extend(_short_findings(value, path))
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            findings.extend(_short_findings(value, f"{prefix}[{index}]"))
    return findings


def _truthy_key_findings(
    payload: Any, keys: set[str], prefix: str = ""
) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if str(key).lower() in keys and _truthy(value):
                findings.append({"path": path})
            findings.extend(_truthy_key_findings(value, keys, path))
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            findings.extend(_truthy_key_findings(value, keys, f"{prefix}[{index}]"))
    return findings


def _has_credential_value(value: Any) -> bool:
    if value is None or value is False:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, list):
        return any(_has_credential_value(item) for item in value)
    if isinstance(value, dict):
        return any(_has_credential_value(item) for item in value.values())
    return True


def _numeric_above_one(value: Any) -> bool:
    if isinstance(value, bool):
        return value is True
    try:
        return float(value) > 1.0
    except (TypeError, ValueError):
        return False


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "short"}
    return bool(value)
