from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence


PAPER_DRIFT_REPORT_NOTICE = (
    "Paper/backtest drift reporting is a no-process-control artifact analysis. "
    "It reads only supplied local historical, walk-forward, training, runtime "
    "validation, and paper metric JSON artifacts; it does not start, stop, "
    "poll, terminate, clean up, promote, or manage freqtrade trade, paper "
    "trading, dry-run trading, live trading, or any bot process."
)

_CREDENTIAL_KEY_RE = re.compile(
    r"(?i)(^key$|api[_-]?key|secret|password|passwd|token|uid|jwt|credential|chat_id)"
)
_PRIVATE_ENV_RE = re.compile(
    r"\$\{[^}]*?(KEY|SECRET|TOKEN|PASSWORD|PASSWD|UID|JWT)[^}]*?\}", re.I
)


@dataclass(frozen=True)
class PaperDriftReportInputs:
    root_dir: Path
    strategy: str
    run_id: str
    historical_metrics_path: Path
    walk_forward_metrics_path: Path
    training_manifest_path: Path
    paper_runtime_validation_path: Path
    paper_metrics_path: Path | None = None
    output_root: Path = Path("data/paper")
    max_return_drift_pct: float = 5.0
    max_drawdown_drift_pct: float = 5.0
    reviewer_notes: Sequence[str] = field(default_factory=list)
    command: Sequence[str] = field(default_factory=list)

    @property
    def output_dir(self) -> Path:
        return self.output_root / self.strategy / self.run_id


@dataclass(frozen=True)
class PaperDriftReportCheck:
    name: str
    status: str
    severity: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_paper_drift_artifact(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Paper drift artifact JSON must contain an object: {path}")
    return payload


def build_paper_drift_report(
    inputs: PaperDriftReportInputs,
    historical_metrics: dict[str, Any],
    walk_forward_metrics: dict[str, Any],
    training_manifest: dict[str, Any],
    paper_runtime_validation: dict[str, Any],
    paper_metrics: dict[str, Any],
) -> dict[str, Any]:
    checks: list[PaperDriftReportCheck] = []
    runtime_validation = _dict_or_empty(
        paper_runtime_validation.get("runtime_validation")
    )
    runtime_summaries = _dict_or_empty(paper_runtime_validation.get("summaries"))
    runtime_scope = _dict_or_empty(paper_runtime_validation.get("safety_scope"))
    paper_scope = _dict_or_empty(paper_metrics.get("safety_scope"))
    walk_forward_summary = _dict_or_empty(walk_forward_metrics.get("summary"))

    reference = _reference_summary(
        historical_metrics, walk_forward_metrics, training_manifest
    )
    paper = _paper_metrics_summary(paper_metrics)
    drift = _drift_summary(inputs, reference, paper)

    checks.extend(_path_checks(inputs))
    checks.extend(
        _source_checks(
            inputs,
            historical_metrics,
            walk_forward_metrics,
            training_manifest,
            paper_runtime_validation,
            paper_metrics,
            runtime_validation,
            runtime_summaries,
        )
    )
    checks.extend(
        _reference_quality_checks(walk_forward_metrics, training_manifest)
    )
    checks.extend(
        _metric_availability_checks(reference, paper, walk_forward_summary)
    )
    checks.extend(_drift_checks(inputs, drift, paper))
    checks.extend(
        _safety_checks(
            historical_metrics,
            walk_forward_metrics,
            training_manifest,
            paper_runtime_validation,
            paper_metrics,
            runtime_scope,
            paper_scope,
        )
    )
    checks.append(
        _check(
            "reviewer_note_present",
            bool(inputs.reviewer_notes),
            "blocker",
            "Paper/backtest drift reporting requires at least one reviewer note.",
            {"note_count": len(inputs.reviewer_notes)},
        )
    )

    status = _resolve_status(checks)
    generated_at = datetime.now(UTC).isoformat()
    artifact_paths = _artifact_paths(inputs)

    return {
        "generated_at": generated_at,
        "phase": "3",
        "factory": "paper_drift_report",
        "status": status,
        "strategy": inputs.strategy,
        "run_id": inputs.run_id,
        "input_paths": {
            "historical_metrics": _safe_relative_path(
                inputs.historical_metrics_path, inputs.root_dir
            ),
            "walk_forward_metrics": _safe_relative_path(
                inputs.walk_forward_metrics_path, inputs.root_dir
            ),
            "training_manifest": _safe_relative_path(
                inputs.training_manifest_path, inputs.root_dir
            ),
            "paper_runtime_validation": _safe_relative_path(
                inputs.paper_runtime_validation_path, inputs.root_dir
            ),
            "paper_metrics": _safe_relative_path(
                inputs.paper_metrics_path, inputs.root_dir
            ),
        },
        "checks": [check.to_dict() for check in checks],
        "blockers": [check.to_dict() for check in checks if check.status == "blocked"],
        "failures": [check.to_dict() for check in checks if check.status == "fail"],
        "reviewer_notes": list(inputs.reviewer_notes),
        "reference_evidence": reference,
        "paper_evidence": paper,
        "drift": drift,
        "drift_report": {
            "valid": status == "pass",
            "paper_promotion_eligible": False,
            "promotion_authorized_by_this_command": False,
            "process_control": False,
            "status_polling_started": False,
            "process_stop_started": False,
            "cleanup_executed": False,
        },
        "artifact_paths": artifact_paths,
        "safety_scope": {
            "command": "paper/backtest drift reporting only",
            "bot_startup_by_reporter": False,
            "freqtrade_trade_executed_by_reporter": False,
            "paper_trading_started_by_reporter": False,
            "dry_run_trading_started_by_reporter": False,
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
        "notice": PAPER_DRIFT_REPORT_NOTICE,
    }


def write_paper_drift_report_artifacts(
    inputs: PaperDriftReportInputs, report: dict[str, Any]
) -> None:
    output_dir = inputs.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "paper_drift_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "drift_metrics.json").write_text(
        json.dumps(report["drift"], indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "command.txt").write_text(" ".join(inputs.command), encoding="utf-8")
    write_paper_drift_markdown_report(report, output_dir / "paper_drift_report.md")


def write_paper_drift_markdown_report(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    reference = report["reference_evidence"]
    paper = report["paper_evidence"]
    drift = report["drift"]
    lines = [
        "# Paper/Backtest Drift Report",
        "",
        "## Summary",
        "",
        f"- Strategy: {report['strategy']}",
        f"- Run ID: {report['run_id']}",
        f"- Status: {report['status']}",
        f"- Historical total return pct: {_format_number(reference['historical_total_return_pct'])}",
        f"- Paper total return pct: {_format_number(paper['total_return_pct'])}",
        f"- Return drift pct points: {_format_number(drift['return_vs_historical_pct_points'])}",
        f"- Historical max drawdown pct: {_format_number(reference['historical_max_drawdown_pct'])}",
        f"- Paper max drawdown pct: {_format_number(paper['max_drawdown_pct'])}",
        f"- Drawdown drift pct points: {_format_number(drift['drawdown_vs_historical_pct_points'])}",
        f"- Promotion authorized by this command: {report['drift_report']['promotion_authorized_by_this_command']}",
        "",
        "## Checks",
        "",
    ]
    for check in report["checks"]:
        lines.append(f"- {check['status'].upper()}: {check['name']} - {check['message']}")

    lines.extend(
        [
            "",
            "## Input Artifacts",
            "",
            f"- historical metrics: `{report['input_paths']['historical_metrics']}`",
            f"- walk-forward metrics: `{report['input_paths']['walk_forward_metrics']}`",
            f"- training manifest: `{report['input_paths']['training_manifest']}`",
            f"- paper runtime validation: `{report['input_paths']['paper_runtime_validation']}`",
            f"- paper metrics: `{report['input_paths']['paper_metrics']}`",
            "",
            "## Reviewer Notes",
            "",
        ]
    )
    if report["reviewer_notes"]:
        lines.extend(f"- {note}" for note in report["reviewer_notes"])
    else:
        lines.append("- No reviewer notes were supplied.")

    lines.extend(
        [
            "",
            "## Reporting Boundary",
            "",
            f"- {PAPER_DRIFT_REPORT_NOTICE}",
            "- This report is not a promotion approval.",
            "- This report does not verify process liveness outside supplied local artifacts.",
            "- Local JSON, CSV, Markdown, and log artifacts remain the source of truth.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _path_checks(inputs: PaperDriftReportInputs) -> list[PaperDriftReportCheck]:
    return [
        _local_existing_path_check(
            "historical_metrics",
            inputs.historical_metrics_path,
            inputs.root_dir,
            "Historical metrics",
        ),
        _local_existing_path_check(
            "walk_forward_metrics",
            inputs.walk_forward_metrics_path,
            inputs.root_dir,
            "Walk-forward metrics",
        ),
        _local_existing_path_check(
            "training_manifest",
            inputs.training_manifest_path,
            inputs.root_dir,
            "Training manifest",
        ),
        _local_existing_path_check(
            "paper_runtime_validation",
            inputs.paper_runtime_validation_path,
            inputs.root_dir,
            "Paper runtime validation",
        ),
        _local_existing_path_check(
            "paper_metrics",
            inputs.paper_metrics_path,
            inputs.root_dir,
            "Paper metrics",
        ),
    ]


def _source_checks(
    inputs: PaperDriftReportInputs,
    historical_metrics: dict[str, Any],
    walk_forward_metrics: dict[str, Any],
    training_manifest: dict[str, Any],
    paper_runtime_validation: dict[str, Any],
    paper_metrics: dict[str, Any],
    runtime_validation: dict[str, Any],
    runtime_summaries: dict[str, Any],
) -> list[PaperDriftReportCheck]:
    expected_runtime_run_id = runtime_summaries.get("process_executor_plan_run_id")
    runtime_input_paths = _dict_or_empty(paper_runtime_validation.get("input_paths"))
    return [
        _check(
            "historical_strategy_matches",
            historical_metrics.get("strategy_name") == inputs.strategy,
            "blocker",
            "Historical metrics strategy must match the drift report candidate.",
            {"historical_strategy": _safe_scalar(historical_metrics.get("strategy_name"))},
        ),
        _check(
            "walk_forward_source_is_phase2_completed",
            walk_forward_metrics.get("phase") == "2"
            and walk_forward_metrics.get("status") == "completed",
            "blocker",
            "Walk-forward metrics must be a completed Phase 2 artifact.",
            {
                "phase": _safe_scalar(walk_forward_metrics.get("phase")),
                "status": _safe_scalar(walk_forward_metrics.get("status")),
            },
        ),
        _check(
            "walk_forward_strategy_matches",
            walk_forward_metrics.get("strategy") == inputs.strategy,
            "blocker",
            "Walk-forward metrics strategy must match the drift report candidate.",
            {"walk_forward_strategy": _safe_scalar(walk_forward_metrics.get("strategy"))},
        ),
        _check(
            "training_source_is_phase2_freqai_training",
            training_manifest.get("phase") == "2"
            and training_manifest.get("factory") == "freqai_training"
            and training_manifest.get("status") == "completed",
            "blocker",
            "Training manifest must be a completed Phase 2 FreqAI training artifact.",
            {
                "phase": _safe_scalar(training_manifest.get("phase")),
                "factory": _safe_scalar(training_manifest.get("factory")),
                "status": _safe_scalar(training_manifest.get("status")),
            },
        ),
        _check(
            "training_strategy_matches",
            training_manifest.get("strategy") == inputs.strategy,
            "blocker",
            "Training manifest strategy must match the drift report candidate.",
            {"training_strategy": _safe_scalar(training_manifest.get("strategy"))},
        ),
        _check(
            "runtime_validation_source_is_phase3_paper_runtime_validation",
            paper_runtime_validation.get("phase") == "3"
            and paper_runtime_validation.get("factory") == "paper_runtime_validation",
            "blocker",
            "Paper drift reporting must consume a Phase 3 paper runtime validation artifact.",
            {
                "phase": _safe_scalar(paper_runtime_validation.get("phase")),
                "factory": _safe_scalar(paper_runtime_validation.get("factory")),
            },
        ),
        _check(
            "runtime_validation_strategy_matches",
            paper_runtime_validation.get("strategy") == inputs.strategy,
            "blocker",
            "Runtime validation strategy must match the drift report candidate.",
            {
                "runtime_validation_strategy": _safe_scalar(
                    paper_runtime_validation.get("strategy")
                )
            },
        ),
        _check(
            "runtime_validation_passed",
            paper_runtime_validation.get("status") == "pass"
            and runtime_validation.get("valid") is True,
            "blocker",
            "Paper runtime validation must pass before drift can be evaluated.",
            {
                "status": _safe_scalar(paper_runtime_validation.get("status")),
                "valid": _safe_scalar(runtime_validation.get("valid")),
            },
        ),
        _path_match_check(
            "paper_metrics_path_matches_runtime_validation",
            runtime_input_paths.get("paper_metrics"),
            inputs.paper_metrics_path,
            inputs.root_dir,
            "Paper metrics path must match the artifact consumed by runtime validation.",
        ),
        _check(
            "paper_metrics_source_is_local",
            paper_metrics.get("source") == "local_paper_artifacts",
            "blocker",
            "Paper metrics must use local paper artifacts as source.",
            {"source": _safe_scalar(paper_metrics.get("source"))},
        ),
        _check(
            "paper_metrics_strategy_matches",
            paper_metrics.get("strategy") == inputs.strategy,
            "blocker",
            "Paper metrics strategy must match the drift report candidate.",
            {"paper_metrics_strategy": _safe_scalar(paper_metrics.get("strategy"))},
        ),
        _check(
            "paper_metrics_run_id_matches_runtime_validation",
            bool(expected_runtime_run_id)
            and paper_metrics.get("run_id") == expected_runtime_run_id,
            "blocker",
            "Paper metrics run ID must match the runtime validation executor plan run ID.",
            {
                "expected_runtime_run_id": _safe_scalar(expected_runtime_run_id),
                "paper_metrics_run_id": _safe_scalar(paper_metrics.get("run_id")),
            },
        ),
    ]


def _reference_quality_checks(
    walk_forward_metrics: dict[str, Any], training_manifest: dict[str, Any]
) -> list[PaperDriftReportCheck]:
    return [
        _check(
            "walk_forward_recommendation_passed",
            walk_forward_metrics.get("recommendation") == "pass",
            "failure",
            "Walk-forward recommendation must pass before drift reporting can support promotion review.",
            {"recommendation": _safe_scalar(walk_forward_metrics.get("recommendation"))},
            failure_status="fail",
        ),
        _check(
            "training_recommendation_passed",
            training_manifest.get("recommendation") == "pass",
            "failure",
            "Training recommendation must pass before drift reporting can support promotion review.",
            {"recommendation": _safe_scalar(training_manifest.get("recommendation"))},
            failure_status="fail",
        ),
    ]


def _metric_availability_checks(
    reference: dict[str, Any],
    paper: dict[str, Any],
    walk_forward_summary: dict[str, Any],
) -> list[PaperDriftReportCheck]:
    return [
        _check(
            "historical_return_metric_present",
            _is_number(reference.get("historical_total_return_pct")),
            "blocker",
            "Historical total_return_pct must be numeric.",
        ),
        _check(
            "historical_drawdown_metric_present",
            _is_number(reference.get("historical_max_drawdown_pct")),
            "blocker",
            "Historical max_drawdown_pct must be numeric.",
        ),
        _check(
            "walk_forward_return_metric_present",
            _is_number(walk_forward_summary.get("total_return_pct")),
            "blocker",
            "Walk-forward summary total_return_pct must be numeric.",
        ),
        _check(
            "walk_forward_drawdown_metric_present",
            _is_number(walk_forward_summary.get("max_drawdown_pct_any_window")),
            "blocker",
            "Walk-forward summary max_drawdown_pct_any_window must be numeric.",
        ),
        _check(
            "paper_return_metric_present",
            _is_number(paper.get("total_return_pct")),
            "blocker",
            "Paper metrics total return percentage must be numeric.",
        ),
        _check(
            "paper_drawdown_metric_present",
            _is_number(paper.get("max_drawdown_pct")),
            "blocker",
            "Paper metrics max drawdown percentage must be numeric.",
        ),
        _check(
            "paper_trade_count_metric_present",
            _non_negative_int(paper.get("trade_count")),
            "blocker",
            "Paper metrics trade count must be a non-negative integer.",
        ),
    ]


def _drift_checks(
    inputs: PaperDriftReportInputs,
    drift: dict[str, Any],
    paper: dict[str, Any],
) -> list[PaperDriftReportCheck]:
    return [
        _check(
            "paper_trade_count_positive",
            _non_negative_int(paper.get("trade_count"))
            and int(paper.get("trade_count")) > 0,
            "failure",
            "Paper metrics must include at least one trade before drift can support review.",
            {"paper_trade_count": _safe_scalar(paper.get("trade_count"))},
            failure_status="fail",
        ),
        _check(
            "paper_return_not_worse_than_historical_threshold",
            _number_gte(
                drift.get("return_vs_historical_pct_points"),
                -inputs.max_return_drift_pct,
            ),
            "failure",
            "Paper total return drift versus historical backtest must stay within threshold.",
            {
                "delta_pct_points": _safe_scalar(
                    drift.get("return_vs_historical_pct_points")
                ),
                "minimum_allowed": -inputs.max_return_drift_pct,
            },
            failure_status="fail",
        ),
        _check(
            "paper_return_not_worse_than_walk_forward_threshold",
            _number_gte(
                drift.get("return_vs_walk_forward_pct_points"),
                -inputs.max_return_drift_pct,
            ),
            "failure",
            "Paper total return drift versus walk-forward evidence must stay within threshold.",
            {
                "delta_pct_points": _safe_scalar(
                    drift.get("return_vs_walk_forward_pct_points")
                ),
                "minimum_allowed": -inputs.max_return_drift_pct,
            },
            failure_status="fail",
        ),
        _check(
            "paper_drawdown_not_worse_than_historical_threshold",
            _number_lte(
                drift.get("drawdown_vs_historical_pct_points"),
                inputs.max_drawdown_drift_pct,
            ),
            "failure",
            "Paper max drawdown drift versus historical backtest must stay within threshold.",
            {
                "delta_pct_points": _safe_scalar(
                    drift.get("drawdown_vs_historical_pct_points")
                ),
                "maximum_allowed": inputs.max_drawdown_drift_pct,
            },
            failure_status="fail",
        ),
        _check(
            "paper_drawdown_not_worse_than_walk_forward_threshold",
            _number_lte(
                drift.get("drawdown_vs_walk_forward_any_window_pct_points"),
                inputs.max_drawdown_drift_pct,
            ),
            "failure",
            "Paper max drawdown drift versus walk-forward max drawdown must stay within threshold.",
            {
                "delta_pct_points": _safe_scalar(
                    drift.get("drawdown_vs_walk_forward_any_window_pct_points")
                ),
                "maximum_allowed": inputs.max_drawdown_drift_pct,
            },
            failure_status="fail",
        ),
    ]


def _safety_checks(
    historical_metrics: dict[str, Any],
    walk_forward_metrics: dict[str, Any],
    training_manifest: dict[str, Any],
    paper_runtime_validation: dict[str, Any],
    paper_metrics: dict[str, Any],
    runtime_scope: dict[str, Any],
    paper_scope: dict[str, Any],
) -> list[PaperDriftReportCheck]:
    credential_findings = _credential_findings(
        {
            "historical_metrics": historical_metrics,
            "walk_forward_metrics": walk_forward_metrics,
            "training_manifest": training_manifest,
            "paper_runtime_validation": paper_runtime_validation,
            "paper_metrics": paper_metrics,
        }
    )
    private_env_findings = _private_env_findings(
        {
            "historical_metrics": historical_metrics,
            "walk_forward_metrics": walk_forward_metrics,
            "training_manifest": training_manifest,
            "paper_runtime_validation": paper_runtime_validation,
            "paper_metrics": paper_metrics,
        }
    )
    return [
        _check(
            "runtime_validation_no_process_control_scope",
            runtime_scope.get("process_control") is False
            and runtime_scope.get("status_polling_started") is False
            and runtime_scope.get("process_stop_started") is False
            and runtime_scope.get("cleanup_executed") is False,
            "blocker",
            "Runtime validation safety scope must record no process control by the validator.",
        ),
        _check(
            "paper_metrics_safe_scope",
            paper_scope.get("live_trading") is False
            and paper_scope.get("canary_live_trading", False) is False
            and paper_scope.get("exchange_order_placement") is False
            and paper_scope.get("uses_api_keys_or_secrets", False) is False
            and paper_scope.get("metadata_contains_secrets") is False
            and paper_scope.get("leverage_above_one") is False
            and paper_scope.get("shorting") is False
            and paper_scope.get("local_artifacts_source_of_truth") is True,
            "blocker",
            "Paper metrics safety scope must be sanitized, long-only, and local-artifact based.",
            {
                "live_trading": paper_scope.get("live_trading"),
                "exchange_order_placement": paper_scope.get("exchange_order_placement"),
                "leverage_above_one": paper_scope.get("leverage_above_one"),
                "shorting": paper_scope.get("shorting"),
            },
        ),
        _check(
            "paper_metrics_no_process_control_scope",
            paper_scope.get("process_control") is False
            and paper_scope.get("status_polling_started") is False
            and paper_scope.get("process_stop_started") is False
            and paper_scope.get("cleanup_executed") is False,
            "blocker",
            "Paper metrics must not record process control, polling, stop, or cleanup execution.",
        ),
        _check(
            "drift_inputs_no_credential_values",
            not credential_findings,
            "blocker",
            "Drift input metadata must not contain non-empty API keys, secrets, tokens, UIDs, or passwords.",
            {"credential_key_paths": [finding["path"] for finding in credential_findings]},
        ),
        _check(
            "drift_inputs_no_private_env_references",
            not private_env_findings,
            "blocker",
            "Drift input metadata must not contain private environment variable references.",
            {"env_reference_paths": private_env_findings},
        ),
    ]


def _reference_summary(
    historical_metrics: dict[str, Any],
    walk_forward_metrics: dict[str, Any],
    training_manifest: dict[str, Any],
) -> dict[str, Any]:
    walk_forward_summary = _dict_or_empty(walk_forward_metrics.get("summary"))
    return {
        "historical_total_return_pct": _number_or_none(
            historical_metrics.get("total_return_pct")
        ),
        "historical_max_drawdown_pct": _number_or_none(
            historical_metrics.get("max_drawdown_pct")
        ),
        "historical_trade_count": _int_or_none(historical_metrics.get("trade_count")),
        "historical_profit_factor": _number_or_none(
            historical_metrics.get("profit_factor")
        ),
        "walk_forward_recommendation": _safe_scalar(
            walk_forward_metrics.get("recommendation")
        ),
        "walk_forward_total_return_pct": _number_or_none(
            walk_forward_summary.get("total_return_pct")
        ),
        "walk_forward_max_drawdown_pct_any_window": _number_or_none(
            walk_forward_summary.get("max_drawdown_pct_any_window")
        ),
        "walk_forward_completed_windows": _safe_scalar(
            walk_forward_summary.get("completed_windows")
        ),
        "walk_forward_pass_rate": _number_or_none(walk_forward_summary.get("pass_rate")),
        "training_recommendation": _safe_scalar(
            training_manifest.get("recommendation")
        ),
        "training_completed_stages": _safe_scalar(
            _dict_or_empty(training_manifest.get("summary")).get("completed_stages")
        ),
    }


def _paper_metrics_summary(paper_metrics: dict[str, Any]) -> dict[str, Any]:
    profit = _dict_or_empty(paper_metrics.get("profit"))
    risk = _dict_or_empty(paper_metrics.get("risk"))
    trade_counts = _dict_or_empty(paper_metrics.get("trade_counts"))
    total_return_pct = _first_number(
        profit.get("total_return_pct"),
        profit.get("total_return_percent"),
        paper_metrics.get("total_return_pct"),
        _dict_or_empty(paper_metrics.get("summary")).get("total_return_pct"),
    )
    max_drawdown_pct = _first_number(
        risk.get("max_drawdown_pct"),
        paper_metrics.get("max_drawdown_pct"),
        _dict_or_empty(paper_metrics.get("summary")).get("max_drawdown_pct"),
    )
    return {
        "status": _safe_scalar(paper_metrics.get("status")),
        "run_id": _safe_scalar(paper_metrics.get("run_id")),
        "total_return_pct": total_return_pct,
        "max_drawdown_pct": max_drawdown_pct,
        "trade_count": _first_int(
            trade_counts.get("total"),
            paper_metrics.get("trade_count"),
            _dict_or_empty(paper_metrics.get("summary")).get("trade_count"),
        ),
        "open_trade_count": _int_or_none(trade_counts.get("open")),
        "closed_trade_count": _int_or_none(trade_counts.get("closed")),
        "realized_profit": _number_or_none(profit.get("realized")),
        "unrealized_profit": _number_or_none(profit.get("unrealized")),
    }


def _drift_summary(
    inputs: PaperDriftReportInputs,
    reference: dict[str, Any],
    paper: dict[str, Any],
) -> dict[str, Any]:
    return {
        "thresholds": {
            "max_return_drift_pct": inputs.max_return_drift_pct,
            "max_drawdown_drift_pct": inputs.max_drawdown_drift_pct,
        },
        "return_vs_historical_pct_points": _delta(
            paper.get("total_return_pct"),
            reference.get("historical_total_return_pct"),
        ),
        "return_vs_walk_forward_pct_points": _delta(
            paper.get("total_return_pct"),
            reference.get("walk_forward_total_return_pct"),
        ),
        "drawdown_vs_historical_pct_points": _delta(
            paper.get("max_drawdown_pct"),
            reference.get("historical_max_drawdown_pct"),
        ),
        "drawdown_vs_walk_forward_any_window_pct_points": _delta(
            paper.get("max_drawdown_pct"),
            reference.get("walk_forward_max_drawdown_pct_any_window"),
        ),
        "paper_trade_count_vs_historical": _delta(
            paper.get("trade_count"),
            reference.get("historical_trade_count"),
        ),
    }


def _local_existing_path_check(
    name: str, path: Path | None, root_dir: Path, label: str
) -> PaperDriftReportCheck:
    return _check(
        f"{name}_within_workspace_and_present",
        path is not None and _path_is_within_root(path, root_dir) and path.is_file(),
        "blocker",
        f"{label} path must resolve inside the workspace and exist locally.",
        {"path": _safe_relative_path(path, root_dir)},
    )


def _path_match_check(
    name: str,
    payload_path: Any,
    expected_path: Path | None,
    root_dir: Path,
    message: str,
) -> PaperDriftReportCheck:
    payload_resolved = path_from_payload(payload_path, root_dir)
    return _check(
        name,
        payload_resolved is not None
        and expected_path is not None
        and _same_resolved_path(payload_resolved, expected_path),
        "blocker",
        message,
        {
            "payload_path": _safe_relative_path(payload_resolved, root_dir),
            "expected_path": _safe_relative_path(expected_path, root_dir),
        },
    )


def _artifact_paths(inputs: PaperDriftReportInputs) -> dict[str, str]:
    output_dir = inputs.output_dir
    return {
        "paper_drift_report": _safe_relative_path(
            output_dir / "paper_drift_report.json", inputs.root_dir
        ),
        "paper_drift_report_markdown": _safe_relative_path(
            output_dir / "paper_drift_report.md", inputs.root_dir
        ),
        "drift_metrics": _safe_relative_path(
            output_dir / "drift_metrics.json", inputs.root_dir
        ),
        "command": _safe_relative_path(output_dir / "command.txt", inputs.root_dir),
    }


def _resolve_status(checks: Sequence[PaperDriftReportCheck]) -> str:
    if any(check.status == "blocked" for check in checks):
        return "blocked"
    if any(check.status == "fail" for check in checks):
        return "fail"
    return "pass"


def _check(
    name: str,
    passed: bool,
    severity: str,
    message: str,
    details: dict[str, Any] | None = None,
    *,
    failure_status: str = "blocked",
) -> PaperDriftReportCheck:
    return PaperDriftReportCheck(
        name=name,
        status="pass" if passed else failure_status,
        severity=severity,
        message=message,
        details=details or {},
    )


def _dict_or_empty(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _path_is_within_root(path: Path, root_dir: Path) -> bool:
    try:
        path.resolve().relative_to(root_dir.resolve())
        return True
    except ValueError:
        return False


def _same_resolved_path(left: Path, right: Path) -> bool:
    return left.resolve() == right.resolve()


def _safe_relative_path(path: Path | None, root_dir: Path) -> str | None:
    if path is None:
        return None
    try:
        return str(path.resolve().relative_to(root_dir.resolve()))
    except ValueError:
        return path.name


def path_from_payload(path_value: Any, root_dir: Path) -> Path | None:
    if isinstance(path_value, Path):
        return path_value
    if not isinstance(path_value, str) or not path_value.strip():
        return None
    path = Path(path_value)
    if path.is_absolute():
        return path
    return root_dir / path


def _safe_scalar(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return type(value).__name__


def _format_number(value: Any) -> str:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{float(value):.6f}"
    return "n/a"


def _first_number(*values: Any) -> float | None:
    for value in values:
        number = _number_or_none(value)
        if number is not None:
            return number
    return None


def _number_or_none(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _first_int(*values: Any) -> int | None:
    for value in values:
        number = _int_or_none(value)
        if number is not None:
            return number
    return None


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        if value is None:
            return None
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number >= 0 else None


def _delta(left: Any, right: Any) -> float | None:
    left_number = _number_or_none(left)
    right_number = _number_or_none(right)
    if left_number is None or right_number is None:
        return None
    return left_number - right_number


def _is_number(value: Any) -> bool:
    return _number_or_none(value) is not None


def _non_negative_int(value: Any) -> bool:
    return _int_or_none(value) is not None


def _number_gte(value: Any, threshold: float) -> bool:
    number = _number_or_none(value)
    return number is not None and number >= threshold


def _number_lte(value: Any, threshold: float) -> bool:
    number = _number_or_none(value)
    return number is not None and number <= threshold


def _credential_findings(payload: Any, prefix: str = "") -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            key_text = str(key)
            if (
                "token_count" not in key_text.lower()
                and _CREDENTIAL_KEY_RE.search(key_text)
                and _has_credential_value(value)
            ):
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
