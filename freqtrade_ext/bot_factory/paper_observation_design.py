from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from freqtrade_ext.bot_factory.regime_promotion import (
    FUTURE_OBSERVATION_SOURCE_TYPES,
    REQUIRED_STATE_OBSERVATION_SCOPE_FIELDS,
    observation_ledger_schema,
    validate_observation_record,
)


PAPER_OBSERVATION_DESIGN_SCHEMA_VERSION = "paper_observation_design_v1"
PAPER_OBSERVATION_DESIGN_REPORT_SCHEMA_VERSION = "paper_observation_design_report_v1"
DEFAULT_DRIFT_THRESHOLDS = {
    "max_state_distribution_l1": 0.4,
    "max_feature_drift_score": 0.35,
    "max_cost_turnover_drift_score": 0.25,
    "max_selector_churn_ratio": 0.3,
    "max_drawdown_envelope_breach_count": 0.0,
    "retirement_quarantine_count": 3.0,
}
FUTURE_REQUIRED_FIELDS = (
    "state_snapshot_id",
    "state_id",
    "horizon_profile_id",
    "state_encoder_version",
    "state_window_id",
    "feature_cutoff_timestamp",
    "label_cutoff_timestamp",
    "decision_window_start",
    "decision_window_end",
    "future_data_used",
)


def build_paper_observation_design(
    *,
    future_observations: Sequence[Mapping[str, Any]] = (),
    paper_observation_metrics: Mapping[str, Any] | None = None,
    run_id: str | None = None,
    generated_at: str | None = None,
    drift_thresholds: Mapping[str, float] | None = None,
    persistent_quarantine_count: int = 0,
    source_artifacts: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    generated_at = generated_at or _utc_now()
    run_id = run_id or "paper_observation_design_" + _compact_timestamp(generated_at)
    thresholds = {
        **DEFAULT_DRIFT_THRESHOLDS,
        **{str(key): float(value) for key, value in (drift_thresholds or {}).items()},
    }
    ledger_schema = observation_ledger_schema()
    observation_validations = [
        _future_observation_validation(observation)
        for observation in future_observations
    ]
    drift_report = _drift_report(
        paper_observation_metrics or {},
        thresholds=thresholds,
    )
    quarantine_report = _quarantine_report(
        observation_validations=observation_validations,
        drift_report=drift_report,
        paper_observation_metrics=paper_observation_metrics or {},
    )
    retirement_report = _retirement_report(
        quarantine_report=quarantine_report,
        persistent_quarantine_count=persistent_quarantine_count,
        thresholds=thresholds,
        paper_observation_metrics=paper_observation_metrics or {},
    )
    summary = _summary_decision(
        observation_validations=observation_validations,
        drift_report=drift_report,
        quarantine_report=quarantine_report,
        retirement_report=retirement_report,
    )
    return {
        "factory": "paper_observation_design",
        "schema_version": PAPER_OBSERVATION_DESIGN_SCHEMA_VERSION,
        "run_id": run_id,
        "generated_at": generated_at,
        "status": summary["status"],
        "paper_observation_schema": _paper_observation_schema(ledger_schema),
        "observation_ledger_compatibility": {
            "base_schema_version": ledger_schema["schema_version"],
            "same_observation_ledger_schema_required": True,
            "allow_future_sources_flag_required": True,
            "future_sources_blocked_by_default": ledger_schema[
                "future_source_types_blocked_by_default"
            ],
        },
        "future_observation_validations": observation_validations,
        "evidence_separation": _evidence_separation(),
        "ranking_policy": {
            "recent_observation_may_influence_ranking": True,
            "strict_state_conditioned_evidence_required_first": True,
            "can_override_failed_historical_or_walk_forward_evidence": False,
            "can_create_paper_readiness_by_itself": False,
            "can_promote_by_itself": False,
            "reason_codes": ["recent_observation_is_additional_evidence_only"],
        },
        "drift_thresholds": thresholds,
        "drift_report": drift_report,
        "quarantine_report": quarantine_report,
        "retirement_report": retirement_report,
        "startup_boundary": _startup_boundary(),
        "summary_decision": summary["summary_decision"],
        "reason_codes": summary["reason_codes"],
        "source_artifacts": dict(source_artifacts or {}),
        "safety_scope": _safety_scope(),
    }


def write_paper_observation_design_artifacts(
    design: Mapping[str, Any],
    *,
    output_root: Path,
) -> dict[str, Path]:
    run_id = _safe_component(str(design.get("run_id") or "paper_observation_design"))
    out_dir = output_root / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "paper_observation_design": out_dir / "paper_observation_design.json",
        "paper_observation_design_report": out_dir / "paper_observation_design_report.md",
        "paper_observation_schema": out_dir / "paper_observation_schema.json",
        "paper_drift_report_schema": out_dir / "paper_drift_report_schema.json",
    }
    paths["paper_observation_design"].write_text(
        json.dumps(design, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    paths["paper_observation_design_report"].write_text(
        render_paper_observation_design_report(design),
        encoding="utf-8",
    )
    paths["paper_observation_schema"].write_text(
        json.dumps(design.get("paper_observation_schema", {}), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    paths["paper_drift_report_schema"].write_text(
        json.dumps(design.get("drift_report", {}).get("schema", {}), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return paths


def render_paper_observation_design_report(design: Mapping[str, Any]) -> str:
    lines = [
        "# Paper Observation Design",
        "",
        f"- Run ID: `{design.get('run_id')}`",
        f"- Status: `{design.get('status')}`",
        f"- Decision: `{design.get('summary_decision')}`",
        f"- Observation validations: `{len(design.get('future_observation_validations', []))}`",
        f"- Drift decision: `{(design.get('drift_report') or {}).get('decision')}`",
        f"- Quarantine decision: `{(design.get('quarantine_report') or {}).get('decision')}`",
        f"- Retirement decision: `{(design.get('retirement_report') or {}).get('decision')}`",
        f"- Reason codes: `{', '.join(design.get('reason_codes', []))}`",
        "",
        "## Boundary",
        "",
        "- Paper observations use the same observation ledger schema with required state scope.",
        "- Paper observations are additional evidence only and cannot directly promote a strategy.",
        "- This artifact does not start paper, dry-run, live trading, `freqtrade trade`, process control, or exchange order placement.",
    ]
    return "\n".join(lines) + "\n"


def _paper_observation_schema(ledger_schema: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "paper_observation_row_v1",
        "base_schema_version": ledger_schema.get("schema_version"),
        "required_base_fields": list(ledger_schema.get("required_fields", [])),
        "required_state_fields": list(REQUIRED_STATE_OBSERVATION_SCOPE_FIELDS),
        "required_future_fields": list(FUTURE_REQUIRED_FIELDS),
        "allowed_future_source_types": sorted(FUTURE_OBSERVATION_SOURCE_TYPES),
        "uses_same_observation_ledger_schema": True,
        "future_data_used_must_be_false": True,
        "state_snapshot_id_required": True,
        "horizon_profile_id_required": True,
        "startup_authorized_by_schema": False,
        "promotion_authorized_by_schema": False,
    }


def _future_observation_validation(observation: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(observation)
    base_validation = validate_observation_record(payload, allow_future_sources=True)
    missing_future_fields = [
        field for field in FUTURE_REQUIRED_FIELDS if payload.get(field) in (None, "")
    ]
    source_type = str(payload.get("source_type") or "")
    future_source_allowed = source_type in FUTURE_OBSERVATION_SOURCE_TYPES
    checks = [
        {
            "name": "same_observation_ledger_schema_valid",
            "passed": base_validation["ok"],
            "details": {"base_checks": base_validation["checks"]},
        },
        {
            "name": "future_observation_source_type_explicit",
            "passed": future_source_allowed,
            "details": {
                "source_type": source_type,
                "allowed_future_source_types": sorted(FUTURE_OBSERVATION_SOURCE_TYPES),
            },
        },
        {
            "name": "future_state_snapshot_scope_present",
            "passed": not missing_future_fields,
            "details": {"missing_fields": missing_future_fields},
        },
        {
            "name": "future_observation_no_future_labels",
            "passed": payload.get("future_data_used") is False,
            "details": {"future_data_used": payload.get("future_data_used")},
        },
    ]
    return {
        "observation_id": payload.get("observation_id"),
        "ok": all(check["passed"] for check in checks),
        "checks": checks,
        "reason_codes": [
            check["name"]
            for check in checks
            if not check["passed"]
        ],
    }


def _evidence_separation() -> dict[str, Any]:
    buckets = {
        "historical_evidence": {"source_types": ["backtest"], "separate_bucket": True},
        "walk_forward_evidence": {"source_types": ["walk_forward"], "separate_bucket": True},
        "training_evidence": {"source_types": ["training"], "separate_bucket": True},
        "readiness_evidence": {"source_types": ["paper_readiness"], "separate_bucket": True},
        "runtime_validation": {"source_types": ["paper_runtime_validation"], "separate_bucket": True},
        "drift_evidence": {"source_types": ["paper_drift_report"], "separate_bucket": True},
        "recent_observation": {
            "source_types": sorted(FUTURE_OBSERVATION_SOURCE_TYPES),
            "separate_bucket": True,
            "can_override_historical_evidence": False,
            "can_override_walk_forward_evidence": False,
            "can_override_readiness_evidence": False,
        },
    }
    return {
        "buckets": buckets,
        "recent_observation_is_additional_evidence_only": True,
        "promotion_authorized_by_recent_observation": False,
    }


def _drift_report(
    metrics: Mapping[str, Any],
    *,
    thresholds: Mapping[str, float],
) -> dict[str, Any]:
    checks = [
        _drift_check(
            "state_distribution_drift",
            _state_distribution_l1(
                metrics.get("reference_state_distribution") or {},
                metrics.get("observed_state_distribution") or {},
            ),
            thresholds["max_state_distribution_l1"],
            lower_is_better=True,
            not_evaluated=not (
                metrics.get("reference_state_distribution")
                and metrics.get("observed_state_distribution")
            ),
        ),
        _drift_check(
            "feature_distribution_drift",
            _number(metrics.get("feature_drift_score"), 0.0),
            thresholds["max_feature_drift_score"],
            lower_is_better=True,
            not_evaluated="feature_drift_score" not in metrics,
        ),
        _drift_check(
            "cost_turnover_drift",
            _number(metrics.get("cost_turnover_drift_score"), 0.0),
            thresholds["max_cost_turnover_drift_score"],
            lower_is_better=True,
            not_evaluated="cost_turnover_drift_score" not in metrics,
        ),
        _drift_check(
            "drawdown_envelope_breach",
            1.0 if metrics.get("drawdown_envelope_breach") else 0.0,
            thresholds["max_drawdown_envelope_breach_count"],
            lower_is_better=True,
            not_evaluated="drawdown_envelope_breach" not in metrics,
        ),
        _drift_check(
            "selector_churn_increase",
            _number(metrics.get("selector_churn_ratio"), 0.0),
            thresholds["max_selector_churn_ratio"],
            lower_is_better=True,
            not_evaluated="selector_churn_ratio" not in metrics,
        ),
    ]
    failed = [check for check in checks if check["status"] == "fail"]
    not_evaluated = [check for check in checks if check["status"] == "not_evaluated"]
    if failed:
        decision = "DRIFT_FAIL"
    elif len(not_evaluated) == len(checks):
        decision = "DRIFT_NOT_EVALUATED"
    else:
        decision = "DRIFT_PASS"
    return {
        "schema": {
            "schema_version": "paper_observation_drift_report_v1",
            "required_checks": [check["name"] for check in checks],
        },
        "decision": decision,
        "checks": checks,
        "promotion_authorized_by_drift_report": False,
        "process_control": False,
    }


def _quarantine_report(
    *,
    observation_validations: Sequence[Mapping[str, Any]],
    drift_report: Mapping[str, Any],
    paper_observation_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    triggers = []
    if any(not validation.get("ok") for validation in observation_validations):
        triggers.append("future_observation_schema_invalid")
    if drift_report.get("decision") == "DRIFT_FAIL":
        triggers.append("paper_observation_drift_failed")
    if paper_observation_metrics.get("state_evidence_contradictions"):
        triggers.append("live_like_observation_contradicts_historical_state_evidence")
    decision = "QUARANTINE" if triggers else "NO_QUARANTINE"
    return {
        "decision": decision,
        "triggers": triggers,
        "state_evidence_contradictions": list(
            paper_observation_metrics.get("state_evidence_contradictions") or []
        ),
        "promotion_authorized_by_quarantine_report": False,
        "process_control": False,
    }


def _retirement_report(
    *,
    quarantine_report: Mapping[str, Any],
    persistent_quarantine_count: int,
    thresholds: Mapping[str, float],
    paper_observation_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    triggers = []
    if (
        quarantine_report.get("decision") == "QUARANTINE"
        and persistent_quarantine_count >= thresholds["retirement_quarantine_count"]
    ):
        triggers.append("persistent_quarantine_threshold_reached")
    if paper_observation_metrics.get("evidence_stale"):
        triggers.append("evidence_stale")
    if paper_observation_metrics.get("state_identity_retired"):
        triggers.append("state_identity_retired")
    decision = "RETIRE_REVIEW" if triggers else "NO_RETIREMENT"
    return {
        "decision": decision,
        "triggers": triggers,
        "persistent_quarantine_count": persistent_quarantine_count,
        "promotion_authorized_by_retirement_report": False,
        "process_control": False,
    }


def _summary_decision(
    *,
    observation_validations: Sequence[Mapping[str, Any]],
    drift_report: Mapping[str, Any],
    quarantine_report: Mapping[str, Any],
    retirement_report: Mapping[str, Any],
) -> dict[str, Any]:
    reason_codes = ["paper_observation_design_local_only"]
    status = "ready"
    decision = "PAPER_OBSERVATION_DESIGN_READY"
    if any(not validation.get("ok") for validation in observation_validations):
        status = "blocked"
        decision = "PAPER_OBSERVATION_DESIGN_BLOCKED"
        reason_codes.append("future_observation_schema_invalid")
        for validation in observation_validations:
            if validation.get("ok"):
                continue
            for reason in validation.get("reason_codes", []):
                if reason not in reason_codes:
                    reason_codes.append(str(reason))
    if drift_report.get("decision") == "DRIFT_FAIL":
        status = "review"
        decision = "PAPER_OBSERVATION_QUARANTINE_REVIEW"
        reason_codes.append("paper_observation_drift_failed")
    if quarantine_report.get("decision") == "QUARANTINE":
        reason_codes.append("paper_observation_quarantine_triggered")
    if retirement_report.get("decision") == "RETIRE_REVIEW":
        status = "review"
        decision = "PAPER_OBSERVATION_RETIRE_REVIEW"
        reason_codes.append("paper_observation_retirement_review_triggered")
    reason_codes.append("paper_observation_additional_evidence_only")
    return {
        "status": status,
        "summary_decision": decision,
        "reason_codes": reason_codes,
    }


def _startup_boundary() -> dict[str, Any]:
    return {
        "requires_explicit_future_approval": True,
        "startup_eligible_by_this_artifact": False,
        "paper_trading_started": False,
        "dry_run_trading_started": False,
        "live_trading_started": False,
        "freqtrade_trade_started": False,
        "exchange_order_placement": False,
        "process_control": False,
        "promotion_authorized_by_this_artifact": False,
    }


def _drift_check(
    name: str,
    value: float,
    threshold: float,
    *,
    lower_is_better: bool,
    not_evaluated: bool,
) -> dict[str, Any]:
    if not_evaluated:
        status = "not_evaluated"
    elif lower_is_better and value > threshold:
        status = "fail"
    elif not lower_is_better and value < threshold:
        status = "fail"
    else:
        status = "pass"
    return {
        "name": name,
        "status": status,
        "value": round(value, 8),
        "threshold": threshold,
    }


def _state_distribution_l1(
    reference: Mapping[str, Any],
    observed: Mapping[str, Any],
) -> float:
    ref_total = sum(max(_number(value), 0.0) for value in reference.values())
    obs_total = sum(max(_number(value), 0.0) for value in observed.values())
    if ref_total <= 0 or obs_total <= 0:
        return 0.0
    keys = set(reference) | set(observed)
    return sum(
        abs(
            max(_number(reference.get(key)), 0.0) / ref_total
            - max(_number(observed.get(key)), 0.0) / obs_total
        )
        for key in keys
    )


def _number(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool) or value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safety_scope() -> dict[str, bool]:
    return {
        "local_artifacts_source_of_truth": True,
        "paper_observation_design_only": True,
        "future_observation_only": True,
        "freqtrade_trade_started": False,
        "paper_trading_started": False,
        "dry_run_trading_started": False,
        "live_trading_started": False,
        "exchange_order_placement": False,
        "uses_api_keys_or_secrets": False,
        "metadata_contains_secrets": False,
        "process_control": False,
        "status_polling_started": False,
        "process_stop_started": False,
        "cleanup_executed": False,
        "leverage_above_one": False,
        "shorting": False,
        "promotion_authorized_by_this_artifact": False,
    }


def _safe_component(value: str) -> str:
    clean = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value)
    return clean.strip("._") or "paper_observation_design"


def _compact_timestamp(value: str) -> str:
    return _safe_component(value.replace("+00:00", "Z").replace(":", "").replace("-", ""))


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()
