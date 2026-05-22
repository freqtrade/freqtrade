from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd


@dataclass(frozen=True)
class FeatureQualityThresholds:
    max_missing_rate: float = 0.02
    max_staleness_seconds: int = 900
    max_outlier_rate: float = 0.05
    max_recent_gap_seconds: int = 900
    min_classifier_confidence: float = 0.6
    max_cost_model_age_seconds: int = 86400


def build_feature_quality_report(
    frame: pd.DataFrame,
    *,
    required_features: Sequence[str],
    timestamp_column: str = "date",
    generated_at: str | None = None,
    now: datetime | None = None,
    classifier_confidence: float | None = None,
    cost_model_updated_at: str | None = None,
    thresholds: FeatureQualityThresholds | None = None,
) -> dict[str, Any]:
    thresholds = thresholds or FeatureQualityThresholds()
    now = now or datetime.now(UTC)
    generated_at = generated_at or now.replace(microsecond=0).isoformat()
    rows = int(len(frame))
    missing_features = [feature for feature in required_features if feature not in frame.columns]
    feature_rows: list[dict[str, Any]] = []
    for feature in required_features:
        if feature not in frame.columns:
            feature_rows.append(
                {
                    "feature": feature,
                    "present": False,
                    "missing_rate": 1.0,
                    "outlier_rate": 1.0,
                    "passed": False,
                }
            )
            continue
        series = pd.to_numeric(frame[feature], errors="coerce")
        missing_rate = float(series.isna().mean()) if rows else 1.0
        outlier_rate = _outlier_rate(series)
        feature_rows.append(
            {
                "feature": feature,
                "present": True,
                "missing_rate": missing_rate,
                "outlier_rate": outlier_rate,
                "passed": (
                    missing_rate <= thresholds.max_missing_rate
                    and outlier_rate <= thresholds.max_outlier_rate
                ),
            }
        )

    timestamps = (
        pd.to_datetime(frame[timestamp_column], utc=True, errors="coerce")
        if timestamp_column in frame.columns
        else pd.Series([], dtype="datetime64[ns, UTC]")
    )
    latest = timestamps.max() if len(timestamps) else pd.NaT
    stale_seconds = (
        float((now - latest.to_pydatetime()).total_seconds())
        if pd.notna(latest)
        else None
    )
    recent_gap_seconds = _max_recent_gap_seconds(timestamps)
    cost_model_age_seconds = _age_seconds(cost_model_updated_at, now)
    checks = [
        _check("required_features_present", not missing_features, {"missing_features": missing_features}),
        _check(
            "feature_missing_rate_within_threshold",
            all(item["missing_rate"] <= thresholds.max_missing_rate for item in feature_rows),
            {"max_missing_rate": thresholds.max_missing_rate},
        ),
        _check(
            "feature_outlier_rate_within_threshold",
            all(item["outlier_rate"] <= thresholds.max_outlier_rate for item in feature_rows),
            {"max_outlier_rate": thresholds.max_outlier_rate},
        ),
        _check(
            "feature_timestamp_fresh",
            stale_seconds is not None and stale_seconds <= thresholds.max_staleness_seconds,
            {
                "stale_seconds": stale_seconds,
                "max_staleness_seconds": thresholds.max_staleness_seconds,
            },
        ),
        _check(
            "recent_data_gaps_within_threshold",
            recent_gap_seconds is not None and recent_gap_seconds <= thresholds.max_recent_gap_seconds,
            {
                "recent_gap_seconds": recent_gap_seconds,
                "max_recent_gap_seconds": thresholds.max_recent_gap_seconds,
            },
        ),
        _check(
            "classifier_confidence_within_threshold",
            classifier_confidence is not None
            and classifier_confidence >= thresholds.min_classifier_confidence,
            {
                "classifier_confidence": classifier_confidence,
                "min_classifier_confidence": thresholds.min_classifier_confidence,
            },
        ),
        _check(
            "cost_model_freshness_within_threshold",
            cost_model_age_seconds is None
            or cost_model_age_seconds <= thresholds.max_cost_model_age_seconds,
            {
                "cost_model_age_seconds": cost_model_age_seconds,
                "max_cost_model_age_seconds": thresholds.max_cost_model_age_seconds,
            },
        ),
    ]
    return {
        "factory": "feature_quality_report",
        "schema_version": "feature_quality_v1",
        "generated_at": generated_at,
        "ok": all(check["passed"] for check in checks),
        "row_count": rows,
        "required_features": list(required_features),
        "feature_quality_thresholds": asdict(thresholds),
        "features": feature_rows,
        "latest_feature_timestamp": latest.isoformat() if pd.notna(latest) else None,
        "stale_seconds": stale_seconds,
        "recent_gap_seconds": recent_gap_seconds,
        "classifier_confidence": classifier_confidence,
        "cost_model_updated_at": cost_model_updated_at,
        "cost_model_age_seconds": cost_model_age_seconds,
        "checks": checks,
        "reason_codes": ["feature_quality_passed"]
        if all(check["passed"] for check in checks)
        else [check["name"] for check in checks if not check["passed"]],
    }


def feature_quality_passes_thresholds(
    report: Mapping[str, Any] | None,
    thresholds: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not isinstance(report, Mapping):
        return {
            "ok": False,
            "reason_codes": ["feature_quality_report_missing"],
            "checks": [_check("feature_quality_report_present", False, {})],
        }
    merged = dict(report.get("feature_quality_thresholds") or {})
    merged.update(dict(thresholds or {}))
    checks = [
        _check("feature_quality_report_ok", bool(report.get("ok")), {}),
        _check(
            "classifier_confidence_above_candidate_threshold",
            float(report.get("classifier_confidence") or 0.0)
            >= float(merged.get("min_classifier_confidence", 0.0)),
            {
                "classifier_confidence": report.get("classifier_confidence"),
                "min_classifier_confidence": merged.get("min_classifier_confidence"),
            },
        ),
    ]
    ok = all(check["passed"] for check in checks)
    return {
        "ok": ok,
        "reason_codes": ["feature_quality_passed"]
        if ok
        else [check["name"] for check in checks if not check["passed"]],
        "checks": checks,
    }


def write_feature_quality_report(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


def _outlier_rate(series: pd.Series) -> float:
    clean = series.dropna()
    if len(clean) < 4:
        return 0.0
    median = clean.median()
    mad = (clean - median).abs().median()
    if not mad:
        return 0.0
    robust_z = (clean - median).abs() / (1.4826 * mad)
    return float((robust_z > 8.0).mean())


def _max_recent_gap_seconds(timestamps: pd.Series) -> float | None:
    clean = timestamps.dropna().sort_values()
    if len(clean) < 2:
        return None
    gaps = clean.diff().dt.total_seconds().dropna()
    return float(gaps.max()) if len(gaps) else None


def _age_seconds(value: str | None, now: datetime) -> float | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return float((now - parsed).total_seconds())


def _check(name: str, passed: bool, details: dict[str, Any]) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "details": details}
