from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence


DIAGNOSTIC_STATE_DISCOVERY_SCHEMA_VERSION = "diagnostic_state_discovery_v1"
DIAGNOSTIC_STATE_DISCOVERY_REPORT_SCHEMA_VERSION = (
    "diagnostic_state_discovery_report_v1"
)
BASE_FEATURE_NAMES = (
    "state_confidence",
    "uncertainty",
    "out_of_distribution_score",
    "horizon_count",
    "mean_horizon_confidence",
    "mean_horizon_uncertainty",
    "mean_horizon_ood_score",
)


def build_diagnostic_state_discovery_report(
    *,
    market_state_snapshots: Sequence[Mapping[str, Any]],
    state_scorecards: Sequence[Mapping[str, Any]] = (),
    strategy_suitability_matrices: Sequence[Mapping[str, Any]] = (),
    run_id: str | None = None,
    generated_at: str | None = None,
    analog_k: int = 3,
    min_cluster_size: int = 2,
    source_artifacts: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    generated_at = generated_at or _utc_now()
    run_id = run_id or "diagnostic_state_discovery_" + _compact_timestamp(generated_at)
    ordered_snapshots = sorted(
        [dict(snapshot) for snapshot in market_state_snapshots],
        key=lambda item: str(item.get("data_asof") or item.get("generated_at") or ""),
    )
    validation = _validate_snapshots(ordered_snapshots)
    if not validation["ok"]:
        return _invalid_report(
            run_id=run_id,
            generated_at=generated_at,
            validation=validation,
            source_artifacts=source_artifacts,
        )

    embeddings = [_embedding_row(snapshot, index=index) for index, snapshot in enumerate(ordered_snapshots)]
    feature_names = _predeclared_feature_names(embeddings)
    vectors = [_vector(row, feature_names) for row in embeddings]
    standardized = _standardize(vectors)
    assignments = _cluster_assignments(standardized)
    clusters = _cluster_rows(
        embeddings,
        assignments=assignments,
        feature_names=feature_names,
        min_cluster_size=min_cluster_size,
    )
    analog_rows = _analog_rows(
        embeddings,
        standardized,
        analog_k=max(1, analog_k),
        min_analog_count=max(1, min(analog_k, min_cluster_size)),
    )
    ood = _ood_calibration(analog_rows, min_analog_count=max(1, min(analog_k, min_cluster_size)))
    suitability_dataset = _suitability_dataset(
        state_scorecards=state_scorecards,
        strategy_suitability_matrices=strategy_suitability_matrices,
        embeddings=embeddings,
    )
    comparison = _deterministic_label_comparison(embeddings, assignments)
    gate = _diagnostic_gate()
    reason_codes = [
        "diagnostic_state_discovery_completed",
        "diagnostic_only_not_selector_eligible",
        "out_of_sample_baseline_not_proven",
    ]
    if any(row["reason_codes"] for row in analog_rows):
        reason_codes.append("some_windows_have_insufficient_analog_evidence")
    return {
        "factory": "diagnostic_state_discovery",
        "schema_version": DIAGNOSTIC_STATE_DISCOVERY_SCHEMA_VERSION,
        "run_id": run_id,
        "generated_at": generated_at,
        "status": "completed",
        "input_validation": validation,
        "predeclared_feature_names": feature_names,
        "diagnostic_model_schemas": _diagnostic_model_schemas(feature_names),
        "state_embedding_dataset": embeddings,
        "diagnostic_state_clusters": clusters,
        "analog_window_search": analog_rows,
        "ood_uncertainty_calibration": ood,
        "suitability_scoring_dataset": suitability_dataset,
        "deterministic_label_comparison": comparison,
        "out_of_sample_selector_replay_evidence": {
            "beats_deterministic_baselines": False,
            "status": "not_proven",
            "selector_candidate_creation_allowed": False,
            "reason_codes": ["out_of_sample_baseline_not_proven"],
        },
        "diagnostic_gate": gate,
        "diagnostic_only": True,
        "manual_review_only": True,
        "selector_candidate_creation_allowed": False,
        "paper_readiness_input_allowed": False,
        "promotion_authorized_by_this_artifact": False,
        "reason_codes": reason_codes,
        "source_artifacts": dict(source_artifacts or {}),
        "safety_scope": _safety_scope(),
    }


def write_diagnostic_state_discovery_artifacts(
    report: Mapping[str, Any],
    *,
    output_root: Path,
) -> dict[str, Path]:
    run_id = _safe_component(str(report.get("run_id") or "diagnostic_state_discovery"))
    out_dir = output_root / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "diagnostic_state_discovery": out_dir / "diagnostic_state_discovery.json",
        "diagnostic_state_discovery_report": out_dir / "diagnostic_state_discovery_report.md",
    }
    paths["diagnostic_state_discovery"].write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    paths["diagnostic_state_discovery_report"].write_text(
        render_diagnostic_state_discovery_report(report),
        encoding="utf-8",
    )
    return paths


def render_diagnostic_state_discovery_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Diagnostic State Discovery",
        "",
        f"- Run ID: `{report.get('run_id')}`",
        f"- Status: `{report.get('status')}`",
        f"- Diagnostic only: `{report.get('diagnostic_only')}`",
        f"- Embedding rows: `{len(report.get('state_embedding_dataset', []))}`",
        f"- Clusters: `{len(report.get('diagnostic_state_clusters', []))}`",
        f"- Suitability rows: `{len(report.get('suitability_scoring_dataset', []))}`",
        f"- Reason codes: `{', '.join(report.get('reason_codes', []))}`",
        "",
        "## Clusters",
        "",
        "| cluster | members | dominant label | purity | selector eligible |",
        "| --- | ---: | --- | ---: | --- |",
    ]
    for row in report.get("diagnostic_state_clusters", []):
        lines.append(
            "| {cluster_id} | {member_count} | {label} | {purity} | {eligible} |".format(
                cluster_id=row.get("cluster_id"),
                member_count=row.get("member_count"),
                label=row.get("dominant_deterministic_label"),
                purity=row.get("deterministic_label_purity"),
                eligible=row.get("selector_candidate_creation_allowed"),
            )
        )
    lines.extend(
        [
            "",
            "## Diagnostic Boundary",
            "",
            "- All ML-like outputs are diagnostic-only.",
            "- High diagnostic scores cannot bypass strict state-conditioned evidence, selector replay, or paper-readiness gates.",
            "- This artifact does not start paper, dry-run, live trading, `freqtrade trade`, process control, or exchange order placement.",
        ]
    )
    return "\n".join(lines) + "\n"


def _invalid_report(
    *,
    run_id: str,
    generated_at: str,
    validation: Mapping[str, Any],
    source_artifacts: Mapping[str, str] | None,
) -> dict[str, Any]:
    return {
        "factory": "diagnostic_state_discovery",
        "schema_version": DIAGNOSTIC_STATE_DISCOVERY_SCHEMA_VERSION,
        "run_id": run_id,
        "generated_at": generated_at,
        "status": "invalid",
        "input_validation": dict(validation),
        "predeclared_feature_names": [],
        "diagnostic_model_schemas": _diagnostic_model_schemas([]),
        "state_embedding_dataset": [],
        "diagnostic_state_clusters": [],
        "analog_window_search": [],
        "ood_uncertainty_calibration": {},
        "suitability_scoring_dataset": [],
        "deterministic_label_comparison": {},
        "diagnostic_gate": _diagnostic_gate(),
        "diagnostic_only": True,
        "manual_review_only": True,
        "selector_candidate_creation_allowed": False,
        "paper_readiness_input_allowed": False,
        "promotion_authorized_by_this_artifact": False,
        "reason_codes": list(validation.get("reason_codes", [])),
        "source_artifacts": dict(source_artifacts or {}),
        "safety_scope": _safety_scope(),
    }


def _validate_snapshots(snapshots: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    checks = [
        _check("market_state_snapshots_present", bool(snapshots), {"count": len(snapshots)}),
        _check(
            "market_state_snapshots_asof_ordered",
            _ordered([snapshot.get("data_asof") for snapshot in snapshots]),
            {"data_asof": [snapshot.get("data_asof") for snapshot in snapshots]},
        ),
        _check(
            "market_state_snapshots_no_future_data",
            not _future_state_leakage(snapshots),
            {"leaks": _future_state_leakage(snapshots)},
        ),
    ]
    return {
        "ok": all(check["passed"] for check in checks),
        "checks": checks,
        "reason_codes": [
            check["name"]
            for check in checks
            if not check["passed"]
        ],
    }


def _embedding_row(snapshot: Mapping[str, Any], *, index: int) -> dict[str, Any]:
    horizons = [row for row in snapshot.get("horizons", []) if isinstance(row, Mapping)]
    features: dict[str, float] = {
        "state_confidence": _number(snapshot.get("state_confidence"), 0.0),
        "uncertainty": _number(snapshot.get("uncertainty"), 1.0),
        "out_of_distribution_score": _number(snapshot.get("out_of_distribution_score"), 1.0),
        "horizon_count": float(len(horizons)),
        "mean_horizon_confidence": _mean(
            [_number(row.get("confidence"), 0.0) for row in horizons]
        ),
        "mean_horizon_uncertainty": _mean(
            [_number(row.get("uncertainty"), 1.0) for row in horizons]
        ),
        "mean_horizon_ood_score": _mean(
            [_number(row.get("out_of_distribution_score"), 1.0) for row in horizons]
        ),
    }
    horizon_state_ids = []
    for horizon_index, horizon in enumerate(horizons):
        prefix = _safe_component(str(horizon.get("horizon") or f"h{horizon_index}"))
        horizon_state_ids.append(str(horizon.get("state_id") or ""))
        features[f"{prefix}.confidence"] = _number(horizon.get("confidence"), 0.0)
        features[f"{prefix}.uncertainty"] = _number(horizon.get("uncertainty"), 1.0)
        features[f"{prefix}.out_of_distribution_score"] = _number(
            horizon.get("out_of_distribution_score"), 1.0
        )
        state_vector = horizon.get("state_vector") or {}
        if isinstance(state_vector, Mapping):
            for key, value in state_vector.items():
                if _is_number(value):
                    features[f"{prefix}.{key}"] = float(value)
    return {
        "row_id": f"state_embedding_{index:04d}",
        "data_asof": snapshot.get("data_asof") or snapshot.get("generated_at"),
        "pair": snapshot.get("pair"),
        "pair_group": snapshot.get("pair_group"),
        "base_timeframe": snapshot.get("base_timeframe"),
        "deterministic_label": snapshot.get("aggregate_label") or "unknown",
        "state_id": snapshot.get("state_id") or next((item for item in horizon_state_ids if item), ""),
        "horizon_profile_id": snapshot.get("horizon_profile_id"),
        "state_encoder_version": snapshot.get("state_encoder_version"),
        "cost_model_id": snapshot.get("cost_model_id"),
        "feature_values": {key: round(value, 10) for key, value in sorted(features.items())},
        "diagnostic_only": True,
        "selector_candidate_creation_allowed": False,
        "paper_readiness_input_allowed": False,
    }


def _predeclared_feature_names(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    names = set(BASE_FEATURE_NAMES)
    for row in rows:
        values = row.get("feature_values") or {}
        if isinstance(values, Mapping):
            names.update(str(key) for key in values)
    return sorted(names)


def _vector(row: Mapping[str, Any], feature_names: Sequence[str]) -> list[float]:
    values = row.get("feature_values") or {}
    return [_number(values.get(name), 0.0) for name in feature_names]


def _standardize(vectors: Sequence[Sequence[float]]) -> list[list[float]]:
    if not vectors:
        return []
    columns = list(zip(*vectors, strict=False))
    means = [_mean(column) for column in columns]
    scales = []
    for column, mean in zip(columns, means, strict=False):
        variance = _mean([(float(value) - mean) ** 2 for value in column])
        scale = math.sqrt(variance)
        scales.append(scale if scale > 0 else 1.0)
    return [
        [
            (float(value) - means[index]) / scales[index]
            for index, value in enumerate(vector)
        ]
        for vector in vectors
    ]


def _cluster_assignments(vectors: Sequence[Sequence[float]]) -> list[int]:
    count = len(vectors)
    if count == 0:
        return []
    if count == 1:
        return [0]
    cluster_count = min(3, max(1, round(math.sqrt(count))))
    ordered_indices = sorted(
        range(count),
        key=lambda index: (sum(float(value) for value in vectors[index]), index),
    )
    if cluster_count == 1:
        centroids = [list(vectors[ordered_indices[0]])]
    else:
        centroids = [
            list(vectors[ordered_indices[round((count - 1) * cluster / (cluster_count - 1))]])
            for cluster in range(cluster_count)
        ]
    assignments = [0] * count
    for _ in range(8):
        next_assignments = [
            min(
                range(cluster_count),
                key=lambda cluster: (_distance(vector, centroids[cluster]), cluster),
            )
            for vector in vectors
        ]
        if next_assignments == assignments:
            break
        assignments = next_assignments
        for cluster in range(cluster_count):
            members = [
                vectors[index]
                for index, assignment in enumerate(assignments)
                if assignment == cluster
            ]
            if members:
                centroids[cluster] = [
                    _mean(column) for column in zip(*members, strict=False)
                ]
    return assignments


def _cluster_rows(
    embeddings: Sequence[Mapping[str, Any]],
    *,
    assignments: Sequence[int],
    feature_names: Sequence[str],
    min_cluster_size: int,
) -> list[dict[str, Any]]:
    output = []
    for cluster in sorted(set(assignments)):
        members = [
            dict(row)
            for row, assignment in zip(embeddings, assignments, strict=False)
            if assignment == cluster
        ]
        label_counts: dict[str, int] = {}
        for member in members:
            label = str(member.get("deterministic_label") or "unknown")
            label_counts[label] = label_counts.get(label, 0) + 1
        dominant_label = max(label_counts, key=lambda label: (label_counts[label], label))
        purity = label_counts[dominant_label] / len(members) if members else 0.0
        centroid = _centroid_features(members, feature_names)
        insufficient = len(members) < min_cluster_size
        output.append(
            {
                "cluster_id": f"diagnostic_cluster_{cluster:03d}",
                "member_count": len(members),
                "member_data_asof": [member.get("data_asof") for member in members],
                "dominant_deterministic_label": dominant_label,
                "deterministic_label_counts": label_counts,
                "deterministic_label_purity": round(purity, 6),
                "centroid_features": centroid,
                "temporal_stability": _temporal_stability(members),
                "diagnostic_only": True,
                "selector_candidate_creation_allowed": False,
                "paper_readiness_input_allowed": False,
                "insufficient_evidence": insufficient,
                "reason_codes": ["INSUFFICIENT_EVIDENCE"] if insufficient else [],
            }
        )
    return output


def _analog_rows(
    embeddings: Sequence[Mapping[str, Any]],
    vectors: Sequence[Sequence[float]],
    *,
    analog_k: int,
    min_analog_count: int,
) -> list[dict[str, Any]]:
    rows = []
    for index, row in enumerate(embeddings):
        data_asof = str(row.get("data_asof") or "")
        candidates = []
        for other_index, other in enumerate(embeddings):
            other_asof = str(other.get("data_asof") or "")
            if other_index == index or other_asof >= data_asof:
                continue
            candidates.append(
                {
                    "data_asof": other.get("data_asof"),
                    "state_id": other.get("state_id"),
                    "deterministic_label": other.get("deterministic_label"),
                    "distance": round(_distance(vectors[index], vectors[other_index]), 8),
                }
            )
        candidates.sort(key=lambda item: (item["distance"], str(item["data_asof"])))
        analogs = candidates[:analog_k]
        insufficient = len(analogs) < min_analog_count
        rows.append(
            {
                "query_data_asof": row.get("data_asof"),
                "query_state_id": row.get("state_id"),
                "query_deterministic_label": row.get("deterministic_label"),
                "analog_count": len(analogs),
                "required_analog_count": min_analog_count,
                "nearest_distance": analogs[0]["distance"] if analogs else None,
                "analogs": analogs,
                "diagnostic_only": True,
                "selector_candidate_creation_allowed": False,
                "paper_readiness_input_allowed": False,
                "reason_codes": ["INSUFFICIENT_EVIDENCE"] if insufficient else [],
            }
        )
    return rows


def _ood_calibration(
    analog_rows: Sequence[Mapping[str, Any]],
    *,
    min_analog_count: int,
) -> dict[str, Any]:
    distances = [
        _number(row.get("nearest_distance"), 0.0)
        for row in analog_rows
        if _number(row.get("analog_count"), 0.0) >= min_analog_count
        and row.get("nearest_distance") is not None
    ]
    threshold = _percentile(distances, 0.9) if distances else 0.0
    rows = []
    for row in analog_rows:
        analog_count = int(_number(row.get("analog_count"), 0.0))
        nearest = row.get("nearest_distance")
        nearest_distance = _number(nearest, 0.0)
        if analog_count < min_analog_count:
            status = "insufficient_analogs"
            uncertainty = 1.0
        elif threshold > 0 and nearest_distance > threshold:
            status = "out_of_distribution"
            uncertainty = min(1.0, nearest_distance / (threshold * 2))
        else:
            status = "in_distribution"
            uncertainty = 0.0 if threshold == 0 else min(1.0, nearest_distance / threshold)
        rows.append(
            {
                "query_data_asof": row.get("query_data_asof"),
                "query_state_id": row.get("query_state_id"),
                "analog_count": analog_count,
                "nearest_distance": nearest,
                "ood_threshold_distance": round(threshold, 8),
                "calibrated_uncertainty": round(uncertainty, 8),
                "calibration_status": status,
                "diagnostic_only": True,
                "selector_candidate_creation_allowed": False,
            }
        )
    return {
        "method": "nearest_historical_analog_distance_p90",
        "min_analog_count": min_analog_count,
        "ood_threshold_distance": round(threshold, 8),
        "rows": rows,
    }


def _suitability_dataset(
    *,
    state_scorecards: Sequence[Mapping[str, Any]],
    strategy_suitability_matrices: Sequence[Mapping[str, Any]],
    embeddings: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    embedding_by_state = {
        str(row.get("state_id")): row for row in embeddings if row.get("state_id")
    }
    output = []
    for scorecard in state_scorecards:
        for row in scorecard.get("rows", []):
            if not isinstance(row, Mapping):
                continue
            state_id = str(row.get("state_id") or "")
            output.append(
                {
                    "source_type": "state_scorecard",
                    "source_run_id": scorecard.get("run_id"),
                    "candidate_id": row.get("candidate_id") or scorecard.get("candidate_id"),
                    "strategy_id": row.get("strategy_id"),
                    "state_id": state_id,
                    "horizon_profile_id": row.get("horizon_profile_id"),
                    "deterministic_label": _label_from_state_id(state_id),
                    "embedding_row_id": (embedding_by_state.get(state_id) or {}).get("row_id"),
                    "target_net_return_stress_cost": _number(
                        row.get("net_return_stress_cost"), 0.0
                    ),
                    "target_lower_confidence_bound": _number(
                        row.get("lower_confidence_bound"), 0.0
                    ),
                    "target_max_drawdown": _number(row.get("max_drawdown"), 0.0),
                    "selector_eligible_source": bool(row.get("selector_eligible")),
                    "diagnostic_only": True,
                    "selector_candidate_creation_allowed": False,
                    "paper_readiness_input_allowed": False,
                    "reason_codes": ["diagnostic_dataset_row_not_selector_evidence"],
                }
            )
    for matrix in strategy_suitability_matrices:
        for row in matrix.get("rows", []):
            if not isinstance(row, Mapping) or row.get("row_type") != "strategy":
                continue
            state_id = str(row.get("state_id") or "")
            output.append(
                {
                    "source_type": "strategy_suitability_matrix",
                    "source_run_id": matrix.get("run_id"),
                    "candidate_id": row.get("candidate_id"),
                    "strategy_id": row.get("strategy_id"),
                    "state_id": state_id,
                    "horizon_profile_id": row.get("horizon_profile_id"),
                    "deterministic_label": _label_from_state_id(state_id),
                    "embedding_row_id": (embedding_by_state.get(state_id) or {}).get("row_id"),
                    "target_net_return_stress_cost": _number(
                        row.get("net_return_stress_cost"), 0.0
                    ),
                    "target_lower_confidence_bound": _number(
                        row.get("lower_confidence_bound"), 0.0
                    ),
                    "target_max_drawdown": _number(row.get("max_drawdown"), 0.0),
                    "selector_eligible_source": bool(row.get("selector_eligible")),
                    "diagnostic_only": True,
                    "selector_candidate_creation_allowed": False,
                    "paper_readiness_input_allowed": False,
                    "reason_codes": ["diagnostic_dataset_row_not_selector_evidence"],
                }
            )
    return output


def _deterministic_label_comparison(
    embeddings: Sequence[Mapping[str, Any]],
    assignments: Sequence[int],
) -> dict[str, Any]:
    cluster_rows = []
    purities = []
    for cluster in sorted(set(assignments)):
        labels: dict[str, int] = {}
        for row, assignment in zip(embeddings, assignments, strict=False):
            if assignment != cluster:
                continue
            label = str(row.get("deterministic_label") or "unknown")
            labels[label] = labels.get(label, 0) + 1
        if not labels:
            continue
        dominant = max(labels, key=lambda label: (labels[label], label))
        purity = labels[dominant] / sum(labels.values())
        purities.append(purity)
        cluster_rows.append(
            {
                "cluster_id": f"diagnostic_cluster_{cluster:03d}",
                "dominant_deterministic_label": dominant,
                "deterministic_label_counts": labels,
                "purity": round(purity, 6),
            }
        )
    return {
        "cluster_label_rows": cluster_rows,
        "mean_cluster_purity": round(_mean(purities), 6) if purities else 0.0,
        "diagnostic_only": True,
        "reason_codes": ["ml_diagnostics_compared_with_deterministic_labels"],
    }


def _diagnostic_model_schemas(feature_names: Sequence[str]) -> dict[str, Any]:
    return {
        "state_encoder_model_v1": {
            "schema_version": "state_encoder_model_v1",
            "diagnostic_only": True,
            "required_feature_names": list(feature_names),
            "outputs": [
                "diagnostic_cluster_id",
                "nearest_analog_distance",
                "calibrated_uncertainty",
                "deterministic_label_comparison",
            ],
            "selector_candidate_creation_allowed": False,
            "paper_readiness_input_allowed": False,
            "promotion_authorized_by_this_artifact": False,
        },
        "strategy_suitability_model_v1": {
            "schema_version": "strategy_suitability_model_v1",
            "diagnostic_only": True,
            "required_training_row_fields": [
                "candidate_id",
                "strategy_id",
                "state_id",
                "horizon_profile_id",
                "target_net_return_stress_cost",
                "target_lower_confidence_bound",
                "target_max_drawdown",
                "selector_eligible_source",
            ],
            "selector_candidate_creation_allowed": False,
            "paper_readiness_input_allowed": False,
            "promotion_authorized_by_this_artifact": False,
        },
    }


def _centroid_features(
    members: Sequence[Mapping[str, Any]],
    feature_names: Sequence[str],
) -> dict[str, float]:
    centroid = {}
    for feature in feature_names:
        centroid[feature] = round(
            _mean(
                [
                    _number((member.get("feature_values") or {}).get(feature), 0.0)
                    for member in members
                ]
            ),
            10,
        )
    return centroid


def _temporal_stability(members: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    ordered = sorted(str(member.get("data_asof") or "") for member in members)
    return {
        "window_count": len(ordered),
        "first_data_asof": ordered[0] if ordered else None,
        "last_data_asof": ordered[-1] if ordered else None,
        "stable_enough_for_selector": False,
        "reason_codes": ["diagnostic_temporal_stability_only"],
    }


def _diagnostic_gate() -> dict[str, Any]:
    return {
        "diagnostic_only": True,
        "manual_review_only": True,
        "selector_candidate_creation_allowed": False,
        "paper_readiness_input_allowed": False,
        "promotion_authorized_by_this_artifact": False,
        "high_diagnostic_score_can_bypass_strict_evidence": False,
        "reason_codes": [
            "diagnostic_state_discovery_cannot_bypass_strict_evidence",
            "out_of_sample_replay_required_before_selector_use",
        ],
    }


def _future_state_leakage(snapshots: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    leaks = []
    for snapshot in snapshots:
        asof = str(snapshot.get("data_asof") or "")
        for horizon in snapshot.get("horizons", []):
            if not isinstance(horizon, Mapping):
                continue
            if horizon.get("future_data_used"):
                leaks.append(
                    {
                        "data_asof": asof,
                        "horizon": horizon.get("horizon"),
                        "field": "future_data_used",
                    }
                )
            for field in ("feature_cutoff_timestamp", "label_cutoff_timestamp"):
                value = str(horizon.get(field) or "")
                if value and asof and value > asof:
                    leaks.append(
                        {
                            "data_asof": asof,
                            "horizon": horizon.get("horizon"),
                            "field": field,
                            "value": value,
                        }
                    )
    return leaks


def _ordered(values: Sequence[Any]) -> bool:
    strings = [str(value or "") for value in values]
    return all(left <= right for left, right in zip(strings, strings[1:], strict=False))


def _check(name: str, passed: bool, details: Mapping[str, Any]) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "details": dict(details)}


def _distance(left: Sequence[float], right: Sequence[float]) -> float:
    return math.sqrt(
        sum((float(a) - float(b)) ** 2 for a, b in zip(left, right, strict=False))
    )


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _label_from_state_id(state_id: str) -> str:
    parts = state_id.split(":")
    return parts[2] if len(parts) >= 3 else "unknown"


def _mean(values: Sequence[float]) -> float:
    numbers = [float(value) for value in values if _is_number(value)]
    return sum(numbers) / len(numbers) if numbers else 0.0


def _number(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool) or value is None:
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(number) or math.isinf(number):
        return default
    return number


def _is_number(value: Any) -> bool:
    if isinstance(value, bool) or value is None:
        return False
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return not math.isnan(number) and not math.isinf(number)


def _safety_scope() -> dict[str, bool]:
    return {
        "local_artifacts_source_of_truth": True,
        "historical_evaluation_only": True,
        "diagnostic_only": True,
        "selector_simulation_only": True,
        "freqtrade_trade_started": False,
        "paper_trading_started": False,
        "dry_run_trading_started": False,
        "live_trading_started": False,
        "exchange_order_placement": False,
        "uses_api_keys_or_secrets": False,
        "metadata_contains_secrets": False,
        "process_control": False,
        "leverage_above_one": False,
        "shorting": False,
        "promotion_authorized_by_this_artifact": False,
    }


def _safe_component(value: str) -> str:
    clean = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value)
    return clean.strip("._") or "diagnostic_state_discovery"


def _compact_timestamp(value: str) -> str:
    return _safe_component(value.replace("+00:00", "Z").replace(":", "").replace("-", ""))


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()
