from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


STRATEGY_CANDIDATE_IDENTITY_SCHEMA_VERSION = "strategy_candidate_identity_v1"
REQUIRED_IDENTITY_FIELDS = (
    "candidate_id",
    "strategy_id",
    "strategy_class_name",
    "strategy_source_path",
    "strategy_version",
    "signal_version",
    "risk_policy_version",
    "regime_classifier_version",
    "cost_model_id",
    "allowed_pairs",
    "allowed_timeframes",
    "created_at",
    "source_artifacts",
)
IDENTITY_COMPARISON_FIELDS = REQUIRED_IDENTITY_FIELDS
IDENTITY_VERSION_FIELDS = (
    "strategy_version",
    "signal_version",
    "risk_policy_version",
    "regime_classifier_version",
    "cost_model_id",
)


@dataclass(frozen=True)
class StrategyCandidateIdentity:
    candidate_id: str
    strategy_id: str
    strategy_class_name: str
    strategy_source_path: str
    strategy_version: str
    signal_version: str
    risk_policy_version: str
    regime_classifier_version: str
    cost_model_id: str
    allowed_pairs: Sequence[str]
    allowed_timeframes: Sequence[str]
    created_at: str
    source_artifacts: Mapping[str, str]

    def to_dict(self) -> dict[str, Any]:
        return canonicalize_candidate_identity(
            {
                "candidate_id": self.candidate_id,
                "strategy_id": self.strategy_id,
                "strategy_class_name": self.strategy_class_name,
                "strategy_source_path": self.strategy_source_path,
                "strategy_version": self.strategy_version,
                "signal_version": self.signal_version,
                "risk_policy_version": self.risk_policy_version,
                "regime_classifier_version": self.regime_classifier_version,
                "cost_model_id": self.cost_model_id,
                "allowed_pairs": list(self.allowed_pairs),
                "allowed_timeframes": list(self.allowed_timeframes),
                "created_at": self.created_at,
                "source_artifacts": dict(self.source_artifacts),
            }
        )


def build_strategy_candidate_identity(
    *,
    candidate_id: str,
    strategy_id: str,
    strategy_class_name: str,
    strategy_source_path: str | Path,
    strategy_version: str,
    signal_version: str,
    risk_policy_version: str,
    regime_classifier_version: str,
    cost_model_id: str,
    allowed_pairs: Iterable[str] = (),
    allowed_timeframes: Iterable[str] = (),
    created_at: str,
    source_artifacts: Mapping[str, Any] | None = None,
    root_dir: Path | None = None,
) -> dict[str, Any]:
    source_path = _safe_relative_path(Path(strategy_source_path), root_dir)
    artifacts = {
        str(key): _safe_artifact_value(value, root_dir)
        for key, value in dict(source_artifacts or {}).items()
        if value is not None
    }
    if "strategy_source" not in artifacts:
        artifacts["strategy_source"] = source_path
    return canonicalize_candidate_identity(
        StrategyCandidateIdentity(
            candidate_id=str(candidate_id),
            strategy_id=str(strategy_id),
            strategy_class_name=str(strategy_class_name),
            strategy_source_path=source_path,
            strategy_version=str(strategy_version),
            signal_version=str(signal_version),
            risk_policy_version=str(risk_policy_version),
            regime_classifier_version=str(regime_classifier_version),
            cost_model_id=str(cost_model_id),
            allowed_pairs=tuple(str(item) for item in allowed_pairs),
            allowed_timeframes=tuple(str(item) for item in allowed_timeframes),
            created_at=str(created_at),
            source_artifacts=artifacts,
        ).to_dict()
    )


def canonicalize_candidate_identity(identity: Mapping[str, Any] | StrategyCandidateIdentity) -> dict[str, Any]:
    if isinstance(identity, StrategyCandidateIdentity):
        identity = identity.to_dict()
    payload = dict(identity)
    canonical: dict[str, Any] = {}
    for field in REQUIRED_IDENTITY_FIELDS:
        value = payload.get(field)
        if field in {"allowed_pairs", "allowed_timeframes"}:
            canonical[field] = _canonical_list(value)
        elif field == "source_artifacts":
            canonical[field] = _canonical_mapping(value)
        else:
            canonical[field] = "" if value is None else str(value).strip()
    return canonical


def extract_candidate_identity(artifact: Mapping[str, Any] | StrategyCandidateIdentity | None) -> dict[str, Any] | None:
    if artifact is None:
        return None
    if isinstance(artifact, StrategyCandidateIdentity):
        return artifact.to_dict()
    if not isinstance(artifact, Mapping):
        return None
    nested = artifact.get("candidate_identity")
    if isinstance(nested, Mapping):
        return canonicalize_candidate_identity(nested)
    if all(field in artifact for field in REQUIRED_IDENTITY_FIELDS):
        return canonicalize_candidate_identity(artifact)
    return None


def validate_candidate_identity(identity: Mapping[str, Any] | StrategyCandidateIdentity | None) -> dict[str, Any]:
    payload = extract_candidate_identity(identity)
    checks: list[dict[str, Any]] = []
    checks.append(
        _check(
            "candidate_identity_present",
            payload is not None,
            {"required_fields": list(REQUIRED_IDENTITY_FIELDS)},
        )
    )
    if payload is None:
        return _validation_result(False, checks, None)

    missing = [field for field in REQUIRED_IDENTITY_FIELDS if field not in payload]
    blank = [
        field
        for field in REQUIRED_IDENTITY_FIELDS
        if field not in {"allowed_pairs", "allowed_timeframes", "source_artifacts"}
        and not str(payload.get(field) or "").strip()
    ]
    checks.extend(
        [
            _check("required_identity_fields_present", not missing, {"missing_fields": missing}),
            _check("string_identity_fields_non_blank", not blank, {"blank_fields": blank}),
            _check(
                "allowed_pairs_list",
                isinstance(payload.get("allowed_pairs"), list),
                {"type": type(payload.get("allowed_pairs")).__name__},
            ),
            _check(
                "allowed_timeframes_list",
                isinstance(payload.get("allowed_timeframes"), list),
                {"type": type(payload.get("allowed_timeframes")).__name__},
            ),
            _check(
                "source_artifacts_object",
                isinstance(payload.get("source_artifacts"), dict),
                {"type": type(payload.get("source_artifacts")).__name__},
            ),
        ]
    )
    return _validation_result(all(item["passed"] for item in checks), checks, payload)


def compare_candidate_identities(
    expected: Mapping[str, Any] | StrategyCandidateIdentity | None,
    observed: Mapping[str, Any] | StrategyCandidateIdentity | None,
    *,
    observed_label: str = "observed",
    allowed_migrations: Mapping[str, Mapping[str, Sequence[str] | str]] | None = None,
) -> dict[str, Any]:
    expected_payload = extract_candidate_identity(expected)
    observed_payload = extract_candidate_identity(observed)
    expected_validation = validate_candidate_identity(expected_payload)
    observed_validation = validate_candidate_identity(observed_payload)
    mismatches: list[dict[str, Any]] = []
    if expected_payload is not None and observed_payload is not None:
        for field in IDENTITY_COMPARISON_FIELDS:
            expected_value = expected_payload.get(field)
            observed_value = observed_payload.get(field)
            if expected_value == observed_value:
                continue
            if _migration_allowed(
                field,
                expected_value,
                observed_value,
                allowed_migrations or {},
            ):
                continue
            mismatches.append(
                {
                    "field": field,
                    "expected": expected_value,
                    observed_label: observed_value,
                }
            )
    ok = expected_validation["ok"] and observed_validation["ok"] and not mismatches
    return {
        "factory": "strategy_candidate_identity_comparison",
        "schema_version": STRATEGY_CANDIDATE_IDENTITY_SCHEMA_VERSION,
        "ok": ok,
        "observed_label": observed_label,
        "expected_validation": expected_validation,
        "observed_validation": observed_validation,
        "mismatches": mismatches,
        "allowed_migrations_used": bool(allowed_migrations),
    }


def validate_artifact_candidate_identity(
    expected_identity: Mapping[str, Any] | StrategyCandidateIdentity | None,
    artifact: Mapping[str, Any] | None,
    *,
    artifact_label: str,
    allowed_migrations: Mapping[str, Mapping[str, Sequence[str] | str]] | None = None,
) -> dict[str, Any]:
    observed = extract_candidate_identity(artifact)
    comparison = compare_candidate_identities(
        expected_identity,
        observed,
        observed_label=artifact_label,
        allowed_migrations=allowed_migrations,
    )
    return {
        "factory": "strategy_candidate_artifact_identity_validation",
        "schema_version": STRATEGY_CANDIDATE_IDENTITY_SCHEMA_VERSION,
        "ok": comparison["ok"],
        "artifact_label": artifact_label,
        "comparison": comparison,
    }


def load_candidate_identity_from_strategy_source(
    strategy_source_path: Path,
    *,
    strategy_class_name: str | None = None,
    root_dir: Path | None = None,
) -> dict[str, Any] | None:
    if not strategy_source_path.is_file():
        return None
    try:
        tree = ast.parse(strategy_source_path.read_text(encoding="utf-8"), filename=str(strategy_source_path))
    except (OSError, SyntaxError):
        return None

    module_identity = _literal_assignment(tree.body, {"BOT_FACTORY_CANDIDATE_IDENTITY"})
    if module_identity is not None:
        return _identity_from_strategy_literal(module_identity, strategy_source_path, root_dir)

    if strategy_class_name:
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name == strategy_class_name:
                class_identity = _literal_assignment(
                    node.body,
                    {"bot_factory_candidate_identity", "BOT_FACTORY_CANDIDATE_IDENTITY"},
                )
                if class_identity is not None:
                    return _identity_from_strategy_literal(class_identity, strategy_source_path, root_dir)
    return None


def _identity_from_strategy_literal(
    payload: Mapping[str, Any],
    strategy_source_path: Path,
    root_dir: Path | None,
) -> dict[str, Any]:
    candidate = dict(payload)
    candidate.setdefault("strategy_source_path", _safe_relative_path(strategy_source_path, root_dir))
    source_artifacts = dict(candidate.get("source_artifacts") or {})
    source_artifacts.setdefault("strategy_source", _safe_relative_path(strategy_source_path, root_dir))
    candidate["source_artifacts"] = source_artifacts
    return canonicalize_candidate_identity(candidate)


def _literal_assignment(nodes: Sequence[ast.stmt], names: set[str]) -> dict[str, Any] | None:
    for node in nodes:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id in names for target in node.targets):
            continue
        try:
            value = ast.literal_eval(node.value)
        except (ValueError, TypeError):
            return None
        return value if isinstance(value, dict) else None
    return None


def _canonical_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        values = [value]
    else:
        try:
            values = list(value)
        except TypeError:
            values = [value]
    cleaned = [str(item).strip() for item in values if str(item).strip()]
    return sorted(dict.fromkeys(cleaned))


def _canonical_mapping(value: Any) -> dict[str, str]:
    if not isinstance(value, Mapping):
        return {}
    return {
        str(key).strip(): str(raw).replace("\\", "/").strip()
        for key, raw in sorted(value.items(), key=lambda item: str(item[0]))
        if str(key).strip() and raw is not None
    }


def _safe_artifact_value(value: Any, root_dir: Path | None) -> str:
    if isinstance(value, Path):
        return _safe_relative_path(value, root_dir)
    text = str(value).strip()
    if not text:
        return text
    return text.replace("\\", "/")


def _safe_relative_path(path: Path, root_dir: Path | None) -> str:
    if root_dir is not None:
        try:
            return path.resolve().relative_to(root_dir.resolve()).as_posix()
        except (OSError, ValueError):
            pass
    return path.as_posix()


def _migration_allowed(
    field: str,
    expected_value: Any,
    observed_value: Any,
    allowed_migrations: Mapping[str, Mapping[str, Sequence[str] | str]],
) -> bool:
    by_field = allowed_migrations.get(field)
    if not by_field:
        return False
    allowed = by_field.get(str(expected_value))
    if allowed is None:
        return False
    if isinstance(allowed, str):
        return allowed == str(observed_value)
    return str(observed_value) in {str(item) for item in allowed}


def _validation_result(
    ok: bool,
    checks: Sequence[dict[str, Any]],
    identity: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "factory": "strategy_candidate_identity_validation",
        "schema_version": STRATEGY_CANDIDATE_IDENTITY_SCHEMA_VERSION,
        "ok": bool(ok),
        "candidate_identity": identity,
        "checks": list(checks),
    }


def _check(name: str, passed: bool, details: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "details": details or {}}
