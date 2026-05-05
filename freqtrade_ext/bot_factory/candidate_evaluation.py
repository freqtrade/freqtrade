from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CandidateEvaluationInputs:
    root_dir: Path
    proposal_metadata_path: Path
    generated_metadata_path: Path
    candidate_id: str
    output_root: Path = Path("registry/strategies/candidates")
    index_path: Path = Path("registry/strategies/candidates/index.jsonl")
    static_check_path: Path | None = None
    freqai_validation_path: Path | None = None
    ohlcv_quality_path: Path | None = None
    backtest_metrics_path: Path | None = None
    walk_forward_metrics_path: Path | None = None
    training_manifest_path: Path | None = None


def evaluate_candidate(inputs: CandidateEvaluationInputs) -> dict[str, Any]:
    root = inputs.root_dir.resolve()
    proposal = _load_json(_resolve(inputs.proposal_metadata_path, root))
    generated = _load_json(_resolve(inputs.generated_metadata_path, root))

    generator_mode = str(generated.get("generator_mode") or proposal.get("generator_mode") or "rule_based").strip()
    ml_mode_required = generator_mode in {"freqai", "hybrid_ml"}
    checks = []
    checks.append(_check("static_strategy_check", inputs.static_check_path, root, key="ok"))
    checks.append(_check("freqai_feature_label_validation", inputs.freqai_validation_path, root, key="ok", required=ml_mode_required))
    checks.append(_check("ohlcv_quality_check", inputs.ohlcv_quality_path, root, key="ok"))
    checks.append(_check("historical_backtest", inputs.backtest_metrics_path, root, key="recommendation", pass_values={"pass"}))
    checks.append(_check("walk_forward", inputs.walk_forward_metrics_path, root, key="recommendation", pass_values={"pass"}))
    checks.append(_check("training_factory", inputs.training_manifest_path, root, key="recommendation", pass_values={"pass"}, required=ml_mode_required))

    failures = [c for c in checks if c["status"] in {"fail", "missing"}]
    failure_codes = list(dict.fromkeys(proposal.get("failure_taxonomy_codes", []) + generated.get("failure_taxonomy_codes", [])))
    recommendation = "pass" if not failures else ("retry" if failure_codes else "fail")
    if proposal.get("code_generation_eligible") is False or generated.get("candidate_evaluation_eligible") is False:
        recommendation = "reject"

    manifest = {
        "generated_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "factory": "candidate_evaluation_pipeline",
        "candidate_id": inputs.candidate_id,
        "strategy_name": generated.get("strategy_name") or proposal.get("strategy_name"),
        "proposal_metadata_path": _rel(_resolve(inputs.proposal_metadata_path, root), root),
        "generated_metadata_path": _rel(_resolve(inputs.generated_metadata_path, root), root),
        "checks": checks,
        "recommendation": recommendation,
        "failure_taxonomy_codes": failure_codes,
        "thesis": {
            "thesis_id": proposal.get("thesis_id") or generated.get("thesis_id"),
            "thesis_type": proposal.get("thesis_type") or generated.get("thesis_type"),
            "falsification_criteria": proposal.get("falsification_criteria") or generated.get("falsification_criteria"),
        },
        "next_candidate_input": {
            "thesis_id": proposal.get("thesis_id") or generated.get("thesis_id"),
            "failure_taxonomy_codes": failure_codes,
            "retry_budget_per_thesis": proposal.get("retry_budget_per_thesis"),
            "thesis_retry_count": proposal.get("thesis_retry_count"),
            "parameter_only_retry_count": proposal.get("parameter_only_retry_count"),
            "force_distinct_hypothesis_family": proposal.get("force_distinct_hypothesis_family", False),
        },
        "safety_scope": {"historical_only": True, "live_trading": False, "process_control": False},
    }
    return manifest


def write_candidate_artifacts(manifest: dict[str, Any], *, root_dir: Path, output_root: Path, index_path: Path) -> tuple[Path, Path]:
    root = root_dir.resolve()
    strategy = _safe_path_component(str(manifest.get("strategy_name") or "unknown_strategy"))
    candidate_id = _safe_path_component(str(manifest["candidate_id"]))
    out_dir = _resolve(output_root, root) / strategy / candidate_id
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "candidate_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    idx_path = _resolve(index_path, root)
    idx_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "created_at": manifest["generated_at"],
        "candidate_id": candidate_id,
        "strategy_name": strategy,
        "recommendation": manifest["recommendation"],
        "failure_taxonomy_codes": manifest.get("failure_taxonomy_codes", []),
        "thesis_id": manifest.get("thesis", {}).get("thesis_id"),
        "manifest_path": _rel(manifest_path, root),
    }
    with idx_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return manifest_path, idx_path


def _safe_path_component(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value.strip())
    cleaned = cleaned.strip("._")
    return cleaned or "unknown"


def _check(name: str, path: Path | None, root: Path, *, key: str, pass_values: set[str] | None = None, required: bool = True) -> dict[str, Any]:
    if path is None:
        return {"name": name, "status": "missing" if required else "skipped", "path": None}
    resolved = _resolve(path, root)
    if not resolved.is_file():
        return {"name": name, "status": "missing" if required else "skipped", "path": _rel(resolved, root)}
    payload = _load_json(resolved)
    value = payload.get(key)
    passed = bool(value) if pass_values is None else str(value) in pass_values
    return {"name": name, "status": "pass" if passed else "fail", "path": _rel(resolved, root), "value": value}


def _resolve(path: Path, root: Path) -> Path:
    return (path if path.is_absolute() else root / path).resolve()


def _rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root)).replace("\\", "/")
    except Exception:
        return str(path)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected object JSON: {path}")
    return payload
