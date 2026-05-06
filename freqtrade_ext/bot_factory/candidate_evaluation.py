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
    config_path: Path | None = None
    strategy_path: Path | None = None
    ohlcv_parquet_paths: list[Path] | None = None
    static_check_path: Path | None = None
    freqai_validation_path: Path | None = None
    ohlcv_quality_path: Path | None = None
    backtest_metrics_path: Path | None = None
    backtest_trades_path: Path | None = None
    backtest_report_path: Path | None = None
    walk_forward_metrics_path: Path | None = None
    walk_forward_report_path: Path | None = None
    training_manifest_path: Path | None = None
    training_report_path: Path | None = None
    reviewer_notes: list[str] | None = None


def evaluate_candidate(inputs: CandidateEvaluationInputs) -> dict[str, Any]:
    root = inputs.root_dir.resolve()
    proposal = _load_json(_resolve(inputs.proposal_metadata_path, root))
    generated = _load_json(_resolve(inputs.generated_metadata_path, root))

    generator_mode = str(generated.get("generator_mode") or proposal.get("generator_mode") or "rule_based").strip()
    ml_mode_required = generator_mode in {"freqai", "hybrid_ml"}
    strategy_name = str(generated.get("strategy_name") or proposal.get("strategy_name") or "")
    steps = [
        _step("static_strategy_check", inputs.static_check_path, root, key="ok", command_preview=["python", "scripts/bot_factory_static_check.py", _display_path(inputs.strategy_path) or "user_data/strategies"]),
        _step("freqai_feature_label_validation", inputs.freqai_validation_path, root, key="ok", required=ml_mode_required, command_preview=["python", "scripts/bot_factory_validate_freqai_strategy.py", _display_path(inputs.strategy_path) or "<strategy_path>"]),
        _step("ohlcv_quality_check", inputs.ohlcv_quality_path, root, key="ok", command_preview=["python", "scripts/bot_factory_check_ohlcv.py", "<ohlcv_parquet>", "--timeframe", str(proposal.get("timeframe") or generated.get("timeframe") or "")]),
        _step("historical_backtest", inputs.backtest_metrics_path, root, key="recommendation", pass_values={"pass"}, command_preview=["python", "scripts/bot_factory_run_backtest.py", "--strategy", str(generated.get("strategy_name") or proposal.get("strategy_name") or "")]),
        _strategy_identity_step("historical_strategy_identity", inputs.backtest_metrics_path, root, expected_strategy=strategy_name, keys=["strategy_name", "strategy"]),
        _file_step("historical_trades_export", inputs.backtest_trades_path, root, command_preview=["python", "scripts/bot_factory_run_backtest.py", "--export", "trades"]),
        _file_step("historical_markdown_report", inputs.backtest_report_path, root, command_preview=["python", "scripts/bot_factory_generate_report.py", "<backtest-result-json>"]),
        _step("walk_forward", inputs.walk_forward_metrics_path, root, key="recommendation", pass_values={"pass"}, command_preview=["python", "scripts/bot_factory_run_walk_forward.py", "--strategy", str(generated.get("strategy_name") or proposal.get("strategy_name") or "")]),
        _strategy_identity_step("walk_forward_strategy_identity", inputs.walk_forward_metrics_path, root, expected_strategy=strategy_name, keys=["strategy"]),
        _file_step("walk_forward_markdown_report", inputs.walk_forward_report_path, root, command_preview=["python", "scripts/bot_factory_run_walk_forward.py", "--write-report"]),
        _step("training_factory", inputs.training_manifest_path, root, key="recommendation", pass_values={"pass"}, required=ml_mode_required, command_preview=["python", "scripts/bot_factory_run_freqai_training.py", "--strategy", str(generated.get("strategy_name") or proposal.get("strategy_name") or "")]),
        _strategy_identity_step("training_strategy_identity", inputs.training_manifest_path, root, expected_strategy=strategy_name, keys=["strategy"], required=ml_mode_required),
        _file_step("training_markdown_report", inputs.training_report_path, root, required=ml_mode_required, command_preview=["python", "scripts/bot_factory_run_freqai_training.py", "--write-report"]),
    ]

    checks = [s["check"] for s in steps]

    failures = [c for c in checks if c["status"] in {"fail", "missing"}]
    failure_codes = list(dict.fromkeys(proposal.get("failure_taxonomy_codes", []) + generated.get("failure_taxonomy_codes", [])))

    if proposal.get("code_generation_eligible") is False or generated.get("candidate_evaluation_eligible") is False:
        recommendation = "reject"
        recommendation_rationale = "Candidate is not eligible for evaluation based on proposal/generated metadata flags."
    elif not failures:
        recommendation = "pass"
        recommendation_rationale = "All required historical-safe checks passed for this generator mode."
    elif any(c["status"] == "missing" for c in failures):
        recommendation = "fail"
        recommendation_rationale = "Required artifacts are missing; regenerate/evaluate required steps before retry."
    elif failure_codes:
        recommendation = "retry"
        recommendation_rationale = "Checks failed with known taxonomy; retry is allowed with thesis-guided iteration input."
    else:
        recommendation = "fail"
        recommendation_rationale = "Checks failed without retry taxonomy guidance."

    manifest = {
        "generated_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "factory": "candidate_evaluation_pipeline",
        "candidate_id": inputs.candidate_id,
        "strategy_name": strategy_name,
        "proposal_metadata_path": _rel(_resolve(inputs.proposal_metadata_path, root), root),
        "generated_metadata_path": _rel(_resolve(inputs.generated_metadata_path, root), root),
        "input_paths": {
            "config": _maybe_rel(inputs.config_path, root),
            "strategy": _maybe_rel(inputs.strategy_path, root) or generated.get("generated_strategy_path"),
            "ohlcv_parquet": [_rel(_resolve(path, root), root) for path in inputs.ohlcv_parquet_paths or []],
        },
        "checks": checks,
        "evaluation_orchestration": {
            "generator_mode": generator_mode,
            "ml_mode_required": ml_mode_required,
            "steps": steps,
            "executed_by_pipeline": False,
            "note": "This pipeline aggregates existing local historical-safe artifacts only and records safe command previews.",
        },
        "recommendation": recommendation,
        "recommendation_rationale": recommendation_rationale,
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
            "thesis_statement": proposal.get("thesis_statement") or generated.get("thesis_statement"),
            "evidence_refs": proposal.get("evidence_refs") or generated.get("evidence_refs") or [],
        },
        "reviewer_notes": inputs.reviewer_notes or [],
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
        "status": manifest["recommendation"],
        "failure_taxonomy_codes": manifest.get("failure_taxonomy_codes", []),
        "thesis_id": manifest.get("thesis", {}).get("thesis_id"),
        "manifest_path": _rel(manifest_path, root),
        "recommendation_rationale": manifest.get("recommendation_rationale"),
        "key_metrics": _extract_metrics_summary(manifest),
        "artifact_paths": _extract_artifact_paths(manifest),
        "candidate_record_path": _rel(out_dir / "candidate_record.json", root),
        "metrics_summary_path": _rel(out_dir / "metrics_summary.json", root),
        "artifact_paths_path": _rel(out_dir / "artifact_paths.json", root),
        "candidate_report_path": _rel(out_dir / "candidate_report.md", root),
    }

    candidate_record = {
        "candidate_id": candidate_id,
        "strategy_name": strategy,
        "recommendation": manifest["recommendation"],
        "failure_taxonomy_codes": manifest.get("failure_taxonomy_codes", []),
        "thesis": manifest.get("thesis", {}),
        "next_candidate_input": manifest.get("next_candidate_input", {}),
        "input_paths": manifest.get("input_paths", {}),
        "artifact_paths": _extract_artifact_paths(manifest),
        "metrics_summary": _extract_metrics_summary(manifest),
        "manifest_path": _rel(manifest_path, root),
        "recommendation_rationale": manifest.get("recommendation_rationale"),
    }
    (out_dir / "candidate_record.json").write_text(json.dumps(candidate_record, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "candidate_report.md").write_text(_render_candidate_report(manifest), encoding="utf-8")
    (out_dir / "metrics_summary.json").write_text(json.dumps(_extract_metrics_summary(manifest), indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "artifact_paths.json").write_text(json.dumps(_extract_artifact_paths(manifest), indent=2, ensure_ascii=False), encoding="utf-8")
    with idx_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return manifest_path, idx_path


def _safe_path_component(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value.strip())
    cleaned = cleaned.strip("._")
    return cleaned or "unknown"


def _render_candidate_report(manifest: dict[str, Any]) -> str:
    lines = [
        "# Candidate Report",
        "",
        f"- candidate_id: {manifest.get('candidate_id')}",
        f"- strategy_name: {manifest.get('strategy_name')}",
        f"- recommendation: {manifest.get('recommendation')}",
        f"- rationale: {manifest.get('recommendation_rationale', '')}",
        "- paper_live_promotion: not authorized by this pipeline",
        "",
        "## Checks",
        "",
    ]
    for check in manifest.get("checks", []):
        lines.append(
            f"- {check.get('name')}: {check.get('status')} ({check.get('path')})"
        )
    lines.extend(["", "## Reviewer Notes", ""])
    notes = manifest.get("reviewer_notes") or []
    if notes:
        lines.extend(f"- {note}" for note in notes)
    else:
        lines.append("- None.")
    lines.append("")
    return "\n".join(lines)


def _check(name: str, path: Path | None, root: Path, *, key: str, pass_values: set[str] | None = None, required: bool = True) -> dict[str, Any]:
    if path is None:
        return {"name": name, "status": "missing" if required else "skipped", "path": None}
    resolved = _resolve(path, root)
    if not resolved.is_file():
        return {"name": name, "status": "missing" if required else "skipped", "path": _rel(resolved, root)}
    payload = _load_json(resolved)
    value = payload.get(key)
    passed = bool(value) if pass_values is None else str(value) in pass_values
    return {
        "name": name,
        "status": "pass" if passed else "fail",
        "path": _rel(resolved, root),
        "value": value,
        "payload_summary": _summarize_payload(name, payload),
    }


def _file_check(name: str, path: Path | None, root: Path, *, required: bool = True) -> dict[str, Any]:
    if path is None:
        return {"name": name, "status": "missing" if required else "skipped", "path": None}
    resolved = _resolve(path, root)
    if not resolved.is_file():
        return {"name": name, "status": "missing" if required else "skipped", "path": _rel(resolved, root)}
    return {
        "name": name,
        "status": "pass",
        "path": _rel(resolved, root),
        "bytes": resolved.stat().st_size,
    }


def _strategy_identity_check(
    name: str,
    path: Path | None,
    root: Path,
    *,
    expected_strategy: str,
    keys: list[str],
    required: bool = True,
) -> dict[str, Any]:
    if path is None:
        return {"name": name, "status": "missing" if required else "skipped", "path": None}
    resolved = _resolve(path, root)
    if not resolved.is_file():
        return {"name": name, "status": "missing" if required else "skipped", "path": _rel(resolved, root)}
    payload = _load_json(resolved)
    observed = None
    for key in keys:
        if payload.get(key):
            observed = str(payload.get(key))
            break
    passed = bool(expected_strategy) and observed == expected_strategy
    return {
        "name": name,
        "status": "pass" if passed else "fail",
        "path": _rel(resolved, root),
        "expected_strategy": expected_strategy,
        "observed_strategy": observed,
    }


def _step(name: str, path: Path | None, root: Path, *, key: str, pass_values: set[str] | None = None, required: bool = True, command_preview: list[str] | None = None) -> dict[str, Any]:
    check = _check(name, path, root, key=key, pass_values=pass_values, required=required)
    return {"name": name, "check": check, "input_path": check.get("path"), "output_path": check.get("path"), "output_status": check.get("status"), "command_preview": command_preview or []}


def _file_step(name: str, path: Path | None, root: Path, *, required: bool = True, command_preview: list[str] | None = None) -> dict[str, Any]:
    check = _file_check(name, path, root, required=required)
    return {"name": name, "check": check, "input_path": check.get("path"), "output_path": check.get("path"), "output_status": check.get("status"), "command_preview": command_preview or []}


def _strategy_identity_step(
    name: str,
    path: Path | None,
    root: Path,
    *,
    expected_strategy: str,
    keys: list[str],
    required: bool = True,
) -> dict[str, Any]:
    check = _strategy_identity_check(
        name,
        path,
        root,
        expected_strategy=expected_strategy,
        keys=keys,
        required=required,
    )
    return {"name": name, "check": check, "input_path": check.get("path"), "output_path": check.get("path"), "output_status": check.get("status"), "command_preview": []}


def _extract_metrics_summary(manifest: dict[str, Any]) -> dict[str, Any]:
    checks = {item["name"]: item for item in manifest.get("checks", [])}
    return {
        "recommendation": manifest.get("recommendation"),
        "historical_backtest": checks.get("historical_backtest", {}).get("value"),
        "walk_forward": checks.get("walk_forward", {}).get("value"),
        "training_factory": checks.get("training_factory", {}).get("value"),
        "historical_summary": checks.get("historical_backtest", {}).get("payload_summary", {}),
        "walk_forward_summary": checks.get("walk_forward", {}).get("payload_summary", {}),
        "training_summary": checks.get("training_factory", {}).get("payload_summary", {}),
    }


def _extract_artifact_paths(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "proposal_metadata_path": manifest.get("proposal_metadata_path"),
        "generated_metadata_path": manifest.get("generated_metadata_path"),
        "input_paths": manifest.get("input_paths", {}),
        "check_artifact_paths": {item.get("name"): item.get("path") for item in manifest.get("checks", [])},
    }


def _resolve(path: Path, root: Path) -> Path:
    return (path if path.is_absolute() else root / path).resolve()


def _maybe_rel(path: Path | None, root: Path) -> str | None:
    if path is None:
        return None
    return _rel(_resolve(path, root), root)


def _display_path(path: Path | None) -> str | None:
    if path is None:
        return None
    return str(path)


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


def _summarize_payload(name: str, payload: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "strategy_name",
        "strategy",
        "recommendation",
        "status",
        "total_return",
        "total_return_pct",
        "trade_count",
        "max_drawdown_pct",
        "profit_factor",
        "sortino",
    ]
    summary = {key: payload.get(key) for key in keys if key in payload}
    nested = payload.get("summary")
    if isinstance(nested, dict):
        summary["summary"] = {
            key: nested.get(key)
            for key in [
                "window_count",
                "pass_rate",
                "profitable_windows_ratio",
                "total_return",
                "total_return_pct",
                "max_drawdown_pct_any_window",
                "max_single_window_profit_dependency",
                "stage_count",
                "completed_stages",
                "failed_stages",
            ]
            if key in nested
        }
    if name == "training_factory" and isinstance(payload.get("stages"), list):
        summary["stage_recommendations"] = [
            stage.get("recommendation") for stage in payload["stages"] if isinstance(stage, dict)
        ]
    return summary
