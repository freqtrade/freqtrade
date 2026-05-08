from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence


RESEARCH_HANDOFF_KEYS = (
    "research_decision_question_handoff",
    "research_decision_novelty_handoff",
    "local_falsification_handoff",
    "structural_data_quality_handoff",
    "structural_data_capability_handoff",
)


@dataclass(frozen=True)
class CandidateRankingInputs:
    root_dir: Path
    candidate_manifest_paths: Sequence[Path]
    output_root: Path = Path("registry/strategies/candidates/rankings")
    ranking_id: str | None = None
    reviewer_notes: Sequence[str] = field(default_factory=list)


def rank_candidates(inputs: CandidateRankingInputs) -> dict[str, Any]:
    root = inputs.root_dir.resolve()
    generated_at = datetime.now(UTC).replace(microsecond=0).isoformat()
    candidates = [
        _candidate_row(_load_json(_resolve(path, root)), _resolve(path, root), root)
        for path in inputs.candidate_manifest_paths
    ]
    hypothesis_diversity = _hypothesis_diversity(candidates)
    _apply_hypothesis_diversity_gate(candidates, hypothesis_diversity)
    ranked = sorted(
        candidates,
        key=lambda item: (
            item["paper_ready_eligible"],
            item["recommendation"] == "pass",
            item["score"],
        ),
        reverse=True,
    )
    for index, item in enumerate(ranked, start=1):
        item["rank"] = index

    return {
        "generated_at": generated_at,
        "factory": "candidate_ranking_registry",
        "ranking_id": inputs.ranking_id or _ranking_id(generated_at),
        "candidate_count": len(ranked),
        "hypothesis_diversity": hypothesis_diversity,
        "ranked_candidates": ranked,
        "best_candidate_id": ranked[0]["candidate_id"] if ranked else None,
        "paper_ready_candidate_ids": [
            item["candidate_id"] for item in ranked if item["paper_ready_eligible"]
        ],
        "reviewer_notes": list(inputs.reviewer_notes),
        "safety_scope": {
            "historical_only": True,
            "paper_trading_started": False,
            "dry_run_trading_started": False,
            "live_trading": False,
            "exchange_order_placement": False,
            "process_control": False,
            "promotion_authorized_by_this_command": False,
        },
    }


def write_candidate_ranking_artifacts(
    ranking: dict[str, Any], *, root_dir: Path, output_root: Path
) -> tuple[Path, Path]:
    root = root_dir.resolve()
    out_dir = _resolve(output_root, root) / _safe_path_component(str(ranking["ranking_id"]))
    out_dir.mkdir(parents=True, exist_ok=True)
    ranking_path = out_dir / "candidate_ranking.json"
    report_path = out_dir / "candidate_ranking_report.md"
    ranking_path.write_text(json.dumps(ranking, indent=2, ensure_ascii=False), encoding="utf-8")
    report_path.write_text(_render_report(ranking), encoding="utf-8")
    return ranking_path, report_path


def _candidate_row(manifest: dict[str, Any], manifest_path: Path, root: Path) -> dict[str, Any]:
    checks = {check.get("name"): check for check in manifest.get("checks", [])}
    historical = _summary(checks.get("historical_backtest"))
    walk_forward = _summary(checks.get("walk_forward"))
    training = _summary(checks.get("training_factory"))
    recommendation = str(manifest.get("recommendation") or "fail")
    required_chain = [
        "historical_backtest",
        "historical_strategy_identity",
        "historical_trades_export",
        "historical_markdown_report",
        "walk_forward",
        "walk_forward_strategy_identity",
        "walk_forward_markdown_report",
        "training_factory",
        "training_strategy_identity",
        "training_markdown_report",
    ]
    missing_or_failed = [
        name for name in required_chain if checks.get(name, {}).get("status") != "pass"
    ]
    paper_ready_eligible = recommendation == "pass" and not missing_or_failed
    reasons = _ranking_reasons(manifest, checks, missing_or_failed)
    research_brief = _research_brief_summary(manifest)
    metrics = {
        "historical_total_return_pct": _number(historical.get("total_return_pct")),
        "historical_trade_count": _number(historical.get("trade_count")),
        "historical_max_drawdown_pct": _number(historical.get("max_drawdown_pct")),
        "historical_profit_factor": _number(historical.get("profit_factor")),
        "walk_forward_total_return_pct": _number(walk_forward.get("total_return_pct")),
        "walk_forward_pass_rate": _number(walk_forward.get("pass_rate")),
        "walk_forward_profitable_windows_ratio": _number(
            walk_forward.get("profitable_windows_ratio")
        ),
        "walk_forward_max_single_window_profit_dependency": _number(
            walk_forward.get("max_single_window_profit_dependency")
        ),
        "training_stage_count": _number(training.get("stage_count")),
        "training_failed_stages": _number(training.get("failed_stages")),
    }
    return {
        "candidate_id": manifest.get("candidate_id"),
        "strategy_name": manifest.get("strategy_name"),
        "manifest_path": _rel(manifest_path, root),
        "recommendation": recommendation,
        "status": recommendation,
        "paper_ready_eligible": paper_ready_eligible,
        "paper_ready_blockers": missing_or_failed,
        "score": _score(recommendation, metrics, paper_ready_eligible),
        "metrics": metrics,
        "failure_taxonomy_codes": manifest.get("failure_taxonomy_codes", []),
        "thesis": manifest.get("thesis", {}),
        "hypothesis_family": _hypothesis_family(manifest),
        "strategy_logic_variant": _strategy_logic_variant(manifest),
        "blocked_next_actions": _blocked_next_actions(research_brief, manifest),
        "research_brief": research_brief,
        "research_handoff_summary": _research_handoff_summary(research_brief),
        "reasons": reasons,
        "artifact_paths": _artifact_paths(manifest),
    }


def _ranking_reasons(
    manifest: dict[str, Any], checks: dict[str, dict[str, Any]], missing_or_failed: list[str]
) -> list[str]:
    reasons: list[str] = []
    rationale = manifest.get("recommendation_rationale")
    if rationale:
        reasons.append(str(rationale))
    for name, check in checks.items():
        if check.get("status") in {"fail", "missing"}:
            reasons.append(f"{name}={check.get('status')}")
    if missing_or_failed:
        reasons.append(
            "paper_ready_blocked_until_full_historical_walk_forward_training_artifact_chain_passes"
        )
    return list(dict.fromkeys(reasons))


def _score(
    recommendation: str, metrics: dict[str, float | None], paper_ready_eligible: bool
) -> float:
    base = {"pass": 100.0, "retry": 40.0, "fail": 10.0, "reject": -100.0}.get(
        recommendation, 0.0
    )
    base += 25.0 if paper_ready_eligible else 0.0
    base += (metrics.get("historical_total_return_pct") or 0.0) * 2.0
    base += (metrics.get("walk_forward_total_return_pct") or 0.0) * 3.0
    base += (metrics.get("walk_forward_pass_rate") or 0.0) * 20.0
    base += (metrics.get("walk_forward_profitable_windows_ratio") or 0.0) * 15.0
    base += min(metrics.get("historical_trade_count") or 0.0, 500.0) / 50.0
    base -= metrics.get("historical_max_drawdown_pct") or 0.0
    concentration = metrics.get("walk_forward_max_single_window_profit_dependency")
    if concentration is not None:
        base -= max(concentration - 0.4, 0.0) * 50.0
    base -= (metrics.get("training_failed_stages") or 0.0) * 25.0
    return round(base, 6)


def _hypothesis_diversity(candidates: Sequence[dict[str, Any]]) -> dict[str, Any]:
    families = sorted(
        {
            str(item.get("hypothesis_family") or "").strip()
            for item in candidates
            if str(item.get("hypothesis_family") or "").strip()
        }
    )
    thesis_ids = sorted(
        {
            thesis_id
            for thesis_id in (_candidate_thesis_id(item) for item in candidates)
            if thesis_id
        }
    )
    variants = sorted(
        {
            str(item.get("strategy_logic_variant") or "").strip()
            for item in candidates
            if str(item.get("strategy_logic_variant") or "").strip()
        }
    )
    return {
        "minimum_distinct_hypothesis_families": 2,
        "distinct_hypothesis_families": families,
        "distinct_thesis_ids": thesis_ids,
        "distinct_strategy_logic_variants": variants,
        "passed": len(families) >= 2,
    }


def _candidate_thesis_id(item: dict[str, Any]) -> str | None:
    thesis = item.get("thesis") if isinstance(item.get("thesis"), dict) else {}
    value = str(thesis.get("thesis_id") or "").strip()
    return value or None


def _apply_hypothesis_diversity_gate(
    candidates: Sequence[dict[str, Any]], diversity: dict[str, Any]
) -> None:
    if diversity.get("passed"):
        return
    for item in candidates:
        if item.get("paper_ready_eligible"):
            item["paper_ready_eligible"] = False
            blockers = list(item.get("paper_ready_blockers", []))
            blockers.append("hypothesis_diversity")
            item["paper_ready_blockers"] = list(dict.fromkeys(blockers))
            reasons = list(item.get("reasons", []))
            reasons.append(
                "paper_ready_blocked_until_multiple_distinct_hypothesis_families_are_evaluated"
            )
            item["reasons"] = list(dict.fromkeys(reasons))
            item["score"] = _score(
                str(item.get("recommendation") or "fail"),
                item.get("metrics", {}),
                False,
            )


def _hypothesis_family(manifest: dict[str, Any]) -> str | None:
    thesis = manifest.get("thesis") if isinstance(manifest.get("thesis"), dict) else {}
    next_input = (
        manifest.get("next_candidate_input")
        if isinstance(manifest.get("next_candidate_input"), dict)
        else {}
    )
    raw = (
        thesis.get("thesis_type")
        or next_input.get("thesis_type")
        or manifest.get("strategy_logic_variant")
    )
    return _safe_token(raw)


def _strategy_logic_variant(manifest: dict[str, Any]) -> str | None:
    next_input = (
        manifest.get("next_candidate_input")
        if isinstance(manifest.get("next_candidate_input"), dict)
        else {}
    )
    return _safe_token(manifest.get("strategy_logic_variant") or next_input.get("strategy_logic_variant"))


def _summary(check: dict[str, Any] | None) -> dict[str, Any]:
    if not check:
        return {}
    payload = check.get("payload_summary") or {}
    nested = payload.get("summary")
    if isinstance(nested, dict):
        merged = dict(payload)
        merged.update(nested)
        merged.pop("summary", None)
        return merged
    return dict(payload)


def _artifact_paths(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "manifest": manifest.get("manifest_path"),
        "proposal_metadata": manifest.get("proposal_metadata_path"),
        "generated_metadata": manifest.get("generated_metadata_path"),
        "checks": {
            check.get("name"): check.get("path")
            for check in manifest.get("checks", [])
            if check.get("path")
        },
    }


def _research_brief_summary(manifest: dict[str, Any]) -> dict[str, Any]:
    raw = manifest.get("research_brief")
    if not isinstance(raw, dict):
        return {}
    keys = (
        "thesis_id",
        "thesis_type",
        "thesis_statement",
        "research_references",
        "evidence_refs",
        "novelty_vs_previous",
        "blocked_next_actions",
        *RESEARCH_HANDOFF_KEYS,
    )
    return {key: _copy_jsonish(raw[key]) for key in keys if key in raw}


def _research_handoff_summary(research_brief: dict[str, Any]) -> dict[str, Any]:
    return {
        key: _copy_jsonish(research_brief[key])
        for key in RESEARCH_HANDOFF_KEYS
        if isinstance(research_brief.get(key), dict)
    }


def _blocked_next_actions(*sources: Any) -> list[str]:
    actions: list[str] = []
    for source in sources:
        if not isinstance(source, dict):
            continue
        _extend_unique_strings(actions, source.get("blocked_next_actions", []))
        next_input = source.get("next_candidate_input")
        if isinstance(next_input, dict):
            _extend_unique_strings(actions, next_input.get("blocked_next_actions", []))
    return actions


def _extend_unique_strings(target: list[str], values: Any) -> None:
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, list):
        return
    for value in values:
        text = str(value).strip()
        if text and text not in target:
            target.append(text)


def _copy_jsonish(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _copy_jsonish(nested) for key, nested in value.items()}
    if isinstance(value, list):
        return [_copy_jsonish(item) for item in value]
    return value


def _number(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _render_report(ranking: dict[str, Any]) -> str:
    lines = [
        "# Candidate Ranking Report",
        "",
        f"- ranking_id: {ranking.get('ranking_id')}",
        f"- candidate_count: {ranking.get('candidate_count')}",
        f"- best_candidate_id: {ranking.get('best_candidate_id')}",
        "- paper_live_promotion: not authorized by this ranking",
        "",
        "## Ranked Candidates",
        "",
    ]
    for item in ranking.get("ranked_candidates", []):
        lines.append(
            f"- {item.get('rank')}. {item.get('candidate_id')} "
            f"{item.get('recommendation')} score={item.get('score')} "
            f"paper_ready={item.get('paper_ready_eligible')}"
        )
    lines.extend(["", "## Reviewer Notes", ""])
    notes = ranking.get("reviewer_notes") or []
    lines.extend([f"- {note}" for note in notes] or ["- None."])
    lines.append("")
    return "\n".join(lines)


def _ranking_id(generated_at: str) -> str:
    parsed = datetime.fromisoformat(generated_at.replace("Z", "+00:00"))
    return parsed.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")


def _safe_path_component(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value)
    return cleaned.strip("._") or "ranking"


def _safe_token(value: Any) -> str | None:
    if value is None:
        return None
    token = "".join(
        ch.lower() if ch.isalnum() else "_" for ch in str(value).strip()
    ).strip("_")
    return token or None


def _resolve(path: Path, root: Path) -> Path:
    return (path if path.is_absolute() else root / path).resolve()


def _rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root)).replace("\\", "/")
    except ValueError:
        return str(path)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected object JSON: {path}")
    return payload
