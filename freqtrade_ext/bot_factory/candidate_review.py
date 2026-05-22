from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping


REVIEW_REASON_CODES = {
    "historical_gate_failed": "Historical backtest did not pass its local gate.",
    "walk_forward_gate_failed": "Walk-forward evidence did not pass.",
    "regime_scorecard_missing": "Regime scorecard artifact is missing.",
    "paper_readiness_blocked": "Paper readiness has blockers.",
    "selector_no_trade": "Runtime selector would return no_trade.",
}


def build_candidate_review(
    *,
    root_dir: Path,
    candidate_id: str,
    strategy: str,
    strategy_source_path: Path | None = None,
    historical_metrics_path: Path | None = None,
    walk_forward_metrics_path: Path | None = None,
    observation_ledger_path: Path | None = None,
    regime_scorecard_path: Path | None = None,
    selector_candidate_path: Path | None = None,
    selector_decision_path: Path | None = None,
    paper_readiness_path: Path | None = None,
    previous_review_path: Path | None = None,
    reviewer_notes: list[str] | None = None,
) -> dict[str, Any]:
    root = root_dir.resolve()
    historical = _load_optional(historical_metrics_path, root)
    walk_forward = _load_optional(walk_forward_metrics_path, root)
    ledger = _load_optional(observation_ledger_path, root)
    scorecard = _load_optional(regime_scorecard_path, root)
    selector_candidate = _load_optional(selector_candidate_path, root)
    selector_decision = _load_optional(selector_decision_path, root)
    readiness = _load_optional(paper_readiness_path, root)
    previous = _load_optional(previous_review_path, root)
    identity = (
        _identity(scorecard)
        or _identity(selector_candidate)
        or _identity(historical)
        or _identity(walk_forward)
    )
    reason_codes = _review_reason_codes(historical, walk_forward, scorecard, selector_decision, readiness)
    return {
        "factory": "candidate_review_report",
        "schema_version": "candidate_review_v1",
        "generated_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "candidate_id": candidate_id,
        "strategy": strategy,
        "candidate_identity": identity,
        "strategy_source": _source_trace(strategy_source_path, root),
        "historical_metrics": _summary(historical),
        "walk_forward_metrics": _walk_forward_summary(walk_forward),
        "observation_ledger_summary": _ledger_summary(ledger),
        "regime_scorecard_summary": _scorecard_summary(scorecard),
        "baseline_comparison": (scorecard or {}).get("baseline_comparison", {}),
        "selector_candidate_summary": _selector_candidate_summary(selector_candidate),
        "selector_decision_summary": _selector_decision_summary(selector_decision),
        "paper_readiness_blockers": _paper_readiness_blockers(readiness),
        "reason_codes": reason_codes,
        "reason_code_glossary": {code: REVIEW_REASON_CODES.get(code, code) for code in reason_codes},
        "architecture_diagram": [
            "strategy source + backtest metrics + trades + OHLCV",
            "-> observation_ledger.json",
            "-> regime_fitness_scorecard.json",
            "-> selector_candidate.json",
            "-> paper_readiness.json",
        ],
        "what_changed_since_last_candidate_version": _what_changed(previous, identity, scorecard),
        "reviewer_notes": list(reviewer_notes or []),
        "safety_scope": {
            "local_artifacts_only": True,
            "paper_trading_started": False,
            "dry_run_trading_started": False,
            "live_trading_started": False,
            "exchange_order_placement": False,
            "promotion_authorized_by_this_command": False,
        },
    }


def render_candidate_review_markdown(review: Mapping[str, Any]) -> str:
    lines = [
        "# Candidate Review Report",
        "",
        f"- candidate_id: {review.get('candidate_id')}",
        f"- strategy: {review.get('strategy')}",
        f"- reason_codes: {', '.join(review.get('reason_codes', [])) or 'none'}",
        "- paper_live_promotion: not authorized by this report",
        "",
        "## Architecture",
        "",
        "```text",
        *review.get("architecture_diagram", []),
        "```",
        "",
        "## Identity",
        "",
    ]
    identity = review.get("candidate_identity") or {}
    for key in (
        "candidate_id",
        "strategy_id",
        "strategy_class_name",
        "strategy_source_path",
        "strategy_version",
        "signal_version",
        "risk_policy_version",
        "regime_classifier_version",
        "cost_model_id",
    ):
        lines.append(f"- {key}: {identity.get(key)}")
    lines.extend(
        [
            "",
            "## Evidence Summary",
            "",
            f"- historical: {review.get('historical_metrics')}",
            f"- walk_forward: {review.get('walk_forward_metrics')}",
            f"- observation_ledger: {review.get('observation_ledger_summary')}",
            f"- regime_scorecard: {review.get('regime_scorecard_summary')}",
            f"- baseline_comparison: {review.get('baseline_comparison')}",
            f"- selector_decision: {review.get('selector_decision_summary')}",
            f"- paper_readiness_blockers: {review.get('paper_readiness_blockers')}",
            "",
            "## What Changed",
            "",
        ]
    )
    changes = review.get("what_changed_since_last_candidate_version") or []
    lines.extend([f"- {item}" for item in changes] or ["- No previous review supplied."])
    lines.extend(["", "## Reviewer Notes", ""])
    lines.extend([f"- {note}" for note in review.get("reviewer_notes", [])] or ["- None."])
    lines.append("")
    return "\n".join(str(line) for line in lines)


def write_candidate_review_artifacts(
    review: dict[str, Any], *, root_dir: Path, output_root: Path
) -> tuple[Path, Path]:
    root = root_dir.resolve()
    out_dir = (
        (output_root if output_root.is_absolute() else root / output_root)
        / _safe_component(str(review.get("strategy")))
        / _safe_component(str(review.get("candidate_id")))
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "candidate_review.json"
    report_path = out_dir / "candidate_review_report.md"
    json_path.write_text(json.dumps(review, indent=2, ensure_ascii=False), encoding="utf-8")
    report_path.write_text(render_candidate_review_markdown(review), encoding="utf-8")
    return json_path, report_path


def _review_reason_codes(
    historical: dict[str, Any] | None,
    walk_forward: dict[str, Any] | None,
    scorecard: dict[str, Any] | None,
    selector_decision: dict[str, Any] | None,
    readiness: dict[str, Any] | None,
) -> list[str]:
    codes: list[str] = []
    if historical and historical.get("recommendation") not in {None, "pass"}:
        codes.append("historical_gate_failed")
    if walk_forward and walk_forward.get("recommendation") not in {None, "pass"}:
        codes.append("walk_forward_gate_failed")
    if not scorecard:
        codes.append("regime_scorecard_missing")
    if selector_decision and selector_decision.get("action") == "no_trade":
        codes.append("selector_no_trade")
    if readiness and readiness.get("readiness") != "pass":
        codes.append("paper_readiness_blocked")
    return codes


def _what_changed(
    previous: dict[str, Any] | None,
    identity: dict[str, Any] | None,
    scorecard: dict[str, Any] | None,
) -> list[str]:
    if not previous:
        return []
    changes: list[str] = []
    previous_identity = previous.get("candidate_identity") or {}
    for key in ("strategy_version", "signal_version", "risk_policy_version", "regime_classifier_version", "cost_model_id"):
        if previous_identity.get(key) != (identity or {}).get(key):
            changes.append(f"{key}: {previous_identity.get(key)} -> {(identity or {}).get(key)}")
    previous_decision = (previous.get("regime_scorecard_summary") or {}).get("decision")
    current_decision = (scorecard or {}).get("decision")
    if previous_decision != current_decision:
        changes.append(f"scorecard_decision: {previous_decision} -> {current_decision}")
    return changes


def _summary(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not payload:
        return {}
    return {
        "strategy_name": payload.get("strategy_name"),
        "total_return_pct": payload.get("total_return_pct"),
        "trade_count": payload.get("trade_count"),
        "max_drawdown_pct": payload.get("max_drawdown_pct"),
        "profit_factor": payload.get("profit_factor"),
        "recommendation": payload.get("recommendation"),
    }


def _walk_forward_summary(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not payload:
        return {}
    summary = dict(payload.get("summary") or {})
    summary["recommendation"] = payload.get("recommendation")
    return summary


def _ledger_summary(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not payload:
        return {}
    return {
        "ledger_id": payload.get("ledger_id"),
        "ok": payload.get("ok"),
        "observation_count": payload.get("observation_count"),
    }


def _scorecard_summary(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not payload:
        return {}
    return {
        "scorecard_id": payload.get("scorecard_id"),
        "decision": payload.get("decision"),
        "eligible_regimes": payload.get("eligible_regimes", []),
        "blocked_regimes": payload.get("blocked_regimes", []),
        "reason_codes": payload.get("reason_codes", []),
    }


def _selector_candidate_summary(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not payload:
        return {}
    return {
        "candidate_id": payload.get("candidate_id"),
        "logic_id": payload.get("logic_id"),
        "scorecard_decision": payload.get("scorecard_decision"),
        "eligible_regimes": payload.get("eligible_regimes", []),
    }


def _selector_decision_summary(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not payload:
        return {}
    return {
        "action": payload.get("action"),
        "selected_candidate_id": payload.get("selected_candidate_id"),
        "reason_codes": payload.get("reason_codes", []),
    }


def _paper_readiness_blockers(payload: dict[str, Any] | None) -> list[str]:
    if not payload:
        return []
    return [str(item.get("name")) for item in payload.get("blockers", []) if isinstance(item, dict)]


def _identity(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    identity = payload.get("candidate_identity")
    return dict(identity) if isinstance(identity, dict) else None


def _source_trace(path: Path | None, root: Path) -> dict[str, Any]:
    if path is None:
        return {"path": None, "exists": False, "sha256": None}
    resolved = path if path.is_absolute() else root / path
    return {
        "path": _rel(resolved, root),
        "exists": resolved.is_file(),
        "sha256": _sha256(resolved) if resolved.is_file() else None,
    }


def _load_optional(path: Path | None, root: Path) -> dict[str, Any] | None:
    if path is None:
        return None
    resolved = path if path.is_absolute() else root / path
    if not resolved.is_file():
        return None
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rel(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _safe_component(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value).strip("._") or "candidate"
