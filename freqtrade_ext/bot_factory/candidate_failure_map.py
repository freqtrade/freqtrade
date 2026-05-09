from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

_MATERIAL_CAUSAL_CATEGORY_MIN_SHARE = 0.70
_DEFAULT_CAUSAL_CATEGORY_SEVERITY = 1.0
_CAUSAL_CATEGORY_SEVERITY_MULTIPLIERS = {
    "walk_forward_fragility": 1.35,
    "regime_fragile_mechanism": 1.30,
    "no_profitable_walk_forward_windows": 1.30,
    "entry_exists_negative_edge": 1.25,
    "generated_entry_negative_edge": 1.25,
    "cost_sensitive_mechanism": 1.20,
    "ml_rule_alignment_failure": 1.15,
    "zero_trade_or_signal_sparsity": 1.10,
    "thesis_rejected_after_entries": 1.10,
    "training_or_artifact_gap": 1.05,
}
_CAUSAL_CATEGORY_RESPONSE_FOCUS = {
    "walk_forward_fragility": [
        "predefined passing and failing walk-forward regimes",
        "local split evidence before proposal generation",
    ],
    "regime_fragile_mechanism": [
        "state definition that is observable on closed candles",
        "why the edge survives regime transitions",
    ],
    "no_profitable_walk_forward_windows": [
        "window-level edge source",
        "stop condition when no historical split is profitable",
    ],
    "entry_exists_negative_edge": [
        "positive expectancy after fees when entries exist",
        "why prior entry logic signs were wrong",
    ],
    "generated_entry_negative_edge": [
        "generated entry set expectancy after costs",
        "why proposal event evidence did not survive code-generation entry masks",
    ],
    "cost_sensitive_mechanism": [
        "fee, slippage, and turnover exposure",
        "expected edge per trade before thresholds are tuned",
    ],
    "ml_rule_alignment_failure": [
        "overlap between model predictions and rule-gate survivor rows",
        "target freshness and candidate-scoped model artifacts",
    ],
    "zero_trade_or_signal_sparsity": [
        "entry-row survival through each gate",
        "avoidance of conjunction-heavy filters",
    ],
}


@dataclass(frozen=True)
class CandidateFailureMapInputs:
    root_dir: Path
    synthesis_path: Path
    output_root: Path = Path("registry/strategies/failure_maps")
    map_id: str | None = None
    reviewer_notes: Sequence[str] = field(default_factory=list)


def build_candidate_failure_map(inputs: CandidateFailureMapInputs) -> dict[str, Any]:
    root = inputs.root_dir.resolve()
    synthesis_path = _resolve_inside(inputs.synthesis_path, root)
    synthesis = _load_json(synthesis_path)
    generated_at = datetime.now(UTC).replace(microsecond=0).isoformat()
    candidates = [
        _candidate_failure_profile(item)
        for item in synthesis.get("candidates", [])
        if isinstance(item, dict)
    ]
    category_summary = _category_summary(candidates)
    guidance = _research_selection_guidance(
        synthesis,
        category_summary,
        candidate_count=len(candidates),
    )
    checks = [
        _check("synthesis_file_present", synthesis_path.is_file()),
        _check(
            "synthesis_factory_valid",
            synthesis.get("factory") == "candidate_failure_synthesis",
        ),
        _check("synthesis_has_candidates", bool(candidates)),
        _check(
            "paper_ready_candidates_absent",
            not bool(
                synthesis.get("aggregate_failure_summary", {}).get(
                    "paper_ready_candidate_ids"
                )
            ),
        ),
    ]
    return {
        "generated_at": generated_at,
        "factory": "candidate_failure_map",
        "map_id": inputs.map_id or _map_id(generated_at),
        "status": "completed" if candidates else "blocked",
        "source_synthesis_path": _rel(synthesis_path, root),
        "source_synthesis_id": synthesis.get("synthesis_id"),
        "candidate_count": len(candidates),
        "checks": checks,
        "causal_failure_categories": category_summary,
        "candidate_failure_profiles": candidates,
        "research_selection_guidance": guidance,
        "reviewer_notes": list(inputs.reviewer_notes),
        "safety_scope": {
            "historical_only": True,
            "backtest_started": False,
            "strategy_code_generated": False,
            "paper_trading_started": False,
            "dry_run_trading_started": False,
            "live_trading": False,
            "exchange_order_placement": False,
            "process_control": False,
            "promotion_authorized_by_this_command": False,
            "local_artifacts_source_of_truth": True,
        },
    }


def write_candidate_failure_map_artifacts(
    failure_map: dict[str, Any], *, root_dir: Path, output_root: Path
) -> tuple[Path, Path]:
    root = root_dir.resolve()
    map_id = _safe_path_component(str(failure_map.get("map_id") or "failure_map"))
    out_dir = _resolve_inside(output_root, root) / map_id
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "causal_failure_map.json"
    report_path = out_dir / "causal_failure_map_report.md"
    json_path.write_text(json.dumps(failure_map, indent=2, ensure_ascii=False), encoding="utf-8")
    report_path.write_text(_render_report(failure_map), encoding="utf-8")
    return json_path, report_path


def _candidate_failure_profile(candidate: dict[str, Any]) -> dict[str, Any]:
    metrics = candidate.get("metrics") if isinstance(candidate.get("metrics"), dict) else {}
    diagnostics = (
        candidate.get("signal_diagnostics")
        if isinstance(candidate.get("signal_diagnostics"), dict)
        else {}
    )
    freqai = (
        candidate.get("freqai_prediction_diagnostics")
        if isinstance(candidate.get("freqai_prediction_diagnostics"), dict)
        else {}
    )
    trade_count = _number(metrics.get("historical_trade_count")) or 0.0
    total_return = _number(metrics.get("historical_total_return_pct")) or 0.0
    wf_pass_rate = _number(metrics.get("walk_forward_pass_rate")) or 0.0
    profitable_windows_ratio = (
        _number(metrics.get("walk_forward_profitable_windows_ratio")) or 0.0
    )
    entry_count = _number(diagnostics.get("entry_count"))
    taxonomy = set(candidate.get("failure_taxonomy_codes") or [])
    failed_checks = set(candidate.get("failed_checks") or [])
    skipped_checks = set(candidate.get("skipped_checks") or [])
    diagnosis_codes = set(diagnostics.get("diagnosis_codes") or [])
    generated_entry_edge = (
        diagnostics.get("generated_entry_edge")
        if isinstance(diagnostics.get("generated_entry_edge"), dict)
        else {}
    )
    freqai_codes = set(freqai.get("diagnosis_codes") or [])
    family = str(candidate.get("hypothesis_family") or "").lower()
    training_relevant = "hybrid" in family or "freqai" in family or bool(
        _number(metrics.get("training_stage_count"))
    )
    tags: list[str] = []
    evidence: list[str] = []

    if trade_count == 0 or entry_count == 0 or "ZERO_ENTRY_SIGNALS" in diagnosis_codes:
        tags.append("zero_trade_or_signal_sparsity")
        evidence.append("No historical trades or zero generated entry signals.")
    if trade_count > 0 and total_return < 0:
        tags.append("entry_exists_negative_edge")
        evidence.append("Entries exist, but historical return is negative.")
    if generated_entry_edge.get("status") == "fail":
        tags.append("generated_entry_negative_edge")
        evidence.append("Generated entry diagnostics show non-positive edge after costs.")
    if wf_pass_rate < 1.0:
        tags.append("walk_forward_fragility")
        evidence.append("Walk-forward pass rate is below the pass threshold.")
    if profitable_windows_ratio == 0.0 and wf_pass_rate < 1.0:
        tags.append("no_profitable_walk_forward_windows")
        evidence.append("No profitable walk-forward windows were recorded.")
    if "FAIL_COST_SENSITIVE" in taxonomy or (trade_count >= 20 and total_return < 0):
        tags.append("cost_sensitive_mechanism")
        evidence.append("Failure taxonomy or trade frequency indicates cost sensitivity.")
    if "FAIL_REGIME_FRAGILE" in taxonomy or wf_pass_rate == 0.0:
        tags.append("regime_fragile_mechanism")
        evidence.append("Failure taxonomy or walk-forward result indicates regime fragility.")
    if "FAIL_OVERFIT_WF_GAP" in taxonomy:
        tags.append("overfit_or_window_dependency")
        evidence.append("Failure taxonomy indicates walk-forward/generalization gap.")
    if (
        diagnostics.get("first_zero_component") == "ml_filter"
        or "PREDICTION_TARGET_MISMATCH" in freqai_codes
        or "ML_FILTER_UNAVAILABLE" in diagnosis_codes
    ):
        tags.append("ml_rule_alignment_failure")
        evidence.append("ML prediction or ML filter does not align with surviving rule rows.")
    if any("training" in check for check in failed_checks) or (
        training_relevant and any("training" in check for check in skipped_checks)
    ):
        tags.append("training_or_artifact_gap")
        evidence.append("Training or training-artifact checks failed or were skipped.")
    if entry_count and entry_count > 0 and trade_count > 0 and total_return < 0 and wf_pass_rate == 0.0:
        tags.append("thesis_rejected_after_entries")
        evidence.append("The thesis produced trades but failed historical and all WF gates.")

    tags = list(dict.fromkeys(tags))
    return {
        "candidate_id": candidate.get("candidate_id"),
        "strategy_name": candidate.get("strategy_name"),
        "rank": candidate.get("rank"),
        "hypothesis_family": candidate.get("hypothesis_family"),
        "thesis_id": candidate.get("thesis_id"),
        "primary_failure_cause": _primary_cause(tags),
        "failure_cause_tags": tags,
        "failure_evidence": evidence,
        "metrics": {
            "historical_trade_count": trade_count,
            "historical_total_return_pct": total_return,
            "walk_forward_pass_rate": wf_pass_rate,
            "walk_forward_profitable_windows_ratio": profitable_windows_ratio,
            "signal_entry_count": entry_count,
            "generated_entry_net_edge_bps": _number(generated_entry_edge.get("net_edge_bps")),
            "generated_entry_profitable_windows_ratio": _number(
                generated_entry_edge.get("profitable_windows_ratio")
            ),
        },
        "signal_bottleneck": {
            "first_zero_component": diagnostics.get("first_zero_component"),
            "rarest_component": diagnostics.get("rarest_component"),
            "top_bottleneck": (diagnostics.get("bottleneck_components") or [{}])[0],
        },
        "freqai_prediction": {
            "available": bool(freqai.get("available")),
            "expected_target_column": freqai.get("expected_target_column"),
            "target_columns": list(freqai.get("target_columns") or []),
            "diagnosis_codes": list(freqai_codes),
        },
    }


def _category_summary(candidates: Sequence[dict[str, Any]]) -> dict[str, Any]:
    by_category: dict[str, list[str]] = defaultdict(list)
    by_primary = Counter()
    family_counter: dict[str, Counter[str]] = defaultdict(Counter)
    for candidate in candidates:
        candidate_id = str(candidate.get("candidate_id") or "")
        family = str(candidate.get("hypothesis_family") or "")
        primary = str(candidate.get("primary_failure_cause") or "unclassified")
        by_primary[primary] += 1
        for tag in candidate.get("failure_cause_tags", []):
            by_category[str(tag)].append(candidate_id)
            if family:
                family_counter[str(tag)][family] += 1
    categories = {
        name: {
            "candidate_count": len(ids),
            "candidate_ids": ids,
            "top_hypothesis_families": [
                {"hypothesis_family": family, "count": count}
                for family, count in family_counter[name].most_common(5)
            ],
        }
        for name, ids in sorted(by_category.items())
    }
    return {
        "category_count": len(categories),
        "categories": categories,
        "primary_failure_counts": dict(sorted(by_primary.items())),
    }


def _research_selection_guidance(
    synthesis: dict[str, Any], category_summary: dict[str, Any], *, candidate_count: int
) -> dict[str, Any]:
    brief = synthesis.get("next_research_brief", {})
    aggregate = synthesis.get("aggregate_failure_summary", {})
    categories = category_summary.get("categories", {})
    local_rejection_summaries = _local_falsification_rejection_summaries(aggregate)
    edge_rejection_summaries = _edge_discovery_rejection_summaries(aggregate)
    dominant = [
        {"category": name, "candidate_count": data.get("candidate_count")}
        for name, data in sorted(
            categories.items(),
            key=lambda item: (-int(item[1].get("candidate_count") or 0), item[0]),
        )[:5]
    ]
    required_categories = _required_research_categories(
        dominant,
        candidate_count=candidate_count,
    )
    causal_risk_weights = _causal_risk_weights(
        categories,
        candidate_count=candidate_count,
        required_categories=required_categories,
    )
    required_questions = [
        "What market mechanism survives after excluding the failed families and thesis IDs?",
        "What local historical data can falsify the mechanism before proposal generation?",
        "Why should expected edge exceed fee, slippage, and turnover costs?",
        "Which walk-forward regimes should be expected to pass or fail, and why?",
    ]
    if "zero_trade_or_signal_sparsity" in categories:
        required_questions.append(
            "How will the thesis avoid conjunction-heavy gates that eliminate entry rows?"
        )
    if "entry_exists_negative_edge" in categories:
        required_questions.append(
            "Why would the thesis have positive expectancy after costs when entries exist?"
        )
    for question in _generated_entry_edge_questions(brief):
        required_questions.append(question)
    for question in _local_falsification_rejection_questions(
        brief, local_rejection_summaries
    ):
        required_questions.append(question)
    for question in _edge_discovery_rejection_questions(
        brief, edge_rejection_summaries
    ):
        required_questions.append(question)
    if "ml_rule_alignment_failure" in categories:
        required_questions.append(
            "How will ML prediction rows overlap the rule-gate survivor rows before entry?"
        )
    blocked_next_actions = _blocked_next_actions(brief, aggregate)
    return {
        "paper_or_live_promotion_allowed": False,
        "parameter_only_retry_allowed": False,
        "requires_research_decision_before_proposal": True,
        "requires_research_question_responses": True,
        "requires_new_thesis_id": bool(brief.get("requires_new_thesis_id", True)),
        "requires_new_research_references": bool(
            brief.get("requires_new_research_references", True)
        ),
        "minimum_research_reference_count": int(
            brief.get("minimum_research_reference_count") or 2
        ),
        "minimum_research_selection_score": 80,
        "research_selection_rubric": [
            {
                "component": "novelty_against_failure_set",
                "max_points": 20,
                "requirement": "New thesis ID and family must not repeat failed evidence.",
            },
            {
                "component": "structured_research_references",
                "max_points": 15,
                "requirement": "References must be structured, dated, relevant, and mapped to the thesis.",
            },
            {
                "component": "local_historical_falsification",
                "max_points": 15,
                "requirement": "The thesis must name local closed-candle artifacts that can falsify it.",
            },
            {
                "component": "causal_failure_response_quality",
                "max_points": 30,
                "requirement": "Responses must cover required failure categories without parameter-only claims.",
            },
            {
                "component": "mechanism_and_falsification_substance",
                "max_points": 20,
                "requirement": "Core thesis fields must describe a market mechanism and falsification path.",
            },
        ],
        "failed_hypothesis_families_to_avoid": list(
            aggregate.get("hypothesis_families_tried")
            or brief.get("prior_hypothesis_families_to_avoid_as_default")
            or []
        ),
        "failed_thesis_ids_to_avoid": list(
            aggregate.get("thesis_ids_tried") or brief.get("failed_thesis_ids") or []
        ),
        "validated_local_falsification_rejections": local_rejection_summaries,
        "validated_edge_discovery_rejections": edge_rejection_summaries,
        "research_handoff_summaries": _research_handoff_summaries(brief, aggregate),
        "dominant_failure_categories": dominant,
        "causal_risk_weights": causal_risk_weights,
        "required_research_questions": list(dict.fromkeys(required_questions)),
        "blocked_next_actions": blocked_next_actions,
    }


def _generated_entry_edge_questions(brief: dict[str, Any]) -> list[str]:
    questions: list[str] = []
    for raw in brief.get("recommended_research_questions", []) or []:
        question = str(raw or "").strip()
        lower = question.lower()
        if "generated entry" in lower or "non-positive edge after costs" in lower:
            questions.append(question)
    return list(dict.fromkeys(questions))


def _local_falsification_rejection_summaries(
    aggregate: dict[str, Any]
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for raw in aggregate.get("local_falsification_rejections", []) or []:
        if not isinstance(raw, dict):
            continue
        summaries.append(
            {
                "path": raw.get("path"),
                "thesis_id": raw.get("thesis_id"),
                "mechanism_class": raw.get("mechanism_class"),
                "net_edge_bps": raw.get("net_edge_bps"),
                "profitable_windows_ratio": raw.get("profitable_windows_ratio"),
                "calendar_window_frequency": raw.get("calendar_window_frequency"),
                "calendar_window_count": raw.get("calendar_window_count"),
                "profitable_calendar_windows_ratio": raw.get(
                    "profitable_calendar_windows_ratio"
                ),
                "calendar_window_summaries": list(
                    raw.get("calendar_window_summaries") or []
                ),
            }
        )
    return summaries


def _edge_discovery_rejection_summaries(
    aggregate: dict[str, Any]
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for raw in aggregate.get("edge_discovery_rejections", []) or []:
        if not isinstance(raw, dict):
            continue
        summaries.append(
            {
                "path": raw.get("path"),
                "edge_discovery_id": raw.get("edge_discovery_id"),
                "thesis_id": raw.get("thesis_id"),
                "mechanism_class": raw.get("mechanism_class"),
                "best_hold_candles": raw.get("best_hold_candles"),
                "net_edge_bps": raw.get("net_edge_bps"),
                "sample_count": raw.get("sample_count"),
                "passing_horizon_count": raw.get("passing_horizon_count"),
                "profitable_windows_ratio": raw.get("profitable_windows_ratio"),
                "profitable_calendar_windows_ratio": raw.get(
                    "profitable_calendar_windows_ratio"
                ),
                "event_count": raw.get("event_count"),
                "data_span_days": raw.get("data_span_days"),
            }
        )
    return summaries


def _local_falsification_rejection_questions(
    brief: dict[str, Any],
    local_rejection_summaries: Sequence[dict[str, Any]],
) -> list[str]:
    questions: list[str] = []
    for raw in brief.get("recommended_research_questions", []) or []:
        question = str(raw or "").strip()
        if question and "local falsification" in question.lower():
            questions.append(question)
    for item in local_rejection_summaries:
        thesis_id = item.get("thesis_id") or "unknown thesis"
        mechanism = item.get("mechanism_class") or "unknown mechanism"
        calendar_ratio = item.get("profitable_calendar_windows_ratio")
        calendar_clause = (
            f" and quarterly calendar-window stability "
            f"(profitable_calendar_windows_ratio={calendar_ratio})"
            if calendar_ratio is not None
            else ""
        )
        questions.append(
            "What materially different market mechanism avoids the validated "
            f"local falsification rejection for {thesis_id} / {mechanism} "
            "while preserving closed-candle evidence, positive post-cost edge"
            f"{calendar_clause}?"
        )
    return list(dict.fromkeys(questions))


def _edge_discovery_rejection_questions(
    brief: dict[str, Any],
    edge_rejection_summaries: Sequence[dict[str, Any]],
) -> list[str]:
    questions: list[str] = []
    for raw in brief.get("recommended_research_questions", []) or []:
        question = str(raw or "").strip()
        if question and "edge discovery" in question.lower():
            questions.append(question)
    for item in edge_rejection_summaries:
        thesis_id = item.get("thesis_id") or "unknown thesis"
        mechanism = item.get("mechanism_class") or "unknown mechanism"
        questions.append(
            "What materially different market mechanism avoids the failed "
            f"edge discovery rejection for {thesis_id} / {mechanism} while "
            "predefining closed-candle evidence, positive post-cost net edge, "
            "and multi-horizon stability before proposal generation?"
        )
    return list(dict.fromkeys(questions))


def _blocked_next_actions(*sources: Any) -> list[str]:
    actions = [
        "parameter_only_threshold_loosen",
        "repeat_failed_hypothesis_family_without_new_evidence",
        "retry_validated_local_rejection_by_parameter_tuning",
        "proposal_generation_without_approved_research_decision",
        "code_generation_from_blocked_or_deferred_research_decision",
        "paper_or_dry_run_or_live_start",
        "exchange_order_endpoint_use",
    ]
    for source in sources:
        if isinstance(source, dict):
            _extend_unique_strings(actions, source.get("blocked_next_actions", []))
    return actions


def _research_handoff_summaries(*sources: Any) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    seen: set[str] = set()
    for source in sources:
        if not isinstance(source, dict):
            continue
        for raw in source.get("research_handoff_summaries", []) or []:
            if not isinstance(raw, dict):
                continue
            copied = _copy_jsonish(raw)
            key = json.dumps(copied, sort_keys=True, ensure_ascii=False)
            if key not in seen:
                seen.add(key)
                summaries.append(copied)
    return summaries


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


def _required_research_categories(
    dominant: Sequence[dict[str, Any]], *, candidate_count: int
) -> list[str]:
    required = [str(item.get("category") or "") for item in dominant[:3]]
    material_threshold = candidate_count * _MATERIAL_CAUSAL_CATEGORY_MIN_SHARE
    for item in dominant[3:]:
        category = str(item.get("category") or "")
        count = _number(item.get("candidate_count")) or 0.0
        if category and candidate_count > 0 and count >= material_threshold:
            required.append(category)
    return list(dict.fromkeys(category for category in required if category))


def _causal_risk_weights(
    categories: dict[str, Any],
    *,
    candidate_count: int,
    required_categories: Sequence[str],
) -> list[dict[str, Any]]:
    required_set = set(required_categories)
    weights: list[dict[str, Any]] = []
    for category, data in categories.items():
        count = _number(data.get("candidate_count")) or 0.0
        share = count / candidate_count if candidate_count > 0 else 0.0
        severity = _CAUSAL_CATEGORY_SEVERITY_MULTIPLIERS.get(
            category,
            _DEFAULT_CAUSAL_CATEGORY_SEVERITY,
        )
        risk_score = min(100.0, share * severity * 100.0)
        weights.append(
            {
                "category": category,
                "candidate_count": int(count),
                "candidate_share": round(share, 4),
                "severity_multiplier": severity,
                "risk_score": round(risk_score, 2),
                "required_for_next_research": category in required_set,
                "response_focus": _CAUSAL_CATEGORY_RESPONSE_FOCUS.get(
                    category,
                    [
                        "category-specific mechanism evidence",
                        "local falsification before proposal generation",
                    ],
                ),
            }
        )
    return sorted(
        weights,
        key=lambda item: (
            -float(item.get("risk_score") or 0.0),
            str(item.get("category") or ""),
        ),
    )


def _primary_cause(tags: Sequence[str]) -> str:
    priority = [
        "ml_rule_alignment_failure",
        "zero_trade_or_signal_sparsity",
        "generated_entry_negative_edge",
        "entry_exists_negative_edge",
        "no_profitable_walk_forward_windows",
        "walk_forward_fragility",
        "training_or_artifact_gap",
        "cost_sensitive_mechanism",
        "regime_fragile_mechanism",
        "overfit_or_window_dependency",
    ]
    for tag in priority:
        if tag in tags:
            return tag
    return "unclassified"


def _render_report(failure_map: dict[str, Any]) -> str:
    guidance = failure_map.get("research_selection_guidance", {})
    categories = failure_map.get("causal_failure_categories", {}).get("categories", {})
    lines = [
        "# Causal Failure Map",
        "",
        f"- map_id: {failure_map.get('map_id')}",
        f"- source_synthesis_id: {failure_map.get('source_synthesis_id')}",
        f"- candidate_count: {failure_map.get('candidate_count')}",
        "- requires_research_decision_before_proposal: "
        f"{guidance.get('requires_research_decision_before_proposal')}",
        "",
        "## Dominant Failure Categories",
        "",
    ]
    for item in guidance.get("dominant_failure_categories", []):
        lines.append(f"- {item.get('category')}: {item.get('candidate_count')}")
    if not guidance.get("dominant_failure_categories"):
        lines.append("- None.")
    lines.extend(["", "## Causal Risk Weights", ""])
    for item in guidance.get("causal_risk_weights", []) or []:
        focus = "; ".join(item.get("response_focus", []) or [])
        lines.append(
            f"- {item.get('category')}: risk_score={item.get('risk_score')}, "
            f"candidate_share={item.get('candidate_share')}, "
            f"severity_multiplier={item.get('severity_multiplier')}, "
            f"required_for_next_research={item.get('required_for_next_research')}, "
            f"response_focus={focus}"
        )
    if not guidance.get("causal_risk_weights"):
        lines.append("- None.")
    lines.extend(["", "## Category Details", ""])
    for name, data in categories.items():
        lines.append(f"- {name}: {data.get('candidate_count')} candidates")
    if not categories:
        lines.append("- None.")
    lines.extend(["", "## Required Research Questions", ""])
    lines.extend(
        [f"- {question}" for question in guidance.get("required_research_questions", [])]
        or ["- None."]
    )
    lines.extend(["", "## Validated Local Falsification Rejections", ""])
    for item in guidance.get("validated_local_falsification_rejections", []) or []:
        lines.append(
            "- "
            f"{item.get('thesis_id')} / {item.get('mechanism_class')}: "
            f"net_edge_bps={item.get('net_edge_bps')}, "
            f"profitable_windows_ratio={item.get('profitable_windows_ratio')}, "
            "profitable_calendar_windows_ratio="
            f"{item.get('profitable_calendar_windows_ratio')}"
        )
    if not guidance.get("validated_local_falsification_rejections"):
        lines.append("- None.")
    lines.extend(["", "## Validated Edge Discovery Rejections", ""])
    for item in guidance.get("validated_edge_discovery_rejections", []) or []:
        lines.append(
            "- "
            f"{item.get('thesis_id')} / {item.get('mechanism_class')}: "
            f"best_hold_candles={item.get('best_hold_candles')}, "
            f"net_edge_bps={item.get('net_edge_bps')}, "
            f"passing_horizon_count={item.get('passing_horizon_count')}"
        )
    if not guidance.get("validated_edge_discovery_rejections"):
        lines.append("- None.")
    lines.extend(
        [
            "",
            "## Research Selection Rubric",
            "",
            "- minimum_research_selection_score: "
            f"{guidance.get('minimum_research_selection_score')}",
        ]
    )
    for item in guidance.get("research_selection_rubric", []) or []:
        lines.append(
            f"- {item.get('component')}: {item.get('max_points')} points - "
            f"{item.get('requirement')}"
        )
    if not guidance.get("research_selection_rubric"):
        lines.append("- None.")
    lines.extend(["", "## Blocked Next Actions", ""])
    lines.extend([f"- {action}" for action in guidance.get("blocked_next_actions", [])])
    lines.append("")
    return "\n".join(lines)


def _check(name: str, passed: bool, *, status_if_false: str = "fail") -> dict[str, str]:
    return {"name": name, "status": "pass" if passed else status_if_false}


def _number(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _map_id(generated_at: str) -> str:
    parsed = datetime.fromisoformat(generated_at.replace("Z", "+00:00"))
    return parsed.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")


def _safe_path_component(value: str) -> str:
    token = "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in value)
    return token.strip("_") or "failure_map"


def _resolve_inside(path: Path, root: Path) -> Path:
    resolved = (path if path.is_absolute() else root / path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Path must resolve inside the workspace: {path}") from exc
    return resolved


def _rel(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root))
    except ValueError:
        return str(path)
