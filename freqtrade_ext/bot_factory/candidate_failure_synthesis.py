from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence


_CONTEXT_MERGE_SEMANTICS = "closed_context_candle_availability_v1"
_LOCAL_FALSIFICATION_REJECTION_STATUSES = {"failed", "rejected", "blocked"}
_EDGE_DISCOVERY_REJECTION_STATUSES = {"failed", "blocked"}
_EDGE_DISCOVERY_METHOD_BLOCKERS = {
    "ohlcv_file_present",
    "ohlcv_parseable",
    "edge_spec_file_present",
    "edge_spec_parseable",
    "edge_spec_factory_valid",
    "edge_spec_thesis_id_present",
    "edge_spec_mechanism_class_present",
    "edge_spec_all_in_cost_bps_non_negative",
    "edge_spec_horizons_present",
    "edge_spec_horizon_count_bounded",
    "edge_spec_no_parameter_search_grid",
    "ohlcv_data_span_sufficient",
    "min_passing_horizon_count_positive",
    "order_book_quality_reports_parseable_when_supplied",
}
RESEARCH_HANDOFF_KEYS = (
    "research_decision_question_handoff",
    "research_decision_novelty_handoff",
    "local_falsification_handoff",
    "structural_data_quality_handoff",
    "structural_data_capability_handoff",
)


@dataclass(frozen=True)
class CandidateFailureSynthesisInputs:
    root_dir: Path
    ranking_path: Path
    signal_diagnostics_paths: Sequence[Path] = field(default_factory=list)
    freqai_prediction_diagnostics_paths: Sequence[Path] = field(default_factory=list)
    local_falsification_paths: Sequence[Path] = field(default_factory=list)
    edge_discovery_paths: Sequence[Path] = field(default_factory=list)
    output_root: Path = Path("registry/strategies/synthesis")
    synthesis_id: str | None = None
    reviewer_notes: Sequence[str] = field(default_factory=list)


def synthesize_candidate_failures(
    inputs: CandidateFailureSynthesisInputs,
) -> dict[str, Any]:
    root = inputs.root_dir.resolve()
    ranking_path = _resolve_inside(inputs.ranking_path, root)
    ranking = _load_json(ranking_path)
    generated_at = datetime.now(UTC).replace(microsecond=0).isoformat()
    diagnostics_by_candidate = _load_signal_diagnostics(inputs.signal_diagnostics_paths, root)
    freqai_prediction_by_candidate = _load_freqai_prediction_diagnostics(
        inputs.freqai_prediction_diagnostics_paths, root
    )
    local_falsification_rejections = _load_local_falsification_rejections(
        inputs.local_falsification_paths, root
    )
    edge_discovery_rejections = _load_edge_discovery_rejections(
        inputs.edge_discovery_paths, root
    )
    candidates = [
        _candidate_summary(item, root, diagnostics_by_candidate, freqai_prediction_by_candidate)
        for item in ranking.get("ranked_candidates", [])
        if isinstance(item, dict)
    ]
    aggregate = _aggregate_failures(
        ranking,
        candidates,
        diagnostics_by_candidate,
        freqai_prediction_by_candidate,
        local_falsification_rejections,
        edge_discovery_rejections,
    )
    next_brief = _next_research_brief(aggregate, candidates, ranking_path, inputs)
    checks = [
        _check("ranking_file_present", ranking_path.is_file()),
        _check("ranking_has_candidates", bool(candidates)),
        _check("paper_ready_candidates_absent", not aggregate["paper_ready_candidate_ids"]),
        _check(
            "signal_diagnostics_linked",
            bool(diagnostics_by_candidate),
            status_if_false="warn",
        ),
        _check(
            "freqai_prediction_diagnostics_linked",
            bool(freqai_prediction_by_candidate),
            status_if_false="warn",
        ),
    ]
    if inputs.local_falsification_paths:
        checks.append(
            _check(
                "local_falsification_rejections_linked",
                any(item.get("rejection_valid") for item in local_falsification_rejections),
                status_if_false="warn",
            )
        )
    if inputs.edge_discovery_paths:
        checks.append(
            _check(
                "edge_discovery_rejections_linked",
                any(item.get("rejection_valid") for item in edge_discovery_rejections),
                status_if_false="warn",
            )
        )
    return {
        "generated_at": generated_at,
        "factory": "candidate_failure_synthesis",
        "synthesis_id": inputs.synthesis_id or _synthesis_id(generated_at),
        "status": "completed" if candidates else "blocked",
        "ranking_path": _rel(ranking_path, root),
        "ranking_id": ranking.get("ranking_id"),
        "candidate_count": len(candidates),
        "checks": checks,
        "candidates": candidates,
        "aggregate_failure_summary": aggregate,
        "next_research_brief": next_brief,
        "reviewer_notes": list(inputs.reviewer_notes),
        "safety_scope": {
            "historical_only": True,
            "backtest_started": False,
            "paper_trading_started": False,
            "dry_run_trading_started": False,
            "live_trading": False,
            "exchange_order_placement": False,
            "process_control": False,
            "promotion_authorized_by_this_command": False,
            "local_artifacts_source_of_truth": True,
        },
    }


def write_candidate_failure_synthesis_artifacts(
    synthesis: dict[str, Any], *, root_dir: Path, output_root: Path
) -> tuple[Path, Path]:
    root = root_dir.resolve()
    synthesis_id = _safe_path_component(str(synthesis.get("synthesis_id") or "synthesis"))
    out_dir = _resolve_inside(output_root, root) / synthesis_id
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "candidate_failure_synthesis.json"
    report_path = out_dir / "candidate_failure_synthesis_report.md"
    json_path.write_text(json.dumps(synthesis, indent=2, ensure_ascii=False), encoding="utf-8")
    report_path.write_text(_render_report(synthesis), encoding="utf-8")
    return json_path, report_path


def _candidate_summary(
    ranking_item: dict[str, Any],
    root: Path,
    diagnostics_by_candidate: dict[str, dict[str, Any]],
    freqai_prediction_by_candidate: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    candidate_id = str(ranking_item.get("candidate_id") or "")
    manifest = _load_manifest_if_available(ranking_item.get("manifest_path"), root)
    checks = {
        str(check.get("name")): check
        for check in manifest.get("checks", [])
        if isinstance(check, dict) and check.get("name")
    }
    metrics = ranking_item.get("metrics") if isinstance(ranking_item.get("metrics"), dict) else {}
    thesis = ranking_item.get("thesis") if isinstance(ranking_item.get("thesis"), dict) else {}
    research_brief = _candidate_research_brief(ranking_item, manifest)
    blocked_next_actions = _blocked_next_actions(ranking_item, research_brief, manifest)
    diagnostics = diagnostics_by_candidate.get(candidate_id, {})
    freqai_prediction = freqai_prediction_by_candidate.get(candidate_id, {})
    return {
        "candidate_id": candidate_id,
        "strategy_name": ranking_item.get("strategy_name"),
        "rank": ranking_item.get("rank"),
        "recommendation": ranking_item.get("recommendation"),
        "paper_ready_eligible": bool(ranking_item.get("paper_ready_eligible")),
        "paper_ready_blockers": list(ranking_item.get("paper_ready_blockers") or []),
        "hypothesis_family": ranking_item.get("hypothesis_family"),
        "thesis_id": thesis.get("thesis_id"),
        "failure_taxonomy_codes": list(ranking_item.get("failure_taxonomy_codes") or []),
        "blocked_next_actions": blocked_next_actions,
        "research_brief": research_brief,
        "research_handoff_summary": _research_handoff_summary(ranking_item, research_brief),
        "failed_checks": [
            name for name, check in checks.items() if check.get("status") in {"fail", "missing"}
        ],
        "skipped_checks": [
            name for name, check in checks.items() if check.get("status") == "skipped"
        ],
        "metrics": {
            "historical_trade_count": _number(metrics.get("historical_trade_count")),
            "historical_total_return_pct": _number(metrics.get("historical_total_return_pct")),
            "historical_profit_factor": _number(metrics.get("historical_profit_factor")),
            "walk_forward_pass_rate": _number(metrics.get("walk_forward_pass_rate")),
            "walk_forward_profitable_windows_ratio": _number(
                metrics.get("walk_forward_profitable_windows_ratio")
            ),
            "walk_forward_max_single_window_profit_dependency": _number(
                metrics.get("walk_forward_max_single_window_profit_dependency")
            ),
            "training_stage_count": _number(metrics.get("training_stage_count")),
        },
        "signal_diagnostics": _diagnostics_summary(diagnostics),
        "freqai_prediction_diagnostics": _freqai_prediction_summary(freqai_prediction),
    }


def _aggregate_failures(
    ranking: dict[str, Any],
    candidates: Sequence[dict[str, Any]],
    diagnostics_by_candidate: dict[str, dict[str, Any]],
    freqai_prediction_by_candidate: dict[str, dict[str, Any]],
    local_falsification_rejections: Sequence[dict[str, Any]],
    edge_discovery_rejections: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    paper_ready_ids = list(ranking.get("paper_ready_candidate_ids") or [])
    taxonomy = Counter(
        code
        for candidate in candidates
        for code in candidate.get("failure_taxonomy_codes", [])
    )
    blockers = Counter(
        blocker
        for candidate in candidates
        for blocker in candidate.get("paper_ready_blockers", [])
    )
    blocked_next_actions: list[str] = []
    for candidate in candidates:
        _extend_unique_strings(blocked_next_actions, candidate.get("blocked_next_actions", []))
    research_handoff_summaries = [
        {
            "candidate_id": candidate.get("candidate_id"),
            "research_handoff_summary": candidate.get("research_handoff_summary"),
        }
        for candidate in candidates
        if candidate.get("research_handoff_summary")
    ]
    local_rejections = [
        item for item in local_falsification_rejections if item.get("rejection_valid")
    ]
    invalid_local_rejections = [
        item for item in local_falsification_rejections if not item.get("rejection_valid")
    ]
    edge_rejections = [
        item for item in edge_discovery_rejections if item.get("rejection_valid")
    ]
    invalid_edge_rejections = [
        item for item in edge_discovery_rejections if not item.get("rejection_valid")
    ]
    local_mechanisms = sorted(
        {
            str(item.get("mechanism_class"))
            for item in local_rejections
            if item.get("mechanism_class")
        }
    )
    local_thesis_ids = sorted(
        {
            str(item.get("thesis_id"))
            for item in local_rejections
            if item.get("thesis_id")
        }
    )
    edge_mechanisms = sorted(
        {
            str(item.get("mechanism_class"))
            for item in edge_rejections
            if item.get("mechanism_class")
        }
    )
    edge_thesis_ids = sorted(
        {
            str(item.get("thesis_id"))
            for item in edge_rejections
            if item.get("thesis_id")
        }
    )
    families = sorted(
        {
            str(candidate.get("hypothesis_family"))
            for candidate in candidates
            if candidate.get("hypothesis_family")
        }
        | set(local_mechanisms)
        | set(edge_mechanisms)
    )
    thesis_ids = sorted(
        {
            str(candidate.get("thesis_id"))
            for candidate in candidates
            if candidate.get("thesis_id")
        }
        | set(local_thesis_ids)
        | set(edge_thesis_ids)
    )
    signal_bottlenecks = [
        {
            "candidate_id": candidate.get("candidate_id"),
            "first_zero_component": candidate.get("signal_diagnostics", {}).get(
                "first_zero_component"
            ),
            "diagnosis_codes": candidate.get("signal_diagnostics", {}).get(
                "diagnosis_codes", []
            ),
            "top_bottleneck": (
                candidate.get("signal_diagnostics", {}).get("bottleneck_components") or [{}]
            )[0],
        }
        for candidate in candidates
        if candidate.get("signal_diagnostics", {}).get("available")
    ]
    generated_entry_edge_failures = [
        {
            "candidate_id": candidate.get("candidate_id"),
            "status": candidate.get("signal_diagnostics", {})
            .get("generated_entry_edge", {})
            .get("status"),
            "sample_count": candidate.get("signal_diagnostics", {})
            .get("generated_entry_edge", {})
            .get("sample_count"),
            "net_edge_bps": candidate.get("signal_diagnostics", {})
            .get("generated_entry_edge", {})
            .get("net_edge_bps"),
            "profitable_windows_ratio": candidate.get("signal_diagnostics", {})
            .get("generated_entry_edge", {})
            .get("profitable_windows_ratio"),
        }
        for candidate in candidates
        if candidate.get("signal_diagnostics", {})
        .get("generated_entry_edge", {})
        .get("status")
        == "fail"
    ]
    freqai_target_mismatches = [
        {
            "candidate_id": candidate.get("candidate_id"),
            "expected_target_column": candidate.get("freqai_prediction_diagnostics", {}).get(
                "expected_target_column"
            ),
            "target_columns": candidate.get("freqai_prediction_diagnostics", {}).get(
                "target_columns", []
            ),
            "model_label_columns": candidate.get("freqai_prediction_diagnostics", {}).get(
                "model_label_columns", []
            ),
            "diagnosis_codes": candidate.get("freqai_prediction_diagnostics", {}).get(
                "diagnosis_codes", []
            ),
        }
        for candidate in candidates
        if "PREDICTION_TARGET_MISMATCH"
        in set(candidate.get("freqai_prediction_diagnostics", {}).get("diagnosis_codes", []))
    ]
    return {
        "paper_ready_candidate_ids": paper_ready_ids,
        "paper_ready_count": len(paper_ready_ids),
        "all_candidates_failed_gates": not paper_ready_ids,
        "hypothesis_families_tried": families,
        "thesis_ids_tried": thesis_ids,
        "zero_trade_candidate_ids": [
            str(candidate["candidate_id"])
            for candidate in candidates
            if (candidate.get("metrics", {}).get("historical_trade_count") or 0) == 0
        ],
        "negative_return_candidate_ids": [
            str(candidate["candidate_id"])
            for candidate in candidates
            if (candidate.get("metrics", {}).get("historical_total_return_pct") or 0) < 0
        ],
        "walk_forward_failed_candidate_ids": [
            str(candidate["candidate_id"])
            for candidate in candidates
            if (candidate.get("metrics", {}).get("walk_forward_pass_rate") or 0) < 1.0
        ],
        "failure_taxonomy_counts": dict(sorted(taxonomy.items())),
        "paper_ready_blocker_counts": dict(sorted(blockers.items())),
        "blocked_next_actions": blocked_next_actions,
        "research_handoff_summaries": research_handoff_summaries,
        "signal_diagnostics_candidate_ids": sorted(diagnostics_by_candidate),
        "signal_bottlenecks": signal_bottlenecks,
        "generated_entry_edge_failures": generated_entry_edge_failures,
        "freqai_prediction_diagnostics_candidate_ids": sorted(freqai_prediction_by_candidate),
        "local_falsification_rejection_artifact_count": len(local_falsification_rejections),
        "local_falsification_rejection_count": len(local_rejections),
        "local_falsification_invalid_rejection_count": len(invalid_local_rejections),
        "local_falsification_rejection_artifacts": list(local_falsification_rejections),
        "local_falsification_rejections": local_rejections,
        "local_falsification_failed_thesis_ids": local_thesis_ids,
        "local_falsification_failed_mechanism_classes": local_mechanisms,
        "edge_discovery_rejection_artifact_count": len(edge_discovery_rejections),
        "edge_discovery_rejection_count": len(edge_rejections),
        "edge_discovery_invalid_rejection_count": len(invalid_edge_rejections),
        "edge_discovery_rejection_artifacts": list(edge_discovery_rejections),
        "edge_discovery_rejections": edge_rejections,
        "edge_discovery_failed_thesis_ids": edge_thesis_ids,
        "edge_discovery_failed_mechanism_classes": edge_mechanisms,
        "freqai_target_mismatch_candidate_ids": [
            str(item.get("candidate_id")) for item in freqai_target_mismatches
        ],
        "freqai_target_mismatches": freqai_target_mismatches,
    }


def _next_research_brief(
    aggregate: dict[str, Any],
    candidates: Sequence[dict[str, Any]],
    ranking_path: Path,
    inputs: CandidateFailureSynthesisInputs,
) -> dict[str, Any]:
    prior_families = aggregate["hypothesis_families_tried"]
    zero_trade_count = len(aggregate["zero_trade_candidate_ids"])
    negative_count = len(aggregate["negative_return_candidate_ids"])
    bottleneck_lines = _bottleneck_questions(aggregate["signal_bottlenecks"])
    generated_entry_edge_lines = _generated_entry_edge_questions(
        aggregate.get("generated_entry_edge_failures") or []
    )
    freqai_lines = _freqai_prediction_questions(aggregate["freqai_target_mismatches"])
    local_rejection_lines = _local_falsification_rejection_questions(
        aggregate.get("local_falsification_rejections") or []
    )
    edge_discovery_rejection_lines = _edge_discovery_rejection_questions(
        aggregate.get("edge_discovery_rejections") or []
    )
    blocked_next_actions = [
        "parameter_only_threshold_loosen",
        "repeat_failed_hypothesis_family_without_new_evidence",
        "paper_or_dry_run_or_live_start",
        "exchange_order_endpoint_use",
        "promotion_from_failed_smoke",
    ]
    _extend_unique_strings(blocked_next_actions, aggregate.get("blocked_next_actions", []))
    return {
        "brief_type": "next_theory_and_code_generation_input",
        "source_ranking_path": _rel(ranking_path, inputs.root_dir.resolve()),
        "requires_new_thesis_id": aggregate["all_candidates_failed_gates"],
        "requires_new_research_references": aggregate["all_candidates_failed_gates"],
        "minimum_research_reference_count": 2,
        "parameter_only_retry_allowed": False,
        "paper_or_live_promotion_allowed": False,
        "prior_hypothesis_families_to_avoid_as_default": prior_families,
        "failed_thesis_ids": aggregate["thesis_ids_tried"],
        "evidence_summary": {
            "candidate_count": len(candidates),
            "paper_ready_count": aggregate["paper_ready_count"],
            "zero_trade_count": zero_trade_count,
            "negative_return_count": negative_count,
            "walk_forward_failed_count": len(aggregate["walk_forward_failed_candidate_ids"]),
            "local_falsification_rejection_count": aggregate[
                "local_falsification_rejection_count"
            ],
            "edge_discovery_rejection_count": aggregate[
                "edge_discovery_rejection_count"
            ],
        },
        "research_handoff_summaries": aggregate.get("research_handoff_summaries", []),
        "recommended_research_questions": [
            "What distinct market mechanism can be falsified with local closed-candle historical artifacts rather than by tuning thresholds inside the failed families?",
            "Can the next thesis explain why pullback recovery and EMA trend alignment conflict on the local BTC 5m sample?",
            "Which safe historical features can separate reversal, breakout, and trend regimes before entry without using future data or order endpoints?",
            *bottleneck_lines,
            *generated_entry_edge_lines,
            *freqai_lines,
            *local_rejection_lines,
            *edge_discovery_rejection_lines,
        ],
        "candidate_generation_constraints": [
            "Use a new thesis_id and structured research_references before generating code.",
            "Do not treat zero drawdown from zero trades as risk evidence.",
            "Keep generated strategies long-only with leverage fixed at 1.0 and no short signals.",
            "Run static checks and OHLCV validation before any historical wrapper.",
            "Use local JSON, CSV, Markdown, and logs as the source of truth.",
        ],
        "blocked_next_actions": blocked_next_actions,
    }


def _load_signal_diagnostics(paths: Sequence[Path], root: Path) -> dict[str, dict[str, Any]]:
    diagnostics: dict[str, dict[str, Any]] = {}
    for path in paths:
        resolved = _resolve_inside(path, root)
        payload = _load_json(resolved)
        candidate_id = str(payload.get("candidate_id") or "").strip()
        if candidate_id:
            payload["diagnostics_path"] = _rel(resolved, root)
            diagnostics[candidate_id] = payload
    return diagnostics


def _load_freqai_prediction_diagnostics(
    paths: Sequence[Path], root: Path
) -> dict[str, dict[str, Any]]:
    diagnostics: dict[str, dict[str, Any]] = {}
    for path in paths:
        resolved = _resolve_inside(path, root)
        payload = _load_json(resolved)
        candidate_id = str(payload.get("candidate_id") or "").strip()
        if candidate_id:
            payload["diagnostics_path"] = _rel(resolved, root)
            diagnostics[candidate_id] = payload
    return diagnostics


def _load_local_falsification_rejections(
    paths: Sequence[Path], root: Path
) -> list[dict[str, Any]]:
    rejections: list[dict[str, Any]] = []
    for path in paths:
        resolved = _resolve_inside(path, root)
        payload = _load_json(resolved)
        status = str(payload.get("status") or "").strip().lower()
        factory = str(payload.get("factory") or "").strip()
        evidence = _local_falsification_cost_payload(payload)
        safety_scope_valid = _local_falsification_safety_scope_valid(
            payload.get("safety_scope")
        )
        event_source = evidence.get("event_source") or payload.get("event_source") or {}
        event_source_valid = _local_falsification_event_source_valid(event_source)
        event_source_context_alignment_valid = (
            _local_falsification_event_source_context_alignment_valid(event_source)
        )
        event_source_failure_synthesis_guard_valid = (
            _local_falsification_event_source_failure_synthesis_guard_valid(event_source)
        )
        blockers = [
            str(item.get("name"))
            for item in payload.get("blockers", []) or []
            if isinstance(item, dict) and item.get("name")
        ]
        factory_valid = factory == "research_local_falsification"
        status_rejected = status in _LOCAL_FALSIFICATION_REJECTION_STATUSES
        rejection_valid = (
            factory_valid
            and status_rejected
            and safety_scope_valid
            and event_source_valid
            and event_source_failure_synthesis_guard_valid
        )
        failure_reasons = _local_falsification_rejection_failure_reasons(
            factory_valid=factory_valid,
            status_rejected=status_rejected,
            safety_scope_valid=safety_scope_valid,
            event_source_valid=event_source_valid,
            event_source_context_alignment_valid=event_source_context_alignment_valid,
            event_source_failure_synthesis_guard_valid=(
                event_source_failure_synthesis_guard_valid
            ),
        )
        rejections.append(
            {
                "path": _rel(resolved, root),
                "factory": factory or None,
                "factory_valid": factory_valid,
                "status": status or None,
                "status_rejected": status_rejected,
                "safety_scope_valid": safety_scope_valid,
                "event_source_valid": event_source_valid,
                "event_source_context_alignment_valid": (
                    event_source_context_alignment_valid
                ),
                "event_source_failure_synthesis_guard_valid": (
                    event_source_failure_synthesis_guard_valid
                ),
                "event_source": _local_falsification_event_source_summary(event_source),
                "rejection_valid": rejection_valid,
                "failure_reasons": failure_reasons,
                "thesis_id": str(evidence.get("thesis_id") or "").strip(),
                "mechanism_class": str(evidence.get("mechanism_class") or "").strip(),
                "expected_edge_bps": _number(evidence.get("expected_edge_bps")),
                "all_in_cost_bps": _number(evidence.get("all_in_cost_bps")),
                "net_edge_bps": _number(evidence.get("net_edge_bps")),
                "sample_count": _number(evidence.get("sample_count")),
                "data_span_days": _number(evidence.get("data_span_days")),
                "profitable_windows_ratio": _number(
                    evidence.get("profitable_windows_ratio")
                ),
                "calendar_window_frequency": (
                    str(evidence.get("calendar_window_frequency") or "").strip()
                    or None
                ),
                "calendar_window_count": _number(
                    evidence.get("calendar_window_count")
                ),
                "profitable_calendar_windows_ratio": _number(
                    evidence.get("profitable_calendar_windows_ratio")
                ),
                "calendar_window_summaries": _local_falsification_calendar_windows(
                    evidence.get("calendar_window_summaries")
                ),
                "blockers": blockers,
            }
        )
    return rejections


def _load_edge_discovery_rejections(
    paths: Sequence[Path], root: Path
) -> list[dict[str, Any]]:
    rejections: list[dict[str, Any]] = []
    for path in paths:
        resolved = _resolve_inside(path, root)
        payload = _load_json(resolved)
        status = str(payload.get("status") or "").strip().lower()
        factory = str(payload.get("factory") or "").strip()
        blocker_names = _check_names(payload.get("blockers"))
        method_blockers = _edge_discovery_method_blocker_names(blocker_names)
        anti_search = payload.get("anti_parameter_search")
        anti_parameter_search_valid = (
            isinstance(anti_search, dict) and anti_search.get("valid") is True
        )
        safety_scope_valid = _edge_discovery_safety_scope_valid(
            payload.get("safety_scope")
        )
        best_horizon = _edge_discovery_best_horizon(payload)
        factory_valid = factory == "research_edge_discovery"
        status_rejected = status in _EDGE_DISCOVERY_REJECTION_STATUSES
        thesis_id = str(payload.get("thesis_id") or "").strip()
        mechanism_class = str(payload.get("mechanism_class") or "").strip()
        rejection_valid = (
            factory_valid
            and status_rejected
            and safety_scope_valid
            and anti_parameter_search_valid
            and bool(thesis_id)
            and bool(mechanism_class)
            and not method_blockers
        )
        rejections.append(
            {
                "path": _rel(resolved, root),
                "factory": factory or None,
                "factory_valid": factory_valid,
                "status": status or None,
                "status_rejected": status_rejected,
                "safety_scope_valid": safety_scope_valid,
                "anti_parameter_search_valid": anti_parameter_search_valid,
                "method_blockers": method_blockers,
                "rejection_valid": rejection_valid,
                "failure_reasons": _edge_discovery_rejection_failure_reasons(
                    factory_valid=factory_valid,
                    status_rejected=status_rejected,
                    safety_scope_valid=safety_scope_valid,
                    anti_parameter_search_valid=anti_parameter_search_valid,
                    thesis_id_present=bool(thesis_id),
                    mechanism_class_present=bool(mechanism_class),
                    method_blockers=method_blockers,
                ),
                "edge_discovery_id": payload.get("edge_discovery_id"),
                "thesis_id": thesis_id,
                "mechanism_class": mechanism_class,
                "source_ohlcv_path": payload.get("source_ohlcv_path"),
                "edge_spec_path": payload.get("edge_spec_path"),
                "event_count": _number(payload.get("event_count")),
                "data_span_days": _number(payload.get("data_span_days")),
                "all_in_cost_bps": _number(payload.get("all_in_cost_bps")),
                "passing_horizon_count": _number(
                    payload.get("passing_horizon_count")
                ),
                "best_horizon": best_horizon,
                "best_hold_candles": best_horizon.get("hold_candles"),
                "expected_edge_bps": _number(best_horizon.get("expected_edge_bps")),
                "net_edge_bps": _number(best_horizon.get("net_edge_bps")),
                "sample_count": _number(best_horizon.get("sample_count")),
                "profitable_windows_ratio": _number(
                    best_horizon.get("profitable_windows_ratio")
                ),
                "calendar_window_frequency": (
                    str(best_horizon.get("calendar_window_frequency") or "").strip()
                    or None
                ),
                "calendar_window_count": _number(
                    best_horizon.get("calendar_window_count")
                ),
                "profitable_calendar_windows_ratio": _number(
                    best_horizon.get("profitable_calendar_windows_ratio")
                ),
                "blockers": blocker_names,
            }
        )
    return rejections


def _check_names(raw: Any) -> list[str]:
    if not isinstance(raw, list):
        return []
    names: list[str] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        if name:
            names.append(name)
    return names


def _edge_discovery_method_blocker_names(blocker_names: Sequence[str]) -> list[str]:
    method_blockers: list[str] = []
    for name in blocker_names:
        if (
            name == "edge_events_generated"
            or name == "passing_horizon_count_sufficient"
            or name.startswith("horizon_")
        ):
            continue
        if name in _EDGE_DISCOVERY_METHOD_BLOCKERS or name:
            method_blockers.append(name)
    return method_blockers


def _edge_discovery_safety_scope_valid(safety_scope: Any) -> bool:
    if not isinstance(safety_scope, dict):
        return False
    unsafe_flags = (
        "backtest_started",
        "strategy_code_generated",
        "paper_trading_started",
        "dry_run_trading_started",
        "live_trading",
        "exchange_order_placement",
        "shorting",
        "process_control",
    )
    leverage = _number(safety_scope.get("leverage"))
    return (
        safety_scope.get("historical_only") is True
        and all(not bool(safety_scope.get(flag)) for flag in unsafe_flags)
        and (leverage is None or leverage <= 1.0)
    )


def _edge_discovery_best_horizon(payload: dict[str, Any]) -> dict[str, Any]:
    best = payload.get("best_horizon_by_net_edge")
    if isinstance(best, dict):
        return _copy_jsonish(best)
    horizons = payload.get("horizon_results")
    if not isinstance(horizons, list):
        return {}
    ranked = [
        item
        for item in horizons
        if isinstance(item, dict) and _number(item.get("net_edge_bps")) is not None
    ]
    if not ranked:
        return {}
    return _copy_jsonish(
        max(ranked, key=lambda item: float(_number(item.get("net_edge_bps")) or 0.0))
    )


def _edge_discovery_rejection_failure_reasons(
    *,
    factory_valid: bool,
    status_rejected: bool,
    safety_scope_valid: bool,
    anti_parameter_search_valid: bool,
    thesis_id_present: bool,
    mechanism_class_present: bool,
    method_blockers: Sequence[str],
) -> list[str]:
    reasons: list[str] = []
    if not factory_valid:
        reasons.append("factory_invalid")
    if not status_rejected:
        reasons.append("status_not_failed_or_blocked")
    if not safety_scope_valid:
        reasons.append("safety_scope_invalid")
    if not anti_parameter_search_valid:
        reasons.append("anti_parameter_search_invalid")
    if not thesis_id_present:
        reasons.append("thesis_id_missing")
    if not mechanism_class_present:
        reasons.append("mechanism_class_missing")
    if method_blockers:
        reasons.append("method_blockers_present")
    return reasons


def _local_falsification_cost_payload(payload: dict[str, Any]) -> dict[str, Any]:
    nested = payload.get("cost_edge_evidence")
    if isinstance(nested, dict):
        merged = dict(payload)
        merged.update(nested)
        return merged
    return payload


def _local_falsification_calendar_windows(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    summaries: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        summaries.append(
            {
                "calendar_window": item.get("calendar_window"),
                "sample_count": _number(item.get("sample_count")),
                "expected_edge_bps": _number(item.get("expected_edge_bps")),
                "net_edge_bps": _number(item.get("net_edge_bps")),
                "win_rate": _number(item.get("win_rate")),
                "profitable": bool(item.get("profitable")),
            }
        )
    return summaries


def _local_falsification_safety_scope_valid(safety_scope: Any) -> bool:
    if not isinstance(safety_scope, dict):
        return False
    unsafe_flags = (
        "backtest_started",
        "strategy_code_generated",
        "paper_trading_started",
        "dry_run_trading_started",
        "live_trading",
        "exchange_order_placement",
        "shorting",
        "process_control",
    )
    leverage = _number(safety_scope.get("leverage"))
    return (
        safety_scope.get("historical_only") is True
        and all(not bool(safety_scope.get(flag)) for flag in unsafe_flags)
        and (leverage is None or leverage <= 1.0)
    )


def _local_falsification_event_source_valid(event_source: Any) -> bool:
    if not isinstance(event_source, dict):
        return False
    return (
        event_source.get("factory_valid") is True
        and event_source.get("status_completed") is True
        and event_source.get("thesis_matches") is True
        and event_source.get("event_path_matches") is True
        and event_source.get("ohlcv_path_matches") is True
        and event_source.get("safety_scope_valid") is True
        and _local_falsification_event_source_context_alignment_valid(event_source)
    )


def _local_falsification_event_source_context_alignment_valid(event_source: Any) -> bool:
    if not isinstance(event_source, dict):
        return False
    if event_source.get("closed_context_candle_alignment_valid") is True:
        return True
    if event_source.get("context_features_used") is False:
        return True
    required_contexts = event_source.get("required_contexts")
    if isinstance(required_contexts, list) and not required_contexts:
        return True
    return (
        event_source.get("context_features_used") is True
        and event_source.get("context_merge_semantics") == _CONTEXT_MERGE_SEMANTICS
        and event_source.get("closed_context_candle_alignment_valid") is True
    )


def _local_falsification_event_source_failure_synthesis_guard_valid(
    event_source: Any,
) -> bool:
    if not isinstance(event_source, dict):
        return False
    if event_source.get("failure_synthesis_guard_valid") is True:
        return True

    nested = event_source.get("failure_synthesis_summary")
    if isinstance(nested, dict):
        used = nested.get("used")
        parseable = nested.get("parseable")
        allow_failed = nested.get("allow_failed_thesis_or_family")
        thesis_repeats = nested.get("thesis_repeats_failed_synthesis")
        mechanism_repeats = nested.get("mechanism_repeats_failed_synthesis")
    else:
        used = event_source.get("failure_synthesis_used")
        parseable = event_source.get("failure_synthesis_parseable")
        allow_failed = event_source.get("failure_synthesis_allow_failed_thesis_or_family")
        thesis_repeats = event_source.get("failure_synthesis_thesis_repeats")
        mechanism_repeats = event_source.get("failure_synthesis_mechanism_repeats")

    return (
        used is True
        and parseable is True
        and allow_failed is not True
        and thesis_repeats is not True
        and mechanism_repeats is not True
    )


def _local_falsification_event_source_summary(event_source: Any) -> dict[str, Any]:
    if not isinstance(event_source, dict):
        return {"valid": False}
    return {
        "valid": _local_falsification_event_source_valid(event_source),
        "context_alignment_valid": (
            _local_falsification_event_source_context_alignment_valid(event_source)
        ),
        "failure_synthesis_guard_valid": (
            _local_falsification_event_source_failure_synthesis_guard_valid(event_source)
        ),
        "path": event_source.get("path"),
        "factory": event_source.get("factory"),
        "status": event_source.get("status"),
        "thesis_id": event_source.get("thesis_id"),
        "events_csv_path": event_source.get("events_csv_path"),
        "source_ohlcv_path": event_source.get("source_ohlcv_path"),
        "event_count": event_source.get("event_count"),
        "context_features_used": event_source.get("context_features_used"),
        "required_contexts": event_source.get("required_contexts"),
        "context_merge_semantics": event_source.get("context_merge_semantics"),
        "closed_context_candle_alignment_valid": event_source.get(
            "closed_context_candle_alignment_valid"
        ),
        "failure_synthesis_used": event_source.get("failure_synthesis_used"),
        "failure_synthesis_parseable": event_source.get("failure_synthesis_parseable"),
        "failure_synthesis_path": event_source.get("failure_synthesis_path"),
        "failure_synthesis_thesis_repeats": event_source.get(
            "failure_synthesis_thesis_repeats"
        ),
        "failure_synthesis_mechanism_repeats": event_source.get(
            "failure_synthesis_mechanism_repeats"
        ),
        "failure_synthesis_allow_failed_thesis_or_family": event_source.get(
            "failure_synthesis_allow_failed_thesis_or_family"
        ),
    }


def _local_falsification_rejection_failure_reasons(
    *,
    factory_valid: bool,
    status_rejected: bool,
    safety_scope_valid: bool,
    event_source_valid: bool,
    event_source_context_alignment_valid: bool,
    event_source_failure_synthesis_guard_valid: bool,
) -> list[str]:
    reasons: list[str] = []
    if not factory_valid:
        reasons.append("factory_invalid")
    if not status_rejected:
        reasons.append("status_not_failed_rejected_or_blocked")
    if not safety_scope_valid:
        reasons.append("safety_scope_invalid")
    if not event_source_valid:
        reasons.append("event_source_invalid")
    if not event_source_context_alignment_valid:
        reasons.append("event_source_context_alignment_missing_or_invalid")
    if not event_source_failure_synthesis_guard_valid:
        reasons.append("event_source_failure_synthesis_guard_missing_or_failed")
    return reasons


def _candidate_research_brief(
    ranking_item: dict[str, Any], manifest: dict[str, Any]
) -> dict[str, Any]:
    brief: dict[str, Any] = {}
    for source in (manifest.get("research_brief"), ranking_item.get("research_brief")):
        if not isinstance(source, dict):
            continue
        for key, value in source.items():
            if key not in brief:
                brief[str(key)] = _copy_jsonish(value)
    return brief


def _research_handoff_summary(*sources: Any) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for source in sources:
        if not isinstance(source, dict):
            continue
        nested = source.get("research_handoff_summary")
        if isinstance(nested, dict):
            for key in RESEARCH_HANDOFF_KEYS:
                if key not in summary and isinstance(nested.get(key), dict):
                    summary[key] = _copy_jsonish(nested[key])
        for key in RESEARCH_HANDOFF_KEYS:
            if key not in summary and isinstance(source.get(key), dict):
                summary[key] = _copy_jsonish(source[key])
    return summary


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


def _diagnostics_summary(diagnostics: dict[str, Any]) -> dict[str, Any]:
    if not diagnostics:
        return {"available": False}
    return {
        "available": True,
        "path": diagnostics.get("diagnostics_path"),
        "status": diagnostics.get("status"),
        "entry_count": diagnostics.get("entry_count"),
        "diagnosis_codes": list(diagnostics.get("diagnosis_codes") or []),
        "first_zero_component": diagnostics.get("first_zero_component"),
        "rarest_component": diagnostics.get("rarest_component"),
        "bottleneck_components": list(diagnostics.get("bottleneck_components") or [])[:3],
        "generated_entry_edge": diagnostics.get("generated_entry_edge") or {},
    }


def _freqai_prediction_summary(diagnostics: dict[str, Any]) -> dict[str, Any]:
    if not diagnostics:
        return {"available": False}
    return {
        "available": True,
        "path": diagnostics.get("diagnostics_path"),
        "status": diagnostics.get("status"),
        "expected_target_column": diagnostics.get("expected_target_column"),
        "expected_target_column_present": diagnostics.get("expected_target_column_present"),
        "target_columns": list(diagnostics.get("target_columns") or []),
        "model_label_columns": list(diagnostics.get("model_label_columns") or []),
        "prediction_file_count": diagnostics.get("prediction_file_count"),
        "row_count": diagnostics.get("row_count"),
        "diagnosis_codes": list(diagnostics.get("diagnosis_codes") or []),
        "alternate_target_summaries": diagnostics.get("alternate_target_summaries") or {},
    }


def _bottleneck_questions(signal_bottlenecks: Sequence[dict[str, Any]]) -> list[str]:
    questions: list[str] = []
    for item in signal_bottlenecks:
        component = item.get("first_zero_component")
        codes = set(item.get("diagnosis_codes") or [])
        if "ML_FILTER_UNAVAILABLE" in codes:
            questions.append(
                "What generated model-prediction diagnostics are needed before "
                f"judging the ML gate for {item.get('candidate_id')}?"
            )
            continue
        if component:
            questions.append(
                "Why does the generated "
                f"{component} condition eliminate the surviving setup rows for "
                f"{item.get('candidate_id')}?"
            )
    return questions


def _generated_entry_edge_questions(edge_failures: Sequence[dict[str, Any]]) -> list[str]:
    questions: list[str] = []
    for item in edge_failures:
        questions.append(
            "Why did the generated entry set for "
            f"{item.get('candidate_id')} have non-positive edge after costs "
            f"(net_edge_bps={item.get('net_edge_bps')}, "
            f"profitable_windows_ratio={item.get('profitable_windows_ratio')}) "
            "even when pre-proposal local falsification looked acceptable?"
        )
    return questions


def _freqai_prediction_questions(
    freqai_target_mismatches: Sequence[dict[str, Any]]
) -> list[str]:
    questions: list[str] = []
    for item in freqai_target_mismatches:
        expected = item.get("expected_target_column")
        found = ", ".join(item.get("target_columns") or [])
        labels = ", ".join(item.get("model_label_columns") or [])
        questions.append(
            "Why did the generated hybrid candidate expect "
            f"{expected} while stored FreqAI predictions expose {found or 'no target'} "
            f"and model labels expose {labels or 'no labels'}?"
        )
    return questions


def _local_falsification_rejection_questions(
    rejections: Sequence[dict[str, Any]]
) -> list[str]:
    questions: list[str] = []
    for item in rejections:
        mechanism = item.get("mechanism_class") or "unknown mechanism"
        thesis_id = item.get("thesis_id") or "unknown thesis"
        questions.append(
            "What materially different mechanism avoids the failed local "
            f"falsification for {thesis_id} / {mechanism} "
            f"(net_edge_bps={item.get('net_edge_bps')}, "
            f"profitable_windows_ratio={item.get('profitable_windows_ratio')})?"
        )
    return questions


def _edge_discovery_rejection_questions(
    rejections: Sequence[dict[str, Any]]
) -> list[str]:
    questions: list[str] = []
    for item in rejections:
        mechanism = item.get("mechanism_class") or "unknown mechanism"
        thesis_id = item.get("thesis_id") or "unknown thesis"
        questions.append(
            "What materially different edge hypothesis avoids the failed "
            f"edge discovery for {thesis_id} / {mechanism} "
            f"(best_hold_candles={item.get('best_hold_candles')}, "
            f"net_edge_bps={item.get('net_edge_bps')}, "
            f"passing_horizon_count={item.get('passing_horizon_count')}) before "
            "any strategy proposal is written?"
        )
    return questions


def _load_manifest_if_available(path: Any, root: Path) -> dict[str, Any]:
    if not path:
        return {}
    resolved = _resolve_inside(Path(str(path)), root)
    if not resolved.is_file():
        return {}
    return _load_json(resolved)


def _check(name: str, passed: bool, *, status_if_false: str = "fail") -> dict[str, str]:
    return {"name": name, "status": "pass" if passed else status_if_false}


def _render_report(synthesis: dict[str, Any]) -> str:
    brief = synthesis.get("next_research_brief", {})
    aggregate = synthesis.get("aggregate_failure_summary", {})
    lines = [
        "# Candidate Failure Synthesis",
        "",
        f"- synthesis_id: {synthesis.get('synthesis_id')}",
        f"- ranking_id: {synthesis.get('ranking_id')}",
        f"- candidate_count: {synthesis.get('candidate_count')}",
        f"- paper_ready_count: {aggregate.get('paper_ready_count')}",
        f"- parameter_only_retry_allowed: {brief.get('parameter_only_retry_allowed')}",
        f"- requires_new_thesis_id: {brief.get('requires_new_thesis_id')}",
        "",
        "## Failed Families",
        "",
    ]
    families = aggregate.get("hypothesis_families_tried") or []
    lines.extend([f"- {family}" for family in families] or ["- None."])
    lines.extend(["", "## Signal Bottlenecks", ""])
    bottlenecks = aggregate.get("signal_bottlenecks") or []
    for item in bottlenecks:
        lines.append(
            "- "
            f"{item.get('candidate_id')}: first_zero_component="
            f"{item.get('first_zero_component')}"
        )
    if not bottlenecks:
        lines.append("- No signal diagnostics supplied.")
    lines.extend(["", "## Generated Entry Edge Failures", ""])
    edge_failures = aggregate.get("generated_entry_edge_failures") or []
    for item in edge_failures:
        lines.append(
            "- "
            f"{item.get('candidate_id')}: status={item.get('status')}, "
            f"net_edge_bps={item.get('net_edge_bps')}, "
            f"profitable_windows_ratio={item.get('profitable_windows_ratio')}"
        )
    if not edge_failures:
        lines.append("- None supplied.")
    lines.extend(["", "## Local Falsification Rejections", ""])
    local_rejections = aggregate.get("local_falsification_rejections") or []
    for item in local_rejections:
        lines.append(
            "- "
            f"{item.get('thesis_id')} / {item.get('mechanism_class')}: "
            f"net_edge_bps={item.get('net_edge_bps')}, "
            f"profitable_windows_ratio={item.get('profitable_windows_ratio')}"
        )
    if not local_rejections:
        lines.append("- None supplied.")
    lines.extend(["", "## Edge Discovery Rejections", ""])
    edge_rejections = aggregate.get("edge_discovery_rejections") or []
    for item in edge_rejections:
        lines.append(
            "- "
            f"{item.get('thesis_id')} / {item.get('mechanism_class')}: "
            f"best_hold_candles={item.get('best_hold_candles')}, "
            f"net_edge_bps={item.get('net_edge_bps')}, "
            f"passing_horizon_count={item.get('passing_horizon_count')}"
        )
    if not edge_rejections:
        lines.append("- None supplied.")
    lines.extend(["", "## FreqAI Prediction Diagnostics", ""])
    mismatches = aggregate.get("freqai_target_mismatches") or []
    for item in mismatches:
        lines.append(
            "- "
            f"{item.get('candidate_id')}: expected={item.get('expected_target_column')}, "
            f"found={', '.join(item.get('target_columns') or [])}"
        )
    if not mismatches:
        lines.append("- No FreqAI target mismatches supplied.")
    lines.extend(["", "## Next Research Questions", ""])
    lines.extend(
        [f"- {question}" for question in brief.get("recommended_research_questions", [])]
        or ["- None."]
    )
    lines.extend(["", "## Blocked Next Actions", ""])
    lines.extend([f"- {action}" for action in brief.get("blocked_next_actions", [])])
    lines.append("")
    return "\n".join(lines)


def _number(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _synthesis_id(generated_at: str) -> str:
    parsed = datetime.fromisoformat(generated_at.replace("Z", "+00:00"))
    return parsed.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")


def _resolve_inside(path: Path, root: Path) -> Path:
    resolved = (path if path.is_absolute() else root / path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Path must resolve inside the workspace: {path}") from exc
    return resolved


def _rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root)).replace("\\", "/")
    except ValueError:
        return str(path)


def _safe_path_component(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value)
    return cleaned.strip("._") or "synthesis"


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected object JSON: {path}")
    return payload
