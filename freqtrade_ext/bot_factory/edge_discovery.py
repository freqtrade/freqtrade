from __future__ import annotations

import json
import random
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from math import sqrt
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from freqtrade_ext.bot_factory.cost_model import (
    CostModelContext,
    cost_context_from_spec,
    cost_scenarios_from_spec,
)
from freqtrade_ext.bot_factory.local_events import (
    _OPS,
    _attach_context_features,
    _condition_checks,
    _condition_diagnostic,
    _context_checks,
    _context_merge_summary,
    _cooldown_candles,
    _events_from_mask,
    _failure_synthesis_checks,
    _feature_column,
    _feature_series,
    _load_funding_rate,
    _load_liquidation,
    _load_long_short_ratio,
    _load_mark_price,
    _load_ohlcv,
    _load_open_interest,
    _load_order_book,
    _required_contexts,
    _safe_spec_summary,
    _source_summary,
    _structural_quality_report_checks,
)
from freqtrade_ext.bot_factory.local_falsification import (
    _calendar_window_summaries,
    _event_returns,
    _mean,
    _median,
    _win_rate,
    _window_summaries,
)


_EDGE_SPEC_FACTORY = "research_edge_discovery_spec"
_HYPOTHESIS_SCOPES = {
    "single_asset",
    "cross_asset",
    "market_neutral",
    "funding_basis",
    "microstructure",
}
_MULTI_INSTRUMENT_SCOPES = {"cross_asset", "market_neutral"}
_PARAMETER_SEARCH_KEYS = {
    "candidates",
    "grid",
    "hyperopt",
    "max",
    "min",
    "optimization",
    "optimize",
    "parameter_grid",
    "search_space",
    "step",
    "threshold_grid",
    "values",
}


@dataclass(frozen=True)
class EdgeDiscoveryInputs:
    root_dir: Path
    ohlcv_path: Path
    edge_spec_path: Path
    funding_rate_path: Path | None = None
    mark_price_path: Path | None = None
    informative_ohlcv_path: Path | None = None
    open_interest_path: Path | None = None
    open_interest_quality_report_paths: Sequence[Path] = field(default_factory=list)
    long_short_ratio_path: Path | None = None
    long_short_ratio_quality_report_paths: Sequence[Path] = field(default_factory=list)
    liquidation_path: Path | None = None
    liquidation_quality_report_paths: Sequence[Path] = field(default_factory=list)
    order_book_path: Path | None = None
    order_book_quality_report_paths: Sequence[Path] = field(default_factory=list)
    failure_synthesis_path: Path | None = None
    allow_failed_thesis_or_family: bool = False
    output_root: Path = Path("registry/strategies/research_decisions")
    edge_discovery_id: str | None = None
    min_sample_count: int = 20
    min_profitable_windows_ratio: float = 0.5
    min_calendar_window_count: int = 0
    min_profitable_calendar_windows_ratio: float = 0.0
    min_data_span_days: float = 0.0
    min_passing_horizon_count: int = 1
    max_horizon_count: int = 5
    min_negative_control_delta_bps: float = 1.0
    reviewer_notes: Sequence[str] = field(default_factory=list)
    created_by_agent: str = "codex"
    created_at: str | None = None
    command: Sequence[str] = field(default_factory=list)


def build_edge_discovery(inputs: EdgeDiscoveryInputs) -> dict[str, Any]:
    root = inputs.root_dir.resolve()
    generated_at = inputs.created_at or datetime.now(UTC).replace(microsecond=0).isoformat()
    ohlcv_path = _resolve_inside(inputs.ohlcv_path, root)
    spec_path = _resolve_inside(inputs.edge_spec_path, root)
    funding_rate_path = _resolve_optional_inside(inputs.funding_rate_path, root)
    mark_price_path = _resolve_optional_inside(inputs.mark_price_path, root)
    informative_ohlcv_path = _resolve_optional_inside(inputs.informative_ohlcv_path, root)
    open_interest_path = _resolve_optional_inside(inputs.open_interest_path, root)
    open_interest_quality_paths = [
        _resolve_inside(path, root) for path in inputs.open_interest_quality_report_paths
    ]
    long_short_ratio_path = _resolve_optional_inside(inputs.long_short_ratio_path, root)
    long_short_ratio_quality_paths = [
        _resolve_inside(path, root) for path in inputs.long_short_ratio_quality_report_paths
    ]
    liquidation_path = _resolve_optional_inside(inputs.liquidation_path, root)
    liquidation_quality_paths = [
        _resolve_inside(path, root) for path in inputs.liquidation_quality_report_paths
    ]
    order_book_path = _resolve_optional_inside(inputs.order_book_path, root)
    failure_synthesis_path = _resolve_optional_inside(inputs.failure_synthesis_path, root)
    order_book_quality_paths = [
        _resolve_inside(path, root) for path in inputs.order_book_quality_report_paths
    ]

    spec, spec_error = _load_json(spec_path)
    spec = spec if isinstance(spec, dict) else {}
    edge_discovery_id = inputs.edge_discovery_id or _edge_discovery_id(
        generated_at,
        str(spec.get("edge_discovery_id") or spec.get("thesis_id") or "edge_discovery"),
    )
    cost_context = cost_context_from_spec(spec)
    cost_scenarios = cost_scenarios_from_spec(spec, context=cost_context)
    all_in_cost_bps = _normal_cost_bps(cost_scenarios)
    horizons = _edge_horizons(spec)
    anti_search = _anti_parameter_search_summary(spec)
    hypothesis_scope = _hypothesis_scope(spec)
    instrument_universe = _instrument_universe(spec)
    market_structure_domains = _market_structure_domains(spec, hypothesis_scope)

    failure_synthesis, failure_synthesis_error = (
        _load_json(failure_synthesis_path)
        if failure_synthesis_path is not None
        else (None, None)
    )
    ohlcv, ohlcv_error = _load_ohlcv(ohlcv_path)
    ohlcv_coverage = _ohlcv_coverage(ohlcv)
    checks: list[dict[str, Any]] = [
        _check("ohlcv_file_present", ohlcv_path.is_file(), {"path": _rel(ohlcv_path, root)}),
        _check("ohlcv_parseable", ohlcv is not None and not ohlcv_error, {"error": ohlcv_error}),
        _check("edge_spec_file_present", spec_path.is_file(), {"path": _rel(spec_path, root)}),
        _check("edge_spec_parseable", bool(spec) and spec_error is None, {"error": spec_error}),
        _check(
            "edge_spec_factory_valid",
            spec.get("factory") == _EDGE_SPEC_FACTORY,
            {"factory": spec.get("factory"), "expected_factory": _EDGE_SPEC_FACTORY},
        ),
        _check("edge_spec_thesis_id_present", bool(str(spec.get("thesis_id") or "").strip()), {}),
        _check(
            "edge_spec_mechanism_class_present",
            bool(str(spec.get("mechanism_class") or "").strip()),
            {},
        ),
        _check(
            "edge_spec_all_in_cost_bps_non_negative",
            all_in_cost_bps is not None and all_in_cost_bps >= 0.0,
            {"all_in_cost_bps": all_in_cost_bps},
        ),
        _check("edge_spec_horizons_present", bool(horizons), {"horizons": horizons}),
        _check(
            "edge_spec_horizon_count_bounded",
            bool(horizons) and len(horizons) <= int(inputs.max_horizon_count),
            {"horizon_count": len(horizons), "max_horizon_count": int(inputs.max_horizon_count)},
        ),
        _check("edge_spec_no_parameter_search_grid", anti_search["valid"], anti_search),
        _check(
            "edge_spec_hypothesis_scope_valid",
            hypothesis_scope in _HYPOTHESIS_SCOPES,
            {
                "hypothesis_scope": hypothesis_scope,
                "allowed_hypothesis_scopes": sorted(_HYPOTHESIS_SCOPES),
            },
        ),
        _check(
            "edge_spec_instrument_universe_sufficient_for_scope",
            hypothesis_scope not in _MULTI_INSTRUMENT_SCOPES
            or len(instrument_universe) >= 2,
            {
                "hypothesis_scope": hypothesis_scope,
                "instrument_universe": instrument_universe,
                "instrument_count": len(instrument_universe),
            },
        ),
        _check(
            "ohlcv_data_span_sufficient",
            float(inputs.min_data_span_days) <= 0.0
            or (
                ohlcv_coverage["data_span_days"] is not None
                and ohlcv_coverage["data_span_days"] >= float(inputs.min_data_span_days)
            ),
            {
                "data_span_days": ohlcv_coverage["data_span_days"],
                "min_data_span_days": float(inputs.min_data_span_days),
                "data_start": ohlcv_coverage["data_start"],
                "data_end": ohlcv_coverage["data_end"],
            },
        ),
        _check(
            "min_passing_horizon_count_positive",
            int(inputs.min_passing_horizon_count) > 0,
            {"min_passing_horizon_count": int(inputs.min_passing_horizon_count)},
        ),
    ]
    checks.extend(_cost_scenario_checks(cost_scenarios, cost_context))

    condition_checks, conditions = _condition_checks(spec)
    checks.extend(condition_checks)
    failure_synthesis_checks, failure_synthesis_summary = _failure_synthesis_checks(
        spec,
        failure_synthesis=failure_synthesis if isinstance(failure_synthesis, dict) else None,
        failure_synthesis_path=failure_synthesis_path,
        failure_synthesis_error=failure_synthesis_error,
        root=root,
        allow_failed_thesis_or_family=inputs.allow_failed_thesis_or_family,
    )
    checks.extend(failure_synthesis_checks)
    required_contexts = _required_contexts(conditions)
    funding_rate, funding_rate_error = (
        _load_funding_rate(funding_rate_path)
        if funding_rate_path is not None
        else (None, "path_not_provided")
    )
    mark_price, mark_price_error = (
        _load_mark_price(mark_price_path)
        if mark_price_path is not None
        else (None, "path_not_provided")
    )
    informative_ohlcv, informative_ohlcv_error = (
        _load_ohlcv(informative_ohlcv_path)
        if informative_ohlcv_path is not None
        else (None, "path_not_provided")
    )
    open_interest, open_interest_error = (
        _load_open_interest(open_interest_path)
        if open_interest_path is not None
        else (None, "path_not_provided")
    )
    long_short_ratio, long_short_ratio_error = (
        _load_long_short_ratio(long_short_ratio_path)
        if long_short_ratio_path is not None
        else (None, "path_not_provided")
    )
    liquidation, liquidation_error = (
        _load_liquidation(liquidation_path)
        if liquidation_path is not None
        else (None, "path_not_provided")
    )
    order_book, order_book_error = (
        _load_order_book(order_book_path)
        if order_book_path is not None
        else (None, "path_not_provided")
    )
    context_checks = _context_checks(
        required_contexts=required_contexts,
        funding_rate_path=funding_rate_path,
        funding_rate=funding_rate,
        funding_rate_error=funding_rate_error,
        mark_price_path=mark_price_path,
        mark_price=mark_price,
        mark_price_error=mark_price_error,
        informative_ohlcv_path=informative_ohlcv_path,
        informative_ohlcv=informative_ohlcv,
        informative_ohlcv_error=informative_ohlcv_error,
        open_interest_path=open_interest_path,
        open_interest=open_interest,
        open_interest_error=open_interest_error,
        long_short_ratio_path=long_short_ratio_path,
        long_short_ratio=long_short_ratio,
        long_short_ratio_error=long_short_ratio_error,
        liquidation_path=liquidation_path,
        liquidation=liquidation,
        liquidation_error=liquidation_error,
        order_book_path=order_book_path,
        order_book=order_book,
        order_book_error=order_book_error,
        root=root,
    )
    checks.extend(context_checks)
    context_merge = _context_merge_summary(
        required_contexts=required_contexts,
        ohlcv=ohlcv,
        funding_rate=funding_rate,
        mark_price=mark_price,
        informative_ohlcv=informative_ohlcv,
        open_interest=open_interest,
        long_short_ratio=long_short_ratio,
        liquidation=liquidation,
        order_book=order_book,
    )
    open_interest_quality_reports = _quality_report_summaries(
        open_interest_quality_paths,
        root=root,
    )
    long_short_ratio_quality_reports = _quality_report_summaries(
        long_short_ratio_quality_paths,
        root=root,
    )
    liquidation_quality_reports = _quality_report_summaries(
        liquidation_quality_paths,
        root=root,
    )
    order_book_quality_reports = _quality_report_summaries(
        order_book_quality_paths,
        root=root,
    )
    quality_checks = _structural_quality_report_checks(
        required_contexts=required_contexts,
        open_interest_quality_reports=open_interest_quality_reports,
        long_short_ratio_quality_reports=long_short_ratio_quality_reports,
        liquidation_quality_reports=liquidation_quality_reports,
        order_book_quality_reports=order_book_quality_reports,
    )
    checks.extend(quality_checks)

    events: list[dict[str, Any]] = []
    feature_columns: list[str] = []
    condition_diagnostics: list[dict[str, Any]] = []
    cumulative_condition_match_counts: list[dict[str, Any]] = []
    combined_match_count = 0
    event_build_prerequisites = (
        ohlcv is not None
        and conditions
        and all(check["status"] == "pass" for check in condition_checks)
        and all(check["status"] == "pass" for check in failure_synthesis_checks)
        and all(check["status"] == "pass" for check in context_checks)
        and all(check["status"] == "pass" for check in quality_checks)
        and anti_search["valid"]
    )
    if event_build_prerequisites:
        enriched = _attach_context_features(
            ohlcv.copy(),
            funding_rate=funding_rate,
            mark_price=mark_price,
            informative_ohlcv=informative_ohlcv,
            open_interest=open_interest,
            long_short_ratio=long_short_ratio,
            liquidation=liquidation,
            order_book=order_book,
        )
        condition_masks: list[pd.Series] = []
        running_mask: pd.Series | None = None
        for condition in conditions:
            column = _feature_column(condition)
            enriched[column] = _feature_series(enriched, condition)
            feature_columns.append(column)
            condition_mask = _OPS[condition["operator"]](enriched[column], condition["value"])
            condition_masks.append(condition_mask)
            running_mask = condition_mask if running_mask is None else running_mask & condition_mask
            condition_diagnostics.append(
                _condition_diagnostic(
                    condition,
                    feature_column=column,
                    series=enriched[column],
                    mask=condition_mask,
                )
            )
            cumulative_condition_match_counts.append(
                {
                    "through_feature_column": column,
                    "match_count": int(running_mask.fillna(False).sum()),
                }
            )
        mask = condition_masks[0]
        for extra_mask in condition_masks[1:]:
            mask = mask & extra_mask
        combined_match_count = int(mask.fillna(False).sum())
        events = _events_from_mask(
            enriched,
            mask.fillna(False),
            feature_columns=feature_columns,
            cooldown_candles=_cooldown_candles(spec),
        )
    checks.append(_check("edge_events_generated", bool(events), {"event_count": len(events)}))

    horizon_results = _horizon_results(
        ohlcv,
        events,
        horizons=horizons,
        cost_scenarios=cost_scenarios,
        funding_rate=funding_rate if funding_rate_path is not None else None,
        min_sample_count=int(inputs.min_sample_count),
        min_profitable_windows_ratio=float(inputs.min_profitable_windows_ratio),
        min_calendar_window_count=int(inputs.min_calendar_window_count),
        min_profitable_calendar_windows_ratio=float(
            inputs.min_profitable_calendar_windows_ratio
        ),
        min_negative_control_delta_bps=float(inputs.min_negative_control_delta_bps),
    )
    passing_horizons = [
        item for item in horizon_results if item.get("status") == "passed"
    ]
    checks.append(
        _check(
            "passing_horizon_count_sufficient",
            len(passing_horizons) >= int(inputs.min_passing_horizon_count),
            {
                "passing_horizon_count": len(passing_horizons),
                "min_passing_horizon_count": int(inputs.min_passing_horizon_count),
            },
        )
    )
    checks.extend(
        _check(
            f"horizon_{item['hold_candles']}_edge_evidence_passed",
            item.get("status") == "passed",
            {
                "hold_candles": item["hold_candles"],
                "net_edge_bps": item.get("net_edge_bps"),
                "sample_count": item.get("sample_count"),
                "profitable_windows_ratio": item.get("profitable_windows_ratio"),
                "profitable_calendar_windows_ratio": item.get(
                    "profitable_calendar_windows_ratio"
                ),
            },
        )
        for item in horizon_results
    )

    structural_checks_passed = all(
        check["status"] == "pass"
        for check in checks
        if not check["name"].startswith("horizon_")
        and check["name"] != "passing_horizon_count_sufficient"
    )
    status = (
        "blocked"
        if not structural_checks_passed
        else "passed"
        if len(passing_horizons) >= int(inputs.min_passing_horizon_count)
        else "failed"
    )
    best_horizon = _best_horizon(horizon_results)
    concentration = _event_concentration_diagnostics(events)
    event_level_report = _event_level_post_cost_report(
        spec,
        best_horizon=best_horizon,
        horizon_results=horizon_results,
        concentration=concentration,
    )
    research_gate = event_level_report["research_gate"]
    candidate_generation_allowed = (
        status == "passed" and research_gate["passes_research_gate"] is True
    )
    blocked_next_actions = _blocked_next_actions(
        status, candidate_generation_allowed=candidate_generation_allowed
    )
    return {
        "generated_at": generated_at,
        "factory": "research_edge_discovery",
        "edge_discovery_id": edge_discovery_id,
        "status": status,
        "thesis_id": str(spec.get("thesis_id") or "").strip(),
        "mechanism_class": str(spec.get("mechanism_class") or "").strip(),
        "source_ohlcv_path": _rel(ohlcv_path, root),
        "source_funding_rate_path": _rel(funding_rate_path, root) if funding_rate_path else None,
        "source_mark_price_path": _rel(mark_price_path, root) if mark_price_path else None,
        "source_informative_ohlcv_path": (
            _rel(informative_ohlcv_path, root) if informative_ohlcv_path else None
        ),
        "source_open_interest_path": (
            _rel(open_interest_path, root) if open_interest_path else None
        ),
        "source_long_short_ratio_path": (
            _rel(long_short_ratio_path, root) if long_short_ratio_path else None
        ),
        "source_liquidation_path": (
            _rel(liquidation_path, root) if liquidation_path else None
        ),
        "source_order_book_path": (
            _rel(order_book_path, root) if order_book_path else None
        ),
        "hypothesis_scope": hypothesis_scope,
        "instrument_universe": instrument_universe,
        "market_structure_domains": market_structure_domains,
        "cost_model_context": _cost_model_context_summary(cost_context),
        "cost_scenarios": cost_scenarios,
        "open_interest_quality_reports": open_interest_quality_reports,
        "long_short_ratio_quality_reports": long_short_ratio_quality_reports,
        "liquidation_quality_reports": liquidation_quality_reports,
        "order_book_quality_reports": order_book_quality_reports,
        "failure_synthesis_path": (
            _rel(failure_synthesis_path, root) if failure_synthesis_path else None
        ),
        "failure_synthesis_summary": failure_synthesis_summary,
        "edge_spec_path": _rel(spec_path, root),
        "edge_spec": _safe_edge_spec_summary(spec),
        "anti_parameter_search": anti_search,
        "ohlcv_row_count": ohlcv_coverage["row_count"],
        "data_start": ohlcv_coverage["data_start"],
        "data_end": ohlcv_coverage["data_end"],
        "data_span_days": ohlcv_coverage["data_span_days"],
        "min_data_span_days": float(inputs.min_data_span_days),
        "auxiliary_sources": {
            "funding_rate": _source_summary(
                funding_rate_path,
                funding_rate,
                funding_rate_error,
                root=root,
                required="funding_rate" in required_contexts,
            ),
            "mark_price": _source_summary(
                mark_price_path,
                mark_price,
                mark_price_error,
                root=root,
                required="mark_price" in required_contexts,
            ),
            "informative_ohlcv": _source_summary(
                informative_ohlcv_path,
                informative_ohlcv,
                informative_ohlcv_error,
                root=root,
                required="informative_ohlcv" in required_contexts,
            ),
            "open_interest": _source_summary(
                open_interest_path,
                open_interest,
                open_interest_error,
                root=root,
                required="open_interest" in required_contexts,
            ),
            "long_short_ratio": _source_summary(
                long_short_ratio_path,
                long_short_ratio,
                long_short_ratio_error,
                root=root,
                required="long_short_ratio" in required_contexts,
            ),
            "liquidation": _source_summary(
                liquidation_path,
                liquidation,
                liquidation_error,
                root=root,
                required="liquidation" in required_contexts,
            ),
            "order_book": _source_summary(
                order_book_path,
                order_book,
                order_book_error,
                root=root,
                required="order_book" in required_contexts,
            ),
        },
        "context_merge": context_merge,
        "horizons": horizons,
        "horizon_count": len(horizons),
        "all_in_cost_bps": all_in_cost_bps,
        "min_sample_count": int(inputs.min_sample_count),
        "min_profitable_windows_ratio": float(inputs.min_profitable_windows_ratio),
        "min_calendar_window_count": int(inputs.min_calendar_window_count),
        "min_profitable_calendar_windows_ratio": float(
            inputs.min_profitable_calendar_windows_ratio
        ),
        "min_passing_horizon_count": int(inputs.min_passing_horizon_count),
        "condition_count": len(conditions),
        "feature_columns": feature_columns,
        "condition_diagnostics": condition_diagnostics,
        "cumulative_condition_match_counts": cumulative_condition_match_counts,
        "combined_match_count_before_cooldown": combined_match_count,
        "cooldown_candles": _cooldown_candles(spec),
        "event_count": len(events),
        "concentration_diagnostics": concentration,
        "event_preview": events[:10],
        "horizon_results": horizon_results,
        "passing_horizon_count": len(passing_horizons),
        "passing_horizons": [
            {
                "hold_candles": item["hold_candles"],
                "net_edge_bps": item.get("net_edge_bps"),
                "sample_count": item.get("sample_count"),
            }
            for item in passing_horizons
        ],
        "best_horizon_by_net_edge": best_horizon,
        "event_level_post_cost_edge_report": event_level_report,
        "research_gate": research_gate,
        "candidate_generation_allowed": candidate_generation_allowed,
        "candidate_generation_result": (
            "candidate generation allowed"
            if candidate_generation_allowed
            else "no candidate generated"
        ),
        "promotion_gate": {
            "proposal_generation_allowed": candidate_generation_allowed,
            "strategy_codegen_allowed": False,
            "candidate_generation_allowed": candidate_generation_allowed,
            "must_pass_research_selection_after_edge_discovery": True,
            "must_pass_research_gate_before_candidate_generation": True,
            "blocked_next_actions": blocked_next_actions,
        },
        "proposal_generation_allowed": candidate_generation_allowed,
        "strategy_codegen_allowed": False,
        "blocked_next_actions": blocked_next_actions,
        "checks": checks,
        "blockers": [check for check in checks if check["status"] != "pass"],
        "reviewer_notes": [str(note) for note in inputs.reviewer_notes],
        "created_by_agent": str(inputs.created_by_agent),
        "command": list(inputs.command),
        "safety_scope": {
            "historical_only": True,
            "closed_candle_ohlcv_only": not bool(required_contexts),
            "closed_candle_local_market_data_only": True,
            "closed_context_candle_alignment": context_merge[
                "closed_context_candle_alignment"
            ],
            "future_data": False,
            "backtest_started": False,
            "strategy_code_generated": False,
            "paper_trading_started": False,
            "dry_run_trading_started": False,
            "live_trading": False,
            "exchange_order_placement": False,
            "shorting": False,
            "leverage": 1.0,
            "process_control": False,
            "promotion_authorized_by_this_command": False,
            "local_artifacts_source_of_truth": True,
        },
    }


def write_edge_discovery_artifacts(
    artifact: dict[str, Any], *, root_dir: Path, output_root: Path
) -> tuple[Path, Path]:
    root = root_dir.resolve()
    edge_discovery_id = _safe_path_component(
        str(artifact.get("edge_discovery_id") or "edge_discovery")
    )
    out_dir = _resolve_inside(output_root, root) / edge_discovery_id
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "edge_discovery.json"
    report_path = out_dir / "edge_discovery_report.md"
    json_path.write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    report_path.write_text(_render_report(artifact), encoding="utf-8")
    return json_path, report_path


def _horizon_results(
    ohlcv: pd.DataFrame | None,
    events: Sequence[dict[str, Any]],
    *,
    horizons: Sequence[int],
    cost_scenarios: dict[str, dict[str, Any]],
    funding_rate: pd.DataFrame | None,
    min_sample_count: int,
    min_profitable_windows_ratio: float,
    min_calendar_window_count: int,
    min_profitable_calendar_windows_ratio: float,
    min_negative_control_delta_bps: float,
) -> list[dict[str, Any]]:
    all_in_cost_bps = _normal_cost_bps(cost_scenarios)
    if ohlcv is None or not events or all_in_cost_bps is None:
        return []
    event_frame = pd.DataFrame(events)
    if "date" not in event_frame.columns:
        return []
    event_frame = event_frame.copy()
    event_frame["date"] = pd.to_datetime(event_frame["date"], utc=True, errors="coerce")
    event_frame = event_frame.dropna(subset=["date"]).sort_values("date")
    event_return_columns = [
        "date",
        *[
            column
            for column in ("pair", "symbol", "instrument")
            if column in event_frame.columns
        ],
    ]
    results: list[dict[str, Any]] = []
    for horizon in horizons:
        rows = _event_returns(
            ohlcv,
            event_frame[event_return_columns],
            hold_candles=int(horizon),
            funding_rate=funding_rate,
            entry_semantics="next_candle_open",
        )
        expected_price_edge_bps = _mean([item["price_return_bps"] for item in rows])
        expected_funding_adjustment_bps = _mean(
            [item["funding_adjustment_bps"] for item in rows]
        )
        expected_edge_bps = _mean([item["gross_return_bps"] for item in rows])
        gross_edge_bps = expected_edge_bps
        cost_bps = _scenario_costs(cost_scenarios)
        net_edge_bps = (
            None
            if expected_edge_bps is None
            else round(expected_edge_bps - float(all_in_cost_bps), 6)
        )
        net_edge_bps_best = _net_edge(expected_edge_bps, cost_bps["best"])
        net_edge_bps_normal = _net_edge(expected_edge_bps, cost_bps["normal"])
        net_edge_bps_stress = _net_edge(expected_edge_bps, cost_bps["stress"])
        windows = _window_summaries(rows, all_in_cost_bps=float(all_in_cost_bps))
        calendar_windows = _calendar_window_summaries(
            rows,
            all_in_cost_bps=float(all_in_cost_bps),
        )
        profitable_window_count = sum(
            1 for item in windows if item["net_edge_bps"] is not None and item["net_edge_bps"] > 0
        )
        profitable_calendar_window_count = sum(
            1
            for item in calendar_windows
            if item["net_edge_bps"] is not None and item["net_edge_bps"] > 0
        )
        window_count = len(windows)
        calendar_window_count = len(calendar_windows)
        profitable_windows_ratio = (
            0.0 if window_count == 0 else round(profitable_window_count / window_count, 4)
        )
        profitable_calendar_windows_ratio = (
            0.0
            if calendar_window_count == 0
            else round(profitable_calendar_window_count / calendar_window_count, 4)
        )
        lower_confidence_bound_bps = _lower_confidence_bound_bps(
            [float(item["gross_return_bps"]) for item in rows],
            cost_bps["normal"],
        )
        pair_alignment = _event_pair_alignment_summary(ohlcv, event_frame)
        pair_price_series = _pair_price_series_summary(ohlcv)
        negative_controls = _negative_control_summary(
            ohlcv,
            event_frame[event_return_columns],
            hold_candles=int(horizon),
            funding_rate=funding_rate,
            normal_cost_bps=cost_bps["normal"],
            real_net_edge_bps=net_edge_bps_normal,
        )
        pair_evidence = _pair_evidence_summary(rows)
        pair_concentration = _effective_pair_concentration(
            pair_evidence,
            pair_price_series,
        )
        calendar_concentration = (
            None
            if calendar_window_count == 0
            else max(
                (
                    item["sample_count"] / len(rows)
                    for item in calendar_windows
                    if len(rows) > 0
                ),
                default=None,
            )
        )
        walk_forward_pass_rate = profitable_calendar_windows_ratio
        research_gate = _research_gate_summary(
            net_edge_bps_normal=net_edge_bps_normal,
            net_edge_bps_stress=net_edge_bps_stress,
            profitable_windows_ratio=profitable_windows_ratio,
            walk_forward_pass_rate=walk_forward_pass_rate,
            lower_confidence_bound_bps=lower_confidence_bound_bps,
            pair_concentration=pair_concentration,
            calendar_concentration=calendar_concentration,
            negative_controls=negative_controls,
            min_negative_control_delta_bps=min_negative_control_delta_bps,
            entry_semantics="next_candle_open",
        )
        horizon_checks = [
            _check(
                "event_sample_count_sufficient",
                len(rows) >= min_sample_count,
                {"sample_count": len(rows), "min_sample_count": min_sample_count},
            ),
            _check(
                "expected_edge_bps_available",
                expected_edge_bps is not None,
                {"expected_edge_bps": expected_edge_bps},
            ),
            _check(
                "expected_edge_exceeds_all_in_cost",
                net_edge_bps is not None and net_edge_bps > 0.0,
                {
                    "expected_edge_bps": expected_edge_bps,
                    "all_in_cost_bps": float(all_in_cost_bps),
                    "net_edge_bps": net_edge_bps,
                },
            ),
            _check(
                "profitable_windows_ratio_sufficient",
                profitable_windows_ratio >= min_profitable_windows_ratio,
                {
                    "profitable_windows_ratio": profitable_windows_ratio,
                    "min_profitable_windows_ratio": min_profitable_windows_ratio,
                },
            ),
            _check(
                "calendar_window_count_sufficient",
                min_calendar_window_count <= 0
                or calendar_window_count >= min_calendar_window_count,
                {
                    "calendar_window_count": calendar_window_count,
                    "min_calendar_window_count": min_calendar_window_count,
                },
            ),
            _check(
                "profitable_calendar_windows_ratio_sufficient",
                min_profitable_calendar_windows_ratio <= 0.0
                or profitable_calendar_windows_ratio
                >= min_profitable_calendar_windows_ratio,
                {
                    "profitable_calendar_windows_ratio": profitable_calendar_windows_ratio,
                    "min_profitable_calendar_windows_ratio": (
                        min_profitable_calendar_windows_ratio
                    ),
                },
            ),
            _check(
                "event_pair_labels_present_for_multi_instrument_ohlcv",
                not (
                    pair_alignment["ohlcv_multi_instrument"]
                    and pair_alignment["missing_event_label_count"] > 0
                ),
                pair_alignment,
            ),
            _check(
                "event_pair_labels_match_ohlcv",
                pair_alignment["unmatched_event_label_count"] == 0,
                pair_alignment,
            ),
        ]
        results.append(
            {
                "hold_candles": int(horizon),
                "status": (
                    "passed"
                    if all(check["status"] == "pass" for check in horizon_checks)
                    else "failed"
                ),
                "sample_count": len(rows),
                "expected_price_edge_bps": expected_price_edge_bps,
                "expected_funding_adjustment_bps": expected_funding_adjustment_bps,
                "expected_edge_bps": expected_edge_bps,
                "gross_edge_bps": gross_edge_bps,
                "median_edge_bps": _median([item["gross_return_bps"] for item in rows]),
                "cost_bps_best": cost_bps["best"],
                "cost_bps_normal": cost_bps["normal"],
                "cost_bps_stress": cost_bps["stress"],
                "all_in_cost_bps": float(all_in_cost_bps),
                "net_edge_bps": net_edge_bps,
                "net_edge_bps_best": net_edge_bps_best,
                "net_edge_bps_normal": net_edge_bps_normal,
                "net_edge_bps_stress": net_edge_bps_stress,
                "win_rate": _win_rate([item["gross_return_bps"] for item in rows]),
                "window_count": window_count,
                "profitable_window_count": profitable_window_count,
                "profitable_windows_ratio": profitable_windows_ratio,
                "calendar_window_frequency": "quarter",
                "calendar_window_count": calendar_window_count,
                "profitable_calendar_window_count": profitable_calendar_window_count,
                "profitable_calendar_windows_ratio": profitable_calendar_windows_ratio,
                "walk_forward_pass_rate": walk_forward_pass_rate,
                "lower_confidence_bound_bps": lower_confidence_bound_bps,
                "pair_concentration": pair_concentration,
                "pair_evidence_count": pair_evidence["pair_evidence_count"],
                "pair_evidence_unique_count": pair_evidence[
                    "pair_evidence_unique_count"
                ],
                "pair_evidence_distribution": pair_evidence[
                    "pair_evidence_distribution"
                ],
                "pair_alignment": pair_alignment,
                "pair_price_series": pair_price_series,
                "calendar_concentration": calendar_concentration,
                "negative_controls": negative_controls,
                "negative_control_random_entry_delta_bps": negative_controls[
                    "random_entry"
                ]["delta_bps"],
                "negative_control_shuffled_signal_delta_bps": negative_controls[
                    "shuffled_signal"
                ]["delta_bps"],
                "negative_control_shifted_signal_delta_bps": negative_controls[
                    "shifted_signal"
                ]["delta_bps"],
                "entry_semantics": "next_candle_open",
                "passes_research_gate": research_gate["passes_research_gate"],
                "rejection_reason": research_gate["rejection_reason"],
                "research_gate": research_gate,
                "window_summaries": windows,
                "calendar_window_summaries": calendar_windows,
                "checks": horizon_checks,
                "blockers": [
                    check for check in horizon_checks if check["status"] != "pass"
                ],
                "sample_preview": rows[:5],
            }
        )
    return results


def _event_pair_alignment_summary(
    ohlcv: pd.DataFrame,
    events: pd.DataFrame,
) -> dict[str, Any]:
    ohlcv_column = _instrument_column(ohlcv)
    ohlcv_labels = _instrument_label_set(ohlcv, ohlcv_column)
    event_labels: list[str] = []
    missing_event_label_count = 0
    for event in events.to_dict("records"):
        label, _column = _event_pair_label_and_column(event)
        if label is None:
            missing_event_label_count += 1
        else:
            event_labels.append(label)
    unmatched = sorted({label for label in event_labels if label not in ohlcv_labels})
    return {
        "ohlcv_instrument_column": ohlcv_column,
        "ohlcv_instrument_count": len(ohlcv_labels),
        "ohlcv_multi_instrument": len(ohlcv_labels) > 1,
        "event_count": int(len(events)),
        "event_pair_evidence_count": len(event_labels),
        "event_pair_evidence_unique_count": len(set(event_labels)),
        "missing_event_label_count": missing_event_label_count,
        "unmatched_event_label_count": len(unmatched),
        "unmatched_event_labels": unmatched,
    }


def _pair_price_series_summary(ohlcv: pd.DataFrame) -> dict[str, Any]:
    column = _instrument_column(ohlcv)
    if column is None:
        return {
            "ohlcv_instrument_column": None,
            "ohlcv_instrument_count": 0,
            "shared_timestamp_count": 0,
            "multi_instrument_price_series_aligned": False,
        }
    frame = ohlcv[["date", column]].copy()
    frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="coerce")
    frame[column] = frame[column].map(_string_label)
    frame = frame.dropna(subset=["date", column])
    if frame.empty:
        return {
            "ohlcv_instrument_column": column,
            "ohlcv_instrument_count": 0,
            "shared_timestamp_count": 0,
            "multi_instrument_price_series_aligned": False,
        }
    instrument_count = int(frame[column].nunique())
    shared_timestamp_count = int(
        (frame.groupby("date")[column].nunique() >= 2).sum()
    )
    return {
        "ohlcv_instrument_column": column,
        "ohlcv_instrument_count": instrument_count,
        "shared_timestamp_count": shared_timestamp_count,
        "multi_instrument_price_series_aligned": (
            instrument_count > 1 and shared_timestamp_count > 0
        ),
    }


def _effective_pair_concentration(
    pair_evidence: dict[str, Any],
    pair_price_series: dict[str, Any],
) -> float:
    concentration = float(pair_evidence["pair_concentration"])
    if (
        int(pair_evidence["pair_evidence_unique_count"]) > 1
        and not pair_price_series["multi_instrument_price_series_aligned"]
    ):
        return 1.0
    return concentration


def _negative_control_summary(
    ohlcv: pd.DataFrame,
    events: pd.DataFrame,
    *,
    hold_candles: int,
    funding_rate: pd.DataFrame | None,
    normal_cost_bps: float,
    real_net_edge_bps: float | None,
) -> dict[str, Any]:
    event_count = int(len(events))
    random_events = _control_events(
        ohlcv,
        events,
        hold_candles=hold_candles,
        event_count=event_count,
        mode="random",
    )
    shuffled_events = _control_events(
        ohlcv,
        events,
        hold_candles=hold_candles,
        event_count=event_count,
        mode="shuffled",
    )
    shifted_past_events = _control_events(
        ohlcv,
        events,
        hold_candles=hold_candles,
        event_count=event_count,
        mode="shifted_past",
    )
    shifted_future_events = _control_events(
        ohlcv,
        events,
        hold_candles=hold_candles,
        event_count=event_count,
        mode="shifted_future",
    )
    random_entry = _control_edge(
        ohlcv,
        random_events,
        hold_candles=hold_candles,
        funding_rate=funding_rate,
        normal_cost_bps=normal_cost_bps,
        real_net_edge_bps=real_net_edge_bps,
    )
    shuffled_signal = _control_edge(
        ohlcv,
        shuffled_events,
        hold_candles=hold_candles,
        funding_rate=funding_rate,
        normal_cost_bps=normal_cost_bps,
        real_net_edge_bps=real_net_edge_bps,
    )
    shifted_past = _control_edge(
        ohlcv,
        shifted_past_events,
        hold_candles=hold_candles,
        funding_rate=funding_rate,
        normal_cost_bps=normal_cost_bps,
        real_net_edge_bps=real_net_edge_bps,
    )
    shifted_future = _control_edge(
        ohlcv,
        shifted_future_events,
        hold_candles=hold_candles,
        funding_rate=funding_rate,
        normal_cost_bps=normal_cost_bps,
        real_net_edge_bps=real_net_edge_bps,
    )
    shifted_candidates = [
        item
        for item in (shifted_past, shifted_future)
        if item.get("net_edge_bps_normal") is not None
    ]
    shifted_signal = (
        max(shifted_candidates, key=lambda item: float(item["net_edge_bps_normal"]))
        if shifted_candidates
        else shifted_future
    )
    shifted_signal = {
        **shifted_signal,
        "past": shifted_past,
        "future": shifted_future,
    }
    return {
        "random_entry": random_entry,
        "shuffled_signal": shuffled_signal,
        "shifted_signal": shifted_signal,
    }


def _control_events(
    ohlcv: pd.DataFrame,
    events: pd.DataFrame,
    *,
    hold_candles: int,
    event_count: int,
    mode: str,
) -> list[dict[str, Any]]:
    if event_count <= 0:
        return []
    event_records = events.to_dict("records")
    grouped: dict[tuple[str | None, str | None], list[dict[str, Any]]] = {}
    ohlcv_multi_instrument = _is_multi_instrument_ohlcv(ohlcv)
    for event in event_records:
        label, column = _event_pair_label_and_column(event)
        if ohlcv_multi_instrument and label is None:
            continue
        grouped.setdefault((label, column), []).append(event)
    controls: list[dict[str, Any]] = []
    for (label, column), group in grouped.items():
        eligible = _eligible_control_dates(
            ohlcv,
            label=label,
            hold_candles=hold_candles,
        )
        if not eligible:
            continue
        count = min(len(group), len(eligible))
        rng = random.Random(
            f"bot-factory:{mode}:{hold_candles}:{event_count}:{len(ohlcv)}:{label or 'none'}"
        )
        if mode == "random":
            selected = sorted(rng.sample(eligible, count))
        elif mode == "shuffled":
            mask = [0] * len(eligible)
            for index in _event_indices_in_eligible(group, eligible)[: len(eligible)]:
                mask[index] = 1
            rng.shuffle(mask)
            selected = [eligible[index] for index, flag in enumerate(mask) if flag][:count]
        else:
            shift = int(hold_candles)
            indices = _event_indices_in_eligible(group, eligible)
            if mode == "shifted_past":
                shifted = [max(0, index - shift) for index in indices]
            elif mode == "shifted_future":
                shifted = [min(len(eligible) - 1, index + shift) for index in indices]
            else:
                shifted = []
            selected = [eligible[index] for index in _dedupe_ints(shifted)[:count]]
        controls.extend(_control_event(date, label=label, column=column) for date in selected)
    return sorted(controls, key=lambda item: str(item["date"]))


def _eligible_control_dates(
    ohlcv: pd.DataFrame, *, label: str | None, hold_candles: int
) -> list[Any]:
    frame = _price_frame_for_label(ohlcv, label)
    if frame.empty:
        return []
    max_start = max(0, len(frame) - int(hold_candles) - 1)
    return list(frame["date"].iloc[:max_start])


def _event_indices_in_eligible(
    events: Sequence[dict[str, Any]], eligible: Sequence[Any]
) -> list[int]:
    if not eligible:
        return []
    eligible_series = pd.Series(pd.to_datetime(list(eligible), utc=True, errors="coerce"))
    indices: list[int] = []
    for event in events:
        event_time = pd.to_datetime(event.get("date"), utc=True, errors="coerce")
        if pd.isna(event_time):
            continue
        index = int(eligible_series.searchsorted(event_time, side="left"))
        if 0 <= index < len(eligible):
            indices.append(index)
    return indices


def _dedupe_ints(values: Sequence[int]) -> list[int]:
    deduped: list[int] = []
    for value in values:
        if value not in deduped:
            deduped.append(value)
    return deduped


def _control_event(date: Any, *, label: str | None, column: str | None) -> dict[str, Any]:
    event: dict[str, Any] = {"date": date}
    if label is not None:
        event[column or "pair"] = label
    return event


def _control_edge(
    ohlcv: pd.DataFrame,
    events: Sequence[dict[str, Any]],
    *,
    hold_candles: int,
    funding_rate: pd.DataFrame | None,
    normal_cost_bps: float,
    real_net_edge_bps: float | None,
) -> dict[str, Any]:
    event_frame = pd.DataFrame(list(events))
    if event_frame.empty:
        event_frame = pd.DataFrame({"date": []})
    rows = _event_returns(
        ohlcv,
        event_frame,
        hold_candles=hold_candles,
        funding_rate=funding_rate,
        entry_semantics="next_candle_open",
    )
    pair_evidence = _pair_evidence_summary(rows)
    gross_edge_bps = _mean([item["gross_return_bps"] for item in rows])
    net_edge_bps = _net_edge(gross_edge_bps, normal_cost_bps)
    delta = (
        None
        if real_net_edge_bps is None or net_edge_bps is None
        else round(float(real_net_edge_bps) - float(net_edge_bps), 6)
    )
    return {
        "sample_count": len(rows),
        "gross_edge_bps": gross_edge_bps,
        "net_edge_bps_normal": net_edge_bps,
        "delta_bps": delta,
        "pair_evidence_count": pair_evidence["pair_evidence_count"],
        "pair_evidence_unique_count": pair_evidence["pair_evidence_unique_count"],
        "pair_evidence_distribution": pair_evidence["pair_evidence_distribution"],
        "sample_preview": rows[:5],
    }


def _research_gate_summary(
    *,
    net_edge_bps_normal: float | None,
    net_edge_bps_stress: float | None,
    profitable_windows_ratio: float,
    walk_forward_pass_rate: float,
    lower_confidence_bound_bps: float | None,
    pair_concentration: float,
    calendar_concentration: float | None,
    negative_controls: dict[str, Any],
    min_negative_control_delta_bps: float,
    entry_semantics: str,
) -> dict[str, Any]:
    shifted = negative_controls.get("shifted_signal", {})
    checks = [
        _check(
            "net_edge_bps_normal_at_least_6",
            net_edge_bps_normal is not None and net_edge_bps_normal >= 6.0,
            {"net_edge_bps_normal": net_edge_bps_normal, "minimum": 6.0},
        ),
        _check(
            "net_edge_bps_stress_positive",
            net_edge_bps_stress is not None and net_edge_bps_stress > 0.0,
            {"net_edge_bps_stress": net_edge_bps_stress},
        ),
        _check(
            "profitable_windows_ratio_at_least_0_7",
            profitable_windows_ratio >= 0.7,
            {"profitable_windows_ratio": profitable_windows_ratio, "minimum": 0.7},
        ),
        _check(
            "walk_forward_pass_rate_at_least_0_6",
            walk_forward_pass_rate >= 0.6,
            {"walk_forward_pass_rate": walk_forward_pass_rate, "minimum": 0.6},
        ),
        _check(
            "lower_confidence_bound_bps_positive",
            lower_confidence_bound_bps is not None and lower_confidence_bound_bps > 0.0,
            {"lower_confidence_bound_bps": lower_confidence_bound_bps},
        ),
        _check(
            "not_single_pair_dependent",
            pair_concentration < 1.0,
            {"pair_concentration": pair_concentration},
        ),
        _check(
            "not_single_calendar_window_dependent",
            calendar_concentration is not None and calendar_concentration < 1.0,
            {"calendar_concentration": calendar_concentration},
        ),
        _negative_control_check(
            "random_entry_control_beaten",
            negative_controls.get("random_entry", {}),
            min_negative_control_delta_bps,
        ),
        _negative_control_check(
            "shuffled_signal_control_beaten",
            negative_controls.get("shuffled_signal", {}),
            min_negative_control_delta_bps,
        ),
        _negative_control_check(
            "shifted_signal_control_beaten",
            shifted,
            min_negative_control_delta_bps,
        ),
        _check(
            "freqtrade_semantics_next_candle_open",
            entry_semantics == "next_candle_open",
            {"entry_semantics": entry_semantics},
        ),
    ]
    blockers = [check for check in checks if check["status"] != "pass"]
    reasons = [check["name"] for check in blockers]
    return {
        "passes_research_gate": not blockers,
        "checks": checks,
        "blockers": blockers,
        "rejection_reasons": reasons,
        "rejection_reason": "; ".join(reasons) if reasons else None,
        "candidate_generation_result": (
            "candidate generation allowed" if not blockers else "no candidate generated"
        ),
    }


def _negative_control_check(
    name: str, control: dict[str, Any], min_negative_control_delta_bps: float
) -> dict[str, Any]:
    delta = _float_or_none(control.get("delta_bps"))
    return _check(
        name,
        delta is not None and delta >= float(min_negative_control_delta_bps),
        {
            "delta_bps": delta,
            "min_negative_control_delta_bps": float(min_negative_control_delta_bps),
            "control_net_edge_bps_normal": control.get("net_edge_bps_normal"),
            "control_sample_count": control.get("sample_count"),
        },
    )


def _event_level_post_cost_report(
    spec: dict[str, Any],
    *,
    best_horizon: dict[str, Any] | None,
    horizon_results: Sequence[dict[str, Any]],
    concentration: dict[str, Any],
) -> dict[str, Any]:
    selected = None
    passing_gate = [item for item in horizon_results if item.get("passes_research_gate")]
    if passing_gate:
        selected = max(
            passing_gate,
            key=lambda item: float(item.get("net_edge_bps_normal") or -1e9),
        )
    elif best_horizon is not None:
        selected = next(
            (
                item
                for item in horizon_results
                if item.get("hold_candles") == best_horizon.get("hold_candles")
            ),
            None,
        )
    if selected is None:
        gate = _research_gate_summary(
            net_edge_bps_normal=None,
            net_edge_bps_stress=None,
            profitable_windows_ratio=0.0,
            walk_forward_pass_rate=0.0,
            lower_confidence_bound_bps=None,
            pair_concentration=1.0,
            calendar_concentration=concentration.get("max_quarter_event_share"),
            negative_controls={
                "random_entry": {},
                "shuffled_signal": {},
                "shifted_signal": {},
            },
            min_negative_control_delta_bps=1.0,
            entry_semantics="next_candle_open",
        )
        return {
            "thesis_id": str(spec.get("thesis_id") or "").strip(),
            "mechanism_class": str(spec.get("mechanism_class") or "").strip(),
            "event_count": 0,
            "entry_signal_count": 0,
            "passes_research_gate": False,
            "rejection_reason": gate["rejection_reason"],
            "pair_concentration": 1.0,
            "pair_evidence_count": 0,
            "pair_evidence_unique_count": 0,
            "pair_alignment": {},
            "pair_price_series": {},
            "research_gate": gate,
            "candidate_generation_result": "no candidate generated",
        }
    gate = selected["research_gate"]
    return {
        "thesis_id": str(spec.get("thesis_id") or "").strip(),
        "mechanism_class": str(spec.get("mechanism_class") or "").strip(),
        "event_count": selected.get("sample_count", 0),
        "entry_signal_count": selected.get("sample_count", 0),
        "gross_edge_bps": selected.get("gross_edge_bps"),
        "cost_bps_best": selected.get("cost_bps_best"),
        "cost_bps_normal": selected.get("cost_bps_normal"),
        "cost_bps_stress": selected.get("cost_bps_stress"),
        "net_edge_bps_best": selected.get("net_edge_bps_best"),
        "net_edge_bps_normal": selected.get("net_edge_bps_normal"),
        "net_edge_bps_stress": selected.get("net_edge_bps_stress"),
        "profitable_windows_ratio": selected.get("profitable_windows_ratio"),
        "walk_forward_pass_rate": selected.get("walk_forward_pass_rate"),
        "lower_confidence_bound_bps": selected.get("lower_confidence_bound_bps"),
        "pair_concentration": selected.get("pair_concentration"),
        "pair_evidence_count": selected.get("pair_evidence_count"),
        "pair_evidence_unique_count": selected.get("pair_evidence_unique_count"),
        "pair_evidence_distribution": selected.get("pair_evidence_distribution"),
        "pair_alignment": selected.get("pair_alignment"),
        "pair_price_series": selected.get("pair_price_series"),
        "calendar_concentration": selected.get("calendar_concentration"),
        "holding_period": selected.get("hold_candles"),
        "negative_control_random_entry_delta_bps": selected.get(
            "negative_control_random_entry_delta_bps"
        ),
        "negative_control_shuffled_signal_delta_bps": selected.get(
            "negative_control_shuffled_signal_delta_bps"
        ),
        "negative_control_shifted_signal_delta_bps": selected.get(
            "negative_control_shifted_signal_delta_bps"
        ),
        "passes_research_gate": gate["passes_research_gate"],
        "rejection_reason": gate["rejection_reason"],
        "research_gate": gate,
        "candidate_generation_result": gate["candidate_generation_result"],
    }


def _scenario_costs(cost_scenarios: dict[str, dict[str, Any]]) -> dict[str, float]:
    return {
        name: float(cost_scenarios.get(name, {}).get("total_cost_bps") or 0.0)
        for name in ("best", "normal", "stress")
    }


def _normal_cost_bps(cost_scenarios: dict[str, dict[str, Any]]) -> float | None:
    value = cost_scenarios.get("normal", {}).get("total_cost_bps")
    return _float_or_none(value)


def _net_edge(gross_edge_bps: float | None, cost_bps: float) -> float | None:
    return None if gross_edge_bps is None else round(float(gross_edge_bps) - cost_bps, 6)


def _lower_confidence_bound_bps(values: Sequence[float], cost_bps: float) -> float | None:
    numbers = [float(value) for value in values]
    if not numbers:
        return None
    mean = sum(numbers) / len(numbers)
    if len(numbers) < 2:
        return round(mean - cost_bps, 6)
    variance = sum((value - mean) ** 2 for value in numbers) / (len(numbers) - 1)
    standard_error = sqrt(variance) / sqrt(len(numbers))
    return round(mean - cost_bps - 1.96 * standard_error, 6)


def _pair_evidence_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    labels = [
        label
        for label in (_pair_evidence_label(row) for row in rows)
        if label is not None
    ]
    if not labels:
        return {
            "pair_concentration": 1.0,
            "pair_evidence_count": 0,
            "pair_evidence_unique_count": 0,
            "pair_evidence_distribution": {},
        }
    counts = Counter(labels)
    return {
        "pair_concentration": round(max(counts.values()) / len(labels), 4),
        "pair_evidence_count": len(labels),
        "pair_evidence_unique_count": len(counts),
        "pair_evidence_distribution": dict(sorted(counts.items())),
    }


def _pair_evidence_label(row: dict[str, Any]) -> str | None:
    label, _column = _event_pair_label_and_column(row)
    return label


def _event_pair_label_and_column(row: dict[str, Any]) -> tuple[str | None, str | None]:
    for column in ("pair", "symbol", "instrument"):
        label = _string_label(row.get(column))
        if label is not None:
            return label, column
    return None, None


def _string_label(value: Any) -> str | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return text or None


def _instrument_column(frame: pd.DataFrame) -> str | None:
    for column in ("pair", "symbol", "instrument"):
        if column in frame.columns:
            return column
    return None


def _instrument_label_set(
    frame: pd.DataFrame, column: str | None = None
) -> set[str]:
    column = column or _instrument_column(frame)
    if column is None:
        return set()
    return {
        label
        for label in frame[column].map(_string_label)
        if label is not None
    }


def _is_multi_instrument_ohlcv(frame: pd.DataFrame) -> bool:
    return len(_instrument_label_set(frame)) > 1


def _price_frame_for_label(
    ohlcv: pd.DataFrame, label: str | None
) -> pd.DataFrame:
    frame = ohlcv.copy()
    frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["date"]).sort_values("date")
    if label is None:
        if _is_multi_instrument_ohlcv(frame):
            return frame.iloc[0:0].copy()
        return frame.reset_index(drop=True)
    column = _instrument_column(frame)
    if column is None:
        return frame.iloc[0:0].copy()
    labels = frame[column].map(_string_label)
    subset = frame[labels == label]
    return subset.sort_values("date").reset_index(drop=True)


def _cost_scenario_checks(
    cost_scenarios: dict[str, dict[str, Any]], context: CostModelContext
) -> list[dict[str, Any]]:
    names = set(cost_scenarios)
    normal = _normal_cost_bps(cost_scenarios)
    stress = _float_or_none(cost_scenarios.get("stress", {}).get("total_cost_bps"))
    maker_context = str(context.order_type or "").strip().lower() == "maker"
    scenario_values = list(cost_scenarios.values())
    return [
        _check(
            "cost_scenarios_best_normal_stress_present",
            {"best", "normal", "stress"} <= names,
            {"scenario_names": sorted(names)},
        ),
        _check(
            "cost_scenario_normal_cost_non_negative",
            normal is not None and normal >= 0.0,
            {"normal_total_cost_bps": normal},
        ),
        _check(
            "cost_scenario_stress_not_less_than_normal",
            normal is not None and stress is not None and stress >= normal,
            {"normal_total_cost_bps": normal, "stress_total_cost_bps": stress},
        ),
        _check(
            "maker_cost_model_includes_fill_risk",
            not maker_context
            or all(
                _float_or_none(item.get("no_fill_rate")) is not None
                and _float_or_none(item.get("partial_fill_rate")) is not None
                and _float_or_none(item.get("adverse_selection_bps")) is not None
                for item in scenario_values
            ),
            {
                "order_type": context.order_type,
                "scenario_names": sorted(names),
            },
        ),
    ]


def _cost_model_context_summary(context: CostModelContext) -> dict[str, Any]:
    return {
        "pair": context.pair,
        "timeframe": context.timeframe,
        "order_type": context.order_type,
        "liquidity_tier": context.liquidity_tier,
        "volatility_regime": context.volatility_regime,
    }


def _quality_report_summaries(paths: Sequence[Path], *, root: Path) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for path in paths:
        payload, error = _load_json(path)
        payload = payload if isinstance(payload, dict) else {}
        raw_reports = payload.get("reports")
        child_reports = raw_reports if isinstance(raw_reports, list) else []
        child_reports_ok = all(
            bool(item.get("ok")) for item in child_reports if isinstance(item, dict)
        )
        summaries.append(
            {
                "path": _rel(path, root),
                "file_present": path.is_file(),
                "parseable": bool(payload) and error is None,
                "error": error,
                "ok": bool(payload.get("ok")) and child_reports_ok,
                "report_count": len(child_reports),
                "rows": _int_or_none(payload.get("rows")),
                "start": payload.get("start"),
                "end": payload.get("end"),
                "finding_count": len(payload.get("findings") or [])
                if isinstance(payload.get("findings"), list)
                else 0,
            }
        )
    return summaries


def _edge_cost_bps(spec: dict[str, Any]) -> float | None:
    return _normal_cost_bps(
        cost_scenarios_from_spec(spec, context=cost_context_from_spec(spec))
    )


def _edge_horizons(spec: dict[str, Any]) -> list[int]:
    raw = spec.get("horizons", spec.get("hold_candles"))
    if raw is None:
        return []
    raw_items = raw if isinstance(raw, list) else [raw]
    horizons: list[int] = []
    for item in raw_items:
        value = item.get("hold_candles") if isinstance(item, dict) else item
        horizon = _int_or_none(value)
        if horizon is not None and horizon > 0 and horizon not in horizons:
            horizons.append(horizon)
    return horizons


def _anti_parameter_search_summary(spec: dict[str, Any]) -> dict[str, Any]:
    violations: list[dict[str, Any]] = []
    for key in _PARAMETER_SEARCH_KEYS:
        if key in spec:
            violations.append({"location": "edge_spec", "key": key})
    for index, condition in enumerate(spec.get("conditions") or [], start=1):
        if not isinstance(condition, dict):
            continue
        for key in _PARAMETER_SEARCH_KEYS:
            if key in condition:
                violations.append({"location": f"conditions[{index}]", "key": key})
    return {
        "valid": not violations,
        "policy": (
            "edge_discovery_accepts_fixed_theory_named_conditions_only; "
            "threshold grids and parameter search artifacts are exploratory "
            "and cannot promote to proposal/codegen evidence"
        ),
        "violations": violations,
    }


def _hypothesis_scope(spec: dict[str, Any]) -> str:
    raw = spec.get("hypothesis_scope", spec.get("edge_scope", "single_asset"))
    return _scope_label(raw)


def _instrument_universe(spec: dict[str, Any]) -> list[str]:
    raw = spec.get("instrument_universe", spec.get("pairs", spec.get("assets", [])))
    if isinstance(raw, str):
        raw_items: Sequence[Any] = [raw]
    elif isinstance(raw, list):
        raw_items = raw
    else:
        raw_items = []
    values: list[str] = []
    for item in raw_items:
        if isinstance(item, dict):
            value = (
                item.get("pair")
                or item.get("symbol")
                or item.get("asset")
                or item.get("instrument")
            )
        else:
            value = item
        text = str(value or "").strip()
        if text and text not in values:
            values.append(text)
    return values


def _market_structure_domains(spec: dict[str, Any], hypothesis_scope: str) -> list[str]:
    raw = spec.get(
        "market_structure_domains",
        spec.get("market_structure_domain", spec.get("data_classes", [])),
    )
    if isinstance(raw, str):
        raw_items: Sequence[Any] = [raw]
    elif isinstance(raw, list):
        raw_items = raw
    else:
        raw_items = []
    values = [_scope_label(item) for item in raw_items if str(item or "").strip()]
    values = list(dict.fromkeys(value for value in values if value))
    if values:
        return values
    if hypothesis_scope in {"funding_basis", "microstructure"}:
        return [hypothesis_scope]
    return ["ohlcv"]


def _scope_label(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = text.replace("-", "_").replace("/", "_").replace(" ", "_")
    while "__" in text:
        text = text.replace("__", "_")
    if text in {"funding", "basis", "funding_rate", "carry"}:
        return "funding_basis"
    return text or "single_asset"


def _safe_edge_spec_summary(spec: dict[str, Any]) -> dict[str, Any]:
    summary = _safe_spec_summary(
        {
            "factory": spec.get("factory"),
            "event_id": spec.get("edge_discovery_id"),
            "thesis_id": spec.get("thesis_id"),
            "mechanism_class": spec.get("mechanism_class"),
            "hypothesis_scope": _hypothesis_scope(spec),
            "instrument_universe": _instrument_universe(spec),
            "market_structure_domains": _market_structure_domains(
                spec, _hypothesis_scope(spec)
            ),
            "conditions": spec.get("conditions", []),
            "cooldown_candles": _cooldown_candles(spec),
        }
    )
    summary["edge_discovery_id"] = spec.get("edge_discovery_id")
    summary["horizons"] = _edge_horizons(spec)
    summary["all_in_cost_bps"] = _edge_cost_bps(spec)
    summary["cost_scenarios"] = cost_scenarios_from_spec(
        spec,
        context=cost_context_from_spec(spec),
    )
    summary["hypothesis"] = spec.get("hypothesis")
    summary["data_classes"] = spec.get("data_classes", spec.get("required_data_classes", []))
    return summary


def _ohlcv_coverage(frame: pd.DataFrame | None) -> dict[str, Any]:
    if frame is None or frame.empty:
        return {
            "row_count": 0,
            "data_start": None,
            "data_end": None,
            "data_span_days": None,
        }
    start = pd.Timestamp(frame["date"].iloc[0])
    end = pd.Timestamp(frame["date"].iloc[-1])
    return {
        "row_count": int(len(frame)),
        "data_start": _timestamp_to_str(start),
        "data_end": _timestamp_to_str(end),
        "data_span_days": round((end - start).total_seconds() / 86400.0, 6),
    }


def _best_horizon(horizon_results: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    ranked = [
        item
        for item in horizon_results
        if item.get("net_edge_bps") is not None and item.get("sample_count", 0) > 0
    ]
    if not ranked:
        return None
    best = max(ranked, key=lambda item: float(item["net_edge_bps"]))
    return {
        "hold_candles": best["hold_candles"],
        "status": best["status"],
        "net_edge_bps": best["net_edge_bps"],
        "net_edge_bps_best": best.get("net_edge_bps_best"),
        "net_edge_bps_normal": best.get("net_edge_bps_normal"),
        "net_edge_bps_stress": best.get("net_edge_bps_stress"),
        "lower_confidence_bound_bps": best.get("lower_confidence_bound_bps"),
        "walk_forward_pass_rate": best.get("walk_forward_pass_rate"),
        "passes_research_gate": best.get("passes_research_gate") is True,
        "rejection_reason": best.get("rejection_reason"),
        "sample_count": best["sample_count"],
        "profitable_windows_ratio": best["profitable_windows_ratio"],
        "profitable_calendar_windows_ratio": best["profitable_calendar_windows_ratio"],
    }


def _event_concentration_diagnostics(events: Sequence[dict[str, Any]]) -> dict[str, Any]:
    raw_dates = [item.get("date") for item in events if isinstance(item, dict)]
    dates = pd.Series(pd.to_datetime(raw_dates, utc=True, errors="coerce")).dropna()
    parsed_count = int(len(dates))
    if parsed_count == 0:
        empty_bucket = _bucket_concentration(pd.Series(dtype="object"))
        return {
            "event_count": len(events),
            "date_parseable_event_count": 0,
            "active_day_count": 0,
            "active_week_count": 0,
            "active_month_count": 0,
            "active_quarter_count": 0,
            "max_day_event_share": None,
            "max_week_event_share": None,
            "max_month_event_share": None,
            "max_quarter_event_share": None,
            "top_day": empty_bucket,
            "top_week": empty_bucket,
            "top_month": empty_bucket,
            "top_quarter": empty_bucket,
        }

    naive_dates = dates.dt.tz_convert("UTC").dt.tz_localize(None)
    day_labels = naive_dates.dt.strftime("%Y-%m-%d")
    week_labels = naive_dates.dt.strftime("%G-W%V")
    month_labels = naive_dates.dt.strftime("%Y-%m")
    quarter_labels = naive_dates.dt.to_period("Q").astype(str)
    day = _bucket_concentration(day_labels)
    week = _bucket_concentration(week_labels)
    month = _bucket_concentration(month_labels)
    quarter = _bucket_concentration(quarter_labels)
    return {
        "event_count": len(events),
        "date_parseable_event_count": parsed_count,
        "active_day_count": day["bucket_count"],
        "active_week_count": week["bucket_count"],
        "active_month_count": month["bucket_count"],
        "active_quarter_count": quarter["bucket_count"],
        "max_day_event_share": day["share"],
        "max_week_event_share": week["share"],
        "max_month_event_share": month["share"],
        "max_quarter_event_share": quarter["share"],
        "top_day": day,
        "top_week": week,
        "top_month": month,
        "top_quarter": quarter,
    }


def _bucket_concentration(labels: pd.Series) -> dict[str, Any]:
    if labels.empty:
        return {
            "bucket": None,
            "event_count": 0,
            "share": None,
            "bucket_count": 0,
        }
    counts = labels.value_counts()
    top_bucket = str(counts.index[0])
    top_count = int(counts.iloc[0])
    total = int(counts.sum())
    return {
        "bucket": top_bucket,
        "event_count": top_count,
        "share": round(top_count / total, 4) if total else None,
        "bucket_count": int(len(counts)),
    }


def _blocked_next_actions(
    status: str, *, candidate_generation_allowed: bool = False
) -> list[str]:
    actions = ["strategy_codegen_directly_from_edge_discovery"]
    if not candidate_generation_allowed:
        actions.append("proposal_generation_without_passing_research_gate")
    if status != "passed":
        actions.extend(
            [
                "proposal_generation_from_unpassed_edge_discovery",
                "research_selection_promotion_from_unpassed_edge_discovery",
                "parameter_only_threshold_loosen_after_failed_edge_discovery",
            ]
        )
    return actions


def _render_report(artifact: dict[str, Any]) -> str:
    research_gate = artifact.get("research_gate", {})
    edge_report = artifact.get("event_level_post_cost_edge_report", {})
    lines = [
        "# Edge Discovery Evidence",
        "",
        f"- edge_discovery_id: {artifact.get('edge_discovery_id')}",
        f"- status: {artifact.get('status')}",
        f"- thesis_id: {artifact.get('thesis_id')}",
        f"- mechanism_class: {artifact.get('mechanism_class')}",
        f"- hypothesis_scope: {artifact.get('hypothesis_scope')}",
        "- instrument_universe: "
        + ", ".join(artifact.get("instrument_universe", []) or ["None"]),
        "- market_structure_domains: "
        + ", ".join(artifact.get("market_structure_domains", []) or ["None"]),
        f"- event_count: {artifact.get('event_count')}",
        f"- passing_horizon_count: {artifact.get('passing_horizon_count')}",
        f"- proposal_generation_allowed: {artifact.get('proposal_generation_allowed')}",
        f"- strategy_codegen_allowed: {artifact.get('strategy_codegen_allowed')}",
        f"- candidate_generation_allowed: {artifact.get('candidate_generation_allowed')}",
        f"- candidate_generation_result: {artifact.get('candidate_generation_result')}",
        f"- passes_research_gate: {research_gate.get('passes_research_gate')}",
        f"- rejection_reason: {research_gate.get('rejection_reason')}",
        "",
        "## Cost Scenarios",
        "",
    ]
    for name, scenario in (artifact.get("cost_scenarios") or {}).items():
        if not isinstance(scenario, dict):
            continue
        lines.append(
            f"- {name}: total_cost_bps={scenario.get('total_cost_bps')}, "
            f"fee_entry={scenario.get('fee_bps_entry')}, "
            f"fee_exit={scenario.get('fee_bps_exit')}, "
            f"spread={scenario.get('spread_bps')}, "
            f"slippage_entry={scenario.get('slippage_bps_entry')}, "
            f"slippage_exit={scenario.get('slippage_bps_exit')}, "
            f"adverse_selection={scenario.get('adverse_selection_bps')}, "
            f"no_fill_rate={scenario.get('no_fill_rate')}, "
            f"partial_fill_rate={scenario.get('partial_fill_rate')}"
        )
    lines.extend(
        [
            "",
            "## Event-Level Post-Cost Report",
            "",
            f"- thesis_id: {edge_report.get('thesis_id')}",
            f"- mechanism_class: {edge_report.get('mechanism_class')}",
            f"- event_count: {edge_report.get('event_count')}",
            f"- entry_signal_count: {edge_report.get('entry_signal_count')}",
            f"- gross_edge_bps: {edge_report.get('gross_edge_bps')}",
            f"- cost_bps_best: {edge_report.get('cost_bps_best')}",
            f"- cost_bps_normal: {edge_report.get('cost_bps_normal')}",
            f"- cost_bps_stress: {edge_report.get('cost_bps_stress')}",
            f"- net_edge_bps_best: {edge_report.get('net_edge_bps_best')}",
            f"- net_edge_bps_normal: {edge_report.get('net_edge_bps_normal')}",
            f"- net_edge_bps_stress: {edge_report.get('net_edge_bps_stress')}",
            f"- profitable_windows_ratio: {edge_report.get('profitable_windows_ratio')}",
            f"- walk_forward_pass_rate: {edge_report.get('walk_forward_pass_rate')}",
            f"- lower_confidence_bound_bps: {edge_report.get('lower_confidence_bound_bps')}",
            f"- pair_concentration: {edge_report.get('pair_concentration')}",
            f"- pair_evidence_count: {edge_report.get('pair_evidence_count')}",
            f"- pair_evidence_unique_count: {edge_report.get('pair_evidence_unique_count')}",
            f"- pair_evidence_distribution: {edge_report.get('pair_evidence_distribution')}",
            f"- pair_alignment: {edge_report.get('pair_alignment')}",
            f"- pair_price_series: {edge_report.get('pair_price_series')}",
            f"- calendar_concentration: {edge_report.get('calendar_concentration')}",
            f"- holding_period: {edge_report.get('holding_period')}",
            f"- negative_control_random_entry_delta_bps: {edge_report.get('negative_control_random_entry_delta_bps')}",
            f"- negative_control_shuffled_signal_delta_bps: {edge_report.get('negative_control_shuffled_signal_delta_bps')}",
            f"- negative_control_shifted_signal_delta_bps: {edge_report.get('negative_control_shifted_signal_delta_bps')}",
            f"- passes_research_gate: {edge_report.get('passes_research_gate')}",
            f"- rejection_reason: {edge_report.get('rejection_reason')}",
            "",
            "## Horizon Results",
            "",
        ]
    )
    for item in artifact.get("horizon_results", []):
        lines.append(
            "- hold_candles="
            f"{item.get('hold_candles')}: status={item.get('status')}, "
            f"sample_count={item.get('sample_count')}, "
            f"net_edge_bps={item.get('net_edge_bps')}, "
            f"net_edge_bps_stress={item.get('net_edge_bps_stress')}, "
            f"profitable_windows_ratio={item.get('profitable_windows_ratio')}, "
            f"pair_concentration={item.get('pair_concentration')}, "
            f"passes_research_gate={item.get('passes_research_gate')}"
        )
    concentration = artifact.get("concentration_diagnostics", {})
    lines.extend(["", "## Concentration Diagnostics", ""])
    lines.extend(
        [
            f"- active_day_count: {concentration.get('active_day_count')}",
            f"- active_week_count: {concentration.get('active_week_count')}",
            f"- active_month_count: {concentration.get('active_month_count')}",
            f"- active_quarter_count: {concentration.get('active_quarter_count')}",
            f"- max_day_event_share: {concentration.get('max_day_event_share')}",
            f"- max_week_event_share: {concentration.get('max_week_event_share')}",
            f"- max_month_event_share: {concentration.get('max_month_event_share')}",
            f"- max_quarter_event_share: {concentration.get('max_quarter_event_share')}",
        ]
    )
    lines.extend(["", "## Checks", ""])
    lines.extend(
        f"- {item.get('name')}: {item.get('status')}"
        for item in artifact.get("checks", [])
    )
    blockers = artifact.get("blockers", [])
    if blockers:
        lines.extend(["", "## Blockers", ""])
        lines.extend(f"- {item.get('name')}" for item in blockers)
    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "- maker_fill_risk is reported through cost fields but is not yet a separate fill-probability gate.",
            "- overlapping_events, cooldown_candles, and effective_sample_count require follow-up diagnostics before promotion.",
        ]
    )
    lines.extend(["", "## Safety Scope", ""])
    safety = artifact.get("safety_scope", {})
    lines.extend(
        [
            f"- historical_only: {safety.get('historical_only')}",
            f"- strategy_code_generated: {safety.get('strategy_code_generated')}",
            f"- paper_trading_started: {safety.get('paper_trading_started')}",
            f"- dry_run_trading_started: {safety.get('dry_run_trading_started')}",
            f"- live_trading: {safety.get('live_trading')}",
            f"- exchange_order_placement: {safety.get('exchange_order_placement')}",
            f"- process_control: {safety.get('process_control')}",
        ]
    )
    lines.append("")
    return "\n".join(lines)


def _load_json(path: Path) -> tuple[Any | None, str | None]:
    if not path.is_file():
        return None, "file_not_found"
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except Exception as exc:
        return None, f"read_error: {exc}"


def _check(name: str, passed: bool, details: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "name": name,
        "status": "pass" if passed else "fail",
        "details": details or {},
    }


def _resolve_inside(path: Path, root: Path) -> Path:
    resolved = (path if path.is_absolute() else root / path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Path must be inside the workspace: {path}") from exc
    return resolved


def _resolve_optional_inside(path: Path | None, root: Path) -> Path | None:
    if path is None:
        return None
    return _resolve_inside(path, root)


def _rel(path: Path | None, root: Path) -> str | None:
    if path is None:
        return None
    try:
        return str(path.resolve().relative_to(root))
    except ValueError:
        return str(path)


def _edge_discovery_id(created_at: str, seed: str) -> str:
    try:
        parsed = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        prefix = parsed.strftime("%Y%m%dT%H%M%SZ")
    except ValueError:
        prefix = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return _safe_path_component(f"{prefix}_{seed}")


def _safe_path_component(value: str) -> str:
    safe = "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in value)
    return safe.strip("_") or "edge_discovery"


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _timestamp_to_str(value: Any) -> str | None:
    if pd.isna(value):
        return None
    return pd.Timestamp(value).isoformat()
