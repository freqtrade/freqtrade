from __future__ import annotations

import json
import operator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Sequence

import pandas as pd


_OPS: dict[str, Callable[[pd.Series, float], pd.Series]] = {
    ">": operator.gt,
    ">=": operator.ge,
    "<": operator.lt,
    "<=": operator.le,
    "==": operator.eq,
}
_SUPPORTED_FEATURES = {
    "funding_rate_bps",
    "funding_rate_delta_bps",
    "hour_utc",
    "informative_range_pct",
    "informative_return_bps",
    "informative_sma_distance_bps",
    "informative_volume_zscore",
    "liquidation_buy_notional",
    "liquidation_count",
    "liquidation_imbalance",
    "liquidation_net_notional",
    "liquidation_sell_notional",
    "liquidation_total_notional",
    "liquidation_total_notional_zscore",
    "mark_price_gap_bps",
    "mark_price_gap_delta_bps",
    "mark_price_return_bps",
    "long_account_ratio",
    "long_account_ratio_delta_bps",
    "long_short_ratio",
    "long_short_ratio_zscore",
    "open_interest",
    "open_interest_delta_pct",
    "open_interest_zscore",
    "order_book_ask_size",
    "order_book_bid_size",
    "order_book_depth_imbalance",
    "order_book_depth_imbalance_zscore",
    "order_book_mid_price_gap_bps",
    "order_book_spread_bps",
    "order_book_spread_bps_zscore",
    "return_bps",
    "range_pct",
    "relative_return_bps",
    "sma_distance_bps",
    "weekday",
    "volume_zscore",
}
_INFORMATIVE_OHLCV_FEATURES = {
    "informative_range_pct",
    "informative_return_bps",
    "informative_sma_distance_bps",
    "informative_volume_zscore",
    "relative_return_bps",
}
_FUNDING_RATE_FEATURES = {
    "funding_rate_bps",
    "funding_rate_delta_bps",
}
_MARK_PRICE_FEATURES = {
    "mark_price_gap_bps",
    "mark_price_gap_delta_bps",
    "mark_price_return_bps",
}
_OPEN_INTEREST_FEATURES = {
    "open_interest",
    "open_interest_delta_pct",
    "open_interest_zscore",
}
_LONG_SHORT_RATIO_FEATURES = {
    "long_account_ratio",
    "long_account_ratio_delta_bps",
    "long_short_ratio",
    "long_short_ratio_zscore",
}
_LIQUIDATION_FEATURES = {
    "liquidation_buy_notional",
    "liquidation_count",
    "liquidation_imbalance",
    "liquidation_net_notional",
    "liquidation_sell_notional",
    "liquidation_total_notional",
    "liquidation_total_notional_zscore",
}
_ORDER_BOOK_FEATURES = {
    "order_book_ask_size",
    "order_book_bid_size",
    "order_book_depth_imbalance",
    "order_book_depth_imbalance_zscore",
    "order_book_mid_price_gap_bps",
    "order_book_spread_bps",
    "order_book_spread_bps_zscore",
}
_EMPTY_INSTRUMENT_LABELS = {
    "-",
    "--",
    "n/a",
    "na",
    "nan",
    "none",
    "null",
    "placeholder",
    "undefined",
    "unknown",
}
_CONTEXT_MERGE_SEMANTICS = "closed_context_candle_availability_v1"


@dataclass(frozen=True)
class LocalEventBuildInputs:
    root_dir: Path
    ohlcv_path: Path
    event_spec_path: Path
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
    event_id: str | None = None
    reviewer_notes: Sequence[str] = field(default_factory=list)
    created_by_agent: str = "codex"
    created_at: str | None = None
    command: Sequence[str] = field(default_factory=list)


def build_local_events(inputs: LocalEventBuildInputs) -> dict[str, Any]:
    root = inputs.root_dir.resolve()
    generated_at = inputs.created_at or datetime.now(UTC).replace(microsecond=0).isoformat()
    ohlcv_path = _resolve_inside(inputs.ohlcv_path, root)
    spec_path = _resolve_inside(inputs.event_spec_path, root)
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
    order_book_quality_paths = [
        _resolve_inside(path, root) for path in inputs.order_book_quality_report_paths
    ]
    failure_synthesis_path = _resolve_optional_inside(inputs.failure_synthesis_path, root)
    spec, spec_error = _load_json(spec_path)
    failure_synthesis, failure_synthesis_error = (
        _load_json(failure_synthesis_path)
        if failure_synthesis_path is not None
        else (None, None)
    )
    ohlcv, ohlcv_error = _load_ohlcv(ohlcv_path)
    event_id = inputs.event_id or _event_id(
        generated_at,
        str((spec or {}).get("event_id") or (spec or {}).get("thesis_id") or "local_events"),
    )
    checks = [
        _check("ohlcv_file_present", ohlcv_path.is_file(), {"path": _rel(ohlcv_path, root)}),
        _check("ohlcv_parseable", ohlcv is not None and not ohlcv_error, {"error": ohlcv_error}),
        _check("event_spec_file_present", spec_path.is_file(), {"path": _rel(spec_path, root)}),
        _check("event_spec_parseable", spec is not None and not spec_error, {"error": spec_error}),
        _check(
            "event_spec_factory_valid",
            bool(spec) and spec.get("factory") == "research_local_event_spec",
            {"factory": (spec or {}).get("factory")},
        ),
    ]
    condition_checks, conditions = _condition_checks(spec or {})
    checks.extend(condition_checks)
    failure_synthesis_checks, failure_synthesis_summary = _failure_synthesis_checks(
        spec or {},
        failure_synthesis=failure_synthesis,
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
    if (
        ohlcv is not None
        and conditions
        and all(check["status"] == "pass" for check in condition_checks)
        and all(check["status"] == "pass" for check in failure_synthesis_checks)
        and all(check["status"] == "pass" for check in context_checks)
        and all(check["status"] == "pass" for check in quality_checks)
    ):
        enriched = ohlcv.copy()
        enriched = _attach_context_features(
            enriched,
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
            cooldown_candles=_cooldown_candles(spec or {}),
        )

    checks.append(_check("events_generated", bool(events), {"event_count": len(events)}))
    status = "completed" if all(check["status"] == "pass" for check in checks) else "blocked"
    return {
        "generated_at": generated_at,
        "factory": "research_local_event_builder",
        "event_id": event_id,
        "status": status,
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
        "open_interest_quality_reports": open_interest_quality_reports,
        "long_short_ratio_quality_reports": long_short_ratio_quality_reports,
        "liquidation_quality_reports": liquidation_quality_reports,
        "order_book_quality_reports": order_book_quality_reports,
        "failure_synthesis_path": (
            _rel(failure_synthesis_path, root) if failure_synthesis_path else None
        ),
        "failure_synthesis_summary": failure_synthesis_summary,
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
        "event_spec_path": _rel(spec_path, root),
        "event_spec": _safe_spec_summary(spec or {}),
        "thesis_id": str((spec or {}).get("thesis_id") or "").strip(),
        "mechanism_class": str((spec or {}).get("mechanism_class") or "").strip(),
        "condition_count": len(conditions),
        "feature_columns": feature_columns,
        "condition_diagnostics": condition_diagnostics,
        "cumulative_condition_match_counts": cumulative_condition_match_counts,
        "combined_match_count_before_cooldown": combined_match_count,
        "cooldown_candles": _cooldown_candles(spec or {}),
        "event_count": len(events),
        "events": events,
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


def write_local_event_artifacts(
    artifact: dict[str, Any], *, root_dir: Path, output_root: Path
) -> tuple[Path, Path, Path]:
    root = root_dir.resolve()
    event_id = _safe_path_component(str(artifact.get("event_id") or "local_events"))
    out_dir = _resolve_inside(output_root, root) / event_id
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "local_events.json"
    report_path = out_dir / "local_events_report.md"
    events_path = out_dir / "events.csv"
    artifact_to_write = dict(artifact)
    artifact_to_write["events_csv_path"] = _rel(events_path, root)
    json_path.write_text(
        json.dumps(artifact_to_write, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    report_path.write_text(_render_report(artifact_to_write), encoding="utf-8")
    pd.DataFrame(artifact_to_write.get("events", []) or [{"date": None}]).to_csv(
        events_path,
        index=False,
    )
    return json_path, report_path, events_path


def _condition_checks(spec: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    raw_conditions = spec.get("conditions", []) if isinstance(spec, dict) else []
    checks = [_check("event_spec_has_conditions", bool(raw_conditions), {})]
    conditions: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_conditions, start=1):
        if not isinstance(raw, dict):
            checks.append(_check(f"condition_{index}_object", False, {"condition": raw}))
            continue
        feature = str(raw.get("feature") or "").strip()
        op = str(raw.get("operator") or raw.get("op") or "").strip()
        value = _float_or_none(raw.get("value"))
        lookback = _int_or_none(raw.get("lookback_candles") or raw.get("lookback") or 1)
        valid = (
            feature in _SUPPORTED_FEATURES
            and op in _OPS
            and value is not None
            and lookback is not None
            and lookback > 0
        )
        checks.append(
            _check(
                f"condition_{index}_valid",
                valid,
                {
                    "feature": feature,
                    "operator": op,
                    "value": value,
                    "lookback_candles": lookback,
                },
            )
        )
        if valid:
            conditions.append(
                {
                    "feature": feature,
                    "operator": op,
                    "value": float(value),
                    "lookback_candles": int(lookback),
                }
            )
    return checks, conditions


def _required_contexts(conditions: Sequence[dict[str, Any]]) -> set[str]:
    contexts: set[str] = set()
    for condition in conditions:
        feature = condition["feature"]
        if feature in _FUNDING_RATE_FEATURES:
            contexts.add("funding_rate")
        if feature in _MARK_PRICE_FEATURES:
            contexts.add("mark_price")
        if feature in _INFORMATIVE_OHLCV_FEATURES:
            contexts.add("informative_ohlcv")
        if feature in _OPEN_INTEREST_FEATURES:
            contexts.add("open_interest")
        if feature in _LONG_SHORT_RATIO_FEATURES:
            contexts.add("long_short_ratio")
        if feature in _LIQUIDATION_FEATURES:
            contexts.add("liquidation")
        if feature in _ORDER_BOOK_FEATURES:
            contexts.add("order_book")
    return contexts


def _failure_synthesis_checks(
    spec: dict[str, Any],
    *,
    failure_synthesis: dict[str, Any] | None,
    failure_synthesis_path: Path | None,
    failure_synthesis_error: str | None,
    root: Path,
    allow_failed_thesis_or_family: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if failure_synthesis_path is None:
        return [], {
            "used": False,
            "path": None,
            "allow_failed_thesis_or_family": allow_failed_thesis_or_family,
        }

    failed = _failed_thesis_and_family_sets(failure_synthesis or {})
    thesis_id = str(spec.get("thesis_id") or "").strip()
    mechanism_class = str(
        spec.get("mechanism_class")
        or spec.get("thesis_family")
        or spec.get("hypothesis_family")
        or ""
    ).strip()
    thesis_repeats = thesis_id in failed["thesis_ids"]
    family_repeats = mechanism_class in failed["families"]
    summary = {
        "used": True,
        "path": _rel(failure_synthesis_path, root),
        "parseable": failure_synthesis is not None and failure_synthesis_error is None,
        "error": failure_synthesis_error,
        "allow_failed_thesis_or_family": allow_failed_thesis_or_family,
        "failed_thesis_id_count": len(failed["thesis_ids"]),
        "failed_family_count": len(failed["families"]),
        "thesis_id": thesis_id,
        "mechanism_class": mechanism_class,
        "thesis_repeats_failed_synthesis": thesis_repeats,
        "mechanism_repeats_failed_synthesis": family_repeats,
    }
    checks = [
        _check(
            "failure_synthesis_file_present",
            failure_synthesis_path.is_file(),
            {"path": _rel(failure_synthesis_path, root)},
        ),
        _check(
            "failure_synthesis_parseable",
            failure_synthesis is not None and failure_synthesis_error is None,
            {"error": failure_synthesis_error},
        ),
        _check(
            "event_spec_thesis_not_in_failure_synthesis",
            allow_failed_thesis_or_family or not thesis_repeats,
            {"thesis_id": thesis_id, "matched": thesis_repeats},
        ),
        _check(
            "event_spec_mechanism_not_in_failure_synthesis",
            allow_failed_thesis_or_family or not family_repeats,
            {"mechanism_class": mechanism_class, "matched": family_repeats},
        ),
    ]
    return checks, summary


def _failed_thesis_and_family_sets(failure_synthesis: dict[str, Any]) -> dict[str, set[str]]:
    summary = failure_synthesis.get("aggregate_failure_summary")
    if not isinstance(summary, dict):
        summary = {}
    failed_families = _string_set(
        summary.get("failed_hypothesis_families_to_avoid_as_default")
        or summary.get("failed_hypothesis_families")
        or summary.get("failed_families")
        or []
    )
    failed_thesis_ids = _string_set(
        summary.get("failed_thesis_ids")
        or summary.get("thesis_ids_to_avoid_as_default")
        or []
    )
    if summary.get("all_candidates_failed_gates") is True:
        failed_families.update(_string_set(summary.get("hypothesis_families_tried") or []))
        failed_thesis_ids.update(_string_set(summary.get("thesis_ids_tried") or []))
    return {
        "families": failed_families,
        "thesis_ids": failed_thesis_ids,
    }


def _string_set(values: Any) -> set[str]:
    if not isinstance(values, list):
        return set()
    return {str(value).strip() for value in values if str(value).strip()}


def _context_checks(
    *,
    required_contexts: set[str],
    funding_rate_path: Path | None,
    funding_rate: pd.DataFrame | None,
    funding_rate_error: str | None,
    mark_price_path: Path | None,
    mark_price: pd.DataFrame | None,
    mark_price_error: str | None,
    informative_ohlcv_path: Path | None,
    informative_ohlcv: pd.DataFrame | None,
    informative_ohlcv_error: str | None,
    open_interest_path: Path | None,
    open_interest: pd.DataFrame | None,
    open_interest_error: str | None,
    long_short_ratio_path: Path | None,
    long_short_ratio: pd.DataFrame | None,
    long_short_ratio_error: str | None,
    liquidation_path: Path | None,
    liquidation: pd.DataFrame | None,
    liquidation_error: str | None,
    order_book_path: Path | None,
    order_book: pd.DataFrame | None,
    order_book_error: str | None,
    root: Path,
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    if "funding_rate" in required_contexts:
        checks.extend(
            [
                _check(
                    "funding_rate_file_present",
                    bool(funding_rate_path is not None and funding_rate_path.is_file()),
                    {"path": _rel(funding_rate_path, root) if funding_rate_path else None},
                ),
                _check(
                    "funding_rate_parseable",
                    funding_rate is not None and not funding_rate_error,
                    {"error": funding_rate_error},
                ),
            ]
        )
    if "mark_price" in required_contexts:
        checks.extend(
            [
                _check(
                    "mark_price_file_present",
                    bool(mark_price_path is not None and mark_price_path.is_file()),
                    {"path": _rel(mark_price_path, root) if mark_price_path else None},
                ),
                _check(
                    "mark_price_parseable",
                    mark_price is not None and not mark_price_error,
                    {"error": mark_price_error},
                ),
            ]
        )
    if "informative_ohlcv" in required_contexts:
        checks.extend(
            [
                _check(
                    "informative_ohlcv_file_present",
                    bool(informative_ohlcv_path is not None and informative_ohlcv_path.is_file()),
                    {
                        "path": (
                            _rel(informative_ohlcv_path, root)
                            if informative_ohlcv_path
                            else None
                        )
                    },
                ),
                _check(
                    "informative_ohlcv_parseable",
                    informative_ohlcv is not None and not informative_ohlcv_error,
                    {"error": informative_ohlcv_error},
                ),
            ]
        )
    if "open_interest" in required_contexts:
        checks.extend(
            [
                _check(
                    "open_interest_file_present",
                    bool(open_interest_path is not None and open_interest_path.is_file()),
                    {"path": _rel(open_interest_path, root) if open_interest_path else None},
                ),
                _check(
                    "open_interest_parseable",
                    open_interest is not None and not open_interest_error,
                    {"error": open_interest_error},
                ),
            ]
        )
    if "long_short_ratio" in required_contexts:
        checks.extend(
            [
                _check(
                    "long_short_ratio_file_present",
                    bool(long_short_ratio_path is not None and long_short_ratio_path.is_file()),
                    {"path": _rel(long_short_ratio_path, root) if long_short_ratio_path else None},
                ),
                _check(
                    "long_short_ratio_parseable",
                    long_short_ratio is not None and not long_short_ratio_error,
                    {"error": long_short_ratio_error},
                ),
            ]
        )
    if "liquidation" in required_contexts:
        checks.extend(
            [
                _check(
                    "liquidation_file_present",
                    bool(liquidation_path is not None and liquidation_path.is_file()),
                    {"path": _rel(liquidation_path, root) if liquidation_path else None},
                ),
                _check(
                    "liquidation_parseable",
                    liquidation is not None and not liquidation_error,
                    {"error": liquidation_error},
                ),
            ]
        )
    if "order_book" in required_contexts:
        checks.extend(
            [
                _check(
                    "order_book_file_present",
                    bool(order_book_path is not None and order_book_path.is_file()),
                    {"path": _rel(order_book_path, root) if order_book_path else None},
                ),
                _check(
                    "order_book_parseable",
                    order_book is not None and not order_book_error,
                    {"error": order_book_error},
                ),
            ]
        )
    return checks


def _structural_quality_report_checks(
    *,
    required_contexts: set[str],
    open_interest_quality_reports: Sequence[dict[str, Any]],
    long_short_ratio_quality_reports: Sequence[dict[str, Any]],
    liquidation_quality_reports: Sequence[dict[str, Any]],
    order_book_quality_reports: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        *_single_structural_quality_report_checks(
            "open_interest",
            required="open_interest" in required_contexts,
            summaries=open_interest_quality_reports,
        ),
        *_single_structural_quality_report_checks(
            "long_short_ratio",
            required="long_short_ratio" in required_contexts,
            summaries=long_short_ratio_quality_reports,
        ),
        *_single_structural_quality_report_checks(
            "liquidation",
            required="liquidation" in required_contexts,
            summaries=liquidation_quality_reports,
        ),
        *_single_structural_quality_report_checks(
            "order_book",
            required="order_book" in required_contexts,
            summaries=order_book_quality_reports,
        ),
    ]


def _single_structural_quality_report_checks(
    context_name: str,
    *,
    required: bool,
    summaries: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    reports = list(summaries)
    all_parseable = all(bool(item.get("parseable")) for item in reports)
    all_ok = all(bool(item.get("ok")) for item in reports)
    return [
        _check(
            f"{context_name}_quality_reports_parseable_when_supplied",
            all_parseable,
            {"report_count": len(reports)},
        ),
        _check(
            f"{context_name}_quality_report_passed_when_required",
            not required or (bool(reports) and all_parseable and all_ok),
            {
                "required": required,
                "report_count": len(reports),
                "all_reports_ok": all_ok,
            },
        ),
    ]


def _attach_context_features(
    frame: pd.DataFrame,
    *,
    funding_rate: pd.DataFrame | None,
    mark_price: pd.DataFrame | None,
    informative_ohlcv: pd.DataFrame | None,
    open_interest: pd.DataFrame | None,
    long_short_ratio: pd.DataFrame | None,
    liquidation: pd.DataFrame | None = None,
    order_book: pd.DataFrame | None = None,
) -> pd.DataFrame:
    enriched = frame.sort_values("date").reset_index(drop=True).copy()
    base_step = _infer_candle_step(enriched["date"])
    if funding_rate is not None:
        funding_features = funding_rate.sort_values("date").reset_index(drop=True).copy()
        funding_features["funding_rate_bps"] = funding_features["funding_rate"] * 10000.0
        funding_features["funding_rate_delta_bps"] = (
            funding_features["funding_rate"].diff() * 10000.0
        )
        funding_features["date_merge"] = _closed_context_merge_dates(
            funding_features["date"],
            context_step=_infer_candle_step(funding_features["date"]),
            base_step=base_step,
        )
        enriched = pd.merge_asof(
            enriched,
            funding_features[["date_merge", "funding_rate_bps", "funding_rate_delta_bps"]],
            left_on="date",
            right_on="date_merge",
            direction="backward",
        )
        enriched = enriched.drop(columns=["date_merge"], errors="ignore")
    if mark_price is not None:
        mark_features = mark_price.sort_values("date").reset_index(drop=True).copy()
        mark_features["mark_price_return_bps"] = (
            mark_features["mark_close"] / mark_features["mark_close"].shift(1) - 1.0
        ) * 10000.0
        mark_features["date_merge"] = _closed_context_merge_dates(
            mark_features["date"],
            context_step=_infer_candle_step(mark_features["date"]),
            base_step=base_step,
        )
        enriched = pd.merge_asof(
            enriched,
            mark_features[["date_merge", "mark_close", "mark_price_return_bps"]],
            left_on="date",
            right_on="date_merge",
            direction="backward",
        )
        enriched = enriched.drop(columns=["date_merge"], errors="ignore")
        enriched["mark_price_gap_bps"] = (
            enriched["close"].astype(float) / enriched["mark_close"].astype(float) - 1.0
        ) * 10000.0
    if informative_ohlcv is not None:
        informative_features = informative_ohlcv.sort_values("date").reset_index(drop=True).copy()
        informative_features = informative_features.rename(
            columns={
                "open": "informative_open",
                "high": "informative_high",
                "low": "informative_low",
                "close": "informative_close",
                "volume": "informative_volume",
            }
        )
        informative_features["date_merge"] = _closed_context_merge_dates(
            informative_features["date"],
            context_step=_infer_candle_step(informative_features["date"]),
            base_step=base_step,
        )
        enriched = pd.merge_asof(
            enriched,
            informative_features[
                [
                    "date_merge",
                    "informative_open",
                    "informative_high",
                    "informative_low",
                    "informative_close",
                    "informative_volume",
                ]
            ],
            left_on="date",
            right_on="date_merge",
            direction="backward",
        )
        enriched = enriched.drop(columns=["date_merge"], errors="ignore")
    if open_interest is not None:
        interest_features = open_interest.sort_values("date").reset_index(drop=True).copy()
        interest_features["date_merge"] = _closed_context_merge_dates(
            interest_features["date"],
            context_step=_infer_candle_step(interest_features["date"]),
            base_step=base_step,
        )
        enriched = pd.merge_asof(
            enriched,
            interest_features[["date_merge", "open_interest"]],
            left_on="date",
            right_on="date_merge",
            direction="backward",
        )
        enriched = enriched.drop(columns=["date_merge"], errors="ignore")
    if long_short_ratio is not None:
        ratio_features = long_short_ratio.sort_values("date").reset_index(drop=True).copy()
        ratio_features["long_account_ratio_delta_bps"] = (
            ratio_features["long_account_ratio"].diff() * 10000.0
        )
        ratio_features["date_merge"] = _closed_context_merge_dates(
            ratio_features["date"],
            context_step=_infer_candle_step(ratio_features["date"]),
            base_step=base_step,
        )
        enriched = pd.merge_asof(
            enriched,
            ratio_features[
                [
                    "date_merge",
                    "long_account_ratio",
                    "short_account_ratio",
                    "long_account_ratio_delta_bps",
                    "long_short_ratio",
                ]
            ],
            left_on="date",
            right_on="date_merge",
            direction="backward",
        )
        enriched = enriched.drop(columns=["date_merge"], errors="ignore")
    if liquidation is not None:
        liquidation_features = _liquidation_features_by_base_candle(
            liquidation,
            base_step=base_step,
        )
        enriched = pd.merge(
            enriched,
            liquidation_features,
            on="date",
            how="left",
        )
        liquidation_columns = [
            "liquidation_count",
            "liquidation_buy_notional",
            "liquidation_sell_notional",
            "liquidation_total_notional",
            "liquidation_net_notional",
            "liquidation_imbalance",
        ]
        for column in liquidation_columns:
            if column in enriched.columns:
                enriched[column] = enriched[column].fillna(0.0)
    if order_book is not None:
        order_book_features = _order_book_features_by_base_candle(
            order_book,
            base_step=base_step,
        )
        enriched = pd.merge(
            enriched,
            order_book_features,
            on="date",
            how="left",
        )
        enriched["order_book_mid_price_gap_bps"] = (
            enriched["close"].astype(float)
            / enriched["order_book_mid_price"].astype(float)
            - 1.0
        ) * 10000.0
    return enriched


def _infer_candle_step(dates: pd.Series) -> pd.Timedelta | None:
    parsed = pd.to_datetime(dates, utc=True, errors="coerce").dropna().sort_values()
    diffs = parsed.diff().dropna()
    positive_diffs = diffs[diffs > pd.Timedelta(0)]
    if positive_diffs.empty:
        return None
    return positive_diffs.median()


def _closed_context_merge_dates(
    dates: pd.Series,
    *,
    context_step: pd.Timedelta | None,
    base_step: pd.Timedelta | None,
) -> pd.Series:
    parsed = pd.to_datetime(dates, utc=True, errors="coerce")
    if context_step is None or base_step is None or context_step <= base_step:
        return parsed
    return parsed + (context_step - base_step)


def _context_merge_summary(
    *,
    required_contexts: set[str],
    ohlcv: pd.DataFrame | None,
    funding_rate: pd.DataFrame | None,
    mark_price: pd.DataFrame | None,
    informative_ohlcv: pd.DataFrame | None,
    open_interest: pd.DataFrame | None,
    long_short_ratio: pd.DataFrame | None,
    liquidation: pd.DataFrame | None,
    order_book: pd.DataFrame | None,
) -> dict[str, Any]:
    base_step = _infer_candle_step(ohlcv["date"]) if ohlcv is not None else None
    contexts = {
        "funding_rate": _single_context_merge_summary(
            required="funding_rate" in required_contexts,
            context_frame=funding_rate,
            base_step=base_step,
        ),
        "mark_price": _single_context_merge_summary(
            required="mark_price" in required_contexts,
            context_frame=mark_price,
            base_step=base_step,
        ),
        "informative_ohlcv": _single_context_merge_summary(
            required="informative_ohlcv" in required_contexts,
            context_frame=informative_ohlcv,
            base_step=base_step,
        ),
        "open_interest": _single_context_merge_summary(
            required="open_interest" in required_contexts,
            context_frame=open_interest,
            base_step=base_step,
        ),
        "long_short_ratio": _single_context_merge_summary(
            required="long_short_ratio" in required_contexts,
            context_frame=long_short_ratio,
            base_step=base_step,
        ),
        "liquidation": _single_context_merge_summary(
            required="liquidation" in required_contexts,
            context_frame=liquidation,
            base_step=base_step,
            event_stream=True,
        ),
        "order_book": _single_context_merge_summary(
            required="order_book" in required_contexts,
            context_frame=order_book,
            base_step=base_step,
            event_stream=True,
        ),
    }
    required = sorted(required_contexts)
    alignment = all(contexts[name]["closed_context_candle_alignment"] for name in required)
    return {
        "semantics": _CONTEXT_MERGE_SEMANTICS,
        "context_features_used": bool(required),
        "required_contexts": required,
        "base_interval_seconds": _timedelta_seconds(base_step),
        "closed_context_candle_alignment": alignment,
        "contexts": contexts,
    }


def _single_context_merge_summary(
    *,
    required: bool,
    context_frame: pd.DataFrame | None,
    base_step: pd.Timedelta | None,
    event_stream: bool = False,
) -> dict[str, Any]:
    context_step = (
        None
        if event_stream
        else _infer_candle_step(context_frame["date"])
        if context_frame is not None
        else None
    )
    shift_seconds = (
        0.0
        if event_stream and context_frame is not None and base_step is not None
        else _closed_context_shift_seconds(
            context_step=context_step,
            base_step=base_step,
        )
    )
    alignment = not required or (
        context_frame is not None
        and base_step is not None
        and shift_seconds is not None
    )
    return {
        "required": required,
        "available": context_frame is not None,
        "context_interval_seconds": _timedelta_seconds(context_step),
        "closed_context_shift_seconds": shift_seconds,
        "closed_context_candle_alignment": alignment,
    }


def _closed_context_shift_seconds(
    *,
    context_step: pd.Timedelta | None,
    base_step: pd.Timedelta | None,
) -> float | None:
    if context_step is None or base_step is None:
        return None
    if context_step <= base_step:
        return 0.0
    return float((context_step - base_step).total_seconds())


def _timedelta_seconds(value: pd.Timedelta | None) -> float | None:
    if value is None:
        return None
    return float(value.total_seconds())


def _condition_diagnostic(
    condition: dict[str, Any],
    *,
    feature_column: str,
    series: pd.Series,
    mask: pd.Series,
) -> dict[str, Any]:
    numeric = pd.to_numeric(series, errors="coerce")
    return {
        "feature": condition["feature"],
        "feature_column": feature_column,
        "operator": condition["operator"],
        "value": condition["value"],
        "lookback_candles": condition["lookback_candles"],
        "non_null_count": int(numeric.notna().sum()),
        "match_count": int(mask.fillna(False).sum()),
        "min": _round_or_none(numeric.min()),
        "max": _round_or_none(numeric.max()),
        "median": _round_or_none(numeric.median()),
    }


def _feature_series(frame: pd.DataFrame, condition: dict[str, Any]) -> pd.Series:
    feature = condition["feature"]
    lookback = int(condition["lookback_candles"])
    close = frame["close"].astype(float)
    if feature == "funding_rate_bps":
        return frame["funding_rate_bps"].astype(float)
    if feature == "funding_rate_delta_bps":
        return frame["funding_rate_delta_bps"].astype(float)
    if feature == "hour_utc":
        return pd.to_datetime(frame["date"], utc=True).dt.hour.astype(float)
    if feature == "informative_range_pct":
        informative_close = frame["informative_close"].astype(float)
        return (
            frame["informative_high"].astype(float) - frame["informative_low"].astype(float)
        ) / informative_close * 100.0
    if feature == "informative_return_bps":
        informative_close = frame["informative_close"].astype(float)
        return _grouped_pct_change_bps(frame, informative_close, lookback)
    if feature == "informative_sma_distance_bps":
        informative_close = frame["informative_close"].astype(float)
        average = _grouped_rolling_mean(frame, informative_close, lookback)
        return (informative_close / average - 1.0) * 10000.0
    if feature == "informative_volume_zscore":
        informative_volume = frame["informative_volume"].astype(float)
        average = _grouped_rolling_mean(frame, informative_volume, lookback)
        stdev = _grouped_rolling_std(frame, informative_volume, lookback).replace(0.0, pd.NA)
        return (informative_volume - average) / stdev
    if feature == "mark_price_gap_bps":
        return frame["mark_price_gap_bps"].astype(float)
    if feature == "mark_price_gap_delta_bps":
        gap = frame["mark_price_gap_bps"].astype(float)
        return gap - _grouped_shift(frame, gap, lookback)
    if feature == "mark_price_return_bps":
        return frame["mark_price_return_bps"].astype(float)
    if feature == "long_account_ratio":
        return frame["long_account_ratio"].astype(float)
    if feature == "long_account_ratio_delta_bps":
        return frame["long_account_ratio_delta_bps"].astype(float)
    if feature == "long_short_ratio":
        return frame["long_short_ratio"].astype(float)
    if feature == "long_short_ratio_zscore":
        ratio = frame["long_short_ratio"].astype(float)
        average = _grouped_rolling_mean(frame, ratio, lookback)
        stdev = _grouped_rolling_std(frame, ratio, lookback).replace(0.0, pd.NA)
        return (ratio - average) / stdev
    if feature == "liquidation_buy_notional":
        return frame["liquidation_buy_notional"].astype(float)
    if feature == "liquidation_count":
        return frame["liquidation_count"].astype(float)
    if feature == "liquidation_imbalance":
        return frame["liquidation_imbalance"].astype(float)
    if feature == "liquidation_net_notional":
        return frame["liquidation_net_notional"].astype(float)
    if feature == "liquidation_sell_notional":
        return frame["liquidation_sell_notional"].astype(float)
    if feature == "liquidation_total_notional":
        return frame["liquidation_total_notional"].astype(float)
    if feature == "liquidation_total_notional_zscore":
        notional = frame["liquidation_total_notional"].astype(float)
        average = _grouped_rolling_mean(frame, notional, lookback)
        stdev = _grouped_rolling_std(frame, notional, lookback).replace(0.0, pd.NA)
        return (notional - average) / stdev
    if feature == "open_interest":
        return frame["open_interest"].astype(float)
    if feature == "open_interest_delta_pct":
        interest = frame["open_interest"].astype(float)
        return (interest / _grouped_shift(frame, interest, lookback) - 1.0) * 100.0
    if feature == "open_interest_zscore":
        interest = frame["open_interest"].astype(float)
        average = _grouped_rolling_mean(frame, interest, lookback)
        stdev = _grouped_rolling_std(frame, interest, lookback).replace(0.0, pd.NA)
        return (interest - average) / stdev
    if feature == "order_book_ask_size":
        return frame["order_book_ask_size"].astype(float)
    if feature == "order_book_bid_size":
        return frame["order_book_bid_size"].astype(float)
    if feature == "order_book_depth_imbalance":
        return frame["order_book_depth_imbalance"].astype(float)
    if feature == "order_book_depth_imbalance_zscore":
        imbalance = frame["order_book_depth_imbalance"].astype(float)
        average = _grouped_rolling_mean(frame, imbalance, lookback)
        stdev = _grouped_rolling_std(frame, imbalance, lookback).replace(0.0, pd.NA)
        return (imbalance - average) / stdev
    if feature == "order_book_mid_price_gap_bps":
        return frame["order_book_mid_price_gap_bps"].astype(float)
    if feature == "order_book_spread_bps":
        return frame["order_book_spread_bps"].astype(float)
    if feature == "order_book_spread_bps_zscore":
        spread = frame["order_book_spread_bps"].astype(float)
        average = _grouped_rolling_mean(frame, spread, lookback)
        stdev = _grouped_rolling_std(frame, spread, lookback).replace(0.0, pd.NA)
        return (spread - average) / stdev
    if feature == "return_bps":
        return _grouped_pct_change_bps(frame, close, lookback)
    if feature == "range_pct":
        return (frame["high"].astype(float) - frame["low"].astype(float)) / close * 100.0
    if feature == "relative_return_bps":
        informative_close = frame["informative_close"].astype(float)
        primary_return = _grouped_pct_change_bps(frame, close, lookback)
        informative_return = _grouped_pct_change_bps(frame, informative_close, lookback)
        return primary_return - informative_return
    if feature == "sma_distance_bps":
        average = _grouped_rolling_mean(frame, close, lookback)
        return (close / average - 1.0) * 10000.0
    if feature == "volume_zscore":
        volume = frame["volume"].astype(float)
        average = _grouped_rolling_mean(frame, volume, lookback)
        stdev = _grouped_rolling_std(frame, volume, lookback).replace(0.0, pd.NA)
        return (volume - average) / stdev
    if feature == "weekday":
        return pd.to_datetime(frame["date"], utc=True).dt.dayofweek.astype(float)
    raise ValueError(f"Unsupported feature: {feature}")


def _grouped_pct_change_bps(
    frame: pd.DataFrame, series: pd.Series, lookback: int
) -> pd.Series:
    shifted = _grouped_shift(frame, series, lookback)
    return (series / shifted - 1.0) * 10000.0


def _grouped_shift(frame: pd.DataFrame, series: pd.Series, periods: int) -> pd.Series:
    labels = _instrument_labels(frame)
    if labels is None:
        return series.shift(periods)
    return series.groupby(labels, group_keys=False).shift(periods)


def _grouped_rolling_mean(
    frame: pd.DataFrame, series: pd.Series, window: int
) -> pd.Series:
    labels = _instrument_labels(frame)
    if labels is None:
        return series.rolling(window, min_periods=window).mean()
    return series.groupby(labels, group_keys=False).transform(
        lambda item: item.rolling(window, min_periods=window).mean()
    )


def _grouped_rolling_std(
    frame: pd.DataFrame, series: pd.Series, window: int
) -> pd.Series:
    labels = _instrument_labels(frame)
    if labels is None:
        return series.rolling(window, min_periods=window).std()
    return series.groupby(labels, group_keys=False).transform(
        lambda item: item.rolling(window, min_periods=window).std()
    )


def _instrument_labels(frame: pd.DataFrame) -> pd.Series | None:
    for column in ("pair", "symbol", "instrument"):
        if column in frame.columns:
            labels = frame[column].map(_instrument_label_or_none)
            if labels.nunique(dropna=True) > 1:
                return labels
    return None


def _instrument_label_or_none(value: Any) -> str | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    if not text or text.lower() in _EMPTY_INSTRUMENT_LABELS:
        return None
    return text


def _events_from_mask(
    frame: pd.DataFrame,
    mask: pd.Series,
    *,
    feature_columns: Sequence[str],
    cooldown_candles: int,
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    next_allowed_index = 0
    for index, matched in enumerate(mask):
        if not bool(matched) or index < next_allowed_index:
            continue
        row = frame.iloc[index]
        event = {
            "date": _timestamp_to_str(row["date"]),
            "row_index": int(index),
        }
        for column in ("pair", "symbol", "instrument"):
            if column in frame.columns:
                value = row.get(column)
                if not pd.isna(value) and str(value).strip():
                    event[column] = str(value).strip()
        for column in feature_columns:
            value = row.get(column)
            event[column] = None if pd.isna(value) else round(float(value), 6)
        events.append(event)
        next_allowed_index = index + max(1, int(cooldown_candles))
    return events


def _load_ohlcv(path: Path) -> tuple[pd.DataFrame | None, str | None]:
    if not path.is_file():
        return None, "file_not_found"
    try:
        frame = pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)
    except Exception as exc:
        return None, f"read_error: {exc}"
    required = ("date", "open", "high", "low", "close", "volume")
    missing = [column for column in required if column not in frame.columns]
    if missing:
        return None, f"missing_columns: {', '.join(missing)}"
    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="coerce")
    for column in ("open", "high", "low", "close", "volume"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=list(required)).sort_values("date")
    if frame.empty:
        return None, "empty_after_ohlcv_cleaning"
    optional = [
        column for column in ("pair", "symbol", "instrument") if column in frame.columns
    ]
    return frame[[*required, *optional]].reset_index(drop=True), None


def _load_funding_rate(path: Path) -> tuple[pd.DataFrame | None, str | None]:
    if not path.is_file():
        return None, "file_not_found"
    try:
        frame = pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)
    except Exception as exc:
        return None, f"read_error: {exc}"
    required = ("date", "open")
    missing = [column for column in required if column not in frame.columns]
    if missing:
        return None, f"missing_columns: {', '.join(missing)}"
    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="coerce")
    frame["funding_rate"] = pd.to_numeric(frame["open"], errors="coerce")
    frame = frame.dropna(subset=["date", "funding_rate"]).sort_values("date")
    if frame.empty:
        return None, "empty_after_funding_rate_cleaning"
    return frame[["date", "funding_rate"]].reset_index(drop=True), None


def _load_mark_price(path: Path) -> tuple[pd.DataFrame | None, str | None]:
    if not path.is_file():
        return None, "file_not_found"
    try:
        frame = pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)
    except Exception as exc:
        return None, f"read_error: {exc}"
    required = ("date", "close")
    missing = [column for column in required if column not in frame.columns]
    if missing:
        return None, f"missing_columns: {', '.join(missing)}"
    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="coerce")
    frame["mark_close"] = pd.to_numeric(frame["close"], errors="coerce")
    frame = frame.dropna(subset=["date", "mark_close"]).sort_values("date")
    if frame.empty:
        return None, "empty_after_mark_price_cleaning"
    return frame[["date", "mark_close"]].reset_index(drop=True), None


def _load_open_interest(path: Path) -> tuple[pd.DataFrame | None, str | None]:
    if not path.is_file():
        return None, "file_not_found"
    try:
        frame = pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)
    except Exception as exc:
        return None, f"read_failed:{exc}"
    if "date" not in frame.columns:
        return None, "missing_date_column"
    interest_column = _open_interest_value_column(frame)
    if interest_column is None:
        return None, "missing_open_interest_column"
    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="coerce")
    frame["open_interest"] = pd.to_numeric(frame[interest_column], errors="coerce")
    frame = frame.dropna(subset=["date", "open_interest"]).sort_values("date")
    frame = frame[frame["open_interest"] >= 0.0]
    if frame.empty:
        return None, "empty_after_open_interest_cleaning"
    return frame[["date", "open_interest"]].reset_index(drop=True), None


def _load_long_short_ratio(path: Path) -> tuple[pd.DataFrame | None, str | None]:
    if not path.is_file():
        return None, "file_not_found"
    try:
        frame = pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)
    except Exception as exc:
        return None, f"read_failed:{exc}"
    if "date" not in frame.columns:
        return None, "missing_date_column"
    ratio_columns = _long_short_ratio_value_columns(frame)
    if ratio_columns is None:
        return None, "missing_long_short_ratio_columns"
    long_column, short_column = ratio_columns
    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="coerce")
    frame["long_account_ratio"] = pd.to_numeric(frame[long_column], errors="coerce")
    frame["short_account_ratio"] = pd.to_numeric(frame[short_column], errors="coerce")
    existing_ratio_column = _long_short_ratio_column(frame)
    if existing_ratio_column:
        frame["long_short_ratio"] = pd.to_numeric(frame[existing_ratio_column], errors="coerce")
    else:
        frame["long_short_ratio"] = (
            frame["long_account_ratio"] / frame["short_account_ratio"].where(frame["short_account_ratio"] > 0.0)
        )
    frame = frame.dropna(subset=["date", "long_account_ratio", "short_account_ratio"]).sort_values("date")
    frame = frame[
        (frame["long_account_ratio"] >= 0.0)
        & (frame["long_account_ratio"] <= 1.0)
        & (frame["short_account_ratio"] >= 0.0)
        & (frame["short_account_ratio"] <= 1.0)
    ]
    if frame.empty:
        return None, "empty_after_long_short_ratio_cleaning"
    return (
        frame[
            [
                "date",
                "long_account_ratio",
                "short_account_ratio",
                "long_short_ratio",
            ]
        ]
        .reset_index(drop=True),
        None,
    )


def _load_liquidation(path: Path) -> tuple[pd.DataFrame | None, str | None]:
    if not path.is_file():
        return None, "file_not_found"
    try:
        frame = pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)
    except Exception as exc:
        return None, f"read_failed:{exc}"
    timestamp_column = _liquidation_column(frame, ("date", "T", "updatedTime", "updateTime", "timestamp", "ts"))
    side_column = _liquidation_column(frame, ("side", "S"))
    size_column = _liquidation_column(frame, ("size", "quantity", "qty", "v"))
    price_column = _liquidation_column(frame, ("price", "bankruptcy_price", "p"))
    missing = [
        name
        for name, column in (
            ("timestamp", timestamp_column),
            ("side", side_column),
            ("size", size_column),
            ("price", price_column),
        )
        if column is None
    ]
    if missing:
        return None, f"missing_liquidation_columns: {', '.join(missing)}"
    frame = frame.copy()
    frame["date"] = _coerce_liquidation_timestamp(frame[timestamp_column])
    frame["liquidation_side"] = frame[side_column].astype(str).str.strip().str.upper()
    frame["liquidation_size"] = pd.to_numeric(frame[size_column], errors="coerce")
    frame["liquidation_price"] = pd.to_numeric(frame[price_column], errors="coerce")
    frame = frame.dropna(subset=["date", "liquidation_size", "liquidation_price"])
    frame = frame[
        frame["liquidation_side"].isin(["BUY", "SELL"])
        & (frame["liquidation_size"] > 0.0)
        & (frame["liquidation_price"] > 0.0)
    ].sort_values("date")
    if frame.empty:
        return None, "empty_after_liquidation_cleaning"
    frame["liquidation_notional"] = frame["liquidation_size"] * frame["liquidation_price"]
    return (
        frame[
            [
                "date",
                "liquidation_side",
                "liquidation_size",
                "liquidation_price",
                "liquidation_notional",
            ]
        ].reset_index(drop=True),
        None,
    )


def _load_order_book(path: Path) -> tuple[pd.DataFrame | None, str | None]:
    if not path.is_file():
        return None, "file_not_found"
    try:
        frame = pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)
    except Exception as exc:
        return None, f"read_failed:{exc}"
    if "date" not in frame.columns:
        return None, "missing_date_column"
    bid_price_column = _order_book_column(frame, ("best_bid", "bid_price", "bidPrice", "bid1Price", "bid"))
    ask_price_column = _order_book_column(frame, ("best_ask", "ask_price", "askPrice", "ask1Price", "ask"))
    bid_size_column = _order_book_column(frame, ("bid_size", "bidSize", "bid1Size", "bid_qty", "bidQty"))
    ask_size_column = _order_book_column(frame, ("ask_size", "askSize", "ask1Size", "ask_qty", "askQty"))
    missing = [
        name
        for name, column in (
            ("best_bid", bid_price_column),
            ("best_ask", ask_price_column),
            ("bid_size", bid_size_column),
            ("ask_size", ask_size_column),
        )
        if column is None
    ]
    if missing:
        return None, f"missing_order_book_columns: {', '.join(missing)}"
    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="coerce")
    frame["order_book_best_bid"] = pd.to_numeric(frame[bid_price_column], errors="coerce")
    frame["order_book_best_ask"] = pd.to_numeric(frame[ask_price_column], errors="coerce")
    frame["order_book_bid_size"] = pd.to_numeric(frame[bid_size_column], errors="coerce")
    frame["order_book_ask_size"] = pd.to_numeric(frame[ask_size_column], errors="coerce")
    frame = frame.dropna(
        subset=[
            "date",
            "order_book_best_bid",
            "order_book_best_ask",
            "order_book_bid_size",
            "order_book_ask_size",
        ]
    )
    frame = frame[
        (frame["order_book_best_bid"] > 0.0)
        & (frame["order_book_best_ask"] > 0.0)
        & (frame["order_book_best_bid"] <= frame["order_book_best_ask"])
        & (frame["order_book_bid_size"] > 0.0)
        & (frame["order_book_ask_size"] > 0.0)
    ].sort_values("date")
    if frame.empty:
        return None, "empty_after_order_book_cleaning"
    imbalance_column = _order_book_column(
        frame,
        ("depth_imbalance", "book_imbalance", "top_of_book_imbalance"),
    )
    if imbalance_column:
        frame["order_book_depth_imbalance"] = pd.to_numeric(
            frame[imbalance_column],
            errors="coerce",
        )
    else:
        denominator = frame["order_book_bid_size"] + frame["order_book_ask_size"]
        frame["order_book_depth_imbalance"] = (
            frame["order_book_bid_size"] - frame["order_book_ask_size"]
        ) / denominator.where(denominator > 0.0)
    frame = frame.dropna(subset=["order_book_depth_imbalance"])
    frame = frame[
        (frame["order_book_depth_imbalance"] >= -1.0)
        & (frame["order_book_depth_imbalance"] <= 1.0)
    ]
    if frame.empty:
        return None, "empty_after_order_book_imbalance_cleaning"
    frame["order_book_mid_price"] = (
        frame["order_book_best_bid"] + frame["order_book_best_ask"]
    ) / 2.0
    frame["order_book_spread_bps"] = (
        (frame["order_book_best_ask"] - frame["order_book_best_bid"])
        / frame["order_book_mid_price"]
        * 10000.0
    )
    return (
        frame[
            [
                "date",
                "order_book_best_bid",
                "order_book_best_ask",
                "order_book_mid_price",
                "order_book_spread_bps",
                "order_book_bid_size",
                "order_book_ask_size",
                "order_book_depth_imbalance",
            ]
        ].reset_index(drop=True),
        None,
    )


def _open_interest_value_column(frame: pd.DataFrame) -> str | None:
    for column in ("open_interest", "open", "close"):
        if column in frame.columns:
            return column
    return None


def _long_short_ratio_value_columns(frame: pd.DataFrame) -> tuple[str, str] | None:
    long_column = next(
        (
            column
            for column in ("long_account_ratio", "buy_ratio", "buyRatio")
            if column in frame.columns
        ),
        None,
    )
    short_column = next(
        (
            column
            for column in ("short_account_ratio", "sell_ratio", "sellRatio")
            if column in frame.columns
        ),
        None,
    )
    if not long_column or not short_column:
        return None
    return long_column, short_column


def _long_short_ratio_column(frame: pd.DataFrame) -> str | None:
    for column in ("long_short_ratio", "account_long_short_ratio"):
        if column in frame.columns:
            return column
    return None


def _liquidation_column(frame: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    for column in candidates:
        if column in frame.columns:
            return column
    return None


def _order_book_column(frame: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    for column in candidates:
        if column in frame.columns:
            return column
    return None


def _coerce_liquidation_timestamp(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.to_datetime(series, utc=True, errors="coerce")
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().any():
        median = numeric.dropna().abs().median()
        if median > 100000000000000:
            return pd.to_datetime(numeric, unit="ns", utc=True, errors="coerce")
        if median > 10000000000:
            return pd.to_datetime(numeric, unit="ms", utc=True, errors="coerce")
        if median > 1000000000:
            return pd.to_datetime(numeric, unit="s", utc=True, errors="coerce")
    return pd.to_datetime(series, utc=True, errors="coerce")


def _liquidation_features_by_base_candle(
    liquidation: pd.DataFrame,
    *,
    base_step: pd.Timedelta | None,
) -> pd.DataFrame:
    frame = liquidation.copy()
    if base_step is not None:
        frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="coerce").dt.floor(base_step)
    frame["buy_notional"] = frame["liquidation_notional"].where(
        frame["liquidation_side"] == "BUY",
        0.0,
    )
    frame["sell_notional"] = frame["liquidation_notional"].where(
        frame["liquidation_side"] == "SELL",
        0.0,
    )
    grouped = (
        frame.groupby("date", as_index=False)
        .agg(
            liquidation_count=("liquidation_notional", "size"),
            liquidation_buy_notional=("buy_notional", "sum"),
            liquidation_sell_notional=("sell_notional", "sum"),
            liquidation_total_notional=("liquidation_notional", "sum"),
        )
        .sort_values("date")
    )
    grouped["liquidation_net_notional"] = (
        grouped["liquidation_buy_notional"] - grouped["liquidation_sell_notional"]
    )
    grouped["liquidation_imbalance"] = grouped["liquidation_net_notional"] / grouped[
        "liquidation_total_notional"
    ].where(grouped["liquidation_total_notional"] > 0.0)
    grouped["liquidation_imbalance"] = grouped["liquidation_imbalance"].fillna(0.0)
    return grouped.reset_index(drop=True)


def _order_book_features_by_base_candle(
    order_book: pd.DataFrame,
    *,
    base_step: pd.Timedelta | None,
) -> pd.DataFrame:
    frame = order_book.copy()
    frame["snapshot_date"] = pd.to_datetime(frame["date"], utc=True, errors="coerce")
    if base_step is not None:
        frame["date"] = frame["snapshot_date"].dt.floor(base_step)
    grouped = (
        frame.sort_values("snapshot_date")
        .groupby("date", as_index=False)
        .agg(
            order_book_mid_price=("order_book_mid_price", "last"),
            order_book_spread_bps=("order_book_spread_bps", "mean"),
            order_book_bid_size=("order_book_bid_size", "last"),
            order_book_ask_size=("order_book_ask_size", "last"),
            order_book_depth_imbalance=("order_book_depth_imbalance", "last"),
        )
        .sort_values("date")
    )
    return grouped.reset_index(drop=True)


def _load_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    if not path.is_file():
        return None, "file_not_found"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return None, f"json_decode_error: {exc}"
    if not isinstance(payload, dict):
        return None, "json_not_object"
    return payload, None


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
            }
        )
    return summaries


def _safe_spec_summary(spec: dict[str, Any]) -> dict[str, Any]:
    return {
        "factory": spec.get("factory"),
        "event_id": spec.get("event_id"),
        "thesis_id": spec.get("thesis_id"),
        "mechanism_class": spec.get("mechanism_class"),
        "conditions": spec.get("conditions", []),
        "cooldown_candles": _cooldown_candles(spec),
    }


def _source_summary(
    path: Path | None,
    frame: pd.DataFrame | None,
    error: str | None,
    *,
    root: Path,
    required: bool,
) -> dict[str, Any]:
    if frame is None or frame.empty:
        return {
            "required": required,
            "used": path is not None,
            "path": _rel(path, root) if path else None,
            "parseable": False if required else error is None,
            "error": error,
            "row_count": 0,
            "start": None,
            "end": None,
        }
    return {
        "required": required,
        "used": True,
        "path": _rel(path, root) if path else None,
        "parseable": error is None,
        "error": error,
        "row_count": int(len(frame)),
        "start": _timestamp_to_str(frame["date"].min()),
        "end": _timestamp_to_str(frame["date"].max()),
    }


def _cooldown_candles(spec: dict[str, Any]) -> int:
    value = _int_or_none(spec.get("cooldown_candles"))
    return max(1, value or 1)


def _feature_column(condition: dict[str, Any]) -> str:
    return f"{condition['feature']}_{condition['lookback_candles']}"


def _check(name: str, passed: bool, details: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "name": name,
        "status": "pass" if passed else "fail",
        "details": details or {},
    }


def _render_report(artifact: dict[str, Any]) -> str:
    lines = [
        "# Local Event Builder",
        "",
        f"- event_id: {artifact.get('event_id')}",
        f"- status: {artifact.get('status')}",
        f"- thesis_id: {artifact.get('thesis_id')}",
        f"- mechanism_class: {artifact.get('mechanism_class')}",
        f"- event_count: {artifact.get('event_count')}",
        "",
        "## Conditions",
        "",
    ]
    for condition in artifact.get("event_spec", {}).get("conditions", []) or []:
        lines.append(f"- {condition}")
    lines.extend(["", "## Checks", ""])
    lines.extend(
        f"- {item.get('name')}: {item.get('status')}"
        for item in artifact.get("checks", [])
    )
    diagnostics = artifact.get("condition_diagnostics", [])
    if diagnostics:
        lines.extend(["", "## Condition Diagnostics", ""])
        lines.extend(
            (
                f"- {item.get('feature_column')}: "
                f"match_count={item.get('match_count')}, "
                f"non_null_count={item.get('non_null_count')}, "
                f"min={item.get('min')}, "
                f"median={item.get('median')}, "
                f"max={item.get('max')}"
            )
            for item in diagnostics
        )
    cumulative = artifact.get("cumulative_condition_match_counts", [])
    if cumulative:
        lines.extend(["", "## Cumulative Condition Matches", ""])
        lines.extend(
            f"- through {item.get('through_feature_column')}: {item.get('match_count')}"
            for item in cumulative
        )
    lines.extend(["", "## Safety Scope", ""])
    safety = artifact.get("safety_scope", {})
    lines.extend(
        [
            f"- historical_only: {safety.get('historical_only')}",
            f"- future_data: {safety.get('future_data')}",
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


def _rel(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root))
    except ValueError:
        return str(path)


def _event_id(created_at: str, seed: str) -> str:
    try:
        parsed = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        prefix = parsed.strftime("%Y%m%dT%H%M%SZ")
    except ValueError:
        prefix = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return _safe_path_component(f"{prefix}_{seed}")


def _safe_path_component(value: str) -> str:
    safe = "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in value)
    return safe.strip("_") or "local_events"


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _round_or_none(value: Any) -> float | None:
    if pd.isna(value):
        return None
    return round(float(value), 6)


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _timestamp_to_str(value: Any) -> str | None:
    if pd.isna(value):
        return None
    return pd.Timestamp(value).isoformat()
