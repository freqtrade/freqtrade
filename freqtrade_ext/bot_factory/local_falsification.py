from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

import pandas as pd


_CONTEXT_MERGE_SEMANTICS = "closed_context_candle_availability_v1"
_FUNDING_RATE_FEATURES = (
    "funding_rate_bps",
    "funding_rate_delta_bps",
)
_MARK_PRICE_FEATURES = (
    "mark_price_gap_bps",
    "mark_price_gap_delta_bps",
    "mark_price_return_bps",
)


@dataclass(frozen=True)
class LocalFalsificationInputs:
    root_dir: Path
    thesis_id: str
    mechanism_class: str
    ohlcv_path: Path
    event_path: Path
    hold_candles: int
    all_in_cost_bps: float
    event_source_path: Path | None = None
    funding_rate_path: Path | None = None
    output_root: Path = Path("registry/strategies/research_decisions")
    falsification_id: str | None = None
    min_sample_count: int = 20
    min_profitable_windows_ratio: float = 0.5
    min_calendar_window_count: int = 0
    min_profitable_calendar_windows_ratio: float = 0.0
    min_data_span_days: float = 0.0
    event_time_column: str = "date"
    reviewer_notes: Sequence[str] = field(default_factory=list)
    created_by_agent: str = "codex"
    created_at: str | None = None
    command: Sequence[str] = field(default_factory=list)


def build_local_falsification(inputs: LocalFalsificationInputs) -> dict[str, Any]:
    root = inputs.root_dir.resolve()
    generated_at = inputs.created_at or datetime.now(UTC).replace(microsecond=0).isoformat()
    falsification_id = inputs.falsification_id or _falsification_id(
        generated_at, inputs.thesis_id
    )
    ohlcv_path = _resolve_inside(inputs.ohlcv_path, root)
    event_path = _resolve_inside(inputs.event_path, root)
    event_source_path = (
        _resolve_inside(inputs.event_source_path, root)
        if inputs.event_source_path is not None
        else None
    )
    funding_rate_path = (
        _resolve_inside(inputs.funding_rate_path, root)
        if inputs.funding_rate_path is not None
        else None
    )
    checks: list[dict[str, Any]] = []

    ohlcv, ohlcv_error = _load_ohlcv(ohlcv_path)
    ohlcv_coverage = _ohlcv_coverage(ohlcv)
    events, events_error = _load_events(event_path, inputs.event_time_column)
    funding_rate, funding_rate_error = (
        _load_funding_rate(funding_rate_path)
        if funding_rate_path is not None
        else (None, None)
    )
    event_source, event_source_error = (
        _load_json(event_source_path) if event_source_path is not None else (None, None)
    )
    event_source_summary = _event_source_summary(
        event_source,
        event_source_path=event_source_path,
        event_source_error=event_source_error,
        root=root,
        thesis_id=str(inputs.thesis_id).strip(),
        event_path=event_path,
        ohlcv_path=ohlcv_path,
    )
    checks.extend(
        [
            _check("ohlcv_file_present", ohlcv_path.is_file(), {"path": _rel(ohlcv_path, root)}),
            _check(
                "ohlcv_parseable",
                ohlcv is not None and not ohlcv_error,
                {"error": ohlcv_error},
            ),
            _check("event_file_present", event_path.is_file(), {"path": _rel(event_path, root)}),
            _check(
                "event_file_parseable",
                events is not None and not events_error,
                {"error": events_error},
            ),
            _check(
                "hold_candles_positive",
                int(inputs.hold_candles) > 0,
                {"hold_candles": inputs.hold_candles},
            ),
            _check(
                "all_in_cost_bps_non_negative",
                float(inputs.all_in_cost_bps) >= 0.0,
                {"all_in_cost_bps": inputs.all_in_cost_bps},
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
        ]
    )
    if event_source_path is not None:
        checks.extend(
            [
                _check(
                    "event_source_file_present",
                    event_source_summary["file_present"],
                    {"path": event_source_summary["path"]},
                ),
                _check(
                    "event_source_parseable",
                    event_source_summary["parseable"],
                    {"error": event_source_summary["error"]},
                ),
                _check(
                    "event_source_factory_valid",
                    event_source_summary["factory_valid"],
                    {"factory": event_source_summary["factory"]},
                ),
                _check(
                    "event_source_completed",
                    event_source_summary["status_completed"],
                    {"status": event_source_summary["status"]},
                ),
                _check(
                    "event_source_thesis_matches",
                    event_source_summary["thesis_matches"],
                    {
                        "event_source_thesis_id": event_source_summary["thesis_id"],
                        "thesis_id": str(inputs.thesis_id).strip(),
                    },
                ),
                _check(
                    "event_source_event_path_matches",
                    event_source_summary["event_path_matches"],
                    {
                        "event_source_events_csv_path": event_source_summary[
                            "events_csv_path"
                        ],
                        "event_path": _rel(event_path, root),
                    },
                ),
                _check(
                    "event_source_ohlcv_path_matches",
                    event_source_summary["ohlcv_path_matches"],
                    {
                        "event_source_ohlcv_path": event_source_summary[
                            "source_ohlcv_path"
                        ],
                        "ohlcv_path": _rel(ohlcv_path, root),
                    },
                ),
                _check(
                    "event_source_safety_scope_valid",
                    event_source_summary["safety_scope_valid"],
                    {},
                ),
                _check(
                    "event_source_closed_context_candle_alignment_valid",
                    event_source_summary["closed_context_candle_alignment_valid"],
                    {
                        "context_features_used": event_source_summary[
                            "context_features_used"
                        ],
                        "context_merge_semantics": event_source_summary[
                            "context_merge_semantics"
                        ],
                        "required_contexts": event_source_summary["required_contexts"],
                    },
                ),
            ]
        )
        funding_required_by_event_source = (
            "funding_rate" in event_source_summary["required_contexts"]
        )
        checks.append(
            _check(
                "funding_rate_path_present_for_funding_event_source",
                not funding_required_by_event_source or funding_rate_path is not None,
                {
                    "funding_rate_required_by_event_source": funding_required_by_event_source,
                    "required_contexts": event_source_summary["required_contexts"],
                    "funding_rate_path": (
                        _rel(funding_rate_path, root) if funding_rate_path else None
                    ),
                },
            )
        )
    if funding_rate_path is not None:
        checks.extend(
            [
                _check(
                    "funding_rate_file_present",
                    funding_rate_path.is_file(),
                    {"path": _rel(funding_rate_path, root)},
                ),
                _check(
                    "funding_rate_parseable",
                    funding_rate is not None and funding_rate_error is None,
                    {"error": funding_rate_error},
                ),
            ]
        )

    event_returns: list[dict[str, Any]] = []
    if ohlcv is not None and events is not None and int(inputs.hold_candles) > 0:
        event_returns = _event_returns(
            ohlcv,
            events,
            hold_candles=int(inputs.hold_candles),
            funding_rate=funding_rate,
        )

    sample_count = len(event_returns)
    expected_price_edge_bps = _mean(
        [item["price_return_bps"] for item in event_returns]
    )
    expected_funding_adjustment_bps = _mean(
        [item["funding_adjustment_bps"] for item in event_returns]
    )
    expected_edge_bps = _mean([item["gross_return_bps"] for item in event_returns])
    median_edge_bps = _median([item["gross_return_bps"] for item in event_returns])
    all_in_cost_bps = float(inputs.all_in_cost_bps)
    net_edge_bps = (
        None if expected_edge_bps is None else round(expected_edge_bps - all_in_cost_bps, 6)
    )
    win_rate = _win_rate([item["gross_return_bps"] for item in event_returns])
    windows = _window_summaries(
        event_returns,
        all_in_cost_bps=all_in_cost_bps,
    )
    calendar_windows = _calendar_window_summaries(
        event_returns,
        all_in_cost_bps=all_in_cost_bps,
    )
    profitable_window_count = sum(1 for item in windows if item["net_edge_bps"] > 0.0)
    window_count = len(windows)
    profitable_windows_ratio = (
        0.0 if window_count == 0 else round(profitable_window_count / window_count, 4)
    )
    profitable_calendar_window_count = sum(
        1 for item in calendar_windows if item["net_edge_bps"] > 0.0
    )
    calendar_window_count = len(calendar_windows)
    profitable_calendar_windows_ratio = (
        0.0
        if calendar_window_count == 0
        else round(profitable_calendar_window_count / calendar_window_count, 4)
    )

    checks.extend(
        [
            _check(
                "event_sample_count_sufficient",
                sample_count >= int(inputs.min_sample_count),
                {
                    "sample_count": sample_count,
                    "min_sample_count": int(inputs.min_sample_count),
                },
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
                    "all_in_cost_bps": all_in_cost_bps,
                    "net_edge_bps": net_edge_bps,
                },
            ),
            _check(
                "profitable_windows_ratio_sufficient",
                profitable_windows_ratio >= float(inputs.min_profitable_windows_ratio),
                {
                    "profitable_windows_ratio": profitable_windows_ratio,
                    "min_profitable_windows_ratio": inputs.min_profitable_windows_ratio,
                },
            ),
            _check(
                "calendar_window_count_sufficient",
                int(inputs.min_calendar_window_count) <= 0
                or calendar_window_count >= int(inputs.min_calendar_window_count),
                {
                    "calendar_window_count": calendar_window_count,
                    "min_calendar_window_count": int(inputs.min_calendar_window_count),
                },
            ),
            _check(
                "profitable_calendar_windows_ratio_sufficient",
                float(inputs.min_profitable_calendar_windows_ratio) <= 0.0
                or profitable_calendar_windows_ratio
                >= float(inputs.min_profitable_calendar_windows_ratio),
                {
                    "profitable_calendar_windows_ratio": profitable_calendar_windows_ratio,
                    "min_profitable_calendar_windows_ratio": float(
                        inputs.min_profitable_calendar_windows_ratio
                    ),
                },
            ),
        ]
    )
    status = "passed" if all(check["status"] == "pass" for check in checks) else "failed"
    return {
        "generated_at": generated_at,
        "factory": "research_local_falsification",
        "falsification_id": falsification_id,
        "status": status,
        "thesis_id": str(inputs.thesis_id).strip(),
        "mechanism_class": str(inputs.mechanism_class).strip(),
        "ohlcv_path": _rel(ohlcv_path, root),
        "ohlcv_row_count": ohlcv_coverage["row_count"],
        "data_start": ohlcv_coverage["data_start"],
        "data_end": ohlcv_coverage["data_end"],
        "data_span_days": ohlcv_coverage["data_span_days"],
        "min_data_span_days": float(inputs.min_data_span_days),
        "event_path": _rel(event_path, root),
        "event_source": event_source_summary,
        "event_time_column": inputs.event_time_column,
        "funding_rate_path": _rel(funding_rate_path, root) if funding_rate_path else None,
        "funding_rate_adjustment": {
            "used": funding_rate_path is not None,
            "required_by_event_source": bool(
                "funding_rate" in event_source_summary["required_contexts"]
            ),
            "path": _rel(funding_rate_path, root) if funding_rate_path else None,
            "parseable": funding_rate is not None and funding_rate_error is None,
            "error": funding_rate_error,
            "long_payment_semantics": (
                "long_funding_adjustment_bps = -sum(funding_rate * 10000) "
                "for funding timestamps after entry and up to exit"
            ),
        },
        "hold_candles": int(inputs.hold_candles),
        "all_in_cost_bps": all_in_cost_bps,
        "expected_price_edge_bps": expected_price_edge_bps,
        "expected_funding_adjustment_bps": expected_funding_adjustment_bps,
        "expected_edge_bps": expected_edge_bps,
        "median_edge_bps": median_edge_bps,
        "net_edge_bps": net_edge_bps,
        "sample_count": sample_count,
        "min_sample_count": int(inputs.min_sample_count),
        "win_rate": win_rate,
        "window_count": window_count,
        "profitable_window_count": profitable_window_count,
        "profitable_windows_ratio": profitable_windows_ratio,
        "min_profitable_windows_ratio": float(inputs.min_profitable_windows_ratio),
        "window_summaries": windows,
        "calendar_window_frequency": "quarter",
        "calendar_window_count": calendar_window_count,
        "min_calendar_window_count": int(inputs.min_calendar_window_count),
        "profitable_calendar_window_count": profitable_calendar_window_count,
        "profitable_calendar_windows_ratio": profitable_calendar_windows_ratio,
        "min_profitable_calendar_windows_ratio": float(
            inputs.min_profitable_calendar_windows_ratio
        ),
        "calendar_window_summaries": calendar_windows,
        "sample_preview": event_returns[:10],
        "checks": checks,
        "blockers": [check for check in checks if check["status"] != "pass"],
        "reviewer_notes": [str(note) for note in inputs.reviewer_notes],
        "created_by_agent": str(inputs.created_by_agent),
        "command": list(inputs.command),
        "safety_scope": {
            "historical_only": True,
            "closed_candle_ohlcv_only": funding_rate_path is None,
            "closed_candle_local_market_data_only": True,
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


def write_local_falsification_artifacts(
    artifact: dict[str, Any], *, root_dir: Path, output_root: Path
) -> tuple[Path, Path]:
    root = root_dir.resolve()
    falsification_id = _safe_path_component(
        str(artifact.get("falsification_id") or "local_falsification")
    )
    out_dir = _resolve_inside(output_root, root) / falsification_id
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "local_falsification.json"
    report_path = out_dir / "local_falsification_report.md"
    json_path.write_text(json.dumps(artifact, indent=2, ensure_ascii=False), encoding="utf-8")
    report_path.write_text(_render_report(artifact), encoding="utf-8")
    return json_path, report_path


def _load_ohlcv(path: Path) -> tuple[pd.DataFrame | None, str | None]:
    if not path.is_file():
        return None, "file_not_found"
    try:
        if path.suffix.lower() == ".parquet":
            frame = pd.read_parquet(path)
        else:
            frame = pd.read_csv(path)
    except Exception as exc:
        return None, f"read_error: {exc}"
    missing = [column for column in ("date", "close") if column not in frame.columns]
    if missing:
        return None, f"missing_columns: {', '.join(missing)}"
    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="coerce")
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    frame = frame.dropna(subset=["date", "close"]).sort_values("date")
    if frame.empty:
        return None, "empty_after_date_close_cleaning"
    return frame[["date", "close"]].reset_index(drop=True), None


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


def _load_events(path: Path, time_column: str) -> tuple[pd.DataFrame | None, str | None]:
    if not path.is_file():
        return None, "file_not_found"
    try:
        if path.suffix.lower() == ".csv":
            frame = pd.read_csv(path)
        elif path.suffix.lower() in {".jsonl", ".ndjson"}:
            frame = pd.read_json(path, lines=True)
        else:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                payload = payload.get("events", [])
            frame = pd.DataFrame(payload)
    except Exception as exc:
        return None, f"read_error: {exc}"
    if time_column not in frame.columns:
        return None, f"missing_event_time_column: {time_column}"
    frame = frame.copy()
    frame[time_column] = pd.to_datetime(frame[time_column], utc=True, errors="coerce")
    frame = frame.dropna(subset=[time_column]).sort_values(time_column)
    if frame.empty:
        return None, "empty_event_file"
    optional = [
        column for column in ("pair", "symbol", "instrument") if column in frame.columns
    ]
    return (
        frame[[time_column, *optional]]
        .rename(columns={time_column: "date"})
        .reset_index(drop=True),
        None,
    )


def _load_funding_rate(path: Path) -> tuple[pd.DataFrame | None, str | None]:
    if not path.is_file():
        return None, "file_not_found"
    try:
        frame = pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)
    except Exception as exc:
        return None, f"read_error: {exc}"
    if "date" not in frame.columns:
        return None, "missing_date_column"
    rate_column = _funding_rate_value_column(frame)
    if rate_column is None:
        return None, "missing_funding_rate_column"
    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="coerce")
    frame["funding_rate"] = pd.to_numeric(frame[rate_column], errors="coerce")
    frame = frame.dropna(subset=["date", "funding_rate"]).sort_values("date")
    if frame.empty:
        return None, "empty_after_funding_rate_cleaning"
    return frame[["date", "funding_rate"]].reset_index(drop=True), None


def _funding_rate_value_column(frame: pd.DataFrame) -> str | None:
    for column in ("funding_rate", "open", "close"):
        if column in frame.columns:
            return column
    return None


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


def _event_source_summary(
    event_source: dict[str, Any] | None,
    *,
    event_source_path: Path | None,
    event_source_error: str | None,
    root: Path,
    thesis_id: str,
    event_path: Path,
    ohlcv_path: Path,
) -> dict[str, Any]:
    source = event_source or {}
    path = _rel(event_source_path, root) if event_source_path is not None else None
    factory = str(source.get("factory") or "").strip()
    status = str(source.get("status") or "").strip()
    source_thesis_id = str(source.get("thesis_id") or "").strip()
    events_csv_path = source.get("events_csv_path")
    source_ohlcv_path = source.get("source_ohlcv_path")
    failure_synthesis = _event_source_failure_synthesis_summary(
        source.get("failure_synthesis_summary")
    )
    context_alignment = _event_source_context_alignment_summary(source)
    return {
        "used": event_source_path is not None,
        "path": path,
        "file_present": bool(event_source_path is not None and event_source_path.is_file()),
        "parseable": bool(event_source is not None and event_source_error is None),
        "error": event_source_error,
        "factory": factory or None,
        "factory_valid": factory == "research_local_event_builder",
        "status": status or None,
        "status_completed": status == "completed",
        "thesis_id": source_thesis_id,
        "thesis_matches": bool(source_thesis_id) and source_thesis_id == thesis_id,
        "events_csv_path": str(events_csv_path) if events_csv_path else None,
        "event_path_matches": _artifact_path_matches(events_csv_path, event_path, root),
        "source_ohlcv_path": str(source_ohlcv_path) if source_ohlcv_path else None,
        "ohlcv_path_matches": _artifact_path_matches(source_ohlcv_path, ohlcv_path, root),
        "event_count": _int_or_none(source.get("event_count")),
        "safety_scope_valid": _event_source_safety_scope_valid(
            source.get("safety_scope")
        ),
        **context_alignment,
        **failure_synthesis,
    }


def _event_source_context_alignment_summary(source: dict[str, Any]) -> dict[str, Any]:
    required_contexts = _event_source_required_contexts(source)
    context_merge = source.get("context_merge")
    context_merge = context_merge if isinstance(context_merge, dict) else {}
    semantics = context_merge.get("semantics")
    context_features_used = bool(required_contexts)
    alignment_valid = (
        not context_features_used
        or (
            semantics == _CONTEXT_MERGE_SEMANTICS
            and context_merge.get("closed_context_candle_alignment") is True
        )
    )
    return {
        "context_features_used": context_features_used,
        "required_contexts": sorted(required_contexts),
        "context_merge_semantics": semantics,
        "closed_context_candle_alignment_valid": alignment_valid,
    }


def _event_source_required_contexts(source: dict[str, Any]) -> set[str]:
    context_merge = source.get("context_merge")
    if isinstance(context_merge, dict):
        raw_required = context_merge.get("required_contexts")
        if isinstance(raw_required, list):
            return {str(item) for item in raw_required if str(item).strip()}

    required: set[str] = set()
    auxiliary = source.get("auxiliary_sources")
    if isinstance(auxiliary, dict):
        for name, summary in auxiliary.items():
            if isinstance(summary, dict) and summary.get("required") is True:
                required.add(str(name))

    for column in source.get("feature_columns") or []:
        column_text = str(column)
        if any(column_text.startswith(feature) for feature in _FUNDING_RATE_FEATURES):
            required.add("funding_rate")
        if any(column_text.startswith(feature) for feature in _MARK_PRICE_FEATURES):
            required.add("mark_price")
    return required


def _event_source_failure_synthesis_summary(summary: Any) -> dict[str, Any]:
    defaults = {
        "failure_synthesis_used": False,
        "failure_synthesis_parseable": False,
        "failure_synthesis_path": None,
        "failure_synthesis_failed_thesis_id_count": None,
        "failure_synthesis_failed_family_count": None,
        "failure_synthesis_thesis_repeats": None,
        "failure_synthesis_mechanism_repeats": None,
        "failure_synthesis_allow_failed_thesis_or_family": None,
        "failure_synthesis_guard_valid": False,
    }
    if not isinstance(summary, dict):
        return defaults

    used = summary.get("used") is True
    parseable = summary.get("parseable") is True
    allow_failed = summary.get("allow_failed_thesis_or_family") is True
    thesis_repeats = summary.get("thesis_repeats_failed_synthesis") is True
    mechanism_repeats = summary.get("mechanism_repeats_failed_synthesis") is True
    return {
        "failure_synthesis_used": used,
        "failure_synthesis_parseable": parseable,
        "failure_synthesis_path": summary.get("path"),
        "failure_synthesis_failed_thesis_id_count": _int_or_none(
            summary.get("failed_thesis_id_count")
        ),
        "failure_synthesis_failed_family_count": _int_or_none(
            summary.get("failed_family_count")
        ),
        "failure_synthesis_thesis_repeats": thesis_repeats,
        "failure_synthesis_mechanism_repeats": mechanism_repeats,
        "failure_synthesis_allow_failed_thesis_or_family": allow_failed,
        "failure_synthesis_guard_valid": bool(
            used
            and parseable
            and not allow_failed
            and not thesis_repeats
            and not mechanism_repeats
        ),
    }


def _artifact_path_matches(raw_path: Any, expected_path: Path, root: Path) -> bool:
    if not raw_path:
        return False
    try:
        candidate = Path(str(raw_path))
        resolved = (candidate if candidate.is_absolute() else root / candidate).resolve()
        resolved.relative_to(root)
    except (OSError, RuntimeError, ValueError):
        return False
    return resolved == expected_path.resolve()


def _event_source_safety_scope_valid(safety_scope: Any) -> bool:
    if not isinstance(safety_scope, dict):
        return False
    unsafe_flags = (
        "future_data",
        "backtest_started",
        "strategy_code_generated",
        "paper_trading_started",
        "dry_run_trading_started",
        "live_trading",
        "exchange_order_placement",
        "shorting",
        "process_control",
    )
    leverage = _float_or_none(safety_scope.get("leverage"))
    closed_candle_local_data = (
        safety_scope.get("closed_candle_ohlcv_only") is True
        or safety_scope.get("closed_candle_local_market_data_only") is True
    )
    return (
        safety_scope.get("historical_only") is True
        and closed_candle_local_data
        and all(not bool(safety_scope.get(flag)) for flag in unsafe_flags)
        and (leverage is None or leverage <= 1.0)
    )


def _event_returns(
    ohlcv: pd.DataFrame,
    events: pd.DataFrame,
    *,
    hold_candles: int,
    funding_rate: pd.DataFrame | None = None,
    entry_semantics: str = "same_candle_close",
) -> list[dict[str, Any]]:
    candle_times = list(ohlcv["date"])
    closes = list(ohlcv["close"])
    opens = list(ohlcv["open"]) if "open" in ohlcv.columns else closes
    rows: list[dict[str, Any]] = []
    semantics = str(entry_semantics or "same_candle_close").strip().lower()
    event_records = events.to_dict("records")
    for event in event_records:
        event_time = event["date"]
        if semantics == "next_candle_open":
            entry_index = int(ohlcv["date"].searchsorted(event_time, side="right"))
            entry_price_type = "open"
            exit_index = entry_index + int(hold_candles) - 1
        else:
            entry_index = int(ohlcv["date"].searchsorted(event_time, side="left"))
            entry_price_type = "close"
            exit_index = entry_index + int(hold_candles)
        if entry_index < 0 or exit_index >= len(ohlcv):
            continue
        entry_price = (
            float(opens[entry_index])
            if entry_price_type == "open"
            else float(closes[entry_index])
        )
        exit_close = float(closes[exit_index])
        if entry_price <= 0.0:
            continue
        price_return_bps = 10000.0 * (exit_close / entry_price - 1.0)
        funding_adjustment_bps = _funding_adjustment_bps(
            funding_rate,
            entry_time=candle_times[entry_index],
            exit_time=candle_times[exit_index],
        )
        gross_return_bps = price_return_bps + funding_adjustment_bps
        row = {
            "event_time": _timestamp_to_str(event_time),
            "entry_time": _timestamp_to_str(candle_times[entry_index]),
            "exit_time": _timestamp_to_str(candle_times[exit_index]),
            "entry_semantics": semantics,
            "entry_price_type": entry_price_type,
            "entry_price": round(entry_price, 8),
            "entry_close": round(float(closes[entry_index]), 8),
            "exit_price_type": "close",
            "exit_price": round(exit_close, 8),
            "exit_close": round(exit_close, 8),
            "price_return_bps": round(price_return_bps, 6),
            "funding_adjustment_bps": round(funding_adjustment_bps, 6),
            "gross_return_bps": round(gross_return_bps, 6),
        }
        for column in ("pair", "symbol", "instrument"):
            value = _string_evidence_or_none(event.get(column))
            if value is not None:
                row[column] = value
        rows.append(row)
    return rows


def _string_evidence_or_none(value: Any) -> str | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return text or None


def _funding_adjustment_bps(
    funding_rate: pd.DataFrame | None, *, entry_time: Any, exit_time: Any
) -> float:
    if funding_rate is None or funding_rate.empty:
        return 0.0
    entry = pd.Timestamp(entry_time)
    exit_ = pd.Timestamp(exit_time)
    mask = (funding_rate["date"] > entry) & (funding_rate["date"] <= exit_)
    if not bool(mask.any()):
        return 0.0
    # Bybit-style funding rates are paid by longs when positive and received by
    # longs when negative. Convert the realized long payment stream to bps.
    return float((-funding_rate.loc[mask, "funding_rate"].astype(float) * 10000.0).sum())


def _window_summaries(
    rows: Sequence[dict[str, Any]], *, all_in_cost_bps: float, window_count: int = 4
) -> list[dict[str, Any]]:
    if not rows:
        return []
    chunk_size = max(1, (len(rows) + window_count - 1) // window_count)
    summaries: list[dict[str, Any]] = []
    for index, start in enumerate(range(0, len(rows), chunk_size), start=1):
        chunk = list(rows[start : start + chunk_size])
        gross_values = [float(item["gross_return_bps"]) for item in chunk]
        expected = _mean(gross_values)
        net = None if expected is None else round(expected - all_in_cost_bps, 6)
        summaries.append(
            {
                "window_index": index,
                "start_event_time": chunk[0]["event_time"],
                "end_event_time": chunk[-1]["event_time"],
                "sample_count": len(chunk),
                "expected_edge_bps": expected,
                "net_edge_bps": net,
                "profitable": bool(net is not None and net > 0.0),
            }
        )
    return summaries


def _calendar_window_summaries(
    rows: Sequence[dict[str, Any]], *, all_in_cost_bps: float
) -> list[dict[str, Any]]:
    if not rows:
        return []
    frame = pd.DataFrame(rows).copy()
    frame["event_time"] = pd.to_datetime(frame["event_time"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["event_time"]).sort_values("event_time")
    if frame.empty:
        return []
    frame["calendar_window"] = (
        frame["event_time"].dt.tz_convert("UTC").dt.tz_localize(None).dt.to_period("Q").astype(str)
    )
    summaries: list[dict[str, Any]] = []
    for window, group in frame.groupby("calendar_window", sort=True):
        gross_values = [float(value) for value in group["gross_return_bps"]]
        price_values = [float(value) for value in group["price_return_bps"]]
        funding_values = [float(value) for value in group["funding_adjustment_bps"]]
        expected = _mean(gross_values)
        net = None if expected is None else round(expected - all_in_cost_bps, 6)
        summaries.append(
            {
                "calendar_window": str(window),
                "start_event_time": _timestamp_to_str(group["event_time"].min()),
                "end_event_time": _timestamp_to_str(group["event_time"].max()),
                "sample_count": int(len(group)),
                "expected_price_edge_bps": _mean(price_values),
                "expected_funding_adjustment_bps": _mean(funding_values),
                "expected_edge_bps": expected,
                "net_edge_bps": net,
                "win_rate": _win_rate(gross_values),
                "profitable": bool(net is not None and net > 0.0),
            }
        )
    return summaries


def _mean(values: Sequence[float]) -> float | None:
    numbers = [float(value) for value in values]
    if not numbers:
        return None
    return round(sum(numbers) / len(numbers), 6)


def _median(values: Sequence[float]) -> float | None:
    numbers = sorted(float(value) for value in values)
    if not numbers:
        return None
    midpoint = len(numbers) // 2
    if len(numbers) % 2:
        return round(numbers[midpoint], 6)
    return round((numbers[midpoint - 1] + numbers[midpoint]) / 2.0, 6)


def _win_rate(values: Sequence[float]) -> float | None:
    numbers = [float(value) for value in values]
    if not numbers:
        return None
    return round(sum(1 for value in numbers if value > 0.0) / len(numbers), 4)


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


def _check(name: str, passed: bool, details: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "name": name,
        "status": "pass" if passed else "fail",
        "details": details or {},
    }


def _render_report(artifact: dict[str, Any]) -> str:
    lines = [
        "# Local Falsification Evidence",
        "",
        f"- falsification_id: {artifact.get('falsification_id')}",
        f"- status: {artifact.get('status')}",
        f"- thesis_id: {artifact.get('thesis_id')}",
        f"- mechanism_class: {artifact.get('mechanism_class')}",
        f"- expected_edge_bps: {artifact.get('expected_edge_bps')}",
        f"- expected_price_edge_bps: {artifact.get('expected_price_edge_bps')}",
        f"- expected_funding_adjustment_bps: {artifact.get('expected_funding_adjustment_bps')}",
        f"- all_in_cost_bps: {artifact.get('all_in_cost_bps')}",
        f"- net_edge_bps: {artifact.get('net_edge_bps')}",
        f"- sample_count: {artifact.get('sample_count')}",
        f"- data_span_days: {artifact.get('data_span_days')}",
        f"- profitable_windows_ratio: {artifact.get('profitable_windows_ratio')}",
        f"- calendar_window_count: {artifact.get('calendar_window_count')}",
        f"- profitable_calendar_windows_ratio: {artifact.get('profitable_calendar_windows_ratio')}",
        "",
        "## Checks",
        "",
    ]
    lines.extend(
        f"- {item.get('name')}: {item.get('status')}"
        for item in artifact.get("checks", [])
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


def _resolve_inside(path: Path, root: Path) -> Path:
    resolved = (path if path.is_absolute() else root / path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Path must be inside the workspace: {path}") from exc
    return resolved


def _rel(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root))
    except ValueError:
        return str(path)


def _falsification_id(created_at: str, thesis_id: str) -> str:
    try:
        parsed = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        prefix = parsed.strftime("%Y%m%dT%H%M%SZ")
    except ValueError:
        prefix = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return _safe_path_component(f"{prefix}_{thesis_id}")


def _safe_path_component(value: str) -> str:
    safe = "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in value)
    return safe.strip("_") or "local_falsification"


def _timestamp_to_str(value: Any) -> str | None:
    if pd.isna(value):
        return None
    return pd.Timestamp(value).isoformat()
