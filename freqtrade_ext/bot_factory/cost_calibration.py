from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from freqtrade_ext.bot_factory.cost_model import CostModelContext


_SCENARIO_NAMES = ("best", "normal", "stress")
_MAKER_REQUIRED_FIELDS = (
    "no_fill_rate",
    "partial_fill_rate",
    "adverse_selection_bps",
    "exit_taker_rate",
)
_COST_TABLE_COLUMNS = (
    "scenario_name",
    "total_cost_bps",
    "fee_bps_entry",
    "fee_bps_exit",
    "spread_bps",
    "slippage_bps_entry",
    "slippage_bps_exit",
    "adverse_selection_bps",
    "no_fill_rate",
    "partial_fill_rate",
    "exit_taker_rate",
    "stress_multiplier",
    "pair",
    "timeframe",
    "order_type",
    "liquidity_tier",
    "volatility_regime",
)


@dataclass(frozen=True)
class CostCalibrationInputs:
    root_dir: Path
    ohlcv_path: Path | None = None
    order_book_path: Path | None = None
    spread_path: Path | None = None
    fills_path: Path | None = None
    pair: str | None = None
    timeframe: str | None = None
    order_type: str | None = None
    liquidity_tier: str | None = None
    volatility_regime: str | None = None
    fee_bps_entry: float = 3.0
    fee_bps_exit: float = 3.0
    cost_calibration_id: str | None = None
    output_root: Path = Path("data/cost_calibration")
    reviewer_notes: list[str] = field(default_factory=list)
    created_at: str | None = None
    command: list[str] = field(default_factory=list)
    created_by_agent: str = "codex"


@dataclass(frozen=True)
class _SourceLoad:
    name: str
    path: Path | None
    status: str
    frame: pd.DataFrame | None = None
    blocker: dict[str, Any] | None = None
    summary: dict[str, Any] = field(default_factory=dict)


def build_cost_calibration(inputs: CostCalibrationInputs) -> dict[str, Any]:
    generated_at = inputs.created_at or datetime.now(UTC).isoformat()
    calibration_id = inputs.cost_calibration_id or _default_calibration_id(generated_at)
    context = CostModelContext(
        pair=_string_or_none(inputs.pair),
        timeframe=_string_or_none(inputs.timeframe),
        order_type=_string_or_none(inputs.order_type),
        liquidity_tier=_string_or_none(inputs.liquidity_tier),
        volatility_regime=_string_or_none(inputs.volatility_regime),
    )
    ohlcv_path = _resolve_input_path(inputs.ohlcv_path, inputs.root_dir)
    order_book_path = _resolve_input_path(inputs.order_book_path, inputs.root_dir)
    spread_path = _resolve_input_path(inputs.spread_path, inputs.root_dir)
    fills_path = _resolve_input_path(inputs.fills_path, inputs.root_dir)
    sources = {
        "ohlcv": _load_frame(
            "ohlcv",
            ohlcv_path,
            required=True,
            required_columns=("date", "open", "high", "low", "close"),
        ),
        "order_book": _load_frame(
            "order_book",
            order_book_path,
            required=False,
            any_column_groups=(
                ("spread_bps",),
                ("best_bid", "best_ask"),
                ("bid_size", "ask_size"),
            ),
        ),
        "spread": _load_frame(
            "spread",
            spread_path,
            required=False,
            any_column_groups=(("spread_bps",), ("best_bid", "best_ask")),
        ),
    }
    fills_source, fills_by_scenario = _load_fills(fills_path, context=context)
    source_blockers = [
        source.blocker
        for source in [*sources.values(), fills_source]
        if source.blocker is not None
    ]
    ohlcv_frame = _loaded_frame(sources["ohlcv"])
    order_book_frame = _loaded_frame(sources["order_book"])
    spread_frame = _loaded_frame(sources["spread"])
    ohlcv_numeric_blocker = _ohlcv_numeric_blocker(ohlcv_frame)
    if ohlcv_numeric_blocker is not None:
        source_blockers.append(ohlcv_numeric_blocker)
    scenarios = _estimate_scenarios(
        inputs=inputs,
        context=context,
        ohlcv=ohlcv_frame,
        order_book=order_book_frame,
        spread=spread_frame,
        fills_by_scenario=fills_by_scenario,
    )
    scenario_blockers = validate_cost_scenarios(scenarios, context=context)
    blockers = [*source_blockers, *scenario_blockers]
    status = "completed" if not blockers else "blocked"
    return {
        "generated_at": generated_at,
        "factory": "cost_calibration",
        "cost_calibration_id": calibration_id,
        "status": status,
        "candidate_generation_allowed": False,
        "proposal_generation_allowed": False,
        "strategy_codegen_allowed": False,
        "candidate_generation_result": "no candidate generated",
        "blocked_next_actions": [
            "strategy_generation_from_cost_calibration",
            "proposal_generation_from_cost_calibration",
            "research_thesis_selection_from_uncalibrated_costs",
            "backtest_from_cost_calibration",
            "paper_or_live_trading_from_cost_calibration",
        ],
        "cost_model_context": _context_summary(context),
        "source_ohlcv_path": _rel(ohlcv_path, inputs.root_dir),
        "source_order_book_path": _rel(order_book_path, inputs.root_dir),
        "source_spread_path": _rel(spread_path, inputs.root_dir),
        "source_fills_path": _rel(fills_path, inputs.root_dir),
        "sources": {
            **{name: _source_summary(source, inputs.root_dir) for name, source in sources.items()},
            "fills": _source_summary(fills_source, inputs.root_dir),
        },
        "cost_scenarios": scenarios,
        "cost_table": [_table_row(scenarios[name], context) for name in _SCENARIO_NAMES if name in scenarios],
        "blockers": blockers,
        "reviewer_notes": [str(note) for note in inputs.reviewer_notes],
        "created_by_agent": str(inputs.created_by_agent),
        "command": list(inputs.command),
        "safety_scope": {
            "local_data_only": True,
            "historical_only": True,
            "strategy_candidate_generated": False,
            "research_thesis_generated": False,
            "backtest_started": False,
            "paper_trading_started": False,
            "dry_run_started": False,
            "live_trading_started": False,
            "exchange_order_endpoint_used": False,
            "api_keys_or_secrets_used": False,
        },
    }


def validate_cost_scenarios(
    scenarios: Mapping[str, Mapping[str, Any]], *, context: CostModelContext
) -> list[dict[str, Any]]:
    blockers: list[dict[str, Any]] = []
    for name in _SCENARIO_NAMES:
        scenario = scenarios.get(name)
        total = _float_or_none((scenario or {}).get("total_cost_bps"))
        if scenario is None or total is None:
            blockers.append(
                _blocker(
                    f"{name}_cost_missing",
                    f"{name} total cost is missing.",
                    severity="blocker",
                    details={"scenario_name": name},
                )
            )
    normal_total = _float_or_none((scenarios.get("normal") or {}).get("total_cost_bps"))
    stress_total = _float_or_none((scenarios.get("stress") or {}).get("total_cost_bps"))
    if normal_total is None and not any(
        blocker["name"] == "normal_cost_missing" for blocker in blockers
    ):
        blockers.append(
            _blocker(
                "normal_cost_missing",
                "normal cost is required before cost calibration can complete.",
                severity="blocker",
            )
        )
    if normal_total is not None and stress_total is not None and stress_total < normal_total:
        blockers.append(
            _blocker(
                "stress_cost_below_normal",
                "stress cost must be greater than or equal to normal cost.",
                severity="blocker",
                details={
                    "normal_total_cost_bps": normal_total,
                    "stress_total_cost_bps": stress_total,
                },
            )
        )
    if _normalize(context.order_type) == "maker":
        for field_name in _MAKER_REQUIRED_FIELDS:
            missing = [
                name
                for name in _SCENARIO_NAMES
                if _float_or_none((scenarios.get(name) or {}).get(field_name)) is None
            ]
            if missing:
                blockers.append(
                    _blocker(
                        f"maker_{field_name}_missing",
                        f"maker context requires {field_name} for every scenario.",
                        severity="blocker",
                        details={"missing_scenarios": missing},
                    )
                )
    return blockers


def write_cost_calibration_artifacts(
    artifact: Mapping[str, Any],
    *,
    root_dir: Path,
    output_root: Path,
) -> tuple[Path, Path, Path]:
    out_dir = _resolve_output_root(output_root, root_dir) / str(
        artifact.get("cost_calibration_id") or "cost_calibration"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "cost_calibration.json"
    report_path = out_dir / "cost_calibration_report.md"
    table_path = out_dir / "cost_table.csv"
    json_path.write_text(json.dumps(artifact, indent=2, ensure_ascii=False), encoding="utf-8")
    report_path.write_text(render_cost_calibration_report(artifact), encoding="utf-8")
    _write_cost_table_csv(artifact.get("cost_table") or [], table_path)
    return json_path, report_path, table_path


def render_cost_calibration_report(artifact: Mapping[str, Any]) -> str:
    context = artifact.get("cost_model_context") or {}
    lines = [
        "# Bot Factory Cost Calibration",
        "",
        f"- cost_calibration_id: {artifact.get('cost_calibration_id')}",
        f"- status: {artifact.get('status')}",
        f"- candidate_generation_result: {artifact.get('candidate_generation_result')}",
        f"- pair: {context.get('pair')}",
        f"- timeframe: {context.get('timeframe')}",
        f"- order_type: {context.get('order_type')}",
        f"- liquidity_tier: {context.get('liquidity_tier')}",
        f"- volatility_regime: {context.get('volatility_regime')}",
        "",
        "## Cost Scenarios",
        "",
        "| scenario | total_cost_bps | fee_entry | fee_exit | spread | slippage_entry | slippage_exit | adverse_selection | no_fill_rate | partial_fill_rate | exit_taker_rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in _SCENARIO_NAMES:
        scenario = (artifact.get("cost_scenarios") or {}).get(name) or {}
        lines.append(
            "| {name} | {total} | {fee_entry} | {fee_exit} | {spread} | "
            "{slip_entry} | {slip_exit} | {adverse} | {no_fill} | "
            "{partial} | {exit_taker} |".format(
                name=name,
                total=_display(scenario.get("total_cost_bps")),
                fee_entry=_display(scenario.get("fee_bps_entry")),
                fee_exit=_display(scenario.get("fee_bps_exit")),
                spread=_display(scenario.get("spread_bps")),
                slip_entry=_display(scenario.get("slippage_bps_entry")),
                slip_exit=_display(scenario.get("slippage_bps_exit")),
                adverse=_display(scenario.get("adverse_selection_bps")),
                no_fill=_display(scenario.get("no_fill_rate")),
                partial=_display(scenario.get("partial_fill_rate")),
                exit_taker=_display(scenario.get("exit_taker_rate")),
            )
        )
    lines.extend(["", "## Source Artifacts", ""])
    for name, source in (artifact.get("sources") or {}).items():
        if not isinstance(source, Mapping):
            continue
        lines.append(
            f"- {name}: status={source.get('status')}, path={source.get('path')}, "
            f"rows={source.get('row_count')}, blocker={source.get('blocker_name')}"
        )
    lines.extend(["", "## Blockers", ""])
    blockers = list(artifact.get("blockers") or [])
    if blockers:
        for blocker in blockers:
            lines.append(
                f"- {blocker.get('name')}: {blocker.get('message')} "
                f"({blocker.get('severity')})"
            )
    else:
        lines.append("- None")
    lines.extend(
        [
            "",
            "## Safety",
            "",
            "- local_data_only: True",
            "- strategy_candidate_generated: False",
            "- research_thesis_generated: False",
            "- backtest_started: False",
            "- paper_or_live_trading_started: False",
            "- candidate_generation_result: no candidate generated",
        ]
    )
    return "\n".join(lines) + "\n"


def _estimate_scenarios(
    *,
    inputs: CostCalibrationInputs,
    context: CostModelContext,
    ohlcv: pd.DataFrame | None,
    order_book: pd.DataFrame | None,
    spread: pd.DataFrame | None,
    fills_by_scenario: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    if ohlcv is None or ohlcv.empty:
        return {}
    ohlcv_metrics = _ohlcv_metrics(ohlcv)
    spread_metrics = _spread_metrics(spread) or _spread_metrics(order_book)
    spread_source = "spread_artifact" if _spread_metrics(spread) else "order_book"
    if spread_metrics is None:
        spread_metrics = _ohlcv_spread_proxy(ohlcv_metrics)
        spread_source = "ohlcv_range_proxy"
    maker_fill_estimates = _maker_fill_estimates(order_book)
    scenarios: dict[str, dict[str, Any]] = {}
    for name in _SCENARIO_NAMES:
        fill_values = fills_by_scenario.get(name, {})
        scenario = _scenario_from_estimates(
            name=name,
            inputs=inputs,
            context=context,
            ohlcv_metrics=ohlcv_metrics,
            spread_metrics=spread_metrics,
            spread_source=spread_source,
            fill_values=fill_values,
            maker_fill_estimates=maker_fill_estimates,
        )
        scenarios[name] = scenario
    return scenarios


def _scenario_from_estimates(
    *,
    name: str,
    inputs: CostCalibrationInputs,
    context: CostModelContext,
    ohlcv_metrics: Mapping[str, float],
    spread_metrics: Mapping[str, float],
    spread_source: str,
    fill_values: Mapping[str, Any],
    maker_fill_estimates: Mapping[str, Mapping[str, float]] | None,
) -> dict[str, Any]:
    order_type = _normalize(context.order_type)
    index = {"best": 0, "normal": 1, "stress": 2}[name]
    spread_values = (
        max(0.0, float(spread_metrics["best"])),
        max(0.0, float(spread_metrics["normal"])),
        max(0.0, float(spread_metrics["stress"])),
    )
    slippage_values = _slippage_values(ohlcv_metrics)
    fee_entry = max(0.0, float(inputs.fee_bps_entry))
    fee_exit = max(0.0, float(inputs.fee_bps_exit))
    fee_discount = 0.75 if name == "best" else 1.0
    scenario = {
        "scenario_name": name,
        "fee_bps_entry": round(fee_entry * fee_discount, 6),
        "fee_bps_exit": round(fee_exit * fee_discount, 6),
        "spread_bps": round(spread_values[index], 6),
        "slippage_bps_entry": round(slippage_values[index], 6),
        "slippage_bps_exit": round(slippage_values[index], 6),
        "adverse_selection_bps": _risk_value(
            "adverse_selection_bps",
            name,
            fill_values,
            order_type=order_type,
            spread_bps=spread_values[index],
            slippage_bps=slippage_values[index],
            maker_fill_estimates=maker_fill_estimates,
        ),
        "no_fill_rate": _risk_value(
            "no_fill_rate",
            name,
            fill_values,
            order_type=order_type,
            spread_bps=spread_values[index],
            slippage_bps=slippage_values[index],
            maker_fill_estimates=maker_fill_estimates,
        ),
        "partial_fill_rate": _risk_value(
            "partial_fill_rate",
            name,
            fill_values,
            order_type=order_type,
            spread_bps=spread_values[index],
            slippage_bps=slippage_values[index],
            maker_fill_estimates=maker_fill_estimates,
        ),
        "exit_taker_rate": _risk_value(
            "exit_taker_rate",
            name,
            fill_values,
            order_type=order_type,
            spread_bps=spread_values[index],
            slippage_bps=slippage_values[index],
            maker_fill_estimates=maker_fill_estimates,
        ),
        "stress_multiplier": 1.0,
        "pair": context.pair,
        "timeframe": context.timeframe,
        "order_type": context.order_type,
        "liquidity_tier": context.liquidity_tier,
        "volatility_regime": context.volatility_regime,
        "provenance": {
            "spread_source": spread_source,
            "slippage_source": "ohlcv_abs_return_distribution",
            "maker_fill_source": (
                "fills_artifact"
                if fill_values
                else "order_book_depth_proxy"
                if maker_fill_estimates is not None
                else "missing"
            ),
        },
    }
    override_total = _float_or_none(fill_values.get("total_cost_bps"))
    if override_total is not None:
        scenario["total_cost_bps"] = round(max(0.0, override_total), 6)
    else:
        total_components = [
            scenario["fee_bps_entry"],
            scenario["fee_bps_exit"],
            scenario["spread_bps"],
            scenario["slippage_bps_entry"],
            scenario["slippage_bps_exit"],
            scenario["adverse_selection_bps"],
        ]
        scenario["total_cost_bps"] = (
            None
            if any(value is None for value in total_components)
            else round(sum(float(value) for value in total_components), 6)
        )
    return scenario


def _load_frame(
    name: str,
    path: Path | None,
    *,
    required: bool,
    required_columns: tuple[str, ...] = (),
    any_column_groups: tuple[tuple[str, ...], ...] = (),
) -> _SourceLoad:
    if path is None:
        blocker = (
            _blocker(
                f"{name}_path_missing",
                f"{name} path is required.",
                details={"source": name},
            )
            if required
            else None
        )
        return _SourceLoad(name=name, path=None, status="missing" if required else "skipped", blocker=blocker)
    try:
        frame = _read_frame(path)
    except Exception as exc:
        return _SourceLoad(
            name=name,
            path=path,
            status="blocked",
            blocker=_blocker(
                f"{name}_parse_error",
                f"{name} artifact could not be parsed.",
                details={"path": str(path), "error": str(exc)},
            ),
        )
    normalized_columns = {_normalize(column): column for column in frame.columns}
    missing_columns = [
        column for column in required_columns if _normalize(column) not in normalized_columns
    ]
    if missing_columns:
        return _SourceLoad(
            name=name,
            path=path,
            status="blocked",
            frame=frame,
            blocker=_blocker(
                f"{name}_required_columns_missing",
                f"{name} artifact is missing required columns.",
                details={"missing_columns": missing_columns},
            ),
            summary={"row_count": int(len(frame)), "columns": list(map(str, frame.columns))},
        )
    if any_column_groups and not any(
        all(_normalize(column) in normalized_columns for column in group)
        for group in any_column_groups
    ):
        return _SourceLoad(
            name=name,
            path=path,
            status="blocked",
            frame=frame,
            blocker=_blocker(
                f"{name}_usable_columns_missing",
                f"{name} artifact is missing usable spread or depth columns.",
                details={"required_any": [list(group) for group in any_column_groups]},
            ),
            summary={"row_count": int(len(frame)), "columns": list(map(str, frame.columns))},
        )
    if frame.empty:
        return _SourceLoad(
            name=name,
            path=path,
            status="blocked",
            frame=frame,
            blocker=_blocker(
                f"{name}_rows_missing",
                f"{name} artifact contains no rows.",
                details={"path": str(path)},
            ),
        )
    return _SourceLoad(
        name=name,
        path=path,
        status="loaded",
        frame=_normalize_columns(frame),
        summary={"row_count": int(len(frame)), "columns": list(map(str, frame.columns))},
    )


def _load_fills(
    path: Path | None, *, context: CostModelContext
) -> tuple[_SourceLoad, dict[str, dict[str, Any]]]:
    if path is None:
        return _SourceLoad(name="fills", path=None, status="skipped"), {}
    try:
        if path.suffix.lower() == ".json":
            raw = json.loads(path.read_text(encoding="utf-8"))
            row_count = _json_fill_candidate_count(raw)
            scenarios = _fill_scenarios_from_json(raw, context=context)
        else:
            frame = _read_frame(path)
            scenarios = _fill_scenarios_from_frame(frame, context=context)
            row_count = int(len(frame))
    except Exception as exc:
        return (
            _SourceLoad(
                name="fills",
                path=path,
                status="blocked",
                blocker=_blocker(
                    "fills_artifact_parse_error",
                    "fills artifact could not be parsed.",
                    details={"path": str(path), "error": str(exc)},
                ),
            ),
            {},
        )
    if row_count > 0 and not scenarios:
        return (
            _SourceLoad(
                name="fills",
                path=path,
                status="blocked",
                blocker=_blocker(
                    "fills_scenarios_missing",
                    "fills artifact loaded but no rows matched the calibration scenarios and context.",
                    details={
                        "path": str(path),
                        "row_count": row_count,
                        "context": _context_summary(context),
                    },
                ),
                summary={"row_count": row_count, "scenario_count": 0},
            ),
            {},
        )
    return (
        _SourceLoad(
            name="fills",
            path=path,
            status="loaded",
            summary={"row_count": row_count, "scenario_count": len(scenarios)},
        ),
        scenarios,
    )


def _read_frame(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix == ".json":
        return pd.read_json(path)
    return pd.read_csv(path)


def _fill_scenarios_from_json(
    raw: Any, *, context: CostModelContext
) -> dict[str, dict[str, Any]]:
    if isinstance(raw, Mapping):
        raw_scenarios = raw.get("scenarios")
        if isinstance(raw_scenarios, Mapping):
            candidates = [
                {"scenario_name": name, **value}
                for name, value in raw_scenarios.items()
                if isinstance(value, Mapping)
            ]
        elif isinstance(raw_scenarios, list):
            candidates = [item for item in raw_scenarios if isinstance(item, Mapping)]
        else:
            candidates = [raw]
    elif isinstance(raw, list):
        candidates = [item for item in raw if isinstance(item, Mapping)]
    else:
        candidates = []
    scenarios: dict[str, dict[str, Any]] = {}
    for item in candidates:
        if not _context_matches(item, context):
            continue
        name = _scenario_name_from_fill_row(item)
        if name in _SCENARIO_NAMES:
            scenarios[name] = {str(key): value for key, value in item.items()}
    return scenarios


def _fill_scenarios_from_frame(
    frame: pd.DataFrame, *, context: CostModelContext
) -> dict[str, dict[str, Any]]:
    normalized = _normalize_columns(frame)
    scenarios: dict[str, dict[str, Any]] = {}
    for item in normalized.to_dict(orient="records"):
        if not _context_matches(item, context):
            continue
        name = _scenario_name_from_fill_row(item)
        if name in _SCENARIO_NAMES:
            scenarios[name] = dict(item)
    return scenarios


def _json_fill_candidate_count(raw: Any) -> int:
    if isinstance(raw, Mapping):
        raw_scenarios = raw.get("scenarios")
        if isinstance(raw_scenarios, Mapping):
            return len(raw_scenarios)
        if isinstance(raw_scenarios, list):
            return len(raw_scenarios)
        return 1
    if isinstance(raw, list):
        return len(raw)
    return 0


def _scenario_name_from_fill_row(item: Mapping[str, Any]) -> str | None:
    raw_name = _string_or_none(item.get("scenario_name"))
    if raw_name is None:
        return "normal"
    return _scenario_name(raw_name)


def _context_matches(item: Mapping[str, Any], context: CostModelContext) -> bool:
    for field_name in (
        "pair",
        "timeframe",
        "order_type",
        "liquidity_tier",
        "volatility_regime",
    ):
        actual = _string_or_none(getattr(context, field_name))
        if actual is None:
            continue
        expected = _string_or_none(item.get(field_name))
        if expected is not None and _normalize(expected) != _normalize(actual):
            return False
    return True


def _ohlcv_metrics(frame: pd.DataFrame) -> dict[str, float]:
    normalized = _normalize_columns(frame)
    close = pd.to_numeric(normalized["close"], errors="coerce")
    high = pd.to_numeric(normalized["high"], errors="coerce")
    low = pd.to_numeric(normalized["low"], errors="coerce")
    valid = close.gt(0) & high.notna() & low.notna()
    range_bps = (((high[valid] - low[valid]).abs() / close[valid]) * 10000.0).dropna()
    abs_return_bps = (
        close.pct_change(fill_method=None).abs() * 10000.0
    ).replace([float("inf")], pd.NA).dropna()
    return {
        "range_p25_bps": _quantile_or_default(range_bps, 0.25, 4.0),
        "range_median_bps": _quantile_or_default(range_bps, 0.50, 8.0),
        "range_p90_bps": _quantile_or_default(range_bps, 0.90, 16.0),
        "abs_return_p25_bps": _quantile_or_default(abs_return_bps, 0.25, 1.0),
        "abs_return_p75_bps": _quantile_or_default(abs_return_bps, 0.75, 4.0),
        "abs_return_p90_bps": _quantile_or_default(abs_return_bps, 0.90, 8.0),
    }


def _ohlcv_numeric_blocker(frame: pd.DataFrame | None) -> dict[str, Any] | None:
    if frame is None or frame.empty:
        return None
    normalized = _normalize_columns(frame)
    close = pd.to_numeric(normalized["close"], errors="coerce")
    high = pd.to_numeric(normalized["high"], errors="coerce")
    low = pd.to_numeric(normalized["low"], errors="coerce")
    valid = close.gt(0) & high.notna() & low.notna()
    if bool(valid.any()):
        return None
    return _blocker(
        "ohlcv_numeric_rows_missing",
        "OHLCV artifact has required columns but no numeric high/low/close rows.",
        details={"required_columns": ["high", "low", "close"]},
    )


def _spread_metrics(frame: pd.DataFrame | None) -> dict[str, float] | None:
    if frame is None or frame.empty:
        return None
    normalized = _normalize_columns(frame)
    if "spread_bps" in normalized.columns:
        spread_bps = pd.to_numeric(normalized["spread_bps"], errors="coerce").dropna()
    elif {"best_bid", "best_ask"}.issubset(set(normalized.columns)):
        bid = pd.to_numeric(normalized["best_bid"], errors="coerce")
        ask = pd.to_numeric(normalized["best_ask"], errors="coerce")
        mid = (bid + ask) / 2.0
        spread_bps = (((ask - bid).abs() / mid) * 10000.0).dropna()
    else:
        return None
    if spread_bps.empty:
        return None
    return {
        "best": _quantile_or_default(spread_bps, 0.25, 1.0),
        "normal": _quantile_or_default(spread_bps, 0.50, 2.0),
        "stress": max(
            _quantile_or_default(spread_bps, 0.90, 4.0),
            _quantile_or_default(spread_bps, 0.50, 2.0) * 1.5,
        ),
    }


def _ohlcv_spread_proxy(ohlcv_metrics: Mapping[str, float]) -> dict[str, float]:
    normal = max(1.0, float(ohlcv_metrics["range_median_bps"]) * 0.08)
    return {
        "best": max(0.5, float(ohlcv_metrics["range_p25_bps"]) * 0.05),
        "normal": normal,
        "stress": max(normal * 1.75, float(ohlcv_metrics["range_p90_bps"]) * 0.12),
    }


def _slippage_values(ohlcv_metrics: Mapping[str, float]) -> tuple[float, float, float]:
    best = max(0.1, float(ohlcv_metrics["abs_return_p25_bps"]) * 0.05)
    normal = max(0.5, float(ohlcv_metrics["abs_return_p75_bps"]) * 0.08)
    stress = max(normal * 1.5, float(ohlcv_metrics["abs_return_p90_bps"]) * 0.12)
    return (round(best, 6), round(normal, 6), round(stress, 6))


def _maker_fill_estimates(frame: pd.DataFrame | None) -> dict[str, dict[str, float]] | None:
    if frame is None or frame.empty:
        return None
    normalized = _normalize_columns(frame)
    if not {"bid_size", "ask_size"}.issubset(set(normalized.columns)):
        return None
    bid_size = pd.to_numeric(normalized["bid_size"], errors="coerce")
    ask_size = pd.to_numeric(normalized["ask_size"], errors="coerce")
    total = (bid_size + ask_size).replace(0, pd.NA)
    imbalance = ((ask_size - bid_size).abs() / total).dropna()
    pressure = _quantile_or_default(imbalance, 0.75, 0.25)
    normal_no_fill = min(0.5, max(0.08, 0.08 + pressure * 0.25))
    return {
        "best": {
            "no_fill_rate": round(max(0.02, normal_no_fill * 0.5), 6),
            "partial_fill_rate": 0.08,
            "exit_taker_rate": 0.25,
        },
        "normal": {
            "no_fill_rate": round(normal_no_fill, 6),
            "partial_fill_rate": 0.16,
            "exit_taker_rate": 0.5,
        },
        "stress": {
            "no_fill_rate": round(min(0.85, normal_no_fill * 2.0), 6),
            "partial_fill_rate": 0.32,
            "exit_taker_rate": 0.8,
        },
    }


def _risk_value(
    field_name: str,
    scenario_name: str,
    fill_values: Mapping[str, Any],
    *,
    order_type: str,
    spread_bps: float,
    slippage_bps: float,
    maker_fill_estimates: Mapping[str, Mapping[str, float]] | None,
) -> float | None:
    parsed = _float_or_none(fill_values.get(field_name))
    if parsed is not None:
        return round(max(0.0, parsed), 6)
    if order_type != "maker":
        if field_name in {"no_fill_rate", "partial_fill_rate"}:
            return 0.0
        if field_name == "exit_taker_rate":
            return 1.0
        multiplier = {"best": 0.0, "normal": 0.25, "stress": 0.45}[scenario_name]
        return round(max(0.0, slippage_bps * multiplier), 6)
    if maker_fill_estimates is None:
        return None
    if field_name == "adverse_selection_bps":
        multiplier = {"best": 0.1, "normal": 0.25, "stress": 0.5}[scenario_name]
        floor = {"best": 0.0, "normal": 0.5, "stress": 1.0}[scenario_name]
        return round(max(floor, spread_bps * multiplier), 6)
    value = maker_fill_estimates.get(scenario_name, {}).get(field_name)
    return None if value is None else round(max(0.0, float(value)), 6)


def _write_cost_table_csv(rows: Any, path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(_COST_TABLE_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in _COST_TABLE_COLUMNS})


def _table_row(scenario: Mapping[str, Any], context: CostModelContext) -> dict[str, Any]:
    return {
        "scenario_name": scenario.get("scenario_name"),
        "total_cost_bps": scenario.get("total_cost_bps"),
        "fee_bps_entry": scenario.get("fee_bps_entry"),
        "fee_bps_exit": scenario.get("fee_bps_exit"),
        "spread_bps": scenario.get("spread_bps"),
        "slippage_bps_entry": scenario.get("slippage_bps_entry"),
        "slippage_bps_exit": scenario.get("slippage_bps_exit"),
        "adverse_selection_bps": scenario.get("adverse_selection_bps"),
        "no_fill_rate": scenario.get("no_fill_rate"),
        "partial_fill_rate": scenario.get("partial_fill_rate"),
        "exit_taker_rate": scenario.get("exit_taker_rate"),
        "stress_multiplier": scenario.get("stress_multiplier"),
        "pair": scenario.get("pair") or context.pair,
        "timeframe": scenario.get("timeframe") or context.timeframe,
        "order_type": scenario.get("order_type") or context.order_type,
        "liquidity_tier": scenario.get("liquidity_tier") or context.liquidity_tier,
        "volatility_regime": scenario.get("volatility_regime") or context.volatility_regime,
    }


def _source_summary(source: _SourceLoad, root_dir: Path) -> dict[str, Any]:
    blocker = source.blocker or {}
    return {
        "status": source.status,
        "path": _rel(source.path, root_dir),
        "row_count": source.summary.get("row_count"),
        "columns": source.summary.get("columns"),
        "blocker_name": blocker.get("name"),
        "blocker_message": blocker.get("message"),
    }


def _loaded_frame(source: _SourceLoad) -> pd.DataFrame | None:
    return source.frame if source.status == "loaded" else None


def _blocker(
    name: str,
    message: str,
    *,
    severity: str = "blocker",
    details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "name": name,
        "status": "fail",
        "severity": severity,
        "message": message,
        "details": dict(details or {}),
    }


def _normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.rename(columns={column: _normalize(column) for column in frame.columns})


def _quantile_or_default(series: pd.Series, quantile: float, default: float) -> float:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return float(default)
    return round(max(0.0, float(numeric.quantile(quantile))), 6)


def _scenario_name(value: Any) -> str | None:
    text = _normalize(value)
    return text if text in _SCENARIO_NAMES else None


def _context_summary(context: CostModelContext) -> dict[str, Any]:
    return {
        "pair": context.pair,
        "timeframe": context.timeframe,
        "order_type": context.order_type,
        "liquidity_tier": context.liquidity_tier,
        "volatility_regime": context.volatility_regime,
    }


def _default_calibration_id(generated_at: str) -> str:
    safe = (
        generated_at.replace("-", "")
        .replace(":", "")
        .replace("+", "")
        .replace(".", "")
        .replace("T", "T")
    )
    return f"cost_calibration_{safe[:15]}"


def _resolve_output_root(output_root: Path, root_dir: Path) -> Path:
    return output_root if output_root.is_absolute() else root_dir / output_root


def _resolve_input_path(path: Path | None, root_dir: Path) -> Path | None:
    if path is None:
        return None
    return path if path.is_absolute() else root_dir / path


def _rel(path: Path | None, root_dir: Path) -> str | None:
    if path is None:
        return None
    try:
        return str(path.resolve().relative_to(root_dir.resolve()))
    except ValueError:
        return str(path)


def _display(value: Any) -> str:
    return "" if value is None else str(value)


def _string_or_none(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _normalize(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _float_or_none(value: Any) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None
