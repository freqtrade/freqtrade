from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from freqtrade_ext.bot_factory.candidate_identity import (
    build_strategy_candidate_identity,
    canonicalize_candidate_identity,
    extract_candidate_identity,
)
from freqtrade_ext.bot_factory.market_regime import (
    RegimeClassifierConfig,
    classify_ohlcv_file,
    write_regime_classifier_artifact,
)
from freqtrade_ext.bot_factory.regime_promotion import (
    RegimePromotionThresholds,
    RegimeStrategyContract,
    RegimeStrategyLogicSpec,
    build_observation_ledger,
    build_regime_fitness_scorecard,
    contract_from_logic_spec,
    render_regime_scorecard_report,
    selection_candidate_from_scorecard,
)


@dataclass(frozen=True)
class BacktestEvidencePipelineInputs:
    root_dir: Path
    metrics_path: Path
    trades_path: Path
    ohlcv_path: Path
    strategy: str
    pair: str
    timeframe: str
    output_root: Path = Path("data/regime_evidence")
    run_id: str | None = None
    candidate_id: str | None = None
    logic_id: str = "backtest_derived_logic_v1"
    candidate_style: str = "intraday_trend_following"
    intended_regimes: Sequence[str] = ("trend_up",)
    excluded_regimes: Sequence[str] = (
        "trend_down",
        "range",
        "high_volatility",
        "liquidity_stress",
        "unknown",
    )
    required_features: Sequence[str] = ("close", "volume", "regime_label", "cost_model")
    normal_cost_bps: float = 10.0
    stress_cost_bps: float = 20.0
    classifier_config: RegimeClassifierConfig = field(default_factory=RegimeClassifierConfig)
    reviewer_notes: Sequence[str] = field(default_factory=list)


def build_backtest_evidence_pipeline(inputs: BacktestEvidencePipelineInputs) -> dict[str, Any]:
    root = inputs.root_dir.resolve()
    metrics_path = _resolve(inputs.metrics_path, root)
    trades_path = _resolve(inputs.trades_path, root)
    ohlcv_path = _resolve(inputs.ohlcv_path, root)
    metrics = _load_json(metrics_path)
    identity = _identity_from_metrics(metrics, inputs, metrics_path, trades_path, ohlcv_path)
    identity = _enrich_identity_source_artifacts(
        identity,
        root=root,
        metrics_path=metrics_path,
        trades_path=trades_path,
        ohlcv_path=ohlcv_path,
    )
    candidate_id = str(identity["candidate_id"])
    run_id = inputs.run_id or _run_id()
    regime_artifact = classify_ohlcv_file(
        ohlcv_path,
        pair=inputs.pair,
        timeframe=inputs.timeframe,
        config=inputs.classifier_config,
    )
    trades = _load_trades(trades_path)
    observations = _candidate_observations(
        inputs,
        identity,
        metrics,
        trades,
        regime_artifact,
        metrics_path=metrics_path,
        trades_path=trades_path,
        ohlcv_path=ohlcv_path,
    )
    baseline_observations = _baseline_observations(inputs, identity, regime_artifact)
    ledger = build_observation_ledger(
        [*observations, *baseline_observations],
        ledger_id=f"{candidate_id}_{run_id}_observation_ledger",
        reviewer_notes=inputs.reviewer_notes,
    )
    logic = _logic_from_identity(inputs, identity)
    contract = contract_from_logic_spec(
        logic,
        minimum_evidence={"min_window_count": 1, "min_trade_count": 0},
        maximum_drawdown_by_regime={regime: 30.0 for regime in inputs.intended_regimes},
    )
    thresholds = RegimePromotionThresholds(
        min_sample_days=0.0,
        min_window_count=1,
        min_trade_count=0,
        min_walk_forward_pass_rate=0.0,
        max_pair_concentration=1.0,
        max_calendar_concentration=1.0,
        max_drawdown=30.0,
        min_global_regime_count=1,
    )
    scorecard = build_regime_fitness_scorecard(
        observations,
        contract=contract,
        baseline_observations=baseline_observations,
        thresholds=thresholds,
        scorecard_id=f"{candidate_id}_{run_id}_regime_scorecard",
        reviewer_notes=inputs.reviewer_notes,
        candidate_identity=identity,
    )
    scorecard["baseline_comparison"] = _baseline_comparison(scorecard, baseline_observations)
    scorecard["candidate_style"] = inputs.candidate_style
    selector_candidate = selection_candidate_from_scorecard(
        logic=logic,
        scorecard=scorecard,
        candidate_id=candidate_id,
    )
    selector_candidate["candidate_style"] = inputs.candidate_style
    selector_candidate["feature_quality_thresholds"] = {
        "min_classifier_confidence": inputs.classifier_config.min_confidence,
    }
    return {
        "factory": "backtest_to_observation_to_scorecard_pipeline",
        "schema_version": "backtest_regime_evidence_pipeline_v1",
        "generated_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "run_id": run_id,
        "strategy": inputs.strategy,
        "pair": inputs.pair,
        "timeframe": inputs.timeframe,
        "candidate_id": candidate_id,
        "candidate_identity": identity,
        "input_artifacts": {
            "metrics": _rel(metrics_path, root),
            "trades": _rel(trades_path, root),
            "ohlcv": _rel(ohlcv_path, root),
        },
        "regime_classifier": regime_artifact,
        "observation_ledger": ledger,
        "regime_fitness_scorecard": scorecard,
        "selector_candidate": selector_candidate,
        "traceability": _traceability(root, metrics_path, trades_path, ohlcv_path, scorecard, selector_candidate),
        "safety_scope": {
            "local_artifacts_only": True,
            "backtest_artifacts_only": True,
            "paper_trading_started": False,
            "dry_run_trading_started": False,
            "live_trading_started": False,
            "exchange_order_placement": False,
            "promotion_authorized_by_this_command": False,
        },
    }


def write_backtest_evidence_pipeline_artifacts(
    pipeline: dict[str, Any],
    *,
    root_dir: Path,
    output_root: Path,
) -> dict[str, Path]:
    root = root_dir.resolve()
    out_dir = (
        _resolve(output_root, root)
        / _safe_component(str(pipeline["strategy"]))
        / _safe_component(str(pipeline["candidate_id"]))
        / _safe_component(str(pipeline["run_id"]))
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "regime_classifier": out_dir / "market_regime_labels.json",
        "observation_ledger": out_dir / "observation_ledger.json",
        "regime_fitness_scorecard": out_dir / "regime_fitness_scorecard.json",
        "regime_fitness_scorecard_report": out_dir / "regime_fitness_scorecard_report.md",
        "selector_candidate": out_dir / "selector_candidate.json",
        "traceability": out_dir / "traceability.json",
        "pipeline": out_dir / "backtest_regime_evidence_pipeline.json",
    }
    write_regime_classifier_artifact(pipeline["regime_classifier"], paths["regime_classifier"])
    _write_json(pipeline["observation_ledger"], paths["observation_ledger"])
    _write_json(pipeline["regime_fitness_scorecard"], paths["regime_fitness_scorecard"])
    paths["regime_fitness_scorecard_report"].write_text(
        render_regime_scorecard_report(pipeline["regime_fitness_scorecard"]),
        encoding="utf-8",
    )
    _write_json(pipeline["selector_candidate"], paths["selector_candidate"])
    _write_json(pipeline["traceability"], paths["traceability"])
    artifact_paths = {key: _rel(value, root) for key, value in paths.items()}
    pipeline_with_paths = dict(pipeline)
    pipeline_with_paths["artifact_paths"] = artifact_paths
    _write_json(pipeline_with_paths, paths["pipeline"])
    return paths


def _candidate_observations(
    inputs: BacktestEvidencePipelineInputs,
    identity: dict[str, Any],
    metrics: dict[str, Any],
    trades: Sequence[dict[str, Any]],
    regime_artifact: dict[str, Any],
    *,
    metrics_path: Path,
    trades_path: Path,
    ohlcv_path: Path,
) -> list[dict[str, Any]]:
    rows = [row for row in regime_artifact.get("rows", []) if row.get("market_regime") != "unknown"]
    if not rows:
        rows = list(regime_artifact.get("rows", []))
    by_regime = _rows_by_regime(rows)
    trades_by_regime = _trades_by_regime(trades, rows)
    observations: list[dict[str, Any]] = []
    dominant_regime = max(by_regime, key=lambda key: len(by_regime[key])) if by_regime else "unknown"
    for regime, regime_rows in sorted(by_regime.items()):
        regime_trades = trades_by_regime.get(regime, [])
        if not regime_trades and regime != dominant_regime:
            gross_return = 0.0
        else:
            gross_return = _trade_return_pct(regime_trades)
            if not regime_trades:
                gross_return = float(metrics.get("total_return_pct") or 0.0)
        trade_count = len(regime_trades) if regime_trades else int(metrics.get("trade_count") or 0 if regime == dominant_regime else 0)
        normal_cost = _cost_return_pct(inputs.normal_cost_bps, trade_count)
        stress_cost = _cost_return_pct(inputs.stress_cost_bps, trade_count)
        observations.append(
            _observation(
                observation_id=f"{identity['candidate_id']}_{regime}_{_window_token(regime_rows)}",
                source_type="backtest",
                identity=identity,
                pair=inputs.pair,
                timeframe=inputs.timeframe,
                regime=regime,
                window_start=str(regime_rows[0]["date"]),
                window_end=str(regime_rows[-1]["date"]),
                baseline_id="candidate",
                normal_cost_bps=inputs.normal_cost_bps,
                stress_cost_bps=inputs.stress_cost_bps,
                trade_count=trade_count,
                exposure_ratio=_exposure_ratio(regime_trades, regime_rows),
                gross_return=gross_return,
                net_return_normal_cost=gross_return - normal_cost,
                net_return_stress_cost=gross_return - stress_cost,
                max_drawdown=float(metrics.get("max_drawdown_pct") or 0.0),
                downside_deviation=_downside_deviation(regime_trades),
                win_rate=_win_rate(regime_trades, metrics),
                profit_factor=_profit_factor(regime_trades, metrics),
                no_trade_opportunity_cost=max(gross_return - normal_cost, 0.0),
                data_quality_flags=_regime_flags(regime_rows),
                reason_codes=["backtest_regime_observation"],
                source_artifacts={
                    "metrics": metrics_path,
                    "trades": trades_path,
                    "ohlcv": ohlcv_path,
                },
            )
        )
    return observations


def _baseline_observations(
    inputs: BacktestEvidencePipelineInputs,
    identity: dict[str, Any],
    regime_artifact: dict[str, Any],
) -> list[dict[str, Any]]:
    observations: list[dict[str, Any]] = []
    for regime, rows in sorted(_rows_by_regime(regime_artifact.get("rows", [])).items()):
        hold_return = _hold_return_pct(rows)
        for baseline_id, gross_return in (("no_trade", 0.0), ("hold", hold_return)):
            observations.append(
                _observation(
                    observation_id=f"{identity['candidate_id']}_{baseline_id}_{regime}_{_window_token(rows)}",
                    source_type="backtest",
                    identity=identity,
                    pair=inputs.pair,
                    timeframe=inputs.timeframe,
                    regime=regime,
                    window_start=str(rows[0]["date"]),
                    window_end=str(rows[-1]["date"]),
                    baseline_id=baseline_id,
                    normal_cost_bps=0.0 if baseline_id == "no_trade" else inputs.normal_cost_bps,
                    stress_cost_bps=0.0 if baseline_id == "no_trade" else inputs.stress_cost_bps,
                    trade_count=0 if baseline_id == "no_trade" else 1,
                    exposure_ratio=0.0 if baseline_id == "no_trade" else 1.0,
                    gross_return=gross_return,
                    net_return_normal_cost=gross_return,
                    net_return_stress_cost=gross_return,
                    max_drawdown=_hold_drawdown_pct(rows) if baseline_id == "hold" else 0.0,
                    downside_deviation=0.0,
                    win_rate=0.0,
                    profit_factor=0.0,
                    no_trade_opportunity_cost=max(gross_return, 0.0),
                    data_quality_flags=_regime_flags(rows),
                    reason_codes=[f"{baseline_id}_baseline"],
                    source_artifacts={},
                )
            )
    return observations


def _observation(
    *,
    observation_id: str,
    source_type: str,
    identity: dict[str, Any],
    pair: str,
    timeframe: str,
    regime: str,
    window_start: str,
    window_end: str,
    baseline_id: str,
    normal_cost_bps: float,
    stress_cost_bps: float,
    trade_count: int,
    exposure_ratio: float,
    gross_return: float,
    net_return_normal_cost: float,
    net_return_stress_cost: float,
    max_drawdown: float,
    downside_deviation: float,
    win_rate: float,
    profit_factor: float | None,
    no_trade_opportunity_cost: float,
    data_quality_flags: Sequence[str],
    reason_codes: Sequence[str],
    source_artifacts: Mapping[str, Any],
) -> dict[str, Any]:
    artifact_paths = {
        key: str(value).replace("\\", "/") for key, value in source_artifacts.items()
    }
    return {
        "observation_id": observation_id,
        "created_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "source_type": source_type,
        "strategy_id": identity["strategy_id"],
        "strategy_version": identity["strategy_version"],
        "candidate_id": identity["candidate_id"],
        "candidate_identity": identity,
        "signal_version": identity["signal_version"],
        "risk_policy_version": identity["risk_policy_version"],
        "pair": pair,
        "timeframe": timeframe,
        "window_start": window_start,
        "window_end": window_end,
        "market_regime": regime,
        "regime_classifier_version": identity["regime_classifier_version"],
        "baseline_id": baseline_id,
        "cost_model_id": identity["cost_model_id"],
        "normal_cost_bps": normal_cost_bps,
        "stress_cost_bps": stress_cost_bps,
        "trade_count": int(trade_count),
        "exposure_ratio": round(float(exposure_ratio), 6),
        "gross_return": round(float(gross_return), 6),
        "net_return_normal_cost": round(float(net_return_normal_cost), 6),
        "net_return_stress_cost": round(float(net_return_stress_cost), 6),
        "max_drawdown": round(float(max_drawdown), 6),
        "downside_deviation": round(float(downside_deviation), 6),
        "win_rate": round(float(win_rate), 6),
        "profit_factor": profit_factor,
        "no_trade_reason": "" if baseline_id == "candidate" else baseline_id,
        "no_trade_opportunity_cost": round(float(no_trade_opportunity_cost), 6),
        "data_quality_flags": sorted(set(data_quality_flags)),
        "reason_codes": list(reason_codes),
        "source_artifacts": artifact_paths,
    }


def _logic_from_identity(
    inputs: BacktestEvidencePipelineInputs, identity: dict[str, Any]
) -> RegimeStrategyLogicSpec:
    return RegimeStrategyLogicSpec(
        logic_id=inputs.logic_id,
        strategy_id=identity["strategy_id"],
        strategy_class_name=identity["strategy_class_name"],
        strategy_source_path=identity["strategy_source_path"],
        strategy_version=identity["strategy_version"],
        signal_version=identity["signal_version"],
        intended_regimes=tuple(inputs.intended_regimes),
        excluded_regimes=tuple(inputs.excluded_regimes),
        entry_conditions=("backtest-derived regime label is inside intended regimes",),
        exit_conditions=("regime label leaves intended regimes",),
        no_trade_conditions=("unknown, excluded, or low-confidence regime",),
        required_features=tuple(inputs.required_features),
        risk_policy_version=identity["risk_policy_version"],
        regime_classifier_version=identity["regime_classifier_version"],
        cost_model_id=identity["cost_model_id"],
        allowed_pairs=tuple(identity.get("allowed_pairs") or [inputs.pair]),
        allowed_timeframes=tuple(identity.get("allowed_timeframes") or [inputs.timeframe]),
        identity_created_at=str(identity["created_at"]),
        source_artifacts=dict(identity.get("source_artifacts") or {}),
    )


def _identity_from_metrics(
    metrics: dict[str, Any],
    inputs: BacktestEvidencePipelineInputs,
    metrics_path: Path,
    trades_path: Path,
    ohlcv_path: Path,
) -> dict[str, Any]:
    identity = extract_candidate_identity(metrics)
    if identity is not None:
        return identity
    candidate_id = inputs.candidate_id or inputs.run_id or _run_id()
    return build_strategy_candidate_identity(
        candidate_id=candidate_id,
        strategy_id=inputs.strategy,
        strategy_class_name=inputs.strategy,
        strategy_source_path=f"user_data/strategies/{inputs.strategy}.py",
        strategy_version=f"{inputs.strategy}_v1",
        signal_version="backtest_derived_signal_v1",
        risk_policy_version="backtest_derived_risk_policy_v1",
        regime_classifier_version=inputs.classifier_config.version,
        cost_model_id="backtest_derived_cost_model_v1",
        allowed_pairs=[inputs.pair],
        allowed_timeframes=[inputs.timeframe],
        created_at=datetime.now(UTC).replace(microsecond=0).isoformat(),
        source_artifacts={"metrics": metrics_path, "trades": trades_path, "ohlcv": ohlcv_path},
        root_dir=inputs.root_dir,
    )


def _enrich_identity_source_artifacts(
    identity: dict[str, Any],
    *,
    root: Path,
    metrics_path: Path,
    trades_path: Path,
    ohlcv_path: Path,
) -> dict[str, Any]:
    enriched = dict(identity)
    artifacts = dict(enriched.get("source_artifacts") or {})
    artifacts.update(
        {
            "metrics": _rel(metrics_path, root),
            "trades": _rel(trades_path, root),
            "ohlcv": _rel(ohlcv_path, root),
        }
    )
    enriched["source_artifacts"] = artifacts
    return canonicalize_candidate_identity(enriched)


def _baseline_comparison(
    scorecard: dict[str, Any], baseline_observations: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    by_regime: dict[str, dict[str, float]] = {}
    for item in baseline_observations:
        regime = str(item.get("market_regime"))
        baseline_id = str(item.get("baseline_id"))
        by_regime.setdefault(regime, {})[baseline_id] = float(item.get("net_return_normal_cost") or 0.0)
    rows = []
    for row in scorecard.get("scorecard_by_regime", []):
        regime = str(row.get("market_regime"))
        candidate = float(row.get("net_pnl_normal_cost") or 0.0)
        hold = by_regime.get(regime, {}).get("hold", 0.0)
        no_trade = by_regime.get(regime, {}).get("no_trade", 0.0)
        rows.append(
            {
                "market_regime": regime,
                "candidate_return": candidate,
                "hold_return": hold,
                "no_trade_return": no_trade,
                "beats_hold": candidate > hold,
                "beats_no_trade": candidate > no_trade,
                "hold_delta": candidate - hold,
                "no_trade_delta": candidate - no_trade,
            }
        )
    return {"by_regime": rows}


def _traceability(
    root: Path,
    metrics_path: Path,
    trades_path: Path,
    ohlcv_path: Path,
    scorecard: dict[str, Any],
    selector_candidate: dict[str, Any],
) -> dict[str, Any]:
    return {
        "strategy_source": (scorecard.get("candidate_identity") or {}).get("strategy_source_path"),
        "metrics": _file_trace(metrics_path, root),
        "trades": _file_trace(trades_path, root),
        "ohlcv": _file_trace(ohlcv_path, root),
        "scorecard_id": scorecard.get("scorecard_id"),
        "selector_candidate_id": selector_candidate.get("candidate_id"),
        "selector_decision": selector_candidate.get("scorecard_decision"),
    }


def _file_trace(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": _rel(path, root),
        "sha256": _sha256(path) if path.is_file() else None,
        "exists": path.is_file(),
    }


def _load_trades(path: Path) -> list[dict[str, Any]]:
    if not path.is_file() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _trades_by_regime(
    trades: Sequence[dict[str, Any]], regime_rows: Sequence[dict[str, Any]]
) -> dict[str, list[dict[str, Any]]]:
    dates = [(pd.to_datetime(row["date"], utc=True), row["market_regime"]) for row in regime_rows]
    grouped: dict[str, list[dict[str, Any]]] = {}
    for trade in trades:
        opened = _trade_timestamp(trade)
        regime = _nearest_regime(opened, dates)
        grouped.setdefault(regime, []).append(dict(trade))
    return grouped


def _nearest_regime(
    opened: pd.Timestamp | None, rows: Sequence[tuple[pd.Timestamp, str]]
) -> str:
    if not rows:
        return "unknown"
    if opened is None:
        return rows[-1][1]
    before = [item for item in rows if item[0] <= opened]
    return before[-1][1] if before else rows[0][1]


def _rows_by_regime(rows: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("market_regime") or "unknown"), []).append(row)
    return {key: value for key, value in grouped.items() if value}


def _trade_timestamp(trade: Mapping[str, Any]) -> pd.Timestamp | None:
    for key in ("open_date", "open_timestamp", "date", "enter_date"):
        value = trade.get(key)
        if value:
            parsed = pd.to_datetime(value, utc=True, errors="coerce")
            return None if pd.isna(parsed) else parsed
    return None


def _trade_return_pct(trades: Sequence[Mapping[str, Any]]) -> float:
    total = 0.0
    for trade in trades:
        value = _number(trade.get("profit_ratio"))
        if value is None:
            value = _number(trade.get("profit_pct"))
            total += value or 0.0
        else:
            total += value * 100.0
    return total


def _cost_return_pct(cost_bps: float, trade_count: int) -> float:
    return (float(cost_bps) / 100.0) * max(int(trade_count), 0)


def _exposure_ratio(trades: Sequence[Mapping[str, Any]], rows: Sequence[dict[str, Any]]) -> float:
    if not trades:
        return 0.0
    return min(1.0, len(trades) / max(len(rows), 1))


def _downside_deviation(trades: Sequence[Mapping[str, Any]]) -> float:
    losses = [min(_number(trade.get("profit_ratio")) or 0.0, 0.0) * 100.0 for trade in trades]
    if not losses:
        return 0.0
    mean_square = sum(value * value for value in losses) / len(losses)
    return mean_square ** 0.5


def _win_rate(trades: Sequence[Mapping[str, Any]], metrics: Mapping[str, Any]) -> float:
    if not trades:
        return float(metrics.get("win_rate") or 0.0)
    wins = sum(1 for trade in trades if (_number(trade.get("profit_ratio")) or 0.0) > 0.0)
    return wins / len(trades)


def _profit_factor(trades: Sequence[Mapping[str, Any]], metrics: Mapping[str, Any]) -> float | None:
    if not trades:
        value = _number(metrics.get("profit_factor"))
        return value
    wins = sum(max(_number(trade.get("profit_ratio")) or 0.0, 0.0) for trade in trades)
    losses = abs(sum(min(_number(trade.get("profit_ratio")) or 0.0, 0.0) for trade in trades))
    if losses == 0:
        metric_value = _number(metrics.get("profit_factor"))
        if metric_value is not None:
            return metric_value
        return 999.0 if wins > 0 else 0.0
    return wins / losses


def _hold_return_pct(rows: Sequence[dict[str, Any]]) -> float:
    if len(rows) < 2:
        return 0.0
    first = _number((rows[0].get("features") or {}).get("close")) or _number(rows[0].get("close"))
    last = _number((rows[-1].get("features") or {}).get("close")) or _number(rows[-1].get("close"))
    if first is None or last is None or first == 0:
        return 0.0
    return (last / first - 1.0) * 100.0


def _hold_drawdown_pct(rows: Sequence[dict[str, Any]]) -> float:
    closes = [
        _number((row.get("features") or {}).get("close")) or _number(row.get("close"))
        for row in rows
    ]
    clean = [value for value in closes if value is not None and value > 0]
    if not clean:
        return 0.0
    peak = clean[0]
    max_dd = 0.0
    for value in clean:
        peak = max(peak, value)
        max_dd = min(max_dd, value / peak - 1.0)
    return abs(max_dd) * 100.0


def _regime_flags(rows: Sequence[dict[str, Any]]) -> list[str]:
    flags: list[str] = []
    for row in rows:
        flags.extend(str(item) for item in row.get("data_quality_flags", []))
    return sorted(set(flags))


def _window_token(rows: Sequence[dict[str, Any]]) -> str:
    if not rows:
        return "empty"
    return f"{_safe_component(str(rows[0].get('date'))[:10])}_{_safe_component(str(rows[-1].get('date'))[:10])}"


def _run_id() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _safe_component(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value).strip("._") or "artifact"


def _number(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _resolve(path: Path, root: Path) -> Path:
    return (path if path.is_absolute() else root / path).resolve()


def _rel(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
