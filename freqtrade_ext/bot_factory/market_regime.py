from __future__ import annotations

import json
import hashlib
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd


REGIME_CLASSIFIER_VERSION = "deterministic_regime_classifier_v1"
MARKET_STATE_ENCODER_VERSION = "deterministic_market_state_encoder_v1"
MARKET_STATE_FEATURE_VERSION = "ohlcv_state_features_v1"
MARKET_STATE_SNAPSHOT_SCHEMA_VERSION = "market_state_snapshot_v1"
MARKET_STATE_WINDOW_SCHEMA_VERSION = "market_state_window_v1"
CURRENT_MARKET_STATE_SCHEMA_VERSION = "current_market_state_v1"
REGIME_LABELS = (
    "trend_up",
    "trend_down",
    "range",
    "high_volatility",
    "low_volatility",
    "liquidity_stress",
    "post_spike_reversion",
    "mixed",
    "unknown",
)
MARKET_STATE_LABELS = tuple(
    dict.fromkeys((*REGIME_LABELS, "transition", "out_of_distribution"))
)
DEFAULT_MARKET_STATE_HORIZONS = ("5m", "15m", "1h", "4h", "1d", "1w")
HORIZON_GROUPS = {
    "5m": "micro",
    "15m": "micro",
    "1h": "intraday",
    "4h": "intraday",
    "1d": "swing",
    "1w": "swing",
}
REQUIRED_OHLCV_COLUMNS = ("date", "open", "high", "low", "close", "volume")


@dataclass(frozen=True)
class RegimeClassifierConfig:
    lookback: int = 12
    min_rows: int = 24
    trend_return_threshold: float = 0.015
    range_return_threshold: float = 0.006
    range_efficiency_threshold: float = 0.35
    trend_efficiency_threshold: float = 0.45
    high_volatility_ratio: float = 1.8
    low_volatility_ratio: float = 0.55
    liquidity_ratio_threshold: float = 0.25
    spike_return_threshold: float = 0.035
    min_confidence: float = 0.45
    max_missing_rate: float = 0.02
    version: str = REGIME_CLASSIFIER_VERSION


@dataclass(frozen=True)
class MarketStateConfig:
    horizons: Sequence[str] = DEFAULT_MARKET_STATE_HORIZONS
    min_horizon_rows: int = 24
    max_staleness_seconds: int = 900
    confidence_threshold: float = 0.5
    out_of_distribution_threshold: float = 0.8
    state_encoder_version: str = MARKET_STATE_ENCODER_VERSION
    feature_version: str = MARKET_STATE_FEATURE_VERSION
    regime_classifier_config: RegimeClassifierConfig = field(
        default_factory=RegimeClassifierConfig
    )


def classify_ohlcv_regimes(
    frame: pd.DataFrame,
    *,
    pair: str,
    timeframe: str,
    config: RegimeClassifierConfig | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    config = config or RegimeClassifierConfig()
    generated_at = generated_at or datetime.now(UTC).replace(microsecond=0).isoformat()
    checks = _input_checks(frame, config)
    if not all(check["passed"] for check in checks):
        rows = [
            _row(
                date=str(item.get("date", "")),
                pair=pair,
                timeframe=timeframe,
                market_regime="unknown",
                confidence=0.0,
                features={},
                data_quality_flags=[check["name"] for check in checks if not check["passed"]],
            )
            for item in frame.to_dict("records")
        ]
        return _artifact(rows, pair, timeframe, config, checks, generated_at)

    data = frame.copy()
    data["date"] = pd.to_datetime(data["date"], utc=True, errors="coerce")
    for column in ("open", "high", "low", "close", "volume"):
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.sort_values("date").reset_index(drop=True)

    lookback = config.lookback
    returns = data["close"].pct_change()
    rolling_return = data["close"] / data["close"].shift(lookback) - 1.0
    realized_volatility = returns.rolling(lookback, min_periods=lookback).std()
    volatility_baseline = realized_volatility.rolling(
        lookback * 4, min_periods=lookback
    ).median()
    rolling_high = data["high"].rolling(lookback, min_periods=lookback).max()
    rolling_low = data["low"].rolling(lookback, min_periods=lookback).min()
    range_width = (rolling_high - rolling_low).abs()
    range_efficiency = (
        (data["close"] - data["close"].shift(lookback)).abs()
        / range_width.replace(0, pd.NA)
    ).fillna(0.0)
    trend_slope = (data["close"] - data["close"].shift(lookback)) / (
        data["close"].shift(lookback).replace(0, pd.NA) * lookback
    )
    local_high = data["close"].rolling(lookback, min_periods=lookback).max()
    drawdown_from_high = data["close"] / local_high.replace(0, pd.NA) - 1.0
    volume_baseline = data["volume"].rolling(lookback, min_periods=lookback).median()
    volume_ratio = data["volume"] / volume_baseline.replace(0, pd.NA)
    candle_gap_proxy = (data["open"] / data["close"].shift(1) - 1.0).abs()

    rows: list[dict[str, Any]] = []
    for index, source in data.iterrows():
        features = {
            "close": _float(source["close"]),
            "rolling_return": _float(rolling_return.iloc[index]),
            "realized_volatility": _float(realized_volatility.iloc[index]),
            "trend_slope": _float(trend_slope.iloc[index]),
            "range_efficiency": _float(range_efficiency.iloc[index]),
            "drawdown_from_local_high": _float(drawdown_from_high.iloc[index]),
            "volume_liquidity_ratio": _float(volume_ratio.iloc[index]),
            "candle_gap_proxy": _float(candle_gap_proxy.iloc[index]),
        }
        flags = _row_quality_flags(source, index, config)
        label, confidence = _classify_row(
            features,
            one_period_return=_float(returns.iloc[index]) or 0.0,
            volatility_baseline=_float(volatility_baseline.iloc[index]),
            config=config,
            flags=flags,
        )
        rows.append(
            _row(
                date=source["date"].isoformat(),
                pair=pair,
                timeframe=timeframe,
                market_regime=label,
                confidence=confidence,
                features=features,
                data_quality_flags=flags,
            )
        )

    return _artifact(rows, pair, timeframe, config, checks, generated_at)


def classify_ohlcv_file(
    path: Path,
    *,
    pair: str,
    timeframe: str,
    config: RegimeClassifierConfig | None = None,
) -> dict[str, Any]:
    frame = pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)
    artifact = classify_ohlcv_regimes(frame, pair=pair, timeframe=timeframe, config=config)
    artifact["source_path"] = str(path)
    return artifact


def build_market_state_snapshot(
    frame: pd.DataFrame,
    *,
    pair: str,
    base_timeframe: str,
    pair_group: str = "single_pair",
    run_id: str | None = None,
    cost_model_id: str = "local_unknown_cost_model",
    source_data_paths: Sequence[str | Path] = (),
    source_data_hashes: Mapping[str, str] | None = None,
    horizon_frames: Mapping[str, pd.DataFrame] | None = None,
    config: MarketStateConfig | None = None,
    generated_at: str | None = None,
    now: datetime | None = None,
    git_commit: str | None = None,
) -> dict[str, Any]:
    config = config or MarketStateConfig()
    now = now or datetime.now(UTC)
    if now.tzinfo is None:
        now = now.replace(tzinfo=UTC)
    generated_at = generated_at or now.replace(microsecond=0).isoformat()
    run_id = run_id or "market_state_" + now.strftime("%Y%m%dT%H%M%SZ")
    source_paths = [str(path).replace("\\", "/") for path in source_data_paths]
    prepared = _prepare_ohlcv(frame)
    latest_local_candle_at = _timestamp_to_iso(prepared["date"].max()) if len(prepared) else None
    stale_data = _is_stale(latest_local_candle_at, now, config.max_staleness_seconds)
    source_hashes = dict(source_data_hashes or {})
    for path in source_data_paths:
        source_hashes.setdefault(str(path).replace("\\", "/"), _file_hash(Path(path)))

    windows = []
    for horizon, horizon_frame in _market_state_horizon_frames(
        prepared,
        base_timeframe=base_timeframe,
        horizons=config.horizons,
        horizon_frames=horizon_frames or {},
    ).items():
        if len(horizon_frame) < config.min_horizon_rows:
            continue
        regime_artifact = classify_ohlcv_regimes(
            horizon_frame,
            pair=pair,
            timeframe=horizon,
            config=config.regime_classifier_config,
            generated_at=generated_at,
        )
        row = _latest_market_state_window(
            regime_artifact,
            horizon_frame,
            pair=pair,
            horizon=horizon,
            run_id=run_id,
            config=config,
            stale_data=stale_data,
            now=now,
        )
        if row is not None:
            windows.append(row)

    aggregate = _aggregate_market_state(windows, config=config, stale_data=stale_data)
    data_quality_flags = sorted(
        {
            flag
            for row in windows
            for flag in row.get("data_quality_flags", [])
            if isinstance(flag, str)
        }
    )
    feature_quality_flags = sorted(
        {
            flag
            for row in windows
            for flag in row.get("feature_quality_flags", [])
            if isinstance(flag, str)
        }
    )
    missing_rates = [
        _number((row.get("state_vector") or {}).get("missing_candle_rate"))
        for row in windows
    ]
    stale_reason = ["stale_local_data"] if stale_data else []
    reason_codes = sorted(set([*aggregate["reason_codes"], *stale_reason]))
    return {
        "factory": "bot_factory",
        "schema_version": MARKET_STATE_SNAPSHOT_SCHEMA_VERSION,
        "run_id": run_id,
        "generated_at": generated_at,
        "data_asof": latest_local_candle_at,
        "latest_local_candle_at": latest_local_candle_at,
        "git_commit": git_commit,
        "source_data_paths": source_paths,
        "source_data_hashes": source_hashes,
        "pair": pair,
        "pair_group": pair_group,
        "base_timeframe": base_timeframe,
        "horizons": windows,
        "state_encoder_version": config.state_encoder_version,
        "regime_classifier_version": config.regime_classifier_config.version,
        "feature_version": config.feature_version,
        "cost_model_id": cost_model_id,
        "data_quality_summary": {
            "horizon_count": len(windows),
            "stale_data": stale_data,
            "latest_local_candle_at": latest_local_candle_at,
            "max_staleness_seconds": config.max_staleness_seconds,
            "max_missing_candle_rate": _max_number(missing_rates),
            "flags": data_quality_flags,
        },
        "feature_quality_summary": {
            "feature_quality_pass": not feature_quality_flags,
            "flags": feature_quality_flags,
        },
        "aggregate_label": aggregate["aggregate_label"],
        "state_confidence": aggregate["state_confidence"],
        "uncertainty": aggregate["uncertainty"],
        "unknown_reason": aggregate["unknown_reason"],
        "out_of_distribution_score": aggregate["out_of_distribution_score"],
        "horizon_profile_id": _horizon_profile_id(windows, config),
        "horizon_conflict": aggregate["horizon_conflict"],
        "no_trade_default": aggregate["no_trade_default"],
        "reason_codes": reason_codes,
        "safety_scope": _market_state_safety_scope(),
    }


def build_market_state_snapshot_file(
    path: Path,
    *,
    pair: str,
    base_timeframe: str,
    pair_group: str = "single_pair",
    run_id: str | None = None,
    cost_model_id: str = "local_unknown_cost_model",
    config: MarketStateConfig | None = None,
    generated_at: str | None = None,
    now: datetime | None = None,
    git_commit: str | None = None,
) -> dict[str, Any]:
    frame = pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)
    return build_market_state_snapshot(
        frame,
        pair=pair,
        base_timeframe=base_timeframe,
        pair_group=pair_group,
        run_id=run_id,
        cost_model_id=cost_model_id,
        source_data_paths=[path],
        config=config,
        generated_at=generated_at,
        now=now,
        git_commit=git_commit,
    )


def build_current_market_state(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    safety_scope = dict(snapshot.get("safety_scope") or {})
    return {
        "factory": "bot_factory",
        "schema_version": CURRENT_MARKET_STATE_SCHEMA_VERSION,
        "snapshot_schema_version": snapshot.get("schema_version"),
        "snapshot_run_id": snapshot.get("run_id"),
        "generated_at": snapshot.get("generated_at"),
        "data_asof": snapshot.get("data_asof"),
        "latest_local_candle_at": snapshot.get("latest_local_candle_at"),
        "pair": snapshot.get("pair"),
        "pair_group": snapshot.get("pair_group"),
        "base_timeframe": snapshot.get("base_timeframe"),
        "aggregate_label": snapshot.get("aggregate_label"),
        "state_confidence": snapshot.get("state_confidence"),
        "uncertainty": snapshot.get("uncertainty"),
        "out_of_distribution_score": snapshot.get("out_of_distribution_score"),
        "stale_data": bool((snapshot.get("data_quality_summary") or {}).get("stale_data")),
        "no_trade_default": bool(snapshot.get("no_trade_default")),
        "horizon_conflict": snapshot.get("horizon_conflict"),
        "horizons": [
            {
                "horizon": row.get("horizon"),
                "label": row.get("label"),
                "confidence": row.get("confidence"),
                "uncertainty": row.get("uncertainty"),
                "reason_codes": row.get("reason_codes", []),
            }
            for row in snapshot.get("horizons", [])
        ],
        "not_allowed_confirmation": {
            "freqtrade_trade_started": safety_scope.get("freqtrade_trade_started") is False,
            "paper_trading_started": safety_scope.get("paper_trading_started") is False,
            "dry_run_trading_started": safety_scope.get("dry_run_trading_started") is False,
            "live_trading_started": safety_scope.get("live_trading_started") is False,
            "exchange_order_placement": safety_scope.get("exchange_order_placement") is False,
            "process_control": safety_scope.get("process_control") is False,
        },
        "reason_codes": list(snapshot.get("reason_codes", [])),
        "safety_scope": safety_scope,
    }


def write_market_state_artifacts(
    snapshot: dict[str, Any],
    *,
    output_root: Path,
) -> dict[str, Path]:
    run_id = _safe_component(str(snapshot.get("run_id") or "market_state"))
    out_dir = output_root / run_id
    current_dir = output_root / "current" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    current_dir.mkdir(parents=True, exist_ok=True)
    current = build_current_market_state(snapshot)
    paths = {
        "market_state_snapshot": out_dir / "market_state_snapshot.json",
        "market_state_windows": out_dir / "market_state_windows.jsonl",
        "market_state_report": out_dir / "market_state_report.md",
        "current_market_state": current_dir / "current_market_state.json",
        "current_market_state_report": current_dir / "current_market_state_report.md",
    }
    _write_json(snapshot, paths["market_state_snapshot"])
    paths["market_state_windows"].write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in snapshot.get("horizons", [])
        ),
        encoding="utf-8",
    )
    paths["market_state_report"].write_text(
        render_market_state_report(snapshot),
        encoding="utf-8",
    )
    _write_json(current, paths["current_market_state"])
    paths["current_market_state_report"].write_text(
        render_current_market_state_report(current),
        encoding="utf-8",
    )
    return paths


def render_market_state_report(snapshot: Mapping[str, Any]) -> str:
    lines = [
        "# Market State Snapshot",
        "",
        "## Summary",
        "",
        f"- Run ID: `{snapshot.get('run_id')}`",
        f"- Pair: `{snapshot.get('pair')}`",
        f"- Data as-of: `{snapshot.get('data_asof')}`",
        f"- Aggregate label: `{snapshot.get('aggregate_label')}`",
        f"- Confidence: `{snapshot.get('state_confidence')}`",
        f"- Uncertainty: `{snapshot.get('uncertainty')}`",
        f"- No-trade default: `{snapshot.get('no_trade_default')}`",
        f"- Reason codes: `{', '.join(snapshot.get('reason_codes', []))}`",
        "",
        "## Multi-Horizon State",
        "",
        "| Horizon | Label | Confidence | Uncertainty | Reason Codes |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    for row in snapshot.get("horizons", []):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("horizon")),
                    str(row.get("label")),
                    str(row.get("confidence")),
                    str(row.get("uncertainty")),
                    ", ".join(row.get("reason_codes", [])),
                ]
            )
            + " |"
        )
    conflict = snapshot.get("horizon_conflict") or {}
    lines.extend(
        [
            "",
            "## Horizon Conflicts",
            "",
            f"- Conflict detected: `{conflict.get('conflict_detected')}`",
            f"- Reason codes: `{', '.join(conflict.get('reason_codes', []))}`",
            "",
            "## Safety Boundary",
            "",
            "- This artifact is local-only and historical/as-of local data only.",
            "- It does not start `freqtrade trade`, paper trading, dry-run trading, live trading, or any bot process.",
            "- It does not use exchange order endpoints, API keys, secrets, leverage above `1.0`, or shorting.",
        ]
    )
    return "\n".join(lines) + "\n"


def render_current_market_state_report(current: Mapping[str, Any]) -> str:
    lines = [
        "# Current Market State",
        "",
        f"Current means current as of local data timestamp `{current.get('data_asof')}`.",
        "",
        "## Summary",
        "",
        f"- Pair: `{current.get('pair')}`",
        f"- Aggregate label: `{current.get('aggregate_label')}`",
        f"- Stale data: `{current.get('stale_data')}`",
        f"- No-trade default: `{current.get('no_trade_default')}`",
        f"- Reason codes: `{', '.join(current.get('reason_codes', []))}`",
        "",
        "## Horizons",
        "",
        "| Horizon | Label | Confidence | Uncertainty | Reason Codes |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    for row in current.get("horizons", []):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("horizon")),
                    str(row.get("label")),
                    str(row.get("confidence")),
                    str(row.get("uncertainty")),
                    ", ".join(row.get("reason_codes", [])),
                ]
            )
            + " |"
        )
    not_allowed = current.get("not_allowed_confirmation") or {}
    lines.extend(
        [
            "",
            "## Not Allowed",
            "",
            f"- Freqtrade trade started: `{not not_allowed.get('freqtrade_trade_started', False)}`",
            f"- Paper trading started: `{not not_allowed.get('paper_trading_started', False)}`",
            f"- Dry-run trading started: `{not not_allowed.get('dry_run_trading_started', False)}`",
            f"- Live trading started: `{not not_allowed.get('live_trading_started', False)}`",
            f"- Exchange order placement: `{not not_allowed.get('exchange_order_placement', False)}`",
            f"- Process control: `{not not_allowed.get('process_control', False)}`",
        ]
    )
    return "\n".join(lines) + "\n"


def regime_churn_report(
    first: dict[str, Any],
    second: dict[str, Any],
    *,
    max_churn_ratio: float = 0.25,
) -> dict[str, Any]:
    first_by_date = {row.get("date"): row.get("market_regime") for row in first.get("rows", [])}
    second_by_date = {row.get("date"): row.get("market_regime") for row in second.get("rows", [])}
    shared = sorted(set(first_by_date) & set(second_by_date))
    changed = [date for date in shared if first_by_date[date] != second_by_date[date]]
    churn_ratio = (len(changed) / len(shared)) if shared else 0.0
    return {
        "factory": "regime_classifier_churn_report",
        "schema_version": "regime_classifier_churn_v1",
        "ok": churn_ratio <= max_churn_ratio,
        "shared_rows": len(shared),
        "changed_rows": len(changed),
        "churn_ratio": churn_ratio,
        "max_churn_ratio": max_churn_ratio,
        "changed_dates": changed[:50],
    }


def write_regime_classifier_artifact(artifact: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, ensure_ascii=False), encoding="utf-8")


def _prepare_ohlcv(frame: pd.DataFrame) -> pd.DataFrame:
    if not all(column in frame.columns for column in REQUIRED_OHLCV_COLUMNS):
        return pd.DataFrame(columns=REQUIRED_OHLCV_COLUMNS)
    data = frame.loc[:, list(REQUIRED_OHLCV_COLUMNS)].copy()
    data["date"] = pd.to_datetime(data["date"], utc=True, errors="coerce")
    for column in ("open", "high", "low", "close", "volume"):
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return data


def _market_state_horizon_frames(
    frame: pd.DataFrame,
    *,
    base_timeframe: str,
    horizons: Sequence[str],
    horizon_frames: Mapping[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    base_seconds = _timeframe_seconds(base_timeframe)
    for horizon in horizons:
        if horizon in horizon_frames:
            frames[horizon] = _prepare_ohlcv(horizon_frames[horizon])
            continue
        horizon_seconds = _timeframe_seconds(horizon)
        if base_seconds is None or horizon_seconds is None or horizon_seconds < base_seconds:
            continue
        frames[horizon] = (
            frame.copy()
            if horizon == base_timeframe
            else _resample_ohlcv(frame, horizon)
        )
    return {key: value for key, value in frames.items() if len(value)}


def _resample_ohlcv(frame: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    rule = _pandas_resample_rule(timeframe)
    if rule is None:
        return pd.DataFrame(columns=REQUIRED_OHLCV_COLUMNS)
    data = frame.set_index("date").sort_index()
    resampled = data.resample(rule, label="left", closed="left").agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    )
    resampled = resampled.dropna(subset=["open", "high", "low", "close"]).reset_index()
    return resampled.loc[:, list(REQUIRED_OHLCV_COLUMNS)]


def _latest_market_state_window(
    regime_artifact: Mapping[str, Any],
    frame: pd.DataFrame,
    *,
    pair: str,
    horizon: str,
    run_id: str,
    config: MarketStateConfig,
    stale_data: bool,
    now: datetime,
) -> dict[str, Any] | None:
    rows = list(regime_artifact.get("rows", []))
    if not rows or frame.empty:
        return None
    latest = rows[-1]
    source_label = str(latest.get("market_regime") or "unknown")
    confidence = float(latest.get("confidence") or 0.0)
    data_quality_flags = list(latest.get("data_quality_flags") or [])
    reason_codes = _window_reason_codes(source_label, confidence, config)
    unknown_reason = None
    label = source_label if source_label in MARKET_STATE_LABELS else "unknown"
    if stale_data:
        label = "unknown"
        confidence = 0.0
        unknown_reason = "stale_local_data"
        data_quality_flags.append("stale_local_data")
        reason_codes.append("stale_local_data")
    elif label == "unknown":
        unknown_reason = "classifier_unknown"
    uncertainty = round(max(0.0, min(1.0, 1.0 - confidence)), 6)
    ood_score = _out_of_distribution_score(label, confidence)
    latest_ts = pd.to_datetime(frame["date"].iloc[-1], utc=True)
    start_index = max(0, len(frame) - config.regime_classifier_config.lookback)
    start_ts = pd.to_datetime(frame["date"].iloc[start_index], utc=True)
    vector = _state_vector(
        latest.get("features") if isinstance(latest.get("features"), dict) else {},
        frame,
        horizon=horizon,
        config=config,
        now=now,
    )
    return {
        "schema_version": MARKET_STATE_WINDOW_SCHEMA_VERSION,
        "run_id": run_id,
        "pair": pair,
        "timeframe": horizon,
        "horizon": horizon,
        "horizon_group": HORIZON_GROUPS.get(horizon, "custom"),
        "lookback_window": {
            "candles": config.regime_classifier_config.lookback,
            "duration": _lookback_duration(horizon, config.regime_classifier_config.lookback),
        },
        "decision_window_start": _timestamp_to_iso(start_ts),
        "decision_window_end": _timestamp_to_iso(latest_ts),
        "label": label,
        "state_id": _state_id(label, horizon, confidence, config),
        "confidence": round(confidence, 6),
        "uncertainty": uncertainty,
        "out_of_distribution_score": ood_score,
        "state_vector": vector,
        "feature_cutoff_timestamp": _timestamp_to_iso(latest_ts),
        "label_cutoff_timestamp": _timestamp_to_iso(latest_ts),
        "future_data_used": False,
        "data_quality_flags": sorted(set(data_quality_flags)),
        "feature_quality_flags": [],
        "unknown_reason": unknown_reason,
        "reason_codes": sorted(set(reason_codes)),
        "regime_classifier_version": config.regime_classifier_config.version,
        "state_encoder_version": config.state_encoder_version,
        "feature_version": config.feature_version,
    }


def _state_vector(
    features: Mapping[str, Any],
    frame: pd.DataFrame,
    *,
    horizon: str,
    config: MarketStateConfig,
    now: datetime,
) -> dict[str, float | None]:
    close = pd.to_numeric(frame["close"], errors="coerce")
    high = pd.to_numeric(frame["high"], errors="coerce")
    low = pd.to_numeric(frame["low"], errors="coerce")
    volume = pd.to_numeric(frame["volume"], errors="coerce")
    lookback = config.regime_classifier_config.lookback
    returns = close.pct_change()
    realized_volatility = returns.rolling(lookback, min_periods=lookback).std()
    volatility_baseline = realized_volatility.rolling(
        lookback * 4, min_periods=lookback
    ).median()
    latest_close = _float(close.iloc[-1]) if len(close) else None
    moving_average = _float(close.rolling(lookback, min_periods=lookback).mean().iloc[-1])
    volume_mean = _float(volume.rolling(lookback, min_periods=lookback).mean().iloc[-1])
    volume_std = _float(volume.rolling(lookback, min_periods=lookback).std().iloc[-1])
    latest_volume = _float(volume.iloc[-1]) if len(volume) else None
    volatility = _float(realized_volatility.iloc[-1]) if len(realized_volatility) else None
    vol_base = _float(volatility_baseline.iloc[-1]) if len(volatility_baseline) else None
    high_low_range_pct = (
        ((float(high.iloc[-1]) - float(low.iloc[-1])) / float(close.iloc[-1])) * 100.0
        if len(frame) and _float(close.iloc[-1]) not in (None, 0.0)
        else None
    )
    latest_ts = pd.to_datetime(frame["date"].iloc[-1], utc=True) if len(frame) else None
    freshness_age_minutes = (
        (now.astimezone(UTC) - latest_ts.to_pydatetime()).total_seconds() / 60.0
        if latest_ts is not None and pd.notna(latest_ts)
        else None
    )
    return {
        "rolling_return_bps": _bps(features.get("rolling_return")),
        "realized_volatility_bps": _bps(features.get("realized_volatility")),
        "volatility_zscore": _rounded(
            (volatility / vol_base - 1.0) if volatility is not None and vol_base else 0.0
        ),
        "trend_slope_bps_per_candle": _bps(features.get("trend_slope")),
        "moving_average_distance_bps": _bps(
            (latest_close / moving_average - 1.0)
            if latest_close is not None and moving_average
            else None
        ),
        "range_efficiency": _rounded(features.get("range_efficiency")),
        "drawdown_from_local_high_bps": _bps(features.get("drawdown_from_local_high")),
        "high_low_range_pct": _rounded(high_low_range_pct),
        "candle_gap_proxy_bps": _bps(features.get("candle_gap_proxy")),
        "volume_liquidity_zscore": _rounded(
            ((latest_volume - volume_mean) / volume_std)
            if latest_volume is not None and volume_mean is not None and volume_std
            else 0.0
        ),
        "turnover_cost_pressure": None,
        "missing_candle_rate": _rounded(_missing_candle_rate(frame, horizon)),
        "freshness_age_minutes": _rounded(freshness_age_minutes),
    }


def _aggregate_market_state(
    windows: Sequence[Mapping[str, Any]],
    *,
    config: MarketStateConfig,
    stale_data: bool,
) -> dict[str, Any]:
    if not windows:
        return {
            "aggregate_label": "unknown",
            "state_confidence": 0.0,
            "uncertainty": 1.0,
            "unknown_reason": "no_supported_horizons",
            "out_of_distribution_score": 1.0,
            "horizon_conflict": {
                "conflict_detected": False,
                "reason_codes": ["no_supported_horizons"],
            },
            "no_trade_default": True,
            "reason_codes": ["no_supported_horizons", "unknown_state"],
        }
    labels = [str(row.get("label") or "unknown") for row in windows]
    confidences = [float(row.get("confidence") or 0.0) for row in windows]
    uncertainties = [float(row.get("uncertainty") or 1.0) for row in windows]
    ood_scores = [float(row.get("out_of_distribution_score") or 0.0) for row in windows]
    unknown_reason = None
    reason_codes: list[str] = []
    conflict_reasons: list[str] = []
    label_set = set(labels)
    if stale_data:
        aggregate_label = "unknown"
        unknown_reason = "stale_local_data"
        reason_codes.append("stale_local_data")
    elif any(label == "unknown" for label in labels):
        aggregate_label = "unknown"
        unknown_reason = "one_or_more_horizons_unknown"
        reason_codes.append("unknown_state")
    elif max(ood_scores) >= config.out_of_distribution_threshold:
        aggregate_label = "out_of_distribution"
        reason_codes.append("out_of_distribution_state")
    elif _labels_conflict(label_set):
        aggregate_label = "mixed"
        reason_codes.append("horizon_conflict")
        conflict_reasons.append("horizon_labels_conflict")
    elif len(label_set) == 1:
        aggregate_label = labels[0]
    else:
        aggregate_label = "transition"
        reason_codes.append("transition_state")
        conflict_reasons.append("horizon_transition")
    state_confidence = _rounded(sum(confidences) / len(confidences))
    uncertainty = _rounded(sum(uncertainties) / len(uncertainties))
    if state_confidence is not None and state_confidence < config.confidence_threshold:
        reason_codes.append("low_state_confidence")
    no_trade_default = aggregate_label in {
        "unknown",
        "mixed",
        "transition",
        "out_of_distribution",
    } or "low_state_confidence" in reason_codes
    return {
        "aggregate_label": aggregate_label,
        "state_confidence": state_confidence,
        "uncertainty": uncertainty,
        "unknown_reason": unknown_reason,
        "out_of_distribution_score": _rounded(max(ood_scores)),
        "horizon_conflict": {
            "conflict_detected": bool(conflict_reasons),
            "reason_codes": conflict_reasons,
        },
        "no_trade_default": no_trade_default,
        "reason_codes": sorted(set(reason_codes or [f"{aggregate_label}_state"])),
    }


def _window_reason_codes(
    label: str, confidence: float, config: MarketStateConfig
) -> list[str]:
    reasons = [f"label_{label}"]
    if confidence < config.confidence_threshold:
        reasons.append("low_state_confidence")
    if label in {"unknown", "mixed", "transition", "out_of_distribution"}:
        reasons.append(f"{label}_state")
    return reasons


def _labels_conflict(labels: set[str]) -> bool:
    clean = labels - {"unknown"}
    if len(clean) <= 1:
        return False
    if {"trend_up", "trend_down"} <= clean:
        return True
    if "range" in clean and ({"trend_up", "trend_down"} & clean):
        return True
    if "high_volatility" in clean and len(clean) > 1:
        return True
    return len(clean) > 1


def _horizon_profile_id(windows: Sequence[Mapping[str, Any]], config: MarketStateConfig) -> str:
    groups = {"micro": "missing", "intraday": "missing", "swing": "missing"}
    for group in groups:
        labels = [
            str(row.get("label"))
            for row in windows
            if row.get("horizon_group") == group and row.get("label")
        ]
        if labels:
            groups[group] = labels[-1] if len(set(labels)) == 1 else "mixed"
    return (
        f"{config.state_encoder_version}:"
        f"micro={groups['micro']}:"
        f"intraday={groups['intraday']}:"
        f"swing={groups['swing']}"
    )


def _state_id(label: str, horizon: str, confidence: float, config: MarketStateConfig) -> str:
    if confidence >= 0.75:
        bucket = "high"
    elif confidence >= config.confidence_threshold:
        bucket = "medium"
    else:
        bucket = "low"
    return f"{config.state_encoder_version}:{horizon}:{label}:{bucket}:{config.feature_version}"


def _out_of_distribution_score(label: str, confidence: float) -> float:
    if label == "out_of_distribution":
        return 1.0
    if label == "unknown":
        return 1.0
    return round(max(0.0, min(1.0, 1.0 - confidence)), 6)


def _market_state_safety_scope() -> dict[str, bool]:
    return {
        "local_artifacts_source_of_truth": True,
        "closed_candle_local_market_data_only": True,
        "live_data_used": False,
        "freqtrade_trade_started": False,
        "paper_trading_started": False,
        "dry_run_trading_started": False,
        "live_trading_started": False,
        "exchange_order_placement": False,
        "uses_api_keys_or_secrets": False,
        "metadata_contains_secrets": False,
        "process_control": False,
        "leverage_above_one": False,
        "shorting": False,
    }


def _write_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _file_hash(path: Path) -> str | None:
    try:
        if not path.is_file():
            return None
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return "sha256:" + digest.hexdigest()
    except OSError:
        return None


def _is_stale(value: str | None, now: datetime, max_staleness_seconds: int) -> bool:
    parsed = _parse_datetime(value)
    if parsed is None:
        return True
    if now.tzinfo is None:
        now = now.replace(tzinfo=UTC)
    return (now - parsed).total_seconds() > max_staleness_seconds


def _parse_datetime(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _timestamp_to_iso(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    parsed = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.isoformat()


def _timeframe_seconds(timeframe: str) -> int | None:
    unit = timeframe[-1:].lower()
    try:
        amount = int(timeframe[:-1])
    except (TypeError, ValueError):
        return None
    multipliers = {"m": 60, "h": 3600, "d": 86400, "w": 604800}
    multiplier = multipliers.get(unit)
    return amount * multiplier if multiplier else None


def _pandas_resample_rule(timeframe: str) -> str | None:
    unit = timeframe[-1:].lower()
    amount = timeframe[:-1]
    if not amount.isdigit():
        return None
    if unit == "m":
        return f"{amount}min"
    if unit == "h":
        return f"{amount}h"
    if unit == "d":
        return f"{amount}D"
    if unit == "w":
        return f"{amount}W"
    return None


def _lookback_duration(timeframe: str, lookback: int) -> str:
    seconds = (_timeframe_seconds(timeframe) or 0) * lookback
    if seconds and seconds % 86400 == 0:
        return f"P{seconds // 86400}D"
    if seconds and seconds % 3600 == 0:
        return f"PT{seconds // 3600}H"
    if seconds and seconds % 60 == 0:
        return f"PT{seconds // 60}M"
    return f"PT{seconds}S"


def _missing_candle_rate(frame: pd.DataFrame, timeframe: str) -> float:
    seconds = _timeframe_seconds(timeframe)
    if seconds is None or len(frame) < 2:
        return 0.0
    dates = pd.to_datetime(frame["date"], utc=True, errors="coerce").dropna().sort_values()
    if len(dates) < 2:
        return 1.0
    span = (dates.iloc[-1] - dates.iloc[0]).total_seconds()
    expected = int(span // seconds) + 1
    if expected <= 0:
        return 0.0
    missing = max(expected - len(dates), 0)
    return missing / expected


def _bps(value: Any) -> float | None:
    number = _number(value)
    return None if number is None else round(number * 10000.0, 6)


def _rounded(value: Any) -> float | None:
    number = _number(value)
    return None if number is None else round(number, 6)


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        if pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _max_number(values: Sequence[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    return round(max(clean), 6) if clean else None


def _safe_component(value: str) -> str:
    clean = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value)
    return clean.strip("._") or "market_state"


def _input_checks(frame: pd.DataFrame, config: RegimeClassifierConfig) -> list[dict[str, Any]]:
    missing_columns = [column for column in REQUIRED_OHLCV_COLUMNS if column not in frame.columns]
    missing_rate = (
        float(frame[list(set(REQUIRED_OHLCV_COLUMNS) & set(frame.columns))].isna().mean().max())
        if len(frame) and not missing_columns
        else 1.0 if missing_columns else 0.0
    )
    dates = pd.to_datetime(frame["date"], utc=True, errors="coerce") if "date" in frame.columns else []
    duplicate_dates = int(pd.Series(dates).duplicated().sum()) if len(frame) and "date" in frame.columns else 0
    return [
        _check("required_ohlcv_columns_present", not missing_columns, {"missing_columns": missing_columns}),
        _check("minimum_rows_present", len(frame) >= config.min_rows, {"rows": len(frame), "min_rows": config.min_rows}),
        _check("missing_rate_within_threshold", missing_rate <= config.max_missing_rate, {"missing_rate": missing_rate}),
        _check("timestamps_unique", duplicate_dates == 0, {"duplicate_timestamps": duplicate_dates}),
    ]


def _classify_row(
    features: dict[str, float | None],
    *,
    one_period_return: float,
    volatility_baseline: float | None,
    config: RegimeClassifierConfig,
    flags: Sequence[str],
) -> tuple[str, float]:
    rolling_return = features.get("rolling_return")
    volatility = features.get("realized_volatility")
    slope = features.get("trend_slope")
    efficiency = features.get("range_efficiency") or 0.0
    volume_ratio = features.get("volume_liquidity_ratio")
    if flags or rolling_return is None or volatility is None or slope is None:
        return "unknown", 0.0
    vol_ratio = volatility / volatility_baseline if volatility_baseline else 1.0
    confidence = min(1.0, max(config.min_confidence, abs(rolling_return) * 10.0 + efficiency * 0.5))
    if volume_ratio is not None and volume_ratio < config.liquidity_ratio_threshold:
        return "liquidity_stress", max(config.min_confidence, 1.0 - volume_ratio)
    if abs(one_period_return) >= config.spike_return_threshold and one_period_return * rolling_return < 0:
        return "post_spike_reversion", confidence
    if vol_ratio >= config.high_volatility_ratio:
        return "high_volatility", min(1.0, vol_ratio / (config.high_volatility_ratio * 2.0))
    if (
        rolling_return >= config.trend_return_threshold
        and slope > 0
        and efficiency >= config.trend_efficiency_threshold
    ):
        return "trend_up", confidence
    if (
        rolling_return <= -config.trend_return_threshold
        and slope < 0
        and efficiency >= config.trend_efficiency_threshold
    ):
        return "trend_down", confidence
    if abs(rolling_return) <= config.range_return_threshold and efficiency <= config.range_efficiency_threshold:
        return "range", max(config.min_confidence, 1.0 - efficiency)
    if vol_ratio <= config.low_volatility_ratio:
        return "low_volatility", max(config.min_confidence, 1.0 - vol_ratio)
    return "mixed", config.min_confidence


def _row_quality_flags(source: pd.Series, index: int, config: RegimeClassifierConfig) -> list[str]:
    flags: list[str] = []
    if index < config.lookback:
        flags.append("insufficient_lookback")
    for column in REQUIRED_OHLCV_COLUMNS:
        if pd.isna(source.get(column)):
            flags.append(f"{column}_missing")
    if any((_float(source.get(column)) or 0.0) <= 0.0 for column in ("open", "high", "low", "close")):
        flags.append("non_positive_price")
    return flags


def _row(
    *,
    date: str,
    pair: str,
    timeframe: str,
    market_regime: str,
    confidence: float,
    features: dict[str, float | None],
    data_quality_flags: Sequence[str],
) -> dict[str, Any]:
    return {
        "date": date,
        "pair": pair,
        "timeframe": timeframe,
        "market_regime": market_regime,
        "confidence": round(float(confidence), 6),
        "features": features,
        "data_quality_flags": list(data_quality_flags),
        "regime_classifier_version": REGIME_CLASSIFIER_VERSION,
    }


def _artifact(
    rows: list[dict[str, Any]],
    pair: str,
    timeframe: str,
    config: RegimeClassifierConfig,
    checks: list[dict[str, Any]],
    generated_at: str,
) -> dict[str, Any]:
    label_counts = {label: 0 for label in REGIME_LABELS}
    for row in rows:
        label_counts[str(row.get("market_regime"))] = label_counts.get(str(row.get("market_regime")), 0) + 1
    return {
        "factory": "deterministic_market_regime_classifier",
        "schema_version": "market_regime_classifier_v1",
        "regime_classifier_version": config.version,
        "generated_at": generated_at,
        "pair": pair,
        "timeframe": timeframe,
        "row_count": len(rows),
        "label_counts": label_counts,
        "checks": checks,
        "ok": all(check["passed"] for check in checks),
        "feature_set": [
            "rolling_return",
            "close",
            "realized_volatility",
            "trend_slope",
            "range_efficiency",
            "drawdown_from_local_high",
            "volume_liquidity_ratio",
            "candle_gap_proxy",
            "data_quality_flags",
        ],
        "config": asdict(config),
        "rows": rows,
    }


def _float(value: Any) -> float | None:
    try:
        if pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _check(name: str, passed: bool, details: dict[str, Any]) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "details": details}
