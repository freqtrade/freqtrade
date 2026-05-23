from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

import pandas as pd


REGIME_CLASSIFIER_VERSION = "deterministic_regime_classifier_v1"
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
