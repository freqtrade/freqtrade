from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

from freqtrade_ext.bot_factory.data_quality import ohlcv_data_path


def load_json_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Config must be a JSON object: {path}")
    return payload


def freqai_model_name(config: dict[str, Any], explicit: str | None = None) -> str | None:
    if explicit:
        return explicit
    freqai = config.get("freqai", {})
    if not isinstance(freqai, dict):
        freqai = {}
    model_name = config.get("freqaimodel") or freqai.get("freqaimodel")
    return str(model_name) if model_name else None


def freqai_identifier(config: dict[str, Any]) -> str | None:
    freqai = config.get("freqai", {})
    if not isinstance(freqai, dict):
        return None
    identifier = freqai.get("identifier")
    return str(identifier) if identifier else None


def freqai_enabled(config: dict[str, Any]) -> bool:
    freqai = config.get("freqai", {})
    return isinstance(freqai, dict) and bool(freqai.get("enabled"))


def selected_pairs(config: dict[str, Any], explicit_pairs: Iterable[str] | None = None) -> list[str]:
    if explicit_pairs:
        return _unique(str(pair) for pair in explicit_pairs)

    exchange = config.get("exchange", {})
    if not isinstance(exchange, dict):
        return []
    whitelist = exchange.get("pair_whitelist", [])
    if not isinstance(whitelist, list):
        return []
    return _unique(str(pair) for pair in whitelist)


def freqai_input_pairs(
    config: dict[str, Any], explicit_pairs: Iterable[str] | None = None
) -> list[str]:
    pairs = selected_pairs(config, explicit_pairs)
    freqai = config.get("freqai", {})
    feature_parameters = freqai.get("feature_parameters", {}) if isinstance(freqai, dict) else {}
    corr_pairs = (
        feature_parameters.get("include_corr_pairlist", [])
        if isinstance(feature_parameters, dict)
        else []
    )
    if isinstance(corr_pairs, list):
        pairs.extend(str(pair) for pair in corr_pairs)
    return _unique(pairs)


def freqai_input_timeframes(
    config: dict[str, Any], explicit_timeframe: str | None = None
) -> list[str]:
    timeframes: list[str] = []
    if explicit_timeframe:
        timeframes.append(explicit_timeframe)
    elif config.get("timeframe"):
        timeframes.append(str(config["timeframe"]))

    freqai = config.get("freqai", {})
    feature_parameters = freqai.get("feature_parameters", {}) if isinstance(freqai, dict) else {}
    include_timeframes = (
        feature_parameters.get("include_timeframes", [])
        if isinstance(feature_parameters, dict)
        else []
    )
    if isinstance(include_timeframes, list):
        timeframes.extend(str(timeframe) for timeframe in include_timeframes)
    return _unique(timeframes)


def resolve_ohlcv_input_paths(
    *,
    config_path: Path,
    config: dict[str, Any],
    userdir: Path,
    pairs: Iterable[str],
    timeframes: Iterable[str],
    trading_mode: str | None = None,
    datadir: Path | None = None,
) -> list[Path]:
    mode = trading_mode or _trading_mode(config)
    paths: list[Path] = []
    for pair in pairs:
        for timeframe in timeframes:
            paths.append(
                ohlcv_data_path(
                    config_path=config_path,
                    userdir=userdir,
                    pair=pair,
                    timeframe=timeframe,
                    trading_mode=mode,
                    datadir=datadir,
                )
            )
    return _unique_paths(paths)


def build_freqai_metadata(
    *,
    root_dir: Path,
    strategy: str,
    run_id: str,
    status: str,
    config_paths: Iterable[Path],
    freqaimodel: str | None,
    freqai_id: str | None,
    timeframe: str | None,
    timerange: str | None,
    pairs: Iterable[str],
    dependency_status: dict[str, Any],
    artifact_paths: dict[str, Path | None],
    notes: Iterable[str] | None = None,
) -> dict[str, Any]:
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "phase": "2",
        "status": status,
        "strategy": strategy,
        "run_id": run_id,
        "freqaimodel": freqaimodel,
        "freqai_identifier": freqai_id,
        "timeframe": timeframe,
        "timerange": timerange,
        "pairs": list(pairs),
        "config_paths": [_safe_relative_path(path, root_dir) for path in config_paths],
        "dependency_status": dependency_status,
        "artifact_paths": {
            name: _safe_relative_path(path, root_dir)
            for name, path in artifact_paths.items()
            if path is not None
        },
        "notes": list(notes or []),
        "safety_scope": {
            "command": "freqtrade backtesting only",
            "paper_trading": False,
            "dry_run_trading": False,
            "live_trading": False,
            "exchange_order_placement": False,
            "metadata_contains_secrets": False,
        },
    }


def write_freqai_metadata(metadata: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")


def _trading_mode(config: dict[str, Any]) -> str:
    mode = config.get("trading_mode", "spot")
    return str(mode or "spot")


def _safe_relative_path(path: Path, root_dir: Path) -> str:
    try:
        return str(path.resolve().relative_to(root_dir.resolve()))
    except ValueError:
        return path.name


def _unique(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    unique_values: list[str] = []
    for value in values:
        if value not in seen:
            unique_values.append(value)
            seen.add(value)
    return unique_values


def _unique_paths(paths: Iterable[Path]) -> list[Path]:
    seen: set[str] = set()
    unique_paths: list[Path] = []
    for path in paths:
        key = str(path)
        if key not in seen:
            unique_paths.append(path)
            seen.add(key)
    return unique_paths
