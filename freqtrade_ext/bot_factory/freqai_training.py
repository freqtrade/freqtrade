from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

from freqtrade_ext.bot_factory.freqai_checks import FREQAI_LABEL_NOTICE


@dataclass(frozen=True)
class TrainingStageResult:
    name: str
    run_id: str
    status: str
    returncode: int | None
    output_dir: Path
    recommendation: str | None = None
    artifacts: dict[str, Path] = field(default_factory=dict)
    command: Sequence[str] = field(default_factory=list)
    error: str | None = None


def training_child_run_id(prefix: str, token: str) -> str:
    safe_token = re.sub(r"[^A-Za-z0-9_.-]+", "_", token.strip())
    safe_token = safe_token.replace("-", "_").strip("_")
    if not safe_token:
        safe_token = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}_{safe_token}"


def build_checked_freqai_backtest_command(
    *,
    python_executable: str,
    runner_script: str,
    config: str,
    strategy: str,
    strategy_path: str,
    output_root: Path,
    run_id: str,
    timerange: str,
    timeframe: str | None = None,
    pairs: Sequence[str] | None = None,
    freqaimodel: str | None = None,
    freqaimodel_path: str | None = None,
    freqai_identifier: str | None = None,
    data_format_ohlcv: str = "parquet",
    userdir: str | None = None,
    datadir: str | None = None,
    trading_mode: str | None = None,
    ohlcv_files: Sequence[str] | None = None,
    gate_config: str | None = None,
    reviewer_notes: Sequence[str] | None = None,
    mlflow: bool = False,
    mlflow_tracking_uri: str | None = None,
    mlflow_experiment: str | None = None,
) -> list[str]:
    cmd = [
        python_executable,
        runner_script,
        "--config",
        config,
        "--strategy",
        strategy,
        "--strategy-path",
        strategy_path,
        "--timerange",
        timerange,
        "--output-root",
        str(output_root),
        "--run-id",
        run_id,
        "--python",
        python_executable,
        "--data-format-ohlcv",
        data_format_ohlcv,
    ]
    _append_common_freqai_args(
        cmd,
        timeframe=timeframe,
        pairs=pairs,
        freqaimodel=freqaimodel,
        freqaimodel_path=freqaimodel_path,
        freqai_identifier=freqai_identifier,
        userdir=userdir,
        datadir=datadir,
        trading_mode=trading_mode,
        ohlcv_files=ohlcv_files,
        gate_config=gate_config,
        reviewer_notes=reviewer_notes,
        mlflow=mlflow,
        mlflow_tracking_uri=mlflow_tracking_uri,
        mlflow_experiment=mlflow_experiment,
    )
    return cmd


def build_checked_walk_forward_command(
    *,
    python_executable: str,
    runner_script: str,
    config: str,
    strategy: str,
    strategy_path: str,
    output_root: Path,
    run_id: str,
    window_specs: Sequence[str] | None = None,
    rolling_args: dict[str, Any] | None = None,
    timeframe: str | None = None,
    pairs: Sequence[str] | None = None,
    freqaimodel: str | None = None,
    freqaimodel_path: str | None = None,
    freqai_identifier: str | None = None,
    data_format_ohlcv: str = "parquet",
    userdir: str | None = None,
    datadir: str | None = None,
    trading_mode: str | None = None,
    ohlcv_files: Sequence[str] | None = None,
    gate_config: str | None = None,
    reviewer_notes: Sequence[str] | None = None,
    mlflow: bool = False,
    mlflow_tracking_uri: str | None = None,
    mlflow_experiment: str | None = None,
    min_pass_rate: float = 0.7,
    min_profitable_windows_ratio: float = 0.6,
    max_drawdown_pct_any_window: float = 20.0,
    max_single_window_profit_dependency: float = 0.4,
) -> list[str]:
    cmd = [
        python_executable,
        runner_script,
        "--config",
        config,
        "--strategy",
        strategy,
        "--strategy-path",
        strategy_path,
        "--output-root",
        str(output_root),
        "--run-id",
        run_id,
        "--python",
        python_executable,
        "--data-format-ohlcv",
        data_format_ohlcv,
        "--min-pass-rate",
        str(min_pass_rate),
        "--min-profitable-windows-ratio",
        str(min_profitable_windows_ratio),
        "--max-drawdown-pct-any-window",
        str(max_drawdown_pct_any_window),
        "--max-single-window-profit-dependency",
        str(max_single_window_profit_dependency),
    ]
    for spec in window_specs or []:
        cmd.extend(["--window", spec])
    for name, value in (rolling_args or {}).items():
        if value is not None:
            cmd.extend([name, str(value)])
    _append_common_freqai_args(
        cmd,
        timeframe=timeframe,
        pairs=pairs,
        freqaimodel=freqaimodel,
        freqaimodel_path=freqaimodel_path,
        freqai_identifier=freqai_identifier,
        userdir=userdir,
        datadir=datadir,
        trading_mode=trading_mode,
        ohlcv_files=ohlcv_files,
        gate_config=gate_config,
        reviewer_notes=reviewer_notes,
        mlflow=mlflow,
        mlflow_tracking_uri=mlflow_tracking_uri,
        mlflow_experiment=mlflow_experiment,
    )
    return cmd


def build_training_manifest(
    *,
    root_dir: Path,
    strategy: str,
    run_id: str,
    config_path: Path,
    timeframe: str | None,
    timerange: str | None,
    pairs: Iterable[str],
    freqaimodel: str | None,
    freqai_identifier: str | None,
    dependency_status: dict[str, Any],
    stages: Sequence[TrainingStageResult],
    artifact_paths: dict[str, Path | None],
    notes: Iterable[str] | None = None,
    status: str | None = None,
) -> dict[str, Any]:
    completed_stages = [stage for stage in stages if stage.status == "completed"]
    failed_stages = [stage for stage in stages if stage.status != "completed"]
    resolved_status = status or ("completed" if stages and not failed_stages else "failed")
    recommendation = _training_recommendation(stages, resolved_status)

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "phase": "2",
        "factory": "freqai_training",
        "status": resolved_status,
        "recommendation": recommendation,
        "strategy": strategy,
        "run_id": run_id,
        "config_path": _safe_relative_path(config_path, root_dir),
        "freqaimodel": freqaimodel,
        "freqai_identifier": freqai_identifier,
        "timeframe": timeframe,
        "timerange": timerange,
        "pairs": list(pairs),
        "dependency_status": dependency_status,
        "summary": {
            "stage_count": len(stages),
            "completed_stages": len(completed_stages),
            "failed_stages": len(failed_stages),
        },
        "stages": [_stage_to_dict(stage, root_dir) for stage in stages],
        "artifact_paths": {
            name: _safe_relative_path(path, root_dir)
            for name, path in artifact_paths.items()
            if path is not None
        },
        "notes": list(notes or []),
        "safety_scope": {
            "command": "freqtrade backtesting only through checked wrappers",
            "paper_trading": False,
            "dry_run_trading": False,
            "live_trading": False,
            "exchange_order_placement": False,
            "leverage_experiments": False,
            "shorting": False,
            "metadata_contains_secrets": False,
            "local_artifacts_source_of_truth": True,
        },
    }


def write_training_manifest(manifest: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def write_training_report(manifest: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = manifest["summary"]
    lines = [
        "# FreqAI Training Factory Report",
        "",
        "## Summary",
        "",
        f"- Strategy: {manifest['strategy']}",
        f"- Run ID: {manifest['run_id']}",
        f"- Status: {manifest['status']}",
        f"- Recommendation: {manifest['recommendation']}",
        f"- Stages completed: {summary['completed_stages']}/{summary['stage_count']}",
        f"- Timerange: {manifest.get('timerange') or 'n/a'}",
        f"- Timeframe: {manifest.get('timeframe') or 'n/a'}",
        f"- FreqAI model: {manifest.get('freqaimodel') or 'n/a'}",
        "",
        "## Stages",
        "",
    ]
    if manifest["stages"]:
        for stage in manifest["stages"]:
            lines.append(
                "- {name}: status={status}, recommendation={recommendation}, "
                "returncode={returncode}".format(
                    name=stage["name"],
                    status=stage["status"],
                    recommendation=stage.get("recommendation") or "n/a",
                    returncode=_fmt(stage.get("returncode")),
                )
            )
            if stage.get("error"):
                lines.append(f"  Error: {stage['error']}")
    else:
        lines.append("- No training stages executed.")

    lines.extend(
        [
            "",
            "## Artifacts",
            "",
        ]
    )
    for name, artifact_path in manifest["artifact_paths"].items():
        lines.append(f"- {name}: `{artifact_path}`")

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- This training factory uses checked historical Freqtrade backtesting wrappers only.",
            "- Local JSON, CSV, log, and Markdown files remain the source of truth.",
            "- Passing training or walk-forward gates does not authorize paper trading or live trading.",
            f"- {FREQAI_LABEL_NOTICE}",
            "",
        ]
    )
    for note in manifest.get("notes", []):
        if note != FREQAI_LABEL_NOTICE:
            lines.append(f"- {note}")

    path.write_text("\n".join(lines), encoding="utf-8")


def _append_common_freqai_args(
    cmd: list[str],
    *,
    timeframe: str | None,
    pairs: Sequence[str] | None,
    freqaimodel: str | None,
    freqaimodel_path: str | None,
    freqai_identifier: str | None,
    userdir: str | None,
    datadir: str | None,
    trading_mode: str | None,
    ohlcv_files: Sequence[str] | None,
    gate_config: str | None,
    reviewer_notes: Sequence[str] | None,
    mlflow: bool,
    mlflow_tracking_uri: str | None,
    mlflow_experiment: str | None,
) -> None:
    if freqaimodel:
        cmd.extend(["--freqaimodel", freqaimodel])
    if freqaimodel_path:
        cmd.extend(["--freqaimodel-path", freqaimodel_path])
    if freqai_identifier:
        cmd.extend(["--freqai-identifier", freqai_identifier])
    if timeframe:
        cmd.extend(["--timeframe", timeframe])
    if pairs:
        cmd.extend(["--pairs", *pairs])
    if userdir:
        cmd.extend(["--userdir", userdir])
    if datadir:
        cmd.extend(["--datadir", datadir])
    if trading_mode:
        cmd.extend(["--trading-mode", trading_mode])
    for path in ohlcv_files or []:
        cmd.extend(["--ohlcv-file", path])
    if gate_config:
        cmd.extend(["--gate-config", gate_config])
    for note in reviewer_notes or []:
        cmd.extend(["--reviewer-note", note])
    if mlflow:
        cmd.append("--mlflow")
    if mlflow_tracking_uri:
        cmd.extend(["--mlflow-tracking-uri", mlflow_tracking_uri])
    if mlflow_experiment:
        cmd.extend(["--mlflow-experiment", mlflow_experiment])


def _training_recommendation(stages: Sequence[TrainingStageResult], status: str) -> str:
    if status != "completed" or not stages:
        return "fail"
    return "pass" if all(stage.recommendation == "pass" for stage in stages) else "fail"


def _stage_to_dict(stage: TrainingStageResult, root_dir: Path) -> dict[str, Any]:
    return {
        "name": stage.name,
        "run_id": stage.run_id,
        "status": stage.status,
        "returncode": stage.returncode,
        "recommendation": stage.recommendation,
        "output_dir": _safe_relative_path(stage.output_dir, root_dir),
        "artifacts": {
            name: _safe_relative_path(path, root_dir) for name, path in stage.artifacts.items()
        },
        "command": list(stage.command),
        "error": stage.error,
    }


def _safe_relative_path(path: Path, root_dir: Path) -> str:
    try:
        return str(path.resolve().relative_to(root_dir.resolve()))
    except ValueError:
        return path.name


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    return str(value)
