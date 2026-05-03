#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from freqtrade_ext.bot_factory.backtest_results import (
    BacktestMetrics,
    evaluate_initial_gate,
    load_gate_thresholds,
)
from freqtrade_ext.bot_factory.freqai_backtest import (
    freqai_enabled,
    freqai_identifier,
    freqai_model_name,
    load_json_config,
    selected_pairs,
)
from freqtrade_ext.bot_factory.freqai_checks import (
    FREQAI_LABEL_NOTICE,
    check_freqai_dependencies,
    missing_required_dependencies,
)
from freqtrade_ext.bot_factory.freqai_training import (
    TrainingStageResult,
    build_checked_freqai_backtest_command,
    build_checked_walk_forward_command,
    build_training_manifest,
    training_child_run_id,
    write_training_manifest,
    write_training_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the Bot Factory FreqAI training factory through checked "
            "historical backtesting paths only."
        )
    )
    parser.add_argument("--config", default="user_data/config.json")
    parser.add_argument("--strategy", required=True)
    parser.add_argument("--strategy-path", default="user_data/strategies")
    parser.add_argument("--freqaimodel", default=None)
    parser.add_argument("--freqaimodel-path", default=None)
    parser.add_argument("--timeframe", default=None)
    parser.add_argument("--timerange", required=True)
    parser.add_argument("--pairs", nargs="*", default=None)
    parser.add_argument("--output-root", default="data/freqai_training")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--freqai-runner-script",
        default="scripts/bot_factory_run_freqai_backtest.py",
        help="Checked FreqAI backtest wrapper to use for the training stage.",
    )
    parser.add_argument(
        "--walk-forward-runner-script",
        default="scripts/bot_factory_run_walk_forward.py",
        help="Checked walk-forward wrapper to use when walk-forward windows are supplied.",
    )
    parser.add_argument("--data-format-ohlcv", default="parquet")
    parser.add_argument("--userdir", default=None)
    parser.add_argument("--datadir", default=None)
    parser.add_argument("--trading-mode", default=None)
    parser.add_argument(
        "--ohlcv-file",
        action="append",
        default=None,
        help="Explicit OHLCV parquet file to quality-check. Can be repeated.",
    )
    parser.add_argument(
        "--gate-config",
        default=None,
        help="Optional JSON file overriding backtest gate thresholds.",
    )
    parser.add_argument(
        "--reviewer-note",
        action="append",
        default=None,
        help="Optional note to include in generated child reports. Can be repeated.",
    )
    parser.add_argument(
        "--mlflow",
        action="store_true",
        help="Optionally pass MLflow logging through to child checked wrappers.",
    )
    parser.add_argument("--mlflow-tracking-uri", default=None)
    parser.add_argument("--mlflow-experiment", default="bot_factory_freqai_training")
    parser.add_argument(
        "--window",
        action="append",
        default=None,
        help=(
            "Optional walk-forward fixed historical window. Use START-END or "
            "TRAIN_START:TRAIN_END:TEST_START:TEST_END. Can be repeated."
        ),
    )
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--train-days", type=int, default=None)
    parser.add_argument("--test-days", type=int, default=None)
    parser.add_argument("--step-days", type=int, default=None)
    parser.add_argument("--min-pass-rate", type=float, default=0.7)
    parser.add_argument("--min-profitable-windows-ratio", type=float, default=0.6)
    parser.add_argument("--max-drawdown-pct-any-window", type=float, default=20.0)
    parser.add_argument("--max-single-window-profit-dependency", type=float, default=0.4)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = Path(args.config)
    _require_file(config_path, "config")
    _require_file(Path(args.freqai_runner_script), "FreqAI runner script")
    if _walk_forward_requested(args):
        _require_file(Path(args.walk_forward_runner_script), "walk-forward runner script")

    config = load_json_config(config_path)
    run_id = args.run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    args.run_id = run_id
    run_dir = Path(args.output_root) / args.strategy / run_id
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    artifact_paths: dict[str, Path | None] = {
        "training_manifest": run_dir / "training_manifest.json",
        "training_report": run_dir / "training_report.md",
        "command": run_dir / "command.txt",
        "freqai_env": run_dir / "freqai_env.json",
        "logs": logs_dir,
    }
    notes = [
        "FreqAI training factory verification only; no paper or live promotion.",
        "Local artifacts remain the source of truth.",
        FREQAI_LABEL_NOTICE,
    ]
    notes.extend(args.reviewer_note or [])
    child_reviewer_notes = [
        "FreqAI training factory verification only; no paper or live promotion."
    ]
    child_reviewer_notes.extend(args.reviewer_note or [])

    dependency_report = check_freqai_dependencies()
    artifact_paths["freqai_env"].write_text(dependency_report.to_json(), encoding="utf-8")

    if missing_required_dependencies(dependency_report):
        manifest = _build_manifest(args, config, dependency_report.to_dict(), [], artifact_paths, notes)
        manifest["status"] = "blocked_dependency_check"
        manifest["recommendation"] = "fail"
        write_training_manifest(manifest, artifact_paths["training_manifest"])
        write_training_report(manifest, artifact_paths["training_report"])
        print(dependency_report.to_json())
        print(f"FreqAI dependency check failed. Report: {artifact_paths['freqai_env']}")
        return 1

    if not freqai_enabled(config):
        manifest = _build_manifest(args, config, dependency_report.to_dict(), [], artifact_paths, notes)
        manifest["status"] = "blocked_freqai_disabled"
        manifest["recommendation"] = "fail"
        write_training_manifest(manifest, artifact_paths["training_manifest"])
        write_training_report(manifest, artifact_paths["training_report"])
        print(f"FreqAI is not enabled in config: {config_path}")
        return 1

    thresholds = load_gate_thresholds(Path(args.gate_config) if args.gate_config else None)
    stage_results: list[TrainingStageResult] = []
    command_lines: list[str] = []

    backtest_run_id = training_child_run_id("train", args.timerange)
    backtest_root = run_dir / "freqai_backtests"
    backtest_cmd = build_checked_freqai_backtest_command(
        python_executable=args.python,
        runner_script=args.freqai_runner_script,
        config=args.config,
        strategy=args.strategy,
        strategy_path=args.strategy_path,
        output_root=backtest_root,
        run_id=backtest_run_id,
        timerange=args.timerange,
        timeframe=args.timeframe,
        pairs=args.pairs,
        freqaimodel=args.freqaimodel,
        freqaimodel_path=args.freqaimodel_path,
        data_format_ohlcv=args.data_format_ohlcv,
        userdir=args.userdir,
        datadir=args.datadir,
        trading_mode=args.trading_mode,
        ohlcv_files=args.ohlcv_file,
        gate_config=args.gate_config,
        reviewer_notes=child_reviewer_notes,
        mlflow=args.mlflow,
        mlflow_tracking_uri=args.mlflow_tracking_uri,
        mlflow_experiment=args.mlflow_experiment,
    )
    command_lines.append(" ".join(backtest_cmd))
    stage_results.append(
        _run_backtest_stage(
            args=args,
            run_id=backtest_run_id,
            output_dir=backtest_root / args.strategy / backtest_run_id,
            cmd=backtest_cmd,
            logs_dir=logs_dir,
            thresholds=thresholds,
        )
    )

    if _walk_forward_requested(args):
        rolling_args = _rolling_args(args)
        walk_forward_run_id = training_child_run_id("wf", run_id)
        walk_forward_root = run_dir / "walk_forward"
        walk_forward_cmd = build_checked_walk_forward_command(
            python_executable=args.python,
            runner_script=args.walk_forward_runner_script,
            config=args.config,
            strategy=args.strategy,
            strategy_path=args.strategy_path,
            output_root=walk_forward_root,
            run_id=walk_forward_run_id,
            window_specs=args.window,
            rolling_args=rolling_args,
            timeframe=args.timeframe,
            pairs=args.pairs,
            freqaimodel=args.freqaimodel,
            freqaimodel_path=args.freqaimodel_path,
            data_format_ohlcv=args.data_format_ohlcv,
            userdir=args.userdir,
            datadir=args.datadir,
            trading_mode=args.trading_mode,
            ohlcv_files=args.ohlcv_file,
            gate_config=args.gate_config,
            reviewer_notes=child_reviewer_notes,
            mlflow=args.mlflow,
            mlflow_tracking_uri=args.mlflow_tracking_uri,
            mlflow_experiment=args.mlflow_experiment,
            min_pass_rate=args.min_pass_rate,
            min_profitable_windows_ratio=args.min_profitable_windows_ratio,
            max_drawdown_pct_any_window=args.max_drawdown_pct_any_window,
            max_single_window_profit_dependency=args.max_single_window_profit_dependency,
        )
        command_lines.append(" ".join(walk_forward_cmd))
        stage_results.append(
            _run_walk_forward_stage(
                args=args,
                run_id=walk_forward_run_id,
                output_dir=walk_forward_root / args.strategy / walk_forward_run_id,
                cmd=walk_forward_cmd,
                logs_dir=logs_dir,
            )
        )

    artifact_paths["command"].write_text("\n".join(command_lines), encoding="utf-8")
    manifest = _build_manifest(args, config, dependency_report.to_dict(), stage_results, artifact_paths, notes)
    write_training_manifest(manifest, artifact_paths["training_manifest"])
    write_training_report(manifest, artifact_paths["training_report"])
    print(f"FreqAI training factory artifacts written: {run_dir}")
    return 1 if manifest["status"] != "completed" else 0


def _run_backtest_stage(
    *,
    args: argparse.Namespace,
    run_id: str,
    output_dir: Path,
    cmd: list[str],
    logs_dir: Path,
    thresholds: Any,
) -> TrainingStageResult:
    completed = _run_stage_command("freqai_backtest", cmd, logs_dir)
    artifacts = _stage_log_artifacts("freqai_backtest", logs_dir)
    metrics_path = output_dir / "metrics.json"
    report_path = output_dir / "report.md"
    artifacts.update(
        {
            "metrics": metrics_path,
            "report": report_path,
            "trades": output_dir / "trades.csv",
            "freqai_metadata": output_dir / "freqai_metadata.json",
            "freqai_validation": output_dir / "freqai_validation.json",
        }
    )
    if completed.returncode == 0 and metrics_path.is_file():
        metrics_payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        gate = evaluate_initial_gate(BacktestMetrics(**metrics_payload), thresholds)
        return TrainingStageResult(
            name="freqai_backtest",
            run_id=run_id,
            status="completed",
            returncode=completed.returncode,
            output_dir=output_dir,
            recommendation=gate["recommendation"],
            artifacts=artifacts,
            command=cmd,
        )

    return TrainingStageResult(
        name="freqai_backtest",
        run_id=run_id,
        status="failed",
        returncode=completed.returncode,
        output_dir=output_dir,
        recommendation=None,
        artifacts=artifacts,
        command=cmd,
        error=_stage_error(completed, metrics_path),
    )


def _run_walk_forward_stage(
    *,
    args: argparse.Namespace,
    run_id: str,
    output_dir: Path,
    cmd: list[str],
    logs_dir: Path,
) -> TrainingStageResult:
    completed = _run_stage_command("walk_forward", cmd, logs_dir)
    artifacts = _stage_log_artifacts("walk_forward", logs_dir)
    metrics_path = output_dir / "walk_forward_metrics.json"
    report_path = output_dir / "walk_forward_report.md"
    artifacts.update({"metrics": metrics_path, "report": report_path})
    if completed.returncode == 0 and metrics_path.is_file():
        metrics_payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        return TrainingStageResult(
            name="walk_forward",
            run_id=run_id,
            status="completed",
            returncode=completed.returncode,
            output_dir=output_dir,
            recommendation=metrics_payload.get("recommendation"),
            artifacts=artifacts,
            command=cmd,
        )

    return TrainingStageResult(
        name="walk_forward",
        run_id=run_id,
        status="failed",
        returncode=completed.returncode,
        output_dir=output_dir,
        recommendation=None,
        artifacts=artifacts,
        command=cmd,
        error=_stage_error(completed, metrics_path),
    )


def _run_stage_command(
    stage_name: str, cmd: list[str], logs_dir: Path
) -> subprocess.CompletedProcess[str]:
    print(f"Running {stage_name}: {' '.join(cmd)}")
    completed = subprocess.run(cmd, text=True, capture_output=True)
    artifacts = _stage_log_artifacts(stage_name, logs_dir)
    artifacts["command"].write_text(" ".join(cmd), encoding="utf-8")
    artifacts["stdout"].write_text(completed.stdout or "", encoding="utf-8")
    artifacts["stderr"].write_text(completed.stderr or "", encoding="utf-8")
    return completed


def _stage_log_artifacts(stage_name: str, logs_dir: Path) -> dict[str, Path]:
    return {
        "command": logs_dir / f"{stage_name}.command.txt",
        "stdout": logs_dir / f"{stage_name}.stdout.log",
        "stderr": logs_dir / f"{stage_name}.stderr.log",
    }


def _stage_error(completed: subprocess.CompletedProcess[str], metrics_path: Path) -> str:
    if completed.returncode == 0:
        return f"Stage command completed but metrics file was not found: {metrics_path}"
    stderr = (completed.stderr or "").strip()
    if stderr:
        return stderr.splitlines()[-1]
    stdout = (completed.stdout or "").strip()
    return stdout.splitlines()[-1] if stdout else f"Stage command failed: {completed.returncode}"


def _build_manifest(
    args: argparse.Namespace,
    config: dict[str, Any],
    dependency_status: dict[str, Any],
    stages: list[TrainingStageResult],
    artifact_paths: dict[str, Path | None],
    notes: list[str],
) -> dict[str, Any]:
    return build_training_manifest(
        root_dir=ROOT_DIR,
        strategy=args.strategy,
        run_id=args.run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"),
        config_path=Path(args.config),
        timeframe=args.timeframe or str(config.get("timeframe") or ""),
        timerange=args.timerange,
        pairs=selected_pairs(config, args.pairs),
        freqaimodel=freqai_model_name(config, args.freqaimodel),
        freqai_identifier=freqai_identifier(config),
        dependency_status=dependency_status,
        stages=stages,
        artifact_paths=artifact_paths,
        notes=notes,
    )


def _walk_forward_requested(args: argparse.Namespace) -> bool:
    return bool(
        args.window
        or args.start
        or args.end
        or args.train_days
        or args.test_days
        or args.step_days
    )


def _rolling_args(args: argparse.Namespace) -> dict[str, Any]:
    if args.window:
        return {}
    values = {
        "--start": args.start,
        "--end": args.end,
        "--train-days": args.train_days,
        "--test-days": args.test_days,
        "--step-days": args.step_days,
    }
    missing = [name for name, value in values.items() if value is None]
    if missing:
        raise SystemExit(
            "Provide all rolling-window arguments or use --window: " + ", ".join(missing)
        )
    return values


def _require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise SystemExit(f"{label} file not found: {path}")


if __name__ == "__main__":
    sys.exit(main())
