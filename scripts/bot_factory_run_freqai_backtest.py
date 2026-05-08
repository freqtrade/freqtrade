#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from freqtrade_ext.bot_factory.backtest_results import (
    BacktestMetrics,
    load_backtest_result,
    load_gate_thresholds,
    summarize,
    write_metrics,
    write_report,
    write_result_json,
    write_trades_csv,
)
from freqtrade_ext.bot_factory.data_quality import check_ohlcv_parquet, write_quality_reports
from freqtrade_ext.bot_factory.freqai_backtest import (
    build_freqai_metadata,
    freqai_enabled,
    freqai_identifier,
    freqai_input_pairs,
    freqai_input_timeframes,
    freqai_model_name,
    load_json_config,
    resolve_ohlcv_input_paths,
    sanitize_freqai_identifier,
    selected_pairs,
    write_freqai_identifier_override_config,
    write_freqai_metadata,
)
from freqtrade_ext.bot_factory.freqai_checks import (
    FREQAI_LABEL_NOTICE,
    check_freqai_dependencies,
    missing_required_dependencies,
    validate_freqai_strategy_paths,
)
from freqtrade_ext.bot_factory.mlflow_tracking import log_backtest_to_mlflow
from freqtrade_ext.bot_factory.safety import scan_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a checked Bot Factory FreqAI backtest. Uses backtesting only."
    )
    parser.add_argument("--config", default="user_data/config.json")
    parser.add_argument("--strategy", required=True)
    parser.add_argument("--strategy-path", default="user_data/strategies")
    parser.add_argument("--freqaimodel", default=None)
    parser.add_argument("--freqaimodel-path", default=None)
    parser.add_argument(
        "--freqai-identifier",
        default=None,
        help=(
            "Candidate-specific FreqAI identifier override. A minimal non-secret "
            "override config is written in the run artifact directory."
        ),
    )
    parser.add_argument("--timeframe", default=None)
    parser.add_argument("--timerange", default=None)
    parser.add_argument("--pairs", nargs="*", default=None)
    parser.add_argument("--output-root", default="data/freqai")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--python", default=sys.executable)
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
        help="Optional note to include in the generated report. Can be repeated.",
    )
    parser.add_argument(
        "--mlflow",
        action="store_true",
        help="Optionally log metrics and artifacts to MLflow. Failures do not fail the backtest.",
    )
    parser.add_argument("--mlflow-tracking-uri", default=None)
    parser.add_argument("--mlflow-experiment", default="bot_factory_freqai_backtests")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.freqai_identifier:
        args.freqai_identifier = sanitize_freqai_identifier(args.freqai_identifier)
    config_path = Path(args.config)
    _require_file(config_path, "config")
    config = load_json_config(config_path)

    strategy_file = _find_strategy_source(Path(args.strategy_path), args.strategy)
    run_id = args.run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(args.output_root) / args.strategy / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    artifacts: dict[str, Path | None] = {
        "command": run_dir / "command.txt",
        "stdout": run_dir / "stdout.log",
        "stderr": run_dir / "stderr.log",
        "result": run_dir / "result.json",
        "metrics": run_dir / "metrics.json",
        "trades": run_dir / "trades.csv",
        "report": run_dir / "report.md",
        "static_check": run_dir / "static_check.json",
        "freqai_validation": run_dir / "freqai_validation.json",
        "freqai_env": run_dir / "freqai_env.json",
        "ohlcv_quality": run_dir / "ohlcv_quality.json",
        "freqai_metadata": run_dir / "freqai_metadata.json",
    }
    if args.freqai_identifier:
        artifacts["freqai_identifier_override_config"] = (
            run_dir / "freqai_identifier_override.json"
        )
        write_freqai_identifier_override_config(
            args.freqai_identifier,
            artifacts["freqai_identifier_override_config"],
        )
    runtime_config_paths = _runtime_config_paths(args, artifacts)

    pairs = selected_pairs(config, args.pairs)
    metadata_notes = [
        "This run path is limited to Freqtrade backtesting.",
        "Passing gates do not authorize paper trading or live trading.",
        FREQAI_LABEL_NOTICE,
    ]
    if args.freqai_identifier:
        metadata_notes.append(
            "FreqAI identifier is candidate-scoped to avoid stale model or prediction cache reuse."
        )

    dependency_report = check_freqai_dependencies()
    artifacts["freqai_env"].write_text(dependency_report.to_json(), encoding="utf-8")
    if missing_required_dependencies(dependency_report):
        print(dependency_report.to_json())
        _write_metadata(
            args,
            config,
            dependency_report.to_dict(),
            artifacts,
            pairs,
            run_id,
            "blocked_dependency_check",
            metadata_notes,
        )
        print(f"FreqAI dependency check failed. Report: {artifacts['freqai_env']}")
        return 1

    if not freqai_enabled(config):
        _write_metadata(
            args,
            config,
            dependency_report.to_dict(),
            artifacts,
            pairs,
            run_id,
            "blocked_freqai_disabled",
            metadata_notes,
        )
        print(f"FreqAI is not enabled in config: {config_path}")
        return 1

    report = scan_paths([strategy_file])
    artifacts["static_check"].write_text(report.to_json(), encoding="utf-8")
    if not report.ok:
        print(report.to_json())
        _write_metadata(
            args,
            config,
            dependency_report.to_dict(),
            artifacts,
            pairs,
            run_id,
            "blocked_static_check",
            metadata_notes,
        )
        print(f"Static check failed. Report: {artifacts['static_check']}")
        return 1

    validation_report = validate_freqai_strategy_paths([strategy_file])
    artifacts["freqai_validation"].write_text(validation_report.to_json(), encoding="utf-8")
    if not validation_report.ok:
        print(validation_report.to_json())
        _write_metadata(
            args,
            config,
            dependency_report.to_dict(),
            artifacts,
            pairs,
            run_id,
            "blocked_freqai_validation",
            metadata_notes,
        )
        print(f"FreqAI validation failed. Report: {artifacts['freqai_validation']}")
        return 1

    quality_ok = _run_ohlcv_quality_checks(args, config_path, config, run_dir)
    if not quality_ok:
        _write_metadata(
            args,
            config,
            dependency_report.to_dict(),
            artifacts,
            pairs,
            run_id,
            "blocked_ohlcv_quality_check",
            metadata_notes,
        )
        print(f"OHLCV quality check failed. Report: {artifacts['ohlcv_quality']}")
        return 1

    result_filename = "result.json"
    cmd = _build_freqai_backtest_command(
        args, run_dir, result_filename, runtime_config_paths
    )
    artifacts["command"].write_text(" ".join(cmd), encoding="utf-8")

    print("Running:", " ".join(cmd))
    completed = subprocess.run(cmd, text=True, capture_output=True)
    artifacts["stdout"].write_text(completed.stdout or "", encoding="utf-8")
    artifacts["stderr"].write_text(completed.stderr or "", encoding="utf-8")
    if completed.returncode != 0:
        print(completed.stdout)
        print(completed.stderr, file=sys.stderr)
        _write_metadata(
            args,
            config,
            dependency_report.to_dict(),
            artifacts,
            pairs,
            run_id,
            "failed_backtest",
            metadata_notes,
        )
        print(f"FreqAI backtest failed. Logs: {run_dir}")
        return int(completed.returncode)

    result_path = _find_result_json(run_dir, result_filename)
    if result_path.name != result_filename:
        shutil.copy2(result_path, run_dir / result_filename)
        result_path = run_dir / result_filename

    result = load_backtest_result(result_path)
    write_result_json(result, run_dir / result_filename)
    metrics = summarize(result, args.strategy)
    write_metrics(metrics, run_dir / "metrics.json")
    write_trades_csv(result, run_dir / "trades.csv", args.strategy)
    thresholds = load_gate_thresholds(Path(args.gate_config) if args.gate_config else None)
    reviewer_notes = list(args.reviewer_note or [])
    reviewer_notes.append(FREQAI_LABEL_NOTICE)
    write_report(metrics, run_dir / "report.md", thresholds, reviewer_notes)
    if args.mlflow:
        _log_mlflow_optional(args, metrics, run_dir)

    _write_metadata(
        args,
        config,
        dependency_report.to_dict(),
        artifacts,
        pairs,
        run_id,
        "completed",
        metadata_notes,
    )
    print(f"FreqAI backtest artifacts written: {run_dir}")
    return 0


def _build_freqai_backtest_command(
    args: argparse.Namespace,
    run_dir: Path,
    result_filename: str,
    config_paths: list[Path],
) -> list[str]:
    cmd = [
        args.python,
        "-m",
        "freqtrade_ext.bot_factory.freqtrade_cli",
        "backtesting",
    ]
    for config_path in config_paths:
        cmd.extend(["-c", str(config_path)])
    cmd.extend(
        [
            "--strategy",
            args.strategy,
            "--strategy-path",
            args.strategy_path,
            "--export",
            "trades",
            "--backtest-directory",
            str(run_dir),
            "--backtest-filename",
            result_filename,
            "--cache",
            "none",
            "--data-format-ohlcv",
            args.data_format_ohlcv,
        ]
    )
    if args.freqaimodel:
        cmd.extend(["--freqaimodel", args.freqaimodel])
    if args.freqaimodel_path:
        cmd.extend(["--freqaimodel-path", args.freqaimodel_path])
    if args.timeframe:
        cmd.extend(["--timeframe", args.timeframe])
    if args.timerange:
        cmd.extend(["--timerange", args.timerange])
    if args.pairs:
        cmd.extend(["--pairs", *args.pairs])
    if args.userdir:
        cmd.extend(["--userdir", args.userdir])
    if args.datadir:
        cmd.extend(["--datadir", args.datadir])
    return cmd


def _runtime_config_paths(
    args: argparse.Namespace, artifacts: dict[str, Path | None]
) -> list[Path]:
    config_paths = [Path(args.config)]
    override = artifacts.get("freqai_identifier_override_config")
    if args.freqai_identifier and override is not None:
        config_paths.append(override)
    return config_paths


def _run_ohlcv_quality_checks(
    args: argparse.Namespace, config_path: Path, config: dict, run_dir: Path
) -> bool:
    if args.ohlcv_file:
        paths = [Path(path) for path in args.ohlcv_file]
        timeframes = [args.timeframe or str(config.get("timeframe") or "")]
    else:
        pairs = freqai_input_pairs(config, args.pairs)
        timeframes = freqai_input_timeframes(config, args.timeframe)
        paths = resolve_ohlcv_input_paths(
            config_path=config_path,
            config=config,
            userdir=Path(args.userdir or "user_data"),
            pairs=pairs,
            timeframes=timeframes,
            trading_mode=args.trading_mode,
            datadir=Path(args.datadir) if args.datadir else None,
        )

    if not paths:
        raise SystemExit("No OHLCV input paths could be resolved for quality checks.")

    expected_timeframe = timeframes[0] if len(set(timeframes)) == 1 else None
    reports = [check_ohlcv_parquet(path, expected_timeframe) for path in paths]
    output = run_dir / "ohlcv_quality.json"
    write_quality_reports(reports, output)
    for quality_report in reports:
        print(quality_report.to_json())
    return all(report.ok for report in reports)


def _write_metadata(
    args: argparse.Namespace,
    config: dict,
    dependency_status: dict,
    artifacts: dict[str, Path | None],
    pairs: list[str],
    run_id: str,
    status: str,
    notes: list[str],
) -> None:
    config_paths = _runtime_config_paths(args, artifacts)
    metadata = build_freqai_metadata(
        root_dir=ROOT_DIR,
        strategy=args.strategy,
        run_id=run_id,
        status=status,
        config_paths=config_paths,
        freqaimodel=freqai_model_name(config, args.freqaimodel),
        freqai_id=args.freqai_identifier or freqai_identifier(config),
        timeframe=args.timeframe or str(config.get("timeframe") or ""),
        timerange=args.timerange,
        pairs=pairs,
        dependency_status=dependency_status,
        artifact_paths=artifacts,
        notes=notes,
        freqai_identifier_source="override" if args.freqai_identifier else "config",
    )
    write_freqai_metadata(metadata, artifacts["freqai_metadata"])


def _find_result_json(run_dir: Path, expected_name: str) -> Path:
    expected = run_dir / expected_name
    if expected.exists():
        return expected

    candidates = [
        p
        for p in run_dir.glob("*.json")
        if not p.name.endswith(".meta.json")
        and p.name
        not in {
            "metrics.json",
            "static_check.json",
            "freqai_validation.json",
            "freqai_env.json",
            "ohlcv_quality.json",
        }
    ]
    if not candidates:
        raise SystemExit(f"No backtest result JSON found in {run_dir}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _find_strategy_source(strategy_path: Path, strategy_name: str) -> Path:
    exact = strategy_path / f"{strategy_name}.py"
    if exact.exists():
        return exact
    if strategy_path.is_file():
        return strategy_path
    if not strategy_path.is_dir():
        return strategy_path

    for file_path in sorted(strategy_path.rglob("*.py")):
        try:
            tree = ast.parse(file_path.read_text(encoding="utf-8"), filename=str(file_path))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == strategy_name:
                return file_path
    return strategy_path


def _require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise SystemExit(f"{label} file not found: {path}")


def _log_mlflow_optional(args: argparse.Namespace, metrics: BacktestMetrics, run_dir: Path) -> None:
    try:
        result = log_backtest_to_mlflow(
            metrics,
            run_dir,
            tracking_uri=args.mlflow_tracking_uri,
            experiment_name=args.mlflow_experiment,
        )
    except Exception as exc:
        error_path = run_dir / "mlflow_error.txt"
        error_path.write_text(str(exc), encoding="utf-8")
        print(f"MLflow logging skipped after error. Details: {error_path}")
        return

    (run_dir / "mlflow_run.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"MLflow run logged: {result['run_id']}")


if __name__ == "__main__":
    sys.exit(main())
