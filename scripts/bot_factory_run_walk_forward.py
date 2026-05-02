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
from freqtrade_ext.bot_factory.walk_forward import (
    WalkForwardRules,
    WalkForwardWindow,
    aggregate_walk_forward_results,
    generate_rolling_windows,
    parse_window_specs,
    window_run_id,
    write_walk_forward_metrics,
    write_walk_forward_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Bot Factory walk-forward evaluation through historical FreqAI "
            "backtests only."
        )
    )
    parser.add_argument("--config", default="user_data/config.json")
    parser.add_argument("--strategy", required=True)
    parser.add_argument("--strategy-path", default="user_data/strategies")
    parser.add_argument("--freqaimodel", default=None)
    parser.add_argument("--freqaimodel-path", default=None)
    parser.add_argument("--timeframe", default=None)
    parser.add_argument("--pairs", nargs="*", default=None)
    parser.add_argument("--output-root", default="data/walk_forward")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--runner-script", default="scripts/bot_factory_run_freqai_backtest.py")
    parser.add_argument("--data-format-ohlcv", default="parquet")
    parser.add_argument("--userdir", default=None)
    parser.add_argument("--datadir", default=None)
    parser.add_argument("--trading-mode", default=None)
    parser.add_argument(
        "--ohlcv-file",
        action="append",
        default=None,
        help="Explicit OHLCV parquet file to quality-check for each window. Can be repeated.",
    )
    parser.add_argument(
        "--gate-config",
        default=None,
        help="Optional JSON file overriding per-window backtest gate thresholds.",
    )
    parser.add_argument(
        "--reviewer-note",
        action="append",
        default=None,
        help="Optional note to include in each generated window report.",
    )
    parser.add_argument(
        "--mlflow",
        action="store_true",
        help="Optionally pass MLflow logging through to each window backtest.",
    )
    parser.add_argument("--mlflow-tracking-uri", default=None)
    parser.add_argument("--mlflow-experiment", default="bot_factory_freqai_walk_forward")
    parser.add_argument(
        "--window",
        action="append",
        default=None,
        help=(
            "Fixed historical window. Use START-END or "
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
    _require_file(Path(args.config), "config")
    _require_file(Path(args.runner_script), "runner script")
    windows = _resolve_windows(args)
    if not windows:
        raise SystemExit("No walk-forward windows were generated.")

    run_id = args.run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(args.output_root) / args.strategy / run_id
    windows_root = run_dir / "windows"
    logs_dir = run_dir / "window_logs"
    run_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    rules = WalkForwardRules(
        min_pass_rate=args.min_pass_rate,
        min_profitable_windows_ratio=args.min_profitable_windows_ratio,
        max_drawdown_pct_any_window=args.max_drawdown_pct_any_window,
        max_single_window_profit_dependency=args.max_single_window_profit_dependency,
    )
    thresholds = load_gate_thresholds(Path(args.gate_config) if args.gate_config else None)

    window_results: list[dict[str, Any]] = []
    command_lines: list[str] = []
    for window in windows:
        child_run_id = window_run_id("wf", window)
        cmd = _build_window_command(args, window, child_run_id, windows_root)
        command_lines.append(" ".join(cmd))
        result = _run_window(args, window, child_run_id, windows_root, logs_dir, cmd, thresholds)
        window_results.append(result)

    (run_dir / "command.txt").write_text("\n".join(command_lines), encoding="utf-8")
    metrics = aggregate_walk_forward_results(window_results, rules)
    metrics["strategy"] = args.strategy
    metrics["run_id"] = run_id
    metrics["config_path"] = args.config
    metrics["window_specs"] = [window.to_dict() for window in windows]
    metrics["artifacts"] = {
        "walk_forward_metrics": str(run_dir / "walk_forward_metrics.json"),
        "walk_forward_report": str(run_dir / "walk_forward_report.md"),
        "command": str(run_dir / "command.txt"),
    }

    write_walk_forward_metrics(metrics, run_dir / "walk_forward_metrics.json")
    write_walk_forward_report(metrics, run_dir / "walk_forward_report.md")
    print(f"Walk-forward artifacts written: {run_dir}")

    failed_windows = metrics["summary"]["failed_windows"]
    return 1 if failed_windows else 0


def _resolve_windows(args: argparse.Namespace) -> list[WalkForwardWindow]:
    if args.window:
        return parse_window_specs(args.window)

    required = {
        "--start": args.start,
        "--end": args.end,
        "--train-days": args.train_days,
        "--test-days": args.test_days,
        "--step-days": args.step_days,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise SystemExit(
            "Provide either at least one --window or all rolling-window arguments: "
            + ", ".join(missing)
        )
    return generate_rolling_windows(
        start=args.start,
        end=args.end,
        train_days=args.train_days,
        test_days=args.test_days,
        step_days=args.step_days,
    )


def _build_window_command(
    args: argparse.Namespace,
    window: WalkForwardWindow,
    child_run_id: str,
    windows_root: Path,
) -> list[str]:
    cmd = [
        args.python,
        args.runner_script,
        "--config",
        args.config,
        "--strategy",
        args.strategy,
        "--strategy-path",
        args.strategy_path,
        "--timerange",
        window.timerange,
        "--output-root",
        str(windows_root),
        "--run-id",
        child_run_id,
        "--python",
        args.python,
        "--data-format-ohlcv",
        args.data_format_ohlcv,
        "--reviewer-note",
        "Walk-forward historical FreqAI verification only; no paper or live promotion.",
    ]
    for note in args.reviewer_note or []:
        cmd.extend(["--reviewer-note", note])
    if args.freqaimodel:
        cmd.extend(["--freqaimodel", args.freqaimodel])
    if args.freqaimodel_path:
        cmd.extend(["--freqaimodel-path", args.freqaimodel_path])
    if args.timeframe:
        cmd.extend(["--timeframe", args.timeframe])
    if args.pairs:
        cmd.extend(["--pairs", *args.pairs])
    if args.userdir:
        cmd.extend(["--userdir", args.userdir])
    if args.datadir:
        cmd.extend(["--datadir", args.datadir])
    if args.trading_mode:
        cmd.extend(["--trading-mode", args.trading_mode])
    for path in args.ohlcv_file or []:
        cmd.extend(["--ohlcv-file", path])
    if args.gate_config:
        cmd.extend(["--gate-config", args.gate_config])
    if args.mlflow:
        cmd.append("--mlflow")
    if args.mlflow_tracking_uri:
        cmd.extend(["--mlflow-tracking-uri", args.mlflow_tracking_uri])
    if args.mlflow_experiment:
        cmd.extend(["--mlflow-experiment", args.mlflow_experiment])
    return cmd


def _run_window(
    args: argparse.Namespace,
    window: WalkForwardWindow,
    child_run_id: str,
    windows_root: Path,
    logs_dir: Path,
    cmd: list[str],
    thresholds: Any,
) -> dict[str, Any]:
    print(f"Running walk-forward window {window.index}: {window.timerange}")
    completed = subprocess.run(cmd, text=True, capture_output=True)
    log_prefix = logs_dir / child_run_id
    (log_prefix.with_suffix(".command.txt")).write_text(" ".join(cmd), encoding="utf-8")
    (log_prefix.with_suffix(".stdout.log")).write_text(completed.stdout or "", encoding="utf-8")
    (log_prefix.with_suffix(".stderr.log")).write_text(completed.stderr or "", encoding="utf-8")

    child_dir = windows_root / args.strategy / child_run_id
    result: dict[str, Any] = {
        "window": window.to_dict(),
        "run_id": child_run_id,
        "run_dir": str(child_dir),
        "returncode": completed.returncode,
        "status": "failed",
    }

    metrics_path = child_dir / "metrics.json"
    if completed.returncode == 0 and metrics_path.is_file():
        metrics_payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        gate = evaluate_initial_gate(BacktestMetrics(**metrics_payload), thresholds)
        result.update(
            {
                "status": "completed",
                "metrics": metrics_payload,
                "gate_recommendation": gate["recommendation"],
                "gate_checks": gate["checks"],
                "artifacts": {
                    "metrics": str(metrics_path),
                    "report": str(child_dir / "report.md"),
                    "freqai_metadata": str(child_dir / "freqai_metadata.json"),
                    "freqai_validation": str(child_dir / "freqai_validation.json"),
                },
            }
        )
    else:
        result["error"] = _window_error(completed, metrics_path)

    return result


def _window_error(completed: subprocess.CompletedProcess[str], metrics_path: Path) -> str:
    if completed.returncode == 0:
        return f"Window command completed but metrics file was not found: {metrics_path}"
    stderr = (completed.stderr or "").strip()
    if stderr:
        return stderr.splitlines()[-1]
    stdout = (completed.stdout or "").strip()
    return stdout.splitlines()[-1] if stdout else f"Window command failed: {completed.returncode}"


def _require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise SystemExit(f"{label} file not found: {path}")


if __name__ == "__main__":
    sys.exit(main())
