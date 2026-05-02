#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
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
    write_trades_csv,
)
from freqtrade_ext.bot_factory.mlflow_tracking import log_backtest_to_mlflow


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Bot Factory metrics and report.")
    parser.add_argument("result_json", help="Freqtrade backtest result JSON.")
    parser.add_argument("--strategy", default=None, help="Strategy name when JSON has many strategies.")
    parser.add_argument("--outdir", default=None, help="Output directory.")
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
        help="Optionally log metrics and artifacts to MLflow. Failures do not fail report generation.",
    )
    parser.add_argument("--mlflow-tracking-uri", default=None)
    parser.add_argument("--mlflow-experiment", default="bot_factory_backtests")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result_path = Path(args.result_json)
    outdir = Path(args.outdir) if args.outdir else result_path.parent
    result = load_backtest_result(result_path)
    metrics = summarize(result, args.strategy)

    write_metrics(metrics, outdir / "metrics.json")
    write_trades_csv(result, outdir / "trades.csv", args.strategy)
    thresholds = load_gate_thresholds(Path(args.gate_config) if args.gate_config else None)
    write_report(metrics, outdir / "report.md", thresholds, args.reviewer_note)
    if args.mlflow:
        _log_mlflow_optional(args, metrics, outdir)

    print(f"Metrics written: {outdir / 'metrics.json'}")
    print(f"Trades written: {outdir / 'trades.csv'}")
    print(f"Report written: {outdir / 'report.md'}")
    return 0


def _log_mlflow_optional(args: argparse.Namespace, metrics: BacktestMetrics, outdir: Path) -> None:
    try:
        result = log_backtest_to_mlflow(
            metrics,
            outdir,
            tracking_uri=args.mlflow_tracking_uri,
            experiment_name=args.mlflow_experiment,
        )
    except Exception as exc:
        error_path = outdir / "mlflow_error.txt"
        error_path.write_text(str(exc), encoding="utf-8")
        print(f"MLflow logging skipped after error. Details: {error_path}")
        return

    (outdir / "mlflow_run.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"MLflow run logged: {result['run_id']}")


if __name__ == "__main__":
    sys.exit(main())
