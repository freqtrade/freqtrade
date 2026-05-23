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
from freqtrade_ext.bot_factory.candidate_identity import (
    build_strategy_candidate_identity,
    extract_candidate_identity,
    load_candidate_identity_from_strategy_source,
    validate_candidate_identity,
)
from freqtrade_ext.bot_factory.data_quality import check_ohlcv_parquet, write_quality_reports
from freqtrade_ext.bot_factory.mlflow_tracking import log_backtest_to_mlflow
from freqtrade_ext.bot_factory.safety import scan_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a checked Bot Factory backtest.")
    parser.add_argument("--config", default="user_data/config.json")
    parser.add_argument("--strategy", required=True)
    parser.add_argument("--strategy-path", default="user_data/strategies")
    parser.add_argument("--timeframe", default=None)
    parser.add_argument("--timerange", default=None)
    parser.add_argument("--pairs", nargs="*", default=None)
    parser.add_argument("--output-root", default="data/backtests")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--candidate-id", default=None)
    parser.add_argument(
        "--candidate-identity-json",
        default=None,
        help="Optional full StrategyCandidateIdentity JSON to embed in this checked backtest.",
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--skip-static-check", action="store_true")
    parser.add_argument("--data-format-ohlcv", default="parquet")
    parser.add_argument(
        "--ohlcv-file",
        action="append",
        default=None,
        help="Explicit OHLCV parquet file to quality-check before backtesting. Can be repeated.",
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
    parser.add_argument("--mlflow-experiment", default="bot_factory_backtests")
    parser.add_argument(
        "--enable-freqai",
        action="store_true",
        help="Use FreqAI settings from config. Disabled by default for Phase 1 backtests.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _require_file(args.config, "config")

    strategy_file = _find_strategy_source(Path(args.strategy_path), args.strategy)

    run_id = args.run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(args.output_root) / args.strategy / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    candidate_identity = _resolve_candidate_identity(args, strategy_file, run_id)
    identity_validation = validate_candidate_identity(candidate_identity)
    (run_dir / "candidate_identity.json").write_text(
        json.dumps(candidate_identity, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if not identity_validation["ok"]:
        print(json.dumps(identity_validation, indent=2, ensure_ascii=False))
        print(f"Candidate identity validation failed. Report: {run_dir / 'candidate_identity.json'}")
        return 1

    if not args.skip_static_check:
        report = scan_paths([strategy_file])
        static_report = run_dir / "static_check.json"
        static_report.write_text(report.to_json(), encoding="utf-8")
        if not report.ok:
            print(report.to_json())
            print(f"Static check failed. Report: {static_report}")
            return 1

    if not _run_ohlcv_quality_checks(args, run_dir):
        print(f"OHLCV quality check failed. Report: {run_dir / 'ohlcv_quality.json'}")
        return 1

    config_args = ["-c", args.config]
    if not args.enable_freqai:
        overlay = run_dir / "freqai_disabled_config.json"
        overlay.write_text(json.dumps({"freqai": {"enabled": False}}, indent=2), encoding="utf-8")
        config_args.extend(["-c", str(overlay)])

    result_filename = "result.json"
    cmd = [
        args.python,
        "-m",
        "freqtrade_ext.bot_factory.freqtrade_cli",
        "backtesting",
        *config_args,
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
    if args.timeframe:
        cmd.extend(["--timeframe", args.timeframe])
    if args.timerange:
        cmd.extend(["--timerange", args.timerange])
    if args.pairs:
        cmd.extend(["--pairs", *args.pairs])

    command_log = run_dir / "command.txt"
    command_log.write_text(" ".join(cmd), encoding="utf-8")

    print("Running:", " ".join(cmd))
    completed = subprocess.run(cmd, text=True, capture_output=True)
    (run_dir / "stdout.log").write_text(completed.stdout or "", encoding="utf-8")
    (run_dir / "stderr.log").write_text(completed.stderr or "", encoding="utf-8")
    if completed.returncode != 0:
        print(completed.stdout)
        print(completed.stderr, file=sys.stderr)
        print(f"Backtest failed. Logs: {run_dir}")
        return int(completed.returncode)

    result_path = _find_result_json(run_dir, result_filename)
    if result_path.name != result_filename:
        shutil.copy2(result_path, run_dir / result_filename)
        result_path = run_dir / result_filename

    result = load_backtest_result(result_path)
    result["candidate_identity"] = candidate_identity
    write_result_json(result, run_dir / result_filename)
    metrics = summarize(result, args.strategy)
    metrics.candidate_identity = candidate_identity
    write_metrics(metrics, run_dir / "metrics.json")
    write_trades_csv(result, run_dir / "trades.csv", args.strategy)
    thresholds = load_gate_thresholds(Path(args.gate_config) if args.gate_config else None)
    write_report(metrics, run_dir / "report.md", thresholds, args.reviewer_note)
    if args.mlflow:
        _log_mlflow_optional(args, metrics, run_dir)

    print(f"Backtest artifacts written: {run_dir}")
    return 0


def _find_result_json(run_dir: Path, expected_name: str) -> Path:
    expected = run_dir / expected_name
    if expected.exists():
        return expected

    candidates = [
        p
        for p in run_dir.glob("*.json")
        if not p.name.endswith(".meta.json")
        and p.name not in {"metrics.json", "static_check.json", "candidate_identity.json"}
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


def _require_file(path: str, label: str) -> None:
    if not Path(path).is_file():
        raise SystemExit(f"{label} file not found: {path}")


def _resolve_candidate_identity(
    args: argparse.Namespace, strategy_file: Path, run_id: str
) -> dict[str, object]:
    provided_identity = _candidate_identity_from_json(args)
    if provided_identity is not None:
        return provided_identity

    strategy_identity = load_candidate_identity_from_strategy_source(
        strategy_file,
        strategy_class_name=args.strategy,
        root_dir=ROOT_DIR,
    )
    if strategy_identity:
        if args.candidate_id and args.candidate_id != strategy_identity.get("candidate_id"):
            raise SystemExit(
                "Provided --candidate-id does not match strategy candidate identity: "
                f"{args.candidate_id} != {strategy_identity.get('candidate_id')}"
            )
        return strategy_identity

    config = _load_json_if_possible(Path(args.config))
    pair_whitelist = (
        config.get("exchange", {}).get("pair_whitelist", []) if isinstance(config, dict) else []
    )
    allowed_pairs = list(args.pairs or pair_whitelist or [])
    allowed_timeframes = [args.timeframe or str(config.get("timeframe") or "")] if isinstance(config, dict) else []
    return build_strategy_candidate_identity(
        candidate_id=args.candidate_id or run_id,
        strategy_id=args.strategy,
        strategy_class_name=args.strategy,
        strategy_source_path=strategy_file,
        strategy_version=f"{args.strategy}_v1",
        signal_version="unspecified_signal_v1",
        risk_policy_version="unspecified_risk_policy_v1",
        regime_classifier_version="unspecified_regime_classifier_v1",
        cost_model_id="unspecified_cost_model_v1",
        allowed_pairs=allowed_pairs,
        allowed_timeframes=[item for item in allowed_timeframes if item],
        created_at=datetime.now(UTC).replace(microsecond=0).isoformat(),
        source_artifacts={"strategy_source": strategy_file},
        root_dir=ROOT_DIR,
    )


def _candidate_identity_from_json(args: argparse.Namespace) -> dict[str, object] | None:
    path_value = getattr(args, "candidate_identity_json", None)
    if not path_value:
        return None
    path = Path(path_value)
    if not path.is_file():
        raise SystemExit(f"candidate identity JSON file not found: {path}")
    identity = extract_candidate_identity(_load_json_if_possible(path))
    if identity is None:
        raise SystemExit(f"candidate identity JSON does not contain a valid identity: {path}")
    if args.candidate_id and args.candidate_id != identity.get("candidate_id"):
        raise SystemExit(
            "Provided --candidate-id does not match candidate identity JSON: "
            f"{args.candidate_id} != {identity.get('candidate_id')}"
        )
    if identity.get("strategy_class_name") != args.strategy:
        raise SystemExit(
            "Candidate identity JSON strategy_class_name does not match --strategy: "
            f"{identity.get('strategy_class_name')} != {args.strategy}"
        )
    return identity


def _load_json_if_possible(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _run_ohlcv_quality_checks(args: argparse.Namespace, run_dir: Path) -> bool:
    if not args.ohlcv_file:
        return True

    reports = [check_ohlcv_parquet(Path(path), args.timeframe) for path in args.ohlcv_file]
    output = run_dir / "ohlcv_quality.json"
    write_quality_reports(reports, output)
    for quality_report in reports:
        print(quality_report.to_json())
    return all(report.ok for report in reports)


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
