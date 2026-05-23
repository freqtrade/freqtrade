#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
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
from freqtrade_ext.bot_factory.candidate_identity import (
    build_strategy_candidate_identity,
    extract_candidate_identity,
    load_candidate_identity_from_strategy_source,
    validate_artifact_candidate_identity,
    validate_candidate_identity,
)
from freqtrade_ext.bot_factory.freqai_backtest import sanitize_freqai_identifier, selected_pairs
from freqtrade_ext.bot_factory.freqai_backtest import freqai_input_timeframes
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
    parser.add_argument(
        "--freqai-identifier",
        default=None,
        help="Candidate-specific FreqAI identifier passed to FreqAI child wrappers.",
    )
    parser.add_argument("--timeframe", default=None)
    parser.add_argument("--pairs", nargs="*", default=None)
    parser.add_argument("--output-root", default="data/walk_forward")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--candidate-id", default=None)
    parser.add_argument(
        "--candidate-identity-json",
        default=None,
        help="Optional full StrategyCandidateIdentity JSON to embed in this checked walk-forward run.",
    )
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
    if args.freqai_identifier:
        args.freqai_identifier = sanitize_freqai_identifier(args.freqai_identifier)
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
    config = _load_json_if_possible(Path(args.config))
    strategy_file = _find_strategy_source(Path(args.strategy_path), args.strategy)
    candidate_identity = _resolve_candidate_identity(args, config, strategy_file, run_id)
    args.candidate_identity = candidate_identity
    identity_validation = validate_candidate_identity(candidate_identity)
    (run_dir / "candidate_identity.json").write_text(
        json.dumps(candidate_identity, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    args.candidate_identity_path = run_dir / "candidate_identity.json"
    if not identity_validation["ok"]:
        print(json.dumps(identity_validation, indent=2, ensure_ascii=False))
        print(f"Candidate identity validation failed. Report: {run_dir / 'candidate_identity.json'}")
        return 1

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
    metrics = aggregate_walk_forward_results(
        window_results,
        rules,
        candidate_identity=candidate_identity,
    )
    metrics["strategy"] = args.strategy
    metrics["run_id"] = run_id
    metrics["config_path"] = args.config
    metrics["window_specs"] = [window.to_dict() for window in windows]
    metrics["artifacts"] = {
        "walk_forward_metrics": str(run_dir / "walk_forward_metrics.json"),
        "walk_forward_report": str(run_dir / "walk_forward_report.md"),
        "command": str(run_dir / "command.txt"),
        "candidate_identity": str(run_dir / "candidate_identity.json"),
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
    if getattr(args, "candidate_identity", None):
        cmd.extend(["--candidate-id", str(args.candidate_identity["candidate_id"])])
    if getattr(args, "candidate_identity_path", None):
        cmd.extend(["--candidate-identity-json", str(args.candidate_identity_path)])
    for note in args.reviewer_note or []:
        cmd.extend(["--reviewer-note", note])
    if args.freqaimodel:
        cmd.extend(["--freqaimodel", args.freqaimodel])
    if args.freqaimodel_path:
        cmd.extend(["--freqaimodel-path", args.freqaimodel_path])
    if args.freqai_identifier:
        cmd.extend(["--freqai-identifier", args.freqai_identifier])
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
        identity_validation = validate_artifact_candidate_identity(
            args.candidate_identity,
            metrics_payload,
            artifact_label=f"walk_forward_window_{window.index:02d}_metrics",
        )
        if not identity_validation["ok"]:
            result.update(
                {
                    "status": "failed_identity_mismatch",
                    "metrics": metrics_payload,
                    "candidate_identity": args.candidate_identity,
                    "identity_validation": identity_validation,
                    "error": "window_candidate_identity_mismatch",
                }
            )
            return result
        gate = evaluate_initial_gate(BacktestMetrics(**metrics_payload), thresholds)
        result.update(
            {
                "status": "completed",
                "metrics": metrics_payload,
                "candidate_identity": args.candidate_identity,
                "identity_validation": identity_validation,
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
        except (OSError, SyntaxError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == strategy_name:
                return file_path
    return strategy_path


def _resolve_candidate_identity(
    args: argparse.Namespace,
    config: dict[str, Any],
    strategy_file: Path,
    run_id: str,
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

    return build_strategy_candidate_identity(
        candidate_id=args.candidate_id or run_id,
        strategy_id=args.strategy,
        strategy_class_name=args.strategy,
        strategy_source_path=strategy_file,
        strategy_version=f"{args.strategy}_v1",
        signal_version=_default_signal_version(args),
        risk_policy_version="unspecified_risk_policy_v1",
        regime_classifier_version="unspecified_regime_classifier_v1",
        cost_model_id="unspecified_cost_model_v1",
        allowed_pairs=selected_pairs(config, args.pairs),
        allowed_timeframes=_identity_timeframes(args, config),
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
        raise SystemExit(f"Candidate identity JSON not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    identity = extract_candidate_identity(payload)
    if identity is None:
        raise SystemExit(f"Candidate identity JSON is invalid: {path}")
    if args.candidate_id and args.candidate_id != identity.get("candidate_id"):
        raise SystemExit(
            "Provided --candidate-id does not match candidate identity JSON: "
            f"{args.candidate_id} != {identity.get('candidate_id')}"
        )
    if identity.get("strategy_class_name") and identity.get("strategy_class_name") != args.strategy:
        raise SystemExit(
            "Candidate identity strategy_class_name does not match --strategy: "
            f"{identity.get('strategy_class_name')} != {args.strategy}"
        )
    return identity


def _default_signal_version(args: argparse.Namespace) -> str:
    if args.freqaimodel or args.freqaimodel_path or args.freqai_identifier:
        return "unspecified_freqai_signal_v1"
    if "freqai" in str(args.runner_script).lower():
        return "unspecified_freqai_signal_v1"
    return "unspecified_signal_v1"


def _identity_timeframes(args: argparse.Namespace, config: dict[str, Any]) -> list[str]:
    if _uses_freqai_child(args):
        return freqai_input_timeframes(config, args.timeframe)
    timeframe = args.timeframe or str(config.get("timeframe") or "")
    return [timeframe] if timeframe else []


def _uses_freqai_child(args: argparse.Namespace) -> bool:
    return (
        bool(args.freqaimodel)
        or bool(args.freqaimodel_path)
        or bool(args.freqai_identifier)
        or "freqai" in str(args.runner_script).lower()
    )


def _load_json_if_possible(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


if __name__ == "__main__":
    sys.exit(main())
