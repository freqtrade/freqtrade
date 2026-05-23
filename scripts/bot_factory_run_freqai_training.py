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
from freqtrade_ext.bot_factory.candidate_identity import (
    build_strategy_candidate_identity,
    extract_candidate_identity,
    load_candidate_identity_from_strategy_source,
    validate_candidate_identity,
)
from freqtrade_ext.bot_factory.freqai_backtest import (
    freqai_enabled,
    freqai_identifier,
    freqai_model_name,
    load_json_config,
    sanitize_freqai_identifier,
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
    parser.add_argument(
        "--freqai-identifier",
        default=None,
        help="Candidate-specific FreqAI identifier passed to checked child wrappers.",
    )
    parser.add_argument("--timeframe", default=None)
    parser.add_argument("--timerange", required=True)
    parser.add_argument("--pairs", nargs="*", default=None)
    parser.add_argument("--output-root", default="data/freqai_training")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--candidate-id", default=None)
    parser.add_argument(
        "--candidate-identity-json",
        default=None,
        help="Optional full StrategyCandidateIdentity JSON to embed in this checked FreqAI training run.",
    )
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
    if args.freqai_identifier:
        args.freqai_identifier = sanitize_freqai_identifier(args.freqai_identifier)
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
        "candidate_identity": run_dir / "candidate_identity.json",
        "logs": logs_dir,
    }
    candidate_identity = _resolve_candidate_identity(args, config, run_id)
    args.candidate_identity = candidate_identity
    identity_validation = validate_candidate_identity(candidate_identity)
    artifact_paths["candidate_identity"].write_text(
        json.dumps(candidate_identity, indent=2, ensure_ascii=False), encoding="utf-8"
    )
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
    if not identity_validation["ok"]:
        manifest = _build_manifest(args, config, {}, [], artifact_paths, notes)
        manifest["status"] = "blocked_candidate_identity"
        manifest["recommendation"] = "fail"
        write_training_manifest(manifest, artifact_paths["training_manifest"])
        write_training_report(manifest, artifact_paths["training_report"])
        print(json.dumps(identity_validation, indent=2, ensure_ascii=False))
        print(f"Candidate identity validation failed. Report: {artifact_paths['candidate_identity']}")
        return 1

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
        freqai_identifier=args.freqai_identifier,
        candidate_id=str(candidate_identity["candidate_id"]),
        candidate_identity_json=artifact_paths["candidate_identity"],
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
            freqai_identifier=args.freqai_identifier,
            candidate_id=str(candidate_identity["candidate_id"]),
            candidate_identity_json=artifact_paths["candidate_identity"],
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


def _resolve_candidate_identity(
    args: argparse.Namespace,
    config: dict[str, Any],
    run_id: str,
) -> dict[str, object]:
    provided_identity = _candidate_identity_from_json(args)
    if provided_identity is not None:
        return provided_identity
    strategy_source = _strategy_source_candidate(args)
    strategy_identity = load_candidate_identity_from_strategy_source(
        strategy_source,
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
        strategy_source_path=strategy_source,
        strategy_version=f"{args.strategy}_v1",
        signal_version="unspecified_freqai_signal_v1",
        risk_policy_version="unspecified_risk_policy_v1",
        regime_classifier_version="unspecified_regime_classifier_v1",
        cost_model_id="unspecified_cost_model_v1",
        allowed_pairs=selected_pairs(config, args.pairs),
        allowed_timeframes=[args.timeframe or str(config.get("timeframe") or "")],
        created_at=datetime.now(UTC).replace(microsecond=0).isoformat(),
        source_artifacts={"strategy_source": strategy_source},
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


def _strategy_source_candidate(args: argparse.Namespace) -> Path:
    strategy_path = Path(args.strategy_path)
    if strategy_path.is_file():
        return strategy_path
    return strategy_path / f"{args.strategy}.py"


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
        freqai_identifier=args.freqai_identifier or freqai_identifier(config),
        dependency_status=dependency_status,
        stages=stages,
        artifact_paths=artifact_paths,
        notes=notes,
        candidate_identity=getattr(args, "candidate_identity", None),
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
