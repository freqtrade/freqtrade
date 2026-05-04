#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from freqtrade_ext.bot_factory.paper_drift import (
    PaperDriftReportInputs,
    build_paper_drift_report,
    load_paper_drift_artifact,
    path_from_payload,
    write_paper_drift_report_artifacts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Write a Bot Factory Phase 3 paper/backtest drift report from local "
            "artifacts only, without starting, stopping, polling, terminating, "
            "cleaning up, promoting, or managing any bot process."
        )
    )
    parser.add_argument("--historical-metrics-json", required=True)
    parser.add_argument("--walk-forward-metrics-json", required=True)
    parser.add_argument("--training-manifest-json", required=True)
    parser.add_argument("--paper-runtime-validation-json", required=True)
    parser.add_argument(
        "--paper-metrics-json",
        default=None,
        help=(
            "Optional paper metrics JSON. If omitted, the path is resolved from "
            "paper_runtime_validation.input_paths.paper_metrics."
        ),
    )
    parser.add_argument("--strategy", required=True)
    parser.add_argument("--output-root", default="data/paper")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--max-return-drift-pct", type=float, default=5.0)
    parser.add_argument("--max-drawdown-drift-pct", type=float, default=5.0)
    parser.add_argument(
        "--reviewer-note",
        action="append",
        default=None,
        help="Reviewer note required before a drift report can pass. Can be repeated.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    historical_metrics_path = Path(args.historical_metrics_json)
    walk_forward_metrics_path = Path(args.walk_forward_metrics_json)
    training_manifest_path = Path(args.training_manifest_json)
    paper_runtime_validation_path = Path(args.paper_runtime_validation_json)

    for path, label in [
        (historical_metrics_path, "historical metrics JSON"),
        (walk_forward_metrics_path, "walk-forward metrics JSON"),
        (training_manifest_path, "training manifest JSON"),
        (paper_runtime_validation_path, "paper runtime validation JSON"),
    ]:
        _require_file(path, label)

    paper_runtime_validation = load_paper_drift_artifact(paper_runtime_validation_path)
    paper_metrics_path = _resolve_paper_metrics_path(args.paper_metrics_json, paper_runtime_validation)

    run_id = args.run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    inputs = PaperDriftReportInputs(
        root_dir=ROOT_DIR,
        strategy=args.strategy,
        run_id=run_id,
        historical_metrics_path=historical_metrics_path,
        walk_forward_metrics_path=walk_forward_metrics_path,
        training_manifest_path=training_manifest_path,
        paper_runtime_validation_path=paper_runtime_validation_path,
        paper_metrics_path=paper_metrics_path,
        output_root=Path(args.output_root),
        max_return_drift_pct=args.max_return_drift_pct,
        max_drawdown_drift_pct=args.max_drawdown_drift_pct,
        reviewer_notes=list(args.reviewer_note or []),
        command=[sys.executable, *sys.argv],
    )
    report = build_paper_drift_report(
        inputs,
        load_paper_drift_artifact(historical_metrics_path),
        load_paper_drift_artifact(walk_forward_metrics_path),
        load_paper_drift_artifact(training_manifest_path),
        paper_runtime_validation,
        _load_optional_artifact(paper_metrics_path),
    )
    write_paper_drift_report_artifacts(inputs, report)

    print(
        json.dumps(
            {"status": report["status"], "output_dir": str(inputs.output_dir)},
            indent=2,
        )
    )
    print(f"Paper/backtest drift report artifacts written: {inputs.output_dir}")
    return 0 if report["status"] == "pass" else 1


def _resolve_paper_metrics_path(
    explicit_path: str | None, runtime_validation: dict
) -> Path | None:
    if explicit_path:
        return Path(explicit_path)
    input_paths = runtime_validation.get("input_paths")
    if not isinstance(input_paths, dict):
        return None
    return path_from_payload(input_paths.get("paper_metrics"), ROOT_DIR)


def _require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise SystemExit(f"{label} file not found: {path}")


def _load_optional_artifact(path: Path | None) -> dict:
    if path is None or not path.is_file():
        return {}
    return load_paper_drift_artifact(path)


if __name__ == "__main__":
    sys.exit(main())
