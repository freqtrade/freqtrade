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

from freqtrade_ext.bot_factory.paper_runtime import (
    PaperRuntimeValidationInputs,
    build_paper_runtime_validation,
    load_paper_runtime_artifact,
    write_paper_runtime_validation_artifacts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate Bot Factory Phase 3 paper runtime artifacts without "
            "starting, stopping, polling, terminating, cleaning up, or "
            "managing any bot process."
        )
    )
    parser.add_argument("--process-executor-plan-json", required=True)
    parser.add_argument("--process-metadata-json", required=True)
    parser.add_argument("--status-snapshot-json", required=True)
    parser.add_argument("--stdout-log", required=True)
    parser.add_argument("--stderr-log", required=True)
    parser.add_argument("--paper-metrics-json", required=True)
    parser.add_argument("--strategy", required=True)
    parser.add_argument("--output-root", default="data/paper")
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--reviewer-note",
        action="append",
        default=None,
        help="Reviewer note required before runtime validation can pass. Can be repeated.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    process_executor_plan_path = Path(args.process_executor_plan_json)
    process_metadata_path = Path(args.process_metadata_json)
    status_snapshot_path = Path(args.status_snapshot_json)
    stdout_path = Path(args.stdout_log)
    stderr_path = Path(args.stderr_log)
    paper_metrics_path = Path(args.paper_metrics_json)

    _require_file(process_executor_plan_path, "paper process executor plan JSON")

    run_id = args.run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    inputs = PaperRuntimeValidationInputs(
        root_dir=ROOT_DIR,
        strategy=args.strategy,
        run_id=run_id,
        process_executor_plan_path=process_executor_plan_path,
        process_metadata_path=process_metadata_path,
        status_snapshot_path=status_snapshot_path,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        paper_metrics_path=paper_metrics_path,
        output_root=Path(args.output_root),
        reviewer_notes=list(args.reviewer_note or []),
        command=[sys.executable, *sys.argv],
    )
    validation = build_paper_runtime_validation(
        inputs,
        load_paper_runtime_artifact(process_executor_plan_path),
        _load_optional_runtime_artifact(process_metadata_path),
        _load_optional_runtime_artifact(status_snapshot_path),
        _load_optional_runtime_artifact(paper_metrics_path),
    )
    write_paper_runtime_validation_artifacts(inputs, validation)

    print(
        json.dumps(
            {"status": validation["status"], "output_dir": str(inputs.output_dir)},
            indent=2,
        )
    )
    print(f"Paper runtime validation artifacts written: {inputs.output_dir}")
    return 0 if validation["status"] == "pass" else 1


def _require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise SystemExit(f"{label} file not found: {path}")


def _load_optional_runtime_artifact(path: Path) -> dict:
    if not path.is_file():
        return {}
    return load_paper_runtime_artifact(path)


if __name__ == "__main__":
    sys.exit(main())
