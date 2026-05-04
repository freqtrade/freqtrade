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

from freqtrade_ext.bot_factory.paper_executor import (
    PaperProcessExecutorPlanInputs,
    build_paper_process_executor_plan,
    load_paper_process_executor_artifact,
    write_paper_process_executor_plan_artifacts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plan a Bot Factory Phase 3 paper process executor without "
            "starting freqtrade trade, paper trading, dry-run trading, live "
            "trading, stopping, polling, terminating, cleaning up, or managing "
            "any bot process."
        )
    )
    parser.add_argument("--execution-request-json", required=True)
    parser.add_argument("--strategy", required=True)
    parser.add_argument("--output-root", default="data/paper")
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--confirm-process-executor-plan",
        action="store_true",
        help=(
            "Explicit acknowledgement required for a ready process executor "
            "plan. This still does not start any bot process."
        ),
    )
    parser.add_argument(
        "--requested-start-command",
        default=None,
        help=(
            "Exact future start command string to compare with the paper "
            "execution request. The command is recorded only and is never executed."
        ),
    )
    parser.add_argument(
        "--reviewer-note",
        action="append",
        default=None,
        help="Reviewer note required before a process executor plan can be ready. Can be repeated.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    execution_request_path = Path(args.execution_request_json)
    _require_file(execution_request_path, "paper execution request JSON")

    run_id = args.run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    inputs = PaperProcessExecutorPlanInputs(
        root_dir=ROOT_DIR,
        strategy=args.strategy,
        run_id=run_id,
        execution_request_path=execution_request_path,
        output_root=Path(args.output_root),
        reviewer_notes=list(args.reviewer_note or []),
        confirm_process_executor_plan=args.confirm_process_executor_plan,
        requested_start_command=args.requested_start_command,
        command=[sys.executable, *sys.argv],
    )
    plan = build_paper_process_executor_plan(
        inputs, load_paper_process_executor_artifact(execution_request_path)
    )
    write_paper_process_executor_plan_artifacts(inputs, plan)

    print(json.dumps({"status": plan["status"], "output_dir": str(inputs.output_dir)}, indent=2))
    print(f"Paper process executor plan artifacts written: {inputs.output_dir}")
    return 0 if plan["status"] == "ready" else 1


def _require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise SystemExit(f"{label} file not found: {path}")


if __name__ == "__main__":
    sys.exit(main())
