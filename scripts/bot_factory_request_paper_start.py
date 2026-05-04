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

from freqtrade_ext.bot_factory.paper_execution import (
    PaperExecutionRequestInputs,
    build_paper_execution_request,
    load_paper_execution_artifact,
    write_paper_execution_request_artifacts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a Bot Factory Phase 3 paper start execution request without "
            "starting freqtrade trade, paper trading, dry-run trading, live "
            "trading, stopping, polling, terminating, or managing any bot process."
        )
    )
    parser.add_argument("--readiness-json", required=True)
    parser.add_argument("--plan-json", required=True)
    parser.add_argument("--startup-preflight-json", required=True)
    parser.add_argument("--monitoring-plan-json", required=True)
    parser.add_argument("--stop-cleanup-plan-json", required=True)
    parser.add_argument("--strategy", required=True)
    parser.add_argument("--output-root", default="data/paper")
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--confirm-paper-execution",
        action="store_true",
        help=(
            "Explicit acknowledgement required for a ready execution request. "
            "This still does not start any bot process."
        ),
    )
    parser.add_argument(
        "--requested-start-command",
        default=None,
        help=(
            "Exact future start command string to compare with the startup "
            "preflight preview. The command is recorded only and is never executed."
        ),
    )
    parser.add_argument(
        "--reviewer-note",
        action="append",
        default=None,
        help="Reviewer note required before an execution request can be ready. Can be repeated.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    readiness_path = Path(args.readiness_json)
    plan_path = Path(args.plan_json)
    startup_preflight_path = Path(args.startup_preflight_json)
    monitoring_plan_path = Path(args.monitoring_plan_json)
    stop_cleanup_plan_path = Path(args.stop_cleanup_plan_json)

    _require_file(readiness_path, "paper readiness JSON")
    _require_file(plan_path, "paper run plan JSON")
    _require_file(startup_preflight_path, "paper startup preflight JSON")
    _require_file(monitoring_plan_path, "paper monitoring plan JSON")
    _require_file(stop_cleanup_plan_path, "paper stop/cleanup plan JSON")

    run_id = args.run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    inputs = PaperExecutionRequestInputs(
        root_dir=ROOT_DIR,
        strategy=args.strategy,
        run_id=run_id,
        readiness_path=readiness_path,
        plan_path=plan_path,
        startup_preflight_path=startup_preflight_path,
        monitoring_plan_path=monitoring_plan_path,
        stop_cleanup_plan_path=stop_cleanup_plan_path,
        output_root=Path(args.output_root),
        reviewer_notes=list(args.reviewer_note or []),
        confirm_paper_execution=args.confirm_paper_execution,
        requested_start_command=args.requested_start_command,
        command=[sys.executable, *sys.argv],
    )
    request = build_paper_execution_request(
        inputs,
        load_paper_execution_artifact(readiness_path),
        load_paper_execution_artifact(plan_path),
        load_paper_execution_artifact(startup_preflight_path),
        load_paper_execution_artifact(monitoring_plan_path),
        load_paper_execution_artifact(stop_cleanup_plan_path),
    )
    write_paper_execution_request_artifacts(inputs, request)

    print(
        json.dumps(
            {"status": request["status"], "output_dir": str(inputs.output_dir)},
            indent=2,
        )
    )
    print(f"Paper execution request artifacts written: {inputs.output_dir}")
    return 0 if request["status"] == "ready" else 1


def _require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise SystemExit(f"{label} file not found: {path}")


if __name__ == "__main__":
    sys.exit(main())
