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

from freqtrade_ext.bot_factory.paper_stop_cleanup import (
    PaperStopCleanupPlanInputs,
    build_paper_stop_cleanup_plan,
    load_paper_monitoring_plan,
    write_paper_stop_cleanup_plan_artifacts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plan Bot Factory Phase 3 paper stop/cleanup artifacts without "
            "starting, stopping, polling, terminating, or managing any bot process."
        )
    )
    parser.add_argument("--monitoring-plan-json", required=True)
    parser.add_argument("--strategy", required=True)
    parser.add_argument("--output-root", default="data/paper")
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--reviewer-note",
        action="append",
        default=None,
        help="Reviewer note required before stop/cleanup planning can be ready. Can be repeated.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    monitoring_plan_path = Path(args.monitoring_plan_json)
    _require_file(monitoring_plan_path, "paper monitoring plan JSON")
    monitoring_plan = load_paper_monitoring_plan(monitoring_plan_path)

    run_id = args.run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    inputs = PaperStopCleanupPlanInputs(
        root_dir=ROOT_DIR,
        strategy=args.strategy,
        run_id=run_id,
        monitoring_plan_path=monitoring_plan_path,
        output_root=Path(args.output_root),
        reviewer_notes=list(args.reviewer_note or []),
        command=[sys.executable, *sys.argv],
    )
    plan = build_paper_stop_cleanup_plan(inputs, monitoring_plan)
    write_paper_stop_cleanup_plan_artifacts(inputs, plan)

    print(json.dumps({"status": plan["status"], "output_dir": str(inputs.output_dir)}, indent=2))
    print(f"Paper stop/cleanup plan artifacts written: {inputs.output_dir}")
    return 0 if plan["status"] == "ready" else 1


def _require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise SystemExit(f"{label} file not found: {path}")


if __name__ == "__main__":
    sys.exit(main())
