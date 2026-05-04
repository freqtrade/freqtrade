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

from freqtrade_ext.bot_factory.paper_monitoring import (
    PaperMonitoringPlanInputs,
    build_paper_monitoring_plan,
    load_paper_startup_preflight,
    write_paper_monitoring_plan_artifacts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plan Bot Factory Phase 3 paper monitoring/status artifacts without "
            "starting, stopping, polling, or managing any bot process."
        )
    )
    parser.add_argument("--startup-preflight-json", required=True)
    parser.add_argument("--strategy", required=True)
    parser.add_argument("--output-root", default="data/paper")
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--reviewer-note",
        action="append",
        default=None,
        help="Reviewer note required before monitoring schemas can be ready. Can be repeated.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    startup_preflight_path = Path(args.startup_preflight_json)
    _require_file(startup_preflight_path, "paper startup preflight JSON")
    startup_preflight = load_paper_startup_preflight(startup_preflight_path)

    run_id = args.run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    inputs = PaperMonitoringPlanInputs(
        root_dir=ROOT_DIR,
        strategy=args.strategy,
        run_id=run_id,
        startup_preflight_path=startup_preflight_path,
        output_root=Path(args.output_root),
        reviewer_notes=list(args.reviewer_note or []),
        command=[sys.executable, *sys.argv],
    )
    plan = build_paper_monitoring_plan(inputs, startup_preflight)
    write_paper_monitoring_plan_artifacts(inputs, plan)

    print(json.dumps({"status": plan["status"], "output_dir": str(inputs.output_dir)}, indent=2))
    print(f"Paper monitoring plan artifacts written: {inputs.output_dir}")
    return 0 if plan["status"] == "ready" else 1


def _require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise SystemExit(f"{label} file not found: {path}")


if __name__ == "__main__":
    sys.exit(main())
