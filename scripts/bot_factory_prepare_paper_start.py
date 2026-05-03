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

from freqtrade_ext.bot_factory.paper_startup import (
    PaperStartupPreflightInputs,
    build_paper_startup_preflight,
    load_paper_run_plan,
    write_paper_startup_preflight_artifacts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a Bot Factory Phase 3 paper startup preflight without "
            "starting freqtrade trade, paper trading, dry-run trading, live "
            "trading, or any bot process."
        )
    )
    parser.add_argument("--plan-json", required=True)
    parser.add_argument("--strategy", required=True)
    parser.add_argument("--output-root", default="data/paper")
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--confirm-paper-start",
        action="store_true",
        help=(
            "Explicit acknowledgement required for a ready startup preflight. "
            "This still does not start any bot process."
        ),
    )
    parser.add_argument(
        "--requested-start-command",
        default=None,
        help=(
            "Exact future start command string to compare with the paper run "
            "plan preview. The command is recorded only and is never executed."
        ),
    )
    parser.add_argument(
        "--reviewer-note",
        action="append",
        default=None,
        help="Reviewer note required before startup preflight can be ready. Can be repeated.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    plan_path = Path(args.plan_json)
    _require_file(plan_path, "paper run plan JSON")
    plan = load_paper_run_plan(plan_path)

    run_id = args.run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    inputs = PaperStartupPreflightInputs(
        root_dir=ROOT_DIR,
        strategy=args.strategy,
        run_id=run_id,
        plan_path=plan_path,
        output_root=Path(args.output_root),
        reviewer_notes=list(args.reviewer_note or []),
        confirm_paper_start=args.confirm_paper_start,
        requested_start_command=args.requested_start_command,
        command=[sys.executable, *sys.argv],
    )
    preflight = build_paper_startup_preflight(inputs, plan)
    write_paper_startup_preflight_artifacts(inputs, preflight)

    print(
        json.dumps(
            {"status": preflight["status"], "output_dir": str(inputs.output_dir)},
            indent=2,
        )
    )
    print(f"Paper startup preflight artifacts written: {inputs.output_dir}")
    return 0 if preflight["status"] == "ready" else 1


def _require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise SystemExit(f"{label} file not found: {path}")


if __name__ == "__main__":
    sys.exit(main())
