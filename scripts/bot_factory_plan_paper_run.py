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

from freqtrade_ext.bot_factory.paper_plan import (
    PaperRunPlanInputs,
    build_paper_run_plan,
    load_paper_readiness,
    write_paper_run_plan_artifacts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plan a future Bot Factory Phase 3 paper run without starting "
            "freqtrade trade, paper trading, dry-run trading, live trading, "
            "or any bot process."
        )
    )
    parser.add_argument("--readiness-json", required=True)
    parser.add_argument("--strategy", required=True)
    parser.add_argument(
        "--config",
        default=None,
        help="Optional config path. Defaults to the config_path recorded in readiness JSON.",
    )
    parser.add_argument("--strategy-path", default="user_data/strategies")
    parser.add_argument("--output-root", default="data/paper")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--freqtrade-bin", default="freqtrade")
    parser.add_argument(
        "--confirm-paper",
        action="store_true",
        help=(
            "Explicit acknowledgement required for a ready plan. This still "
            "does not start any bot process."
        ),
    )
    parser.add_argument(
        "--reviewer-note",
        action="append",
        default=None,
        help="Reviewer note required before a paper run plan can be ready. Can be repeated.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    readiness_path = Path(args.readiness_json)
    _require_file(readiness_path, "readiness JSON")
    readiness = load_paper_readiness(readiness_path)

    run_id = args.run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    inputs = PaperRunPlanInputs(
        root_dir=ROOT_DIR,
        strategy=args.strategy,
        run_id=run_id,
        readiness_path=readiness_path,
        config_path=Path(args.config) if args.config else None,
        strategy_path=Path(args.strategy_path),
        output_root=Path(args.output_root),
        reviewer_notes=list(args.reviewer_note or []),
        confirm_paper=args.confirm_paper,
        command=[sys.executable, *sys.argv],
        freqtrade_binary=args.freqtrade_bin,
    )
    plan = build_paper_run_plan(inputs, readiness)
    write_paper_run_plan_artifacts(inputs, plan)

    print(json.dumps({"status": plan["status"], "output_dir": str(inputs.output_dir)}, indent=2))
    print(f"Paper run plan artifacts written: {inputs.output_dir}")
    return 0 if plan["status"] == "ready" else 1


def _require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise SystemExit(f"{label} file not found: {path}")


if __name__ == "__main__":
    sys.exit(main())
