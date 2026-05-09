#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from freqtrade_ext.bot_factory.candidate_failure_map import (
    CandidateFailureMapInputs,
    build_candidate_failure_map,
    write_candidate_failure_map_artifacts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a local causal failure map from a Bot Factory "
            "candidate_failure_synthesis.json artifact. This command does not "
            "generate code, run backtests, start paper/live trading, or manage "
            "any bot process."
        )
    )
    parser.add_argument("--synthesis-json", required=True)
    parser.add_argument("--map-id", default=None)
    parser.add_argument("--output-root", default="registry/strategies/failure_maps")
    parser.add_argument("--reviewer-note", action="append", default=[])
    return parser.parse_args()


def build_inputs_from_args(
    args: argparse.Namespace, *, root_dir: Path = ROOT_DIR
) -> CandidateFailureMapInputs:
    return CandidateFailureMapInputs(
        root_dir=root_dir,
        synthesis_path=Path(args.synthesis_json),
        output_root=Path(args.output_root),
        map_id=args.map_id,
        reviewer_notes=list(args.reviewer_note or []),
    )


def main() -> int:
    inputs = build_inputs_from_args(parse_args())
    failure_map = build_candidate_failure_map(inputs)
    map_path, report_path = write_candidate_failure_map_artifacts(
        failure_map,
        root_dir=ROOT_DIR,
        output_root=inputs.output_root,
    )
    guidance = failure_map.get("research_selection_guidance", {})
    print(
        json.dumps(
            {
                "causal_failure_map_path": str(map_path),
                "causal_failure_map_report_path": str(report_path),
                "status": failure_map["status"],
                "candidate_count": failure_map["candidate_count"],
                "category_count": failure_map["causal_failure_categories"][
                    "category_count"
                ],
                "requires_research_decision_before_proposal": guidance.get(
                    "requires_research_decision_before_proposal"
                ),
            },
            indent=2,
        )
    )
    return 0 if failure_map.get("status") == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
