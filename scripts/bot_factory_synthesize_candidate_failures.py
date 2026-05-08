#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from freqtrade_ext.bot_factory.candidate_failure_synthesis import (
    CandidateFailureSynthesisInputs,
    synthesize_candidate_failures,
    write_candidate_failure_synthesis_artifacts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Synthesize failed Bot Factory candidate evidence into a local "
            "theory-first next research/code-generation brief. This command "
            "does not generate code, run backtests, start paper/live trading, "
            "or manage any bot process."
        )
    )
    parser.add_argument("--ranking-json", required=True)
    parser.add_argument("--signal-diagnostics-json", action="append", default=[])
    parser.add_argument("--freqai-prediction-diagnostics-json", action="append", default=[])
    parser.add_argument(
        "--local-falsification-json",
        action="append",
        default=[],
        help=(
            "Optional research_local_falsification JSON artifacts that failed "
            "before proposal generation and should be carried into rejection memory."
        ),
    )
    parser.add_argument(
        "--edge-discovery-json",
        action="append",
        default=[],
        help=(
            "Optional research_edge_discovery JSON artifacts that failed or were "
            "blocked before proposal generation and should be carried into "
            "rejection memory."
        ),
    )
    parser.add_argument("--synthesis-id", default=None)
    parser.add_argument("--output-root", default="registry/strategies/synthesis")
    parser.add_argument("--reviewer-note", action="append", default=[])
    return parser.parse_args()


def build_inputs_from_args(
    args: argparse.Namespace, *, root_dir: Path = ROOT_DIR
) -> CandidateFailureSynthesisInputs:
    return CandidateFailureSynthesisInputs(
        root_dir=root_dir,
        ranking_path=Path(args.ranking_json),
        signal_diagnostics_paths=[Path(path) for path in args.signal_diagnostics_json],
        freqai_prediction_diagnostics_paths=[
            Path(path) for path in args.freqai_prediction_diagnostics_json
        ],
        local_falsification_paths=[Path(path) for path in args.local_falsification_json],
        edge_discovery_paths=[Path(path) for path in args.edge_discovery_json],
        output_root=Path(args.output_root),
        synthesis_id=args.synthesis_id,
        reviewer_notes=args.reviewer_note,
    )


def main() -> int:
    inputs = build_inputs_from_args(parse_args())
    synthesis = synthesize_candidate_failures(inputs)
    synthesis_path, report_path = write_candidate_failure_synthesis_artifacts(
        synthesis,
        root_dir=ROOT_DIR,
        output_root=inputs.output_root,
    )
    brief = synthesis.get("next_research_brief", {})
    print(
        json.dumps(
            {
                "candidate_failure_synthesis_path": str(synthesis_path),
                "candidate_failure_synthesis_report_path": str(report_path),
                "status": synthesis["status"],
                "candidate_count": synthesis["candidate_count"],
                "paper_ready_count": synthesis["aggregate_failure_summary"][
                    "paper_ready_count"
                ],
                "parameter_only_retry_allowed": brief.get("parameter_only_retry_allowed"),
                "requires_new_thesis_id": brief.get("requires_new_thesis_id"),
                "local_falsification_rejection_count": synthesis[
                    "aggregate_failure_summary"
                ].get("local_falsification_rejection_count"),
                "edge_discovery_rejection_count": synthesis[
                    "aggregate_failure_summary"
                ].get("edge_discovery_rejection_count"),
            },
            indent=2,
        )
    )
    return 0 if synthesis.get("status") == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
