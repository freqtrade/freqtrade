#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from freqtrade_ext.bot_factory.candidate_iteration import (
    CandidateIterationInputs,
    build_candidate_iteration_plan,
    write_candidate_iteration_artifacts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a local Bot Factory candidate iteration plan. This command "
            "does not generate code, run backtests, start paper/live trading, "
            "or manage any bot process."
        )
    )
    parser.add_argument("--candidate-manifest-json", required=True)
    parser.add_argument("--proposal-metadata-json", required=True)
    parser.add_argument("--generated-metadata-json", default=None)
    parser.add_argument("--revision-id", default=None)
    parser.add_argument("--output-root", default="registry/strategies/reviews")
    parser.add_argument("--reviewer-finding", action="append", required=True)
    parser.add_argument("--changed-assumption", action="append", default=[])
    parser.add_argument("--changed-parameter", action="append", default=[])
    parser.add_argument("--changed-data-requirement", action="append", default=[])
    parser.add_argument("--unchanged-rejection-rule", action="append", required=True)
    parser.add_argument("--prior-timerange", default=None)
    parser.add_argument("--proposed-timerange", default=None)
    parser.add_argument("--max-parameter-changes", type=int, default=4)
    parser.add_argument("--max-attempts-per-strategy-family", type=int, default=5)
    parser.add_argument("--timeout-minutes", type=int, default=60)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    inputs = CandidateIterationInputs(
        root_dir=ROOT_DIR,
        candidate_manifest_path=Path(args.candidate_manifest_json),
        proposal_metadata_path=Path(args.proposal_metadata_json),
        generated_metadata_path=Path(args.generated_metadata_json)
        if args.generated_metadata_json
        else None,
        output_root=Path(args.output_root),
        revision_id=args.revision_id,
        reviewer_findings=args.reviewer_finding,
        changed_assumptions=args.changed_assumption,
        changed_parameters=args.changed_parameter,
        changed_data_requirements=args.changed_data_requirement,
        unchanged_rejection_rules=args.unchanged_rejection_rule,
        prior_timerange=args.prior_timerange,
        proposed_timerange=args.proposed_timerange,
        max_parameter_changes=args.max_parameter_changes,
        max_attempts_per_strategy_family=args.max_attempts_per_strategy_family,
        timeout_minutes=args.timeout_minutes,
    )
    plan = build_candidate_iteration_plan(inputs)
    plan_path, revision_input_path, report_path = write_candidate_iteration_artifacts(
        plan,
        root_dir=ROOT_DIR,
        output_root=inputs.output_root,
    )
    print(
        json.dumps(
            {
                "iteration_plan_path": str(plan_path),
                "proposal_revision_input_path": str(revision_input_path),
                "iteration_report_path": str(report_path),
                "action": plan["action"],
                "evaluation_allowed_by_this_plan": plan["evaluation_allowed_by_this_plan"],
            },
            indent=2,
        )
    )
    return 0 if plan["action"] in {"revise", "reject"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
