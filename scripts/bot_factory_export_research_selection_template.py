#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from freqtrade_ext.bot_factory.research_selection_template import (
    ResearchSelectionTemplateInputs,
    build_research_selection_template,
    write_research_selection_template_artifacts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export a local response template for Bot Factory research "
            "selection from a causal_failure_map.json. This command writes "
            "local artifacts only; it does not select a thesis, generate code, "
            "run backtests, start trading, or manage processes."
        )
    )
    parser.add_argument("--causal-failure-map-json", required=True)
    parser.add_argument("--template-id", default=None)
    parser.add_argument(
        "--output-root",
        default="registry/strategies/research_decisions",
    )
    parser.add_argument("--reviewer-note", action="append", default=[])
    parser.add_argument("--created-at", default=None)
    return parser.parse_args()


def build_inputs_from_args(
    args: argparse.Namespace, *, root_dir: Path = ROOT_DIR
) -> ResearchSelectionTemplateInputs:
    return ResearchSelectionTemplateInputs(
        root_dir=root_dir,
        causal_failure_map_path=Path(args.causal_failure_map_json),
        output_root=Path(args.output_root),
        template_id=args.template_id,
        reviewer_notes=list(args.reviewer_note or []),
        created_at=args.created_at,
        command=sys.argv,
    )


def main() -> int:
    inputs = build_inputs_from_args(parse_args())
    artifact = build_research_selection_template(inputs)
    json_path, report_path = write_research_selection_template_artifacts(
        artifact,
        root_dir=ROOT_DIR,
        output_root=inputs.output_root,
    )
    print(
        json.dumps(
            {
                "research_selection_template_path": str(json_path),
                "research_selection_template_report_path": str(report_path),
                "status": artifact["status"],
                "required_causal_failure_response_count": artifact[
                    "required_causal_failure_response_count"
                ],
                "required_research_question_response_count": artifact[
                    "required_research_question_response_count"
                ],
                "validated_local_falsification_rejection_count": artifact[
                    "validated_local_falsification_rejection_count"
                ],
                "blocker_count": len(artifact["blockers"]),
            },
            indent=2,
        )
    )
    return 0 if artifact.get("status") == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
