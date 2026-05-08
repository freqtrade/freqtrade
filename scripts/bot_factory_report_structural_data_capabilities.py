#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from freqtrade_ext.bot_factory.structural_data_capabilities import (
    StructuralDataCapabilityInputs,
    build_structural_data_capability_report,
    default_structural_data_capability_output_path,
    write_structural_data_capability_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Report which structural market-data classes are available for "
            "Bot Factory local research and which remain blocked before "
            "proposal/codegen. This command writes a local JSON artifact only."
        )
    )
    parser.add_argument("--open-interest-path", default=None)
    parser.add_argument("--open-interest-quality-report-json", action="append", default=[])
    parser.add_argument("--long-short-ratio-path", default=None)
    parser.add_argument("--long-short-ratio-quality-report-json", action="append", default=[])
    parser.add_argument("--liquidation-path", action="append", default=[])
    parser.add_argument("--liquidation-quality-report-json", action="append", default=[])
    parser.add_argument("--order-book-path", action="append", default=[])
    parser.add_argument("--order-book-quality-report-json", action="append", default=[])
    parser.add_argument(
        "--output",
        default=None,
        help="Optional JSON report path. Defaults to registry/strategies/checks/.",
    )
    parser.add_argument("--created-at", default=None)
    return parser.parse_args()


def inputs_from_args(args: argparse.Namespace) -> StructuralDataCapabilityInputs:
    return StructuralDataCapabilityInputs(
        root_dir=ROOT_DIR,
        open_interest_path=Path(args.open_interest_path) if args.open_interest_path else None,
        open_interest_quality_report_paths=[
            Path(item) for item in args.open_interest_quality_report_json or []
        ],
        long_short_ratio_path=Path(args.long_short_ratio_path) if args.long_short_ratio_path else None,
        long_short_ratio_quality_report_paths=[
            Path(item) for item in args.long_short_ratio_quality_report_json or []
        ],
        liquidation_paths=[Path(item) for item in args.liquidation_path or []],
        liquidation_quality_report_paths=[
            Path(item) for item in args.liquidation_quality_report_json or []
        ],
        order_book_paths=[Path(item) for item in args.order_book_path or []],
        order_book_quality_report_paths=[
            Path(item) for item in args.order_book_quality_report_json or []
        ],
        output_path=Path(args.output) if args.output else None,
        created_at=args.created_at,
        command=sys.argv,
    )


def main() -> int:
    inputs = inputs_from_args(parse_args())
    artifact = build_structural_data_capability_report(inputs)
    output_path = inputs.output_path or default_structural_data_capability_output_path()
    write_structural_data_capability_report(artifact, output_path)
    print(
        json.dumps(
            {
                "structural_data_capability_report_path": str(output_path),
                "local_research_usable": artifact["proposal_guidance"][
                    "local_research_usable"
                ],
                "blocked_without_new_data": artifact["proposal_guidance"][
                    "blocked_without_new_data"
                ],
                "must_not_codegen": artifact["proposal_guidance"]["must_not_codegen"],
                "blocker_count": len(artifact["blockers"]),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
