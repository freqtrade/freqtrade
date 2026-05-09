#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from freqtrade_ext.bot_factory.candidate_ranking import (
    CandidateRankingInputs,
    rank_candidates,
    write_candidate_ranking_artifacts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rank local Bot Factory candidate manifests. This command reads "
            "historical-safe artifacts only and does not start paper/live trading "
            "or manage any bot process."
        )
    )
    parser.add_argument("--candidate-manifest-json", action="append", required=True)
    parser.add_argument("--ranking-id", default=None)
    parser.add_argument("--output-root", default="registry/strategies/candidates/rankings")
    parser.add_argument("--reviewer-note", action="append", default=[])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    inputs = CandidateRankingInputs(
        root_dir=ROOT_DIR,
        candidate_manifest_paths=[Path(path) for path in args.candidate_manifest_json],
        output_root=Path(args.output_root),
        ranking_id=args.ranking_id,
        reviewer_notes=args.reviewer_note,
    )
    ranking = rank_candidates(inputs)
    ranking_path, report_path = write_candidate_ranking_artifacts(
        ranking,
        root_dir=ROOT_DIR,
        output_root=inputs.output_root,
    )
    print(
        json.dumps(
            {
                "ranking_path": str(ranking_path),
                "ranking_report_path": str(report_path),
                "candidate_count": ranking["candidate_count"],
                "best_candidate_id": ranking["best_candidate_id"],
                "paper_ready_candidate_ids": ranking["paper_ready_candidate_ids"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
