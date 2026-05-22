#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from freqtrade_ext.bot_factory.candidate_review import (
    build_candidate_review,
    write_candidate_review_artifacts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a local candidate review report from Bot Factory artifacts."
    )
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--strategy", required=True)
    parser.add_argument("--strategy-source", default=None)
    parser.add_argument("--historical-metrics", default=None)
    parser.add_argument("--walk-forward-metrics", default=None)
    parser.add_argument("--observation-ledger", default=None)
    parser.add_argument("--regime-scorecard", default=None)
    parser.add_argument("--selector-candidate", default=None)
    parser.add_argument("--selector-decision", default=None)
    parser.add_argument("--paper-readiness", default=None)
    parser.add_argument("--previous-review", default=None)
    parser.add_argument("--output-root", default="registry/strategies/reviews")
    parser.add_argument("--reviewer-note", action="append", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    review = build_candidate_review(
        root_dir=ROOT_DIR,
        candidate_id=args.candidate_id,
        strategy=args.strategy,
        strategy_source_path=_path(args.strategy_source),
        historical_metrics_path=_path(args.historical_metrics),
        walk_forward_metrics_path=_path(args.walk_forward_metrics),
        observation_ledger_path=_path(args.observation_ledger),
        regime_scorecard_path=_path(args.regime_scorecard),
        selector_candidate_path=_path(args.selector_candidate),
        selector_decision_path=_path(args.selector_decision),
        paper_readiness_path=_path(args.paper_readiness),
        previous_review_path=_path(args.previous_review),
        reviewer_notes=list(args.reviewer_note or []),
    )
    json_path, report_path = write_candidate_review_artifacts(
        review,
        root_dir=ROOT_DIR,
        output_root=Path(args.output_root),
    )
    print(json.dumps({"candidate_review": str(json_path), "report": str(report_path)}, indent=2))
    return 0


def _path(value: str | None) -> Path | None:
    return Path(value) if value else None


if __name__ == "__main__":
    sys.exit(main())
