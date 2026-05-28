#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from freqtrade_ext.bot_factory.state_conditioning import (
    build_state_conditioned_scorecard,
    write_state_conditioned_scorecard_artifacts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a local state-conditioned scorecard from a checked regime "
            "fitness scorecard and market-state snapshot. This does not start "
            "paper, dry-run, live, bot, or exchange-facing processes."
        )
    )
    parser.add_argument("--regime-scorecard-json", required=True)
    parser.add_argument("--market-state-snapshot-json", required=True)
    parser.add_argument("--output-root", default="data/state_scorecards")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--generated-at", default=None)
    parser.add_argument("--proxy-evidence", action="store_true")
    parser.add_argument("--relaxed-thresholds-used", action="store_true")
    parser.add_argument("--allow-missing-walk-forward", action="store_true")
    parser.add_argument("--reviewer-note", action="append", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    regime_path = Path(args.regime_scorecard_json)
    snapshot_path = Path(args.market_state_snapshot_json)
    regime_scorecard = json.loads(regime_path.read_text(encoding="utf-8"))
    market_state_snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    scorecard = build_state_conditioned_scorecard(
        regime_scorecard=regime_scorecard,
        market_state_snapshot=market_state_snapshot,
        run_id=args.run_id,
        generated_at=args.generated_at,
        source_artifacts={
            "regime_scorecard": str(regime_path).replace("\\", "/"),
            "market_state_snapshot": str(snapshot_path).replace("\\", "/"),
        },
        proxy_evidence=args.proxy_evidence,
        relaxed_thresholds_used=args.relaxed_thresholds_used,
        require_walk_forward_evidence=not args.allow_missing_walk_forward,
        reviewer_notes=tuple(args.reviewer_note or []),
    )
    paths = write_state_conditioned_scorecard_artifacts(
        scorecard,
        output_root=ROOT_DIR / Path(args.output_root),
    )
    print(json.dumps({key: str(value) for key, value in paths.items()}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
