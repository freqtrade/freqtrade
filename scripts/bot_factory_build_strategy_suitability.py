#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from freqtrade_ext.bot_factory.strategy_suitability import (
    build_strategy_suitability_matrix,
    write_strategy_suitability_artifacts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a local-only Bot Factory strategy suitability matrix from "
            "state-conditioned scorecards. This does not start any bot process."
        )
    )
    parser.add_argument(
        "--state-scorecard-json",
        action="append",
        required=True,
        help="state_conditioned_scorecard.json path. Can be repeated.",
    )
    parser.add_argument("--market-state-snapshot-json", default=None)
    parser.add_argument("--output-root", default="data/strategy_suitability")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--generated-at", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    state_scorecards = [_load_json(Path(path)) for path in args.state_scorecard_json]
    market_state_snapshot = (
        _load_json(Path(args.market_state_snapshot_json))
        if args.market_state_snapshot_json
        else None
    )
    matrix = build_strategy_suitability_matrix(
        state_scorecards=state_scorecards,
        market_state_snapshot=market_state_snapshot,
        run_id=args.run_id,
        generated_at=args.generated_at,
        source_artifacts={
            f"state_scorecard_{index}": str(path)
            for index, path in enumerate(args.state_scorecard_json, start=1)
        },
    )
    paths = write_strategy_suitability_artifacts(
        matrix,
        output_root=Path(args.output_root),
    )
    print(
        json.dumps(
            {
                "run_id": matrix["run_id"],
                "selector_row_count": matrix["selector_row_count"],
                "paths": {name: str(path) for name, path in paths.items()},
            },
            indent=2,
        )
    )
    return 0


def _load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"JSON file must contain an object: {path}")
    return payload


if __name__ == "__main__":
    sys.exit(main())
