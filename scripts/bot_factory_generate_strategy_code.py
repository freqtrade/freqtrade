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

from freqtrade_ext.bot_factory.strategy_code import (
    StrategyCodeInputs,
    build_strategy_code,
    write_strategy_code_artifacts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a Bot Factory long-only Freqtrade strategy from an "
            "accepted proposal metadata artifact. This command writes local "
            "strategy, metadata, and static-check artifacts only; it does not "
            "run backtests, start paper/live trading, call exchange order "
            "endpoints, promote candidates, or manage any bot process."
        )
    )
    parser.add_argument("--proposal-metadata-json", required=True)
    parser.add_argument("--candidate-id", default=None)
    parser.add_argument("--output-root", default="registry/strategies/generated")
    parser.add_argument("--created-by-agent", default="codex")
    parser.add_argument("--created-at", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    created_at = args.created_at or datetime.now(UTC).replace(microsecond=0).isoformat()
    inputs = StrategyCodeInputs(
        root_dir=ROOT_DIR,
        proposal_metadata_path=Path(args.proposal_metadata_json),
        candidate_id=args.candidate_id,
        output_root=Path(args.output_root),
        created_by_agent=args.created_by_agent,
        created_at=created_at,
        command=[sys.executable, *sys.argv],
    )
    artifacts = build_strategy_code(inputs)
    write_strategy_code_artifacts(artifacts)

    print(
        json.dumps(
            {
                "status": artifacts.metadata["status"],
                "strategy_path": str(artifacts.strategy_path),
                "metadata_path": str(artifacts.metadata_path),
                "static_check_path": str(artifacts.static_check_path),
                "strategy_code_generated": artifacts.metadata[
                    "strategy_code_generated"
                ],
                "candidate_evaluation_eligible": artifacts.metadata[
                    "candidate_evaluation_eligible"
                ],
                "static_check_ok": artifacts.metadata["static_check"]["ok"],
            },
            indent=2,
        )
    )
    print(f"Strategy code artifacts written: {artifacts.metadata_path.parent}")
    return 0 if artifacts.metadata["status"] == "generated" else 1


if __name__ == "__main__":
    sys.exit(main())
