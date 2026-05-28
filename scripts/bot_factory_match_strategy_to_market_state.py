#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from freqtrade_ext.bot_factory.selector_matching import (
    build_no_trade_scorecard,
    build_selector_matching_decision,
    write_selector_matching_artifacts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Match the current local market state to a strategy suitability matrix. "
            "This is selector simulation only and does not start paper, dry-run, "
            "live trading, or exchange order placement."
        )
    )
    state_group = parser.add_mutually_exclusive_group(required=True)
    state_group.add_argument("--current-market-state-json")
    state_group.add_argument("--market-state-snapshot-json")
    parser.add_argument("--strategy-suitability-matrix-json", required=True)
    parser.add_argument("--selector-state-json", default=None)
    parser.add_argument("--output-root", default="data/selector_matching")
    parser.add_argument("--decision-id", default=None)
    parser.add_argument("--generated-at", default=None)
    parser.add_argument("--min-state-confidence", type=float, default=0.5)
    parser.add_argument("--max-out-of-distribution-score", type=float, default=0.8)
    parser.add_argument("--cooldown-observations", type=int, default=0)
    parser.add_argument("--hysteresis-margin", type=float, default=0.1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    current_state_path = Path(
        args.current_market_state_json or args.market_state_snapshot_json
    )
    current_state = _load_json(current_state_path)
    matrix_path = Path(args.strategy_suitability_matrix_json)
    matrix = _load_json(matrix_path)
    selector_state = _load_json(Path(args.selector_state_json)) if args.selector_state_json else None
    decision = build_selector_matching_decision(
        current_market_state=current_state,
        strategy_suitability_matrix=matrix,
        selector_state=selector_state,
        generated_at=args.generated_at,
        decision_id=args.decision_id,
        min_state_confidence=args.min_state_confidence,
        max_out_of_distribution_score=args.max_out_of_distribution_score,
        cooldown_observations=args.cooldown_observations,
        hysteresis_margin=args.hysteresis_margin,
        source_artifacts={
            "current_market_state": str(current_state_path),
            "strategy_suitability_matrix": str(matrix_path),
        },
    )
    no_trade_scorecard = build_no_trade_scorecard(
        current_market_state=current_state,
        strategy_suitability_matrix=matrix,
        selector_decision=decision,
        generated_at=args.generated_at,
    )
    paths = write_selector_matching_artifacts(
        decision,
        output_root=Path(args.output_root),
        no_trade_scorecard=no_trade_scorecard,
    )
    print(
        json.dumps(
            {
                "decision_id": decision["decision_id"],
                "selected_action": decision["selected_action"],
                "selected_candidate_id": decision["selected_candidate_id"],
                "no_trade_reason": decision["no_trade_reason"],
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
