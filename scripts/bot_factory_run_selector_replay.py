from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from freqtrade_ext.bot_factory.selector_replay import (
    build_historical_selector_replay,
    write_historical_selector_replay_artifacts,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a local historical as-of selector/no_trade replay."
    )
    parser.add_argument(
        "--market-state-snapshot-json",
        action="append",
        required=True,
        help="Historical market_state_snapshot.json path. Repeat in decision order or any order.",
    )
    parser.add_argument(
        "--strategy-suitability-matrix-json",
        action="append",
        required=True,
        help="strategy_state_suitability_matrix.json path. Repeat for each evidence version.",
    )
    parser.add_argument(
        "--realized-returns-json",
        help=(
            "Optional JSON mapping data_asof timestamp to returns by candidate_id and hold. "
            "Returns must be local historical evaluation data."
        ),
    )
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--generated-at", default=None)
    parser.add_argument("--incumbent-candidate-id", default=None)
    parser.add_argument("--normal-turnover-cost", type=float, default=0.0)
    parser.add_argument("--stress-turnover-cost", type=float, default=0.0)
    parser.add_argument("--output-root", default="data/selector_replay")
    args = parser.parse_args()

    snapshots = [_read_json(Path(path)) for path in args.market_state_snapshot_json]
    matrices = [_read_json(Path(path)) for path in args.strategy_suitability_matrix_json]
    realized_returns = (
        _read_json(Path(args.realized_returns_json))
        if args.realized_returns_json
        else {}
    )
    replay = build_historical_selector_replay(
        market_state_snapshots=snapshots,
        strategy_suitability_matrices=matrices,
        realized_returns_by_timestamp=realized_returns,
        run_id=args.run_id,
        generated_at=args.generated_at,
        incumbent_candidate_id=args.incumbent_candidate_id,
        normal_turnover_cost=args.normal_turnover_cost,
        stress_turnover_cost=args.stress_turnover_cost,
        source_artifacts={
            "market_state_snapshots": ",".join(args.market_state_snapshot_json),
            "strategy_suitability_matrices": ",".join(args.strategy_suitability_matrix_json),
            "realized_returns": args.realized_returns_json or "",
        },
    )
    paths = write_historical_selector_replay_artifacts(
        replay,
        output_root=Path(args.output_root),
    )
    print(
        json.dumps(
            {
                "status": replay.get("status"),
                "run_id": replay.get("run_id"),
                "decision_count": replay.get("decision_count"),
                "selector_net_return_normal_cost": (
                    replay.get("metrics_summary") or {}
                ).get("selector_net_return_normal_cost"),
                "paths": {key: str(value) for key, value in paths.items()},
                "safety_scope": replay.get("safety_scope"),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if replay.get("status") == "completed" else 1


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    raise SystemExit(main())
