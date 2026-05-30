from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from freqtrade_ext.bot_factory.diagnostic_state_discovery import (
    build_diagnostic_state_discovery_report,
    write_diagnostic_state_discovery_artifacts,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build diagnostic-only state discovery artifacts from local market-state snapshots."
    )
    parser.add_argument(
        "--market-state-snapshot-json",
        action="append",
        required=True,
        help="Local market_state_snapshot.json. Repeat for historical windows.",
    )
    parser.add_argument(
        "--state-scorecard-json",
        action="append",
        default=[],
        help="Optional local state_conditioned_scorecard.json. Repeat as needed.",
    )
    parser.add_argument(
        "--strategy-suitability-matrix-json",
        action="append",
        default=[],
        help="Optional local strategy_state_suitability_matrix.json. Repeat as needed.",
    )
    parser.add_argument("--run-id")
    parser.add_argument("--generated-at")
    parser.add_argument("--analog-k", type=int, default=3)
    parser.add_argument("--min-cluster-size", type=int, default=2)
    parser.add_argument("--output-root", default="data/diagnostic_state_discovery")
    args = parser.parse_args()

    snapshot_paths = [Path(path) for path in args.market_state_snapshot_json]
    scorecard_paths = [Path(path) for path in args.state_scorecard_json]
    matrix_paths = [Path(path) for path in args.strategy_suitability_matrix_json]
    report = build_diagnostic_state_discovery_report(
        market_state_snapshots=[_read_json(path) for path in snapshot_paths],
        state_scorecards=[_read_json(path) for path in scorecard_paths],
        strategy_suitability_matrices=[_read_json(path) for path in matrix_paths],
        run_id=args.run_id,
        generated_at=args.generated_at,
        analog_k=args.analog_k,
        min_cluster_size=args.min_cluster_size,
        source_artifacts={
            "market_state_snapshots": ";".join(str(path) for path in snapshot_paths),
            "state_scorecards": ";".join(str(path) for path in scorecard_paths),
            "strategy_suitability_matrices": ";".join(str(path) for path in matrix_paths),
        },
    )
    paths = write_diagnostic_state_discovery_artifacts(
        report,
        output_root=Path(args.output_root),
    )
    print(
        json.dumps(
            {
                "run_id": report.get("run_id"),
                "status": report.get("status"),
                "diagnostic_only": report.get("diagnostic_only"),
                "reason_codes": report.get("reason_codes"),
                "paths": {key: str(value) for key, value in paths.items()},
                "safety_scope": report.get("safety_scope"),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    raise SystemExit(main())
