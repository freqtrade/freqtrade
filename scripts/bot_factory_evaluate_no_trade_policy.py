from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from freqtrade_ext.bot_factory.no_trade_evaluation import (
    build_no_trade_policy_evaluation,
    write_no_trade_policy_evaluation_artifacts,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate local no_trade policy quality from selector replay."
    )
    parser.add_argument("--selector-replay-json", required=True)
    parser.add_argument(
        "--opportunity-cost-thresholds-json",
        help="Optional JSON object keyed by state type.",
    )
    parser.add_argument("--run-id")
    parser.add_argument("--generated-at")
    parser.add_argument("--output-root", default="data/no_trade_evaluations")
    args = parser.parse_args()

    replay = _read_json(Path(args.selector_replay_json))
    thresholds = (
        _read_json(Path(args.opportunity_cost_thresholds_json))
        if args.opportunity_cost_thresholds_json
        else {}
    )
    evaluation = build_no_trade_policy_evaluation(
        selector_replay=replay,
        opportunity_cost_thresholds=thresholds,
        run_id=args.run_id,
        generated_at=args.generated_at,
        source_artifacts={
            "selector_replay": args.selector_replay_json,
            "opportunity_cost_thresholds": args.opportunity_cost_thresholds_json or "",
        },
    )
    paths = write_no_trade_policy_evaluation_artifacts(
        evaluation,
        output_root=Path(args.output_root),
    )
    print(
        json.dumps(
            {
                "run_id": evaluation.get("run_id"),
                "summary_decision": evaluation.get("summary_decision"),
                "reason_codes": evaluation.get("reason_codes"),
                "paths": {key: str(value) for key, value in paths.items()},
                "safety_scope": evaluation.get("safety_scope"),
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
