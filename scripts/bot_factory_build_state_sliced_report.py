from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from freqtrade_ext.bot_factory.state_sliced_reporting import (
    build_state_sliced_evaluation_report,
    write_state_sliced_evaluation_artifacts,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a local state-sliced strategy evaluation report."
    )
    parser.add_argument("--state-scorecard-json", required=True)
    parser.add_argument("--historical-metrics-json")
    parser.add_argument("--walk-forward-metrics-json")
    parser.add_argument("--expected-state-id", action="append", default=[])
    parser.add_argument("--candidate-style", default="scalp")
    parser.add_argument("--incumbent-baseline-json")
    parser.add_argument("--style-baseline-json")
    parser.add_argument("--max-state-drawdown-pct", type=float)
    parser.add_argument("--run-id")
    parser.add_argument("--generated-at")
    parser.add_argument("--output-root", default="data/state_sliced_evaluations")
    args = parser.parse_args()

    state_scorecard = _read_json(Path(args.state_scorecard_json))
    historical_metrics = (
        _read_json(Path(args.historical_metrics_json))
        if args.historical_metrics_json
        else {}
    )
    walk_forward_metrics = (
        _read_json(Path(args.walk_forward_metrics_json))
        if args.walk_forward_metrics_json
        else {}
    )
    incumbent_baseline = (
        _read_json(Path(args.incumbent_baseline_json))
        if args.incumbent_baseline_json
        else {}
    )
    style_baseline = (
        _read_json(Path(args.style_baseline_json))
        if args.style_baseline_json
        else {}
    )
    evaluation = build_state_sliced_evaluation_report(
        state_scorecard=state_scorecard,
        historical_metrics=historical_metrics,
        walk_forward_metrics=walk_forward_metrics,
        expected_state_ids=args.expected_state_id,
        candidate_style=args.candidate_style,
        incumbent_baseline_by_state=incumbent_baseline,
        style_baseline_by_state=style_baseline,
        max_state_drawdown_pct=args.max_state_drawdown_pct,
        run_id=args.run_id,
        generated_at=args.generated_at,
        source_artifacts={
            "state_scorecard": args.state_scorecard_json,
            "historical_metrics": args.historical_metrics_json or "",
            "walk_forward_metrics": args.walk_forward_metrics_json or "",
            "incumbent_baseline": args.incumbent_baseline_json or "",
            "style_baseline": args.style_baseline_json or "",
        },
    )
    paths = write_state_sliced_evaluation_artifacts(
        evaluation,
        output_root=Path(args.output_root),
    )
    print(
        json.dumps(
            {
                "run_id": evaluation.get("run_id"),
                "candidate_id": evaluation.get("candidate_id"),
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
