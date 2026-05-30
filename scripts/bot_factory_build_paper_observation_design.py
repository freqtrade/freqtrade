from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from freqtrade_ext.bot_factory.paper_observation_design import (
    build_paper_observation_design,
    write_paper_observation_design_artifacts,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build local paper observation design artifacts without starting any process."
    )
    parser.add_argument(
        "--future-observations-json",
        help="Optional JSON array or object with future_observations rows.",
    )
    parser.add_argument(
        "--paper-observation-metrics-json",
        help="Optional JSON metrics object for drift/quarantine design checks.",
    )
    parser.add_argument("--run-id")
    parser.add_argument("--generated-at")
    parser.add_argument("--persistent-quarantine-count", type=int, default=0)
    parser.add_argument("--output-root", default="data/paper_observation_design")
    args = parser.parse_args()

    future_observations = (
        _read_future_observations(Path(args.future_observations_json))
        if args.future_observations_json
        else []
    )
    metrics = (
        _read_json(Path(args.paper_observation_metrics_json))
        if args.paper_observation_metrics_json
        else {}
    )
    design = build_paper_observation_design(
        future_observations=future_observations,
        paper_observation_metrics=metrics,
        run_id=args.run_id,
        generated_at=args.generated_at,
        persistent_quarantine_count=args.persistent_quarantine_count,
        source_artifacts={
            "future_observations": args.future_observations_json or "",
            "paper_observation_metrics": args.paper_observation_metrics_json or "",
        },
    )
    paths = write_paper_observation_design_artifacts(
        design,
        output_root=Path(args.output_root),
    )
    print(
        json.dumps(
            {
                "run_id": design.get("run_id"),
                "status": design.get("status"),
                "summary_decision": design.get("summary_decision"),
                "reason_codes": design.get("reason_codes"),
                "paths": {key: str(value) for key, value in paths.items()},
                "safety_scope": design.get("safety_scope"),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_future_observations(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("future_observations"), list):
        payload = payload["future_observations"]
    if not isinstance(payload, list):
        raise ValueError("Future observations JSON must be an array or object with future_observations.")
    return [item for item in payload if isinstance(item, dict)]


if __name__ == "__main__":
    raise SystemExit(main())
