#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from freqtrade_ext.bot_factory.local_falsification import (
    LocalFalsificationInputs,
    build_local_falsification,
    write_local_falsification_artifacts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build local pre-proposal falsification evidence from closed-candle "
            "OHLCV and event timestamps. This command writes local artifacts only; "
            "it does not generate strategy code, run backtests, start paper/live "
            "trading, or manage bot processes."
        )
    )
    parser.add_argument("--thesis-id", required=True)
    parser.add_argument("--mechanism-class", required=True)
    parser.add_argument("--ohlcv-path", required=True)
    parser.add_argument("--event-file", required=True)
    parser.add_argument(
        "--event-source-json",
        default=None,
        help=(
            "Optional local_events.json produced by bot_factory_build_local_events.py. "
            "When supplied, the falsification artifact verifies that the event CSV "
            "and OHLCV source match this Bot Factory event source."
        ),
    )
    parser.add_argument(
        "--funding-rate-path",
        default=None,
        help=(
            "Optional closed-candle funding-rate parquet/CSV. When supplied, "
            "the local falsification edge includes realized long funding "
            "payments between entry and exit timestamps."
        ),
    )
    parser.add_argument("--event-time-column", default="date")
    parser.add_argument("--hold-candles", type=int, required=True)
    parser.add_argument("--all-in-cost-bps", type=float, required=True)
    parser.add_argument("--min-sample-count", type=int, default=20)
    parser.add_argument("--min-profitable-windows-ratio", type=float, default=0.5)
    parser.add_argument(
        "--min-calendar-window-count",
        type=int,
        default=0,
        help=(
            "Optional minimum number of quarterly calendar windows required "
            "for local falsification evidence."
        ),
    )
    parser.add_argument(
        "--min-profitable-calendar-windows-ratio",
        type=float,
        default=0.0,
        help=(
            "Optional minimum ratio of profitable quarterly calendar windows "
            "required for local falsification evidence."
        ),
    )
    parser.add_argument(
        "--min-data-span-days",
        type=float,
        default=0.0,
        help="Optional minimum OHLCV coverage span required by the evidence artifact.",
    )
    parser.add_argument("--falsification-id", default=None)
    parser.add_argument(
        "--output-root",
        default="registry/strategies/research_decisions",
    )
    parser.add_argument("--reviewer-note", action="append", default=[])
    parser.add_argument("--created-at", default=None)
    return parser.parse_args()


def build_inputs_from_args(
    args: argparse.Namespace, *, root_dir: Path = ROOT_DIR
) -> LocalFalsificationInputs:
    return LocalFalsificationInputs(
        root_dir=root_dir,
        thesis_id=args.thesis_id,
        mechanism_class=args.mechanism_class,
        ohlcv_path=Path(args.ohlcv_path),
        event_path=Path(args.event_file),
        event_source_path=Path(args.event_source_json) if args.event_source_json else None,
        funding_rate_path=Path(args.funding_rate_path) if args.funding_rate_path else None,
        event_time_column=args.event_time_column,
        hold_candles=args.hold_candles,
        all_in_cost_bps=args.all_in_cost_bps,
        min_sample_count=args.min_sample_count,
        min_profitable_windows_ratio=args.min_profitable_windows_ratio,
        min_calendar_window_count=args.min_calendar_window_count,
        min_profitable_calendar_windows_ratio=(
            args.min_profitable_calendar_windows_ratio
        ),
        min_data_span_days=args.min_data_span_days,
        falsification_id=args.falsification_id,
        output_root=Path(args.output_root),
        reviewer_notes=list(args.reviewer_note or []),
        created_at=args.created_at,
        command=sys.argv,
    )


def main() -> int:
    inputs = build_inputs_from_args(parse_args())
    artifact = build_local_falsification(inputs)
    json_path, report_path = write_local_falsification_artifacts(
        artifact,
        root_dir=ROOT_DIR,
        output_root=inputs.output_root,
    )
    print(
        json.dumps(
            {
                "local_falsification_path": str(json_path),
                "local_falsification_report_path": str(report_path),
                "status": artifact["status"],
                "thesis_id": artifact["thesis_id"],
                "expected_edge_bps": artifact["expected_edge_bps"],
                "expected_price_edge_bps": artifact["expected_price_edge_bps"],
                "expected_funding_adjustment_bps": artifact[
                    "expected_funding_adjustment_bps"
                ],
                "all_in_cost_bps": artifact["all_in_cost_bps"],
                "net_edge_bps": artifact["net_edge_bps"],
                "sample_count": artifact["sample_count"],
                "data_span_days": artifact["data_span_days"],
                "profitable_windows_ratio": artifact["profitable_windows_ratio"],
                "blocker_count": len(artifact["blockers"]),
            },
            indent=2,
        )
    )
    return 0 if artifact.get("status") == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
