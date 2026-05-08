#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from freqtrade_ext.bot_factory.edge_discovery import (
    EdgeDiscoveryInputs,
    build_edge_discovery,
    write_edge_discovery_artifacts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build local Edge Discovery evidence from fixed theory conditions "
            "and closed-candle market data. This command writes local artifacts "
            "only; it does not generate strategy code, run backtests, start "
            "paper/live trading, or manage bot processes."
        )
    )
    parser.add_argument("--ohlcv-path", required=True)
    parser.add_argument("--edge-spec-json", required=True)
    parser.add_argument(
        "--funding-rate-path",
        default=None,
        help="Optional local funding-rate parquet/CSV for features and long funding adjustment.",
    )
    parser.add_argument(
        "--mark-price-path",
        default=None,
        help="Optional local mark-price parquet/CSV for mark_price_* features.",
    )
    parser.add_argument(
        "--informative-ohlcv-path",
        default=None,
        help="Optional local informative OHLCV parquet/CSV for cross-asset features.",
    )
    parser.add_argument(
        "--open-interest-path",
        default=None,
        help="Optional local open-interest parquet/CSV for open_interest* features.",
    )
    parser.add_argument(
        "--open-interest-quality-report-json",
        action="append",
        default=[],
        help=(
            "Required passing open-interest quality report JSON when the edge "
            "spec references open_interest* features."
        ),
    )
    parser.add_argument(
        "--long-short-ratio-path",
        default=None,
        help="Optional local account long/short-ratio parquet/CSV for ratio features.",
    )
    parser.add_argument(
        "--long-short-ratio-quality-report-json",
        action="append",
        default=[],
        help=(
            "Required passing long/short ratio quality report JSON when the "
            "edge spec references ratio features."
        ),
    )
    parser.add_argument(
        "--liquidation-path",
        default=None,
        help="Optional local historical liquidation parquet/CSV for liquidation_* features.",
    )
    parser.add_argument(
        "--liquidation-quality-report-json",
        action="append",
        default=[],
        help=(
            "Required passing liquidation quality report JSON when the edge "
            "spec references liquidation_* features."
        ),
    )
    parser.add_argument(
        "--order-book-path",
        default=None,
        help="Optional local historical order-book snapshot parquet/CSV for order_book_* features.",
    )
    parser.add_argument(
        "--order-book-quality-report-json",
        action="append",
        default=[],
        help=(
            "Required passing order-book quality report JSON when the edge "
            "spec references order_book_* features. Current Bybit REST order "
            "book snapshots are not treated as historical features by this runner."
        ),
    )
    parser.add_argument(
        "--failure-synthesis-json",
        default=None,
        help=(
            "Optional candidate_failure_synthesis.json. When supplied, the edge "
            "spec is blocked if its thesis_id or mechanism_class repeats a "
            "failed thesis/family."
        ),
    )
    parser.add_argument(
        "--allow-failed-thesis-or-family",
        action="store_true",
        help="Allow diagnostics for a thesis/family already listed in failure synthesis.",
    )
    parser.add_argument("--min-sample-count", type=int, default=20)
    parser.add_argument("--min-profitable-windows-ratio", type=float, default=0.5)
    parser.add_argument("--min-calendar-window-count", type=int, default=0)
    parser.add_argument(
        "--min-profitable-calendar-windows-ratio",
        type=float,
        default=0.0,
    )
    parser.add_argument("--min-data-span-days", type=float, default=0.0)
    parser.add_argument("--min-passing-horizon-count", type=int, default=1)
    parser.add_argument("--max-horizon-count", type=int, default=5)
    parser.add_argument("--min-negative-control-delta-bps", type=float, default=1.0)
    parser.add_argument("--edge-discovery-id", default=None)
    parser.add_argument(
        "--output-root",
        default="registry/strategies/research_decisions",
    )
    parser.add_argument("--reviewer-note", action="append", default=[])
    parser.add_argument("--created-at", default=None)
    return parser.parse_args()


def build_inputs_from_args(
    args: argparse.Namespace, *, root_dir: Path = ROOT_DIR
) -> EdgeDiscoveryInputs:
    return EdgeDiscoveryInputs(
        root_dir=root_dir,
        ohlcv_path=Path(args.ohlcv_path),
        edge_spec_path=Path(args.edge_spec_json),
        funding_rate_path=Path(args.funding_rate_path) if args.funding_rate_path else None,
        mark_price_path=Path(args.mark_price_path) if args.mark_price_path else None,
        informative_ohlcv_path=(
            Path(args.informative_ohlcv_path) if args.informative_ohlcv_path else None
        ),
        open_interest_path=Path(args.open_interest_path) if args.open_interest_path else None,
        open_interest_quality_report_paths=[
            Path(path) for path in args.open_interest_quality_report_json or []
        ],
        long_short_ratio_path=Path(args.long_short_ratio_path)
        if args.long_short_ratio_path
        else None,
        long_short_ratio_quality_report_paths=[
            Path(path) for path in args.long_short_ratio_quality_report_json or []
        ],
        liquidation_path=Path(args.liquidation_path) if args.liquidation_path else None,
        liquidation_quality_report_paths=[
            Path(path) for path in args.liquidation_quality_report_json or []
        ],
        order_book_path=Path(args.order_book_path) if args.order_book_path else None,
        order_book_quality_report_paths=[
            Path(path) for path in args.order_book_quality_report_json or []
        ],
        failure_synthesis_path=(
            Path(args.failure_synthesis_json) if args.failure_synthesis_json else None
        ),
        allow_failed_thesis_or_family=bool(args.allow_failed_thesis_or_family),
        min_sample_count=args.min_sample_count,
        min_profitable_windows_ratio=args.min_profitable_windows_ratio,
        min_calendar_window_count=args.min_calendar_window_count,
        min_profitable_calendar_windows_ratio=(
            args.min_profitable_calendar_windows_ratio
        ),
        min_data_span_days=args.min_data_span_days,
        min_passing_horizon_count=args.min_passing_horizon_count,
        max_horizon_count=args.max_horizon_count,
        min_negative_control_delta_bps=getattr(
            args,
            "min_negative_control_delta_bps",
            1.0,
        ),
        edge_discovery_id=args.edge_discovery_id,
        output_root=Path(args.output_root),
        reviewer_notes=list(args.reviewer_note or []),
        created_at=args.created_at,
        command=sys.argv,
    )


def main() -> int:
    inputs = build_inputs_from_args(parse_args())
    artifact = build_edge_discovery(inputs)
    json_path, report_path = write_edge_discovery_artifacts(
        artifact,
        root_dir=ROOT_DIR,
        output_root=inputs.output_root,
    )
    print(
        json.dumps(
            {
                "edge_discovery_path": str(json_path),
                "edge_discovery_report_path": str(report_path),
                "status": artifact["status"],
                "edge_discovery_id": artifact["edge_discovery_id"],
                "thesis_id": artifact["thesis_id"],
                "passing_horizon_count": artifact["passing_horizon_count"],
                "best_horizon_by_net_edge": artifact["best_horizon_by_net_edge"],
                "proposal_generation_allowed": artifact["proposal_generation_allowed"],
                "strategy_codegen_allowed": artifact["strategy_codegen_allowed"],
                "blocker_count": len(artifact["blockers"]),
            },
            indent=2,
        )
    )
    return 0 if artifact.get("status") == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
