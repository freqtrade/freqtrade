#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from freqtrade_ext.bot_factory.cost_calibration import (
    CostCalibrationInputs,
    build_cost_calibration,
    write_cost_calibration_artifacts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calibrate Bot Factory execution costs from local artifacts only. "
            "This command writes cost calibration artifacts; it does not "
            "generate strategy candidates, explore theses, run backtests, "
            "start paper/live trading, or call exchange order endpoints."
        )
    )
    parser.add_argument("--ohlcv-path", default=None)
    parser.add_argument("--order-book-path", default=None)
    parser.add_argument("--spread-path", default=None)
    parser.add_argument("--fills-path", default=None)
    parser.add_argument("--pair", default=None)
    parser.add_argument("--timeframe", default=None)
    parser.add_argument("--order-type", default=None)
    parser.add_argument("--liquidity-tier", default=None)
    parser.add_argument("--volatility-regime", default=None)
    parser.add_argument("--fee-bps-entry", type=float, default=3.0)
    parser.add_argument("--fee-bps-exit", type=float, default=3.0)
    parser.add_argument("--cost-calibration-id", default=None)
    parser.add_argument("--output-root", default="data/cost_calibration")
    parser.add_argument("--reviewer-note", action="append", default=[])
    parser.add_argument("--created-at", default=None)
    return parser.parse_args()


def build_inputs_from_args(
    args: argparse.Namespace, *, root_dir: Path = ROOT_DIR
) -> CostCalibrationInputs:
    return CostCalibrationInputs(
        root_dir=root_dir,
        ohlcv_path=Path(args.ohlcv_path) if args.ohlcv_path else None,
        order_book_path=Path(args.order_book_path) if args.order_book_path else None,
        spread_path=Path(args.spread_path) if args.spread_path else None,
        fills_path=Path(args.fills_path) if args.fills_path else None,
        pair=args.pair,
        timeframe=args.timeframe,
        order_type=args.order_type,
        liquidity_tier=args.liquidity_tier,
        volatility_regime=args.volatility_regime,
        fee_bps_entry=args.fee_bps_entry,
        fee_bps_exit=args.fee_bps_exit,
        cost_calibration_id=args.cost_calibration_id,
        output_root=Path(args.output_root),
        reviewer_notes=list(args.reviewer_note or []),
        created_at=args.created_at,
        command=sys.argv,
    )


def main() -> int:
    inputs = build_inputs_from_args(parse_args())
    artifact = build_cost_calibration(inputs)
    json_path, report_path, table_path = write_cost_calibration_artifacts(
        artifact,
        root_dir=ROOT_DIR,
        output_root=inputs.output_root,
    )
    print(
        json.dumps(
            {
                "cost_calibration_path": str(json_path),
                "cost_calibration_report_path": str(report_path),
                "cost_table_path": str(table_path),
                "status": artifact["status"],
                "cost_calibration_id": artifact["cost_calibration_id"],
                "candidate_generation_result": artifact["candidate_generation_result"],
                "candidate_generation_allowed": artifact["candidate_generation_allowed"],
                "blocker_count": len(artifact["blockers"]),
            },
            indent=2,
        )
    )
    return 0 if artifact.get("status") == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
