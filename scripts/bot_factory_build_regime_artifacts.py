#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from freqtrade_ext.bot_factory.evidence_pipeline import (
    BacktestEvidencePipelineInputs,
    build_backtest_evidence_pipeline,
    write_backtest_evidence_pipeline_artifacts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build local regime evidence artifacts from checked backtest outputs. "
            "This does not start paper, dry-run, live, or exchange-facing processes."
        )
    )
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--trades", required=True)
    parser.add_argument("--ohlcv", required=True)
    parser.add_argument("--strategy", required=True)
    parser.add_argument("--pair", required=True)
    parser.add_argument("--timeframe", required=True)
    parser.add_argument("--output-root", default="data/regime_evidence")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--candidate-id", default=None)
    parser.add_argument("--logic-id", default="backtest_derived_logic_v1")
    parser.add_argument("--candidate-style", default="intraday_trend_following")
    parser.add_argument("--intended-regime", action="append", default=None)
    parser.add_argument("--excluded-regime", action="append", default=None)
    parser.add_argument("--normal-cost-bps", type=float, default=10.0)
    parser.add_argument("--stress-cost-bps", type=float, default=20.0)
    parser.add_argument("--reviewer-note", action="append", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    inputs = BacktestEvidencePipelineInputs(
        root_dir=ROOT_DIR,
        metrics_path=Path(args.metrics),
        trades_path=Path(args.trades),
        ohlcv_path=Path(args.ohlcv),
        strategy=args.strategy,
        pair=args.pair,
        timeframe=args.timeframe,
        output_root=Path(args.output_root),
        run_id=args.run_id,
        candidate_id=args.candidate_id,
        logic_id=args.logic_id,
        candidate_style=args.candidate_style,
        intended_regimes=tuple(args.intended_regime or ["trend_up"]),
        excluded_regimes=tuple(
            args.excluded_regime
            or ["trend_down", "range", "high_volatility", "liquidity_stress", "unknown"]
        ),
        normal_cost_bps=args.normal_cost_bps,
        stress_cost_bps=args.stress_cost_bps,
        reviewer_notes=tuple(args.reviewer_note or []),
    )
    pipeline = build_backtest_evidence_pipeline(inputs)
    paths = write_backtest_evidence_pipeline_artifacts(
        pipeline,
        root_dir=ROOT_DIR,
        output_root=Path(args.output_root),
    )
    print(json.dumps({key: str(value) for key, value in paths.items()}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
