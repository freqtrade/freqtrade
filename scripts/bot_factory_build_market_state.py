#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from freqtrade_ext.bot_factory.market_regime import (
    MarketStateConfig,
    RegimeClassifierConfig,
    build_market_state_snapshot_file,
    write_market_state_artifacts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build local multi-horizon market-state artifacts from closed-candle "
            "OHLCV data. This does not start paper, dry-run, live, bot, or "
            "exchange-facing processes."
        )
    )
    parser.add_argument("--ohlcv", required=True)
    parser.add_argument("--pair", required=True)
    parser.add_argument("--base-timeframe", required=True)
    parser.add_argument("--pair-group", default="single_pair")
    parser.add_argument("--output-root", default="data/market_state")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--cost-model-id", default="local_unknown_cost_model")
    parser.add_argument("--horizon", action="append", default=None)
    parser.add_argument("--min-horizon-rows", type=int, default=24)
    parser.add_argument("--classifier-lookback", type=int, default=12)
    parser.add_argument("--classifier-min-rows", type=int, default=24)
    parser.add_argument("--max-staleness-seconds", type=int, default=900)
    parser.add_argument("--confidence-threshold", type=float, default=0.5)
    parser.add_argument("--generated-at", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    now = (
        datetime.fromisoformat(args.generated_at.replace("Z", "+00:00"))
        if args.generated_at
        else datetime.now(UTC)
    )
    if now.tzinfo is None:
        now = now.replace(tzinfo=UTC)
    classifier_config = RegimeClassifierConfig(
        lookback=args.classifier_lookback,
        min_rows=args.classifier_min_rows,
    )
    config = MarketStateConfig(
        horizons=tuple(args.horizon or ("5m", "15m", "1h", "4h", "1d", "1w")),
        min_horizon_rows=args.min_horizon_rows,
        max_staleness_seconds=args.max_staleness_seconds,
        confidence_threshold=args.confidence_threshold,
        regime_classifier_config=classifier_config,
    )
    snapshot = build_market_state_snapshot_file(
        Path(args.ohlcv),
        pair=args.pair,
        base_timeframe=args.base_timeframe,
        pair_group=args.pair_group,
        run_id=args.run_id,
        cost_model_id=args.cost_model_id,
        config=config,
        generated_at=args.generated_at,
        now=now,
    )
    paths = write_market_state_artifacts(
        snapshot,
        output_root=ROOT_DIR / Path(args.output_root),
    )
    print(json.dumps({key: str(value) for key, value in paths.items()}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
