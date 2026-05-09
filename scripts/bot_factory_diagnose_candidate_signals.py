#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from freqtrade_ext.bot_factory.signal_diagnostics import (
    CandidateSignalDiagnosticsInputs,
    diagnose_candidate_signals,
    write_signal_diagnostics_artifacts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose generated candidate entry-signal bottlenecks from local "
            "historical OHLCV and optional FreqAI prediction artifacts. This "
            "command does not start backtesting, "
            "paper trading, live trading, exchange order placement, or bot "
            "process control."
        )
    )
    parser.add_argument("--generated-metadata-json", required=True)
    ohlcv_group = parser.add_mutually_exclusive_group(required=True)
    ohlcv_group.add_argument("--ohlcv-parquet", dest="ohlcv_path")
    ohlcv_group.add_argument("--ohlcv-csv", dest="ohlcv_path")
    informative_group = parser.add_mutually_exclusive_group(required=False)
    informative_group.add_argument("--informative-ohlcv-parquet", dest="informative_ohlcv_path")
    informative_group.add_argument("--informative-ohlcv-csv", dest="informative_ohlcv_path")
    funding_group = parser.add_mutually_exclusive_group(required=False)
    funding_group.add_argument("--funding-rate-parquet", dest="funding_rate_path")
    funding_group.add_argument("--funding-rate-csv", dest="funding_rate_path")
    parser.add_argument("--freqai-predictions-dir", default=None)
    parser.add_argument("--timerange", default=None)
    parser.add_argument(
        "--entry-edge-hold-candles",
        type=int,
        default=None,
        help=(
            "Optional fixed closed-candle hold horizon used to estimate the "
            "generated entry set's forward edge."
        ),
    )
    parser.add_argument(
        "--entry-edge-all-in-cost-bps",
        type=float,
        default=None,
        help=(
            "Optional all-in cost in basis points subtracted from generated "
            "entry forward returns."
        ),
    )
    parser.add_argument(
        "--entry-edge-min-profitable-windows-ratio",
        type=float,
        default=0.5,
        help="Minimum ratio of chronological windows whose generated-entry net edge is positive.",
    )
    parser.add_argument("--diagnostics-id", default=None)
    parser.add_argument("--output-root", default="registry/strategies/diagnostics")
    parser.add_argument("--reviewer-note", action="append", default=[])
    return parser.parse_args()


def build_inputs_from_args(
    args: argparse.Namespace, *, root_dir: Path = ROOT_DIR
) -> CandidateSignalDiagnosticsInputs:
    return CandidateSignalDiagnosticsInputs(
        root_dir=root_dir,
        generated_metadata_path=Path(args.generated_metadata_json),
        ohlcv_path=Path(args.ohlcv_path),
        informative_ohlcv_path=(
            Path(args.informative_ohlcv_path) if args.informative_ohlcv_path else None
        ),
        funding_rate_path=(
            Path(args.funding_rate_path) if args.funding_rate_path else None
        ),
        freqai_predictions_dir=(
            Path(args.freqai_predictions_dir) if args.freqai_predictions_dir else None
        ),
        output_root=Path(args.output_root),
        diagnostics_id=args.diagnostics_id,
        timerange=args.timerange,
        entry_edge_hold_candles=args.entry_edge_hold_candles,
        entry_edge_all_in_cost_bps=args.entry_edge_all_in_cost_bps,
        entry_edge_min_profitable_windows_ratio=args.entry_edge_min_profitable_windows_ratio,
        reviewer_notes=args.reviewer_note,
    )


def main() -> int:
    inputs = build_inputs_from_args(parse_args())
    diagnostics = diagnose_candidate_signals(inputs)
    diagnostics_path, report_path = write_signal_diagnostics_artifacts(
        diagnostics,
        root_dir=ROOT_DIR,
        output_root=inputs.output_root,
    )
    print(
        json.dumps(
            {
                "signal_diagnostics_path": str(diagnostics_path),
                "signal_diagnostics_report_path": str(report_path),
                "status": diagnostics["status"],
                "entry_count": diagnostics.get("entry_count"),
                "zero_entry_signal": diagnostics.get("zero_entry_signal"),
                "first_zero_component": diagnostics.get("first_zero_component"),
                "diagnosis_codes": diagnostics.get("diagnosis_codes", []),
                "generated_entry_edge": {
                    "status": (diagnostics.get("generated_entry_edge") or {}).get("status"),
                    "sample_count": (diagnostics.get("generated_entry_edge") or {}).get(
                        "sample_count"
                    ),
                    "net_edge_bps": (diagnostics.get("generated_entry_edge") or {}).get(
                        "net_edge_bps"
                    ),
                    "profitable_windows_ratio": (
                        diagnostics.get("generated_entry_edge") or {}
                    ).get("profitable_windows_ratio"),
                },
            },
            indent=2,
        )
    )
    return 0 if diagnostics.get("status") == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
