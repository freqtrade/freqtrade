#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from freqtrade_ext.bot_factory.candidate_evaluation import (
    CandidateEvaluationInputs,
    evaluate_candidate,
    write_candidate_artifacts,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate Bot Factory candidate artifacts (historical-safe only).")
    p.add_argument("--proposal-metadata-json", required=True)
    p.add_argument("--generated-metadata-json", required=True)
    p.add_argument("--candidate-id", required=True)
    p.add_argument("--config", default=None)
    p.add_argument("--strategy-path", default=None)
    p.add_argument("--ohlcv-parquet", action="append", default=None)
    p.add_argument("--static-check-json", default=None)
    p.add_argument("--freqai-validation-json", default=None)
    p.add_argument("--ohlcv-quality-json", default=None)
    p.add_argument("--funding-rate-quality-json", default=None)
    p.add_argument("--mark-price-quality-json", default=None)
    p.add_argument("--backtest-metrics-json", default=None)
    p.add_argument("--backtest-trades-csv", default=None)
    p.add_argument("--backtest-report-md", default=None)
    p.add_argument("--walk-forward-metrics-json", default=None)
    p.add_argument("--walk-forward-report-md", default=None)
    p.add_argument("--training-manifest-json", default=None)
    p.add_argument("--training-report-md", default=None)
    p.add_argument("--reviewer-note", action="append", default=[])
    p.add_argument(
        "--execute-historical-chain",
        action="store_true",
        help=(
            "Opt in to running the checked historical-safe wrapper chain. "
            "This never starts paper/dry-run/live trading."
        ),
    )
    p.add_argument("--execution-run-id", default=None)
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--timeframe", default=None)
    p.add_argument("--timerange", default=None)
    p.add_argument("--pairs", action="append", default=None)
    p.add_argument("--walk-forward-window", action="append", default=None)
    p.add_argument("--training-timerange", default=None)
    p.add_argument(
        "--freqai-identifier",
        default=None,
        help=(
            "Optional candidate-specific FreqAI identifier. If omitted for ML "
            "candidates, generated metadata or a deterministic candidate id is used."
        ),
    )
    p.add_argument("--execution-output-root", default="registry/strategies/candidates/executions")
    p.add_argument("--backtest-output-root", default="data/backtests")
    p.add_argument("--freqai-output-root", default="data/freqai")
    p.add_argument("--walk-forward-output-root", default="data/walk_forward")
    p.add_argument("--training-output-root", default="data/freqai_training")
    return p.parse_args()


def main() -> int:
    a = parse_args()
    inputs = build_inputs_from_args(a)
    manifest = evaluate_candidate(inputs)
    manifest_path, index_path = write_candidate_artifacts(
        manifest,
        root_dir=ROOT_DIR,
        output_root=inputs.output_root,
        index_path=inputs.index_path,
    )
    print(f"candidate_manifest={manifest_path}")
    print(f"candidate_index={index_path}")
    print(f"recommendation={manifest['recommendation']}")
    return 0 if manifest["recommendation"] == "pass" else 1


def build_inputs_from_args(
    a: argparse.Namespace,
    *,
    root_dir: Path = ROOT_DIR,
) -> CandidateEvaluationInputs:
    return CandidateEvaluationInputs(
        root_dir=root_dir,
        proposal_metadata_path=Path(a.proposal_metadata_json),
        generated_metadata_path=Path(a.generated_metadata_json),
        candidate_id=a.candidate_id,
        config_path=Path(a.config) if a.config else None,
        strategy_path=Path(a.strategy_path) if a.strategy_path else None,
        ohlcv_parquet_paths=[Path(path) for path in a.ohlcv_parquet or []],
        static_check_path=Path(a.static_check_json) if a.static_check_json else None,
        freqai_validation_path=Path(a.freqai_validation_json) if a.freqai_validation_json else None,
        ohlcv_quality_path=Path(a.ohlcv_quality_json) if a.ohlcv_quality_json else None,
        funding_rate_quality_path=(
            Path(getattr(a, "funding_rate_quality_json", None))
            if getattr(a, "funding_rate_quality_json", None)
            else None
        ),
        mark_price_quality_path=(
            Path(getattr(a, "mark_price_quality_json", None))
            if getattr(a, "mark_price_quality_json", None)
            else None
        ),
        backtest_metrics_path=Path(a.backtest_metrics_json) if a.backtest_metrics_json else None,
        backtest_trades_path=Path(a.backtest_trades_csv) if a.backtest_trades_csv else None,
        backtest_report_path=Path(a.backtest_report_md) if a.backtest_report_md else None,
        walk_forward_metrics_path=Path(a.walk_forward_metrics_json) if a.walk_forward_metrics_json else None,
        walk_forward_report_path=Path(a.walk_forward_report_md) if a.walk_forward_report_md else None,
        training_manifest_path=Path(a.training_manifest_json) if a.training_manifest_json else None,
        training_report_path=Path(a.training_report_md) if a.training_report_md else None,
        reviewer_notes=a.reviewer_note,
        execute_historical_chain=a.execute_historical_chain,
        execution_run_id=a.execution_run_id,
        python_executable=a.python,
        timeframe=a.timeframe,
        timerange=a.timerange,
        pairs=a.pairs,
        walk_forward_windows=a.walk_forward_window,
        training_timerange=a.training_timerange,
        freqai_identifier=a.freqai_identifier,
        execution_output_root=Path(a.execution_output_root),
        backtest_output_root=Path(a.backtest_output_root),
        freqai_output_root=Path(a.freqai_output_root),
        walk_forward_output_root=Path(a.walk_forward_output_root),
        training_output_root=Path(a.training_output_root),
    )


if __name__ == "__main__":
    raise SystemExit(main())
