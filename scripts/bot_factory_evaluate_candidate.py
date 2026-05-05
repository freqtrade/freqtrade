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
    p.add_argument("--static-check-json", default=None)
    p.add_argument("--freqai-validation-json", default=None)
    p.add_argument("--ohlcv-quality-json", default=None)
    p.add_argument("--backtest-metrics-json", default=None)
    p.add_argument("--walk-forward-metrics-json", default=None)
    p.add_argument("--training-manifest-json", default=None)
    return p.parse_args()


def main() -> int:
    a = parse_args()
    inputs = CandidateEvaluationInputs(
        root_dir=ROOT_DIR,
        proposal_metadata_path=Path(a.proposal_metadata_json),
        generated_metadata_path=Path(a.generated_metadata_json),
        candidate_id=a.candidate_id,
        static_check_path=Path(a.static_check_json) if a.static_check_json else None,
        freqai_validation_path=Path(a.freqai_validation_json) if a.freqai_validation_json else None,
        ohlcv_quality_path=Path(a.ohlcv_quality_json) if a.ohlcv_quality_json else None,
        backtest_metrics_path=Path(a.backtest_metrics_json) if a.backtest_metrics_json else None,
        walk_forward_metrics_path=Path(a.walk_forward_metrics_json) if a.walk_forward_metrics_json else None,
        training_manifest_path=Path(a.training_manifest_json) if a.training_manifest_json else None,
    )
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


if __name__ == "__main__":
    raise SystemExit(main())
