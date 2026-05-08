#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from freqtrade_ext.bot_factory.freqai_prediction_diagnostics import (
    FreqAIPredictionDiagnosticsInputs,
    diagnose_freqai_predictions,
    write_freqai_prediction_diagnostics_artifacts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose local FreqAI prediction artifacts for a generated "
            "candidate. This command reads local files only and does not run "
            "backtests, training, paper/live trading, or bot process control."
        )
    )
    parser.add_argument("--generated-metadata-json", required=True)
    parser.add_argument("--predictions-dir", required=True)
    parser.add_argument("--signal-diagnostics-json", default=None)
    parser.add_argument("--freqai-metadata-json", default=None)
    parser.add_argument("--training-manifest-json", default=None)
    parser.add_argument("--diagnostics-id", default=None)
    parser.add_argument("--output-root", default="registry/strategies/diagnostics")
    parser.add_argument("--reviewer-note", action="append", default=[])
    return parser.parse_args()


def build_inputs_from_args(
    args: argparse.Namespace, *, root_dir: Path = ROOT_DIR
) -> FreqAIPredictionDiagnosticsInputs:
    return FreqAIPredictionDiagnosticsInputs(
        root_dir=root_dir,
        generated_metadata_path=Path(args.generated_metadata_json),
        predictions_dir=Path(args.predictions_dir),
        output_root=Path(args.output_root),
        diagnostics_id=args.diagnostics_id,
        signal_diagnostics_path=Path(args.signal_diagnostics_json)
        if args.signal_diagnostics_json
        else None,
        freqai_metadata_path=Path(args.freqai_metadata_json)
        if args.freqai_metadata_json
        else None,
        training_manifest_path=Path(args.training_manifest_json)
        if args.training_manifest_json
        else None,
        reviewer_notes=args.reviewer_note,
    )


def main() -> int:
    inputs = build_inputs_from_args(parse_args())
    diagnostics = diagnose_freqai_predictions(inputs)
    diagnostics_path, report_path = write_freqai_prediction_diagnostics_artifacts(
        diagnostics,
        root_dir=ROOT_DIR,
        output_root=inputs.output_root,
    )
    print(
        json.dumps(
            {
                "freqai_prediction_diagnostics_path": str(diagnostics_path),
                "freqai_prediction_diagnostics_report_path": str(report_path),
                "status": diagnostics["status"],
                "expected_target_column": diagnostics["expected_target_column"],
                "expected_target_column_present": diagnostics[
                    "expected_target_column_present"
                ],
                "prediction_file_count": diagnostics["prediction_file_count"],
                "diagnosis_codes": diagnostics["diagnosis_codes"],
            },
            indent=2,
        )
    )
    return 0 if diagnostics.get("status") == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
