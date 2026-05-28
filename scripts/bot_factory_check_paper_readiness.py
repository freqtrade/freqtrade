#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from freqtrade_ext.bot_factory.freqai_backtest import load_json_config
from freqtrade_ext.bot_factory.paper import (
    PaperReadinessInputs,
    evaluate_paper_readiness,
    load_json_file,
    write_paper_readiness_artifacts,
)
from freqtrade_ext.bot_factory.safety import Finding, SafetyReport, scan_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check Bot Factory Phase 3 paper readiness from local artifacts only. "
            "This command does not start paper, dry-run, live, or any bot process."
        )
    )
    parser.add_argument("--config", required=True, help="Proposed dry-run paper config to inspect.")
    parser.add_argument("--strategy", required=True)
    parser.add_argument("--strategy-path", default="user_data/strategies")
    parser.add_argument("--historical-dir", required=True)
    parser.add_argument("--walk-forward-dir", required=True)
    parser.add_argument("--training-dir", required=True)
    parser.add_argument("--regime-scorecard", default=None)
    parser.add_argument("--requires-regime-scorecard", action="store_true")
    parser.add_argument("--market-state-scorecard", default=None)
    parser.add_argument("--requires-market-state-scorecard", action="store_true")
    parser.add_argument("--strategy-suitability-matrix", default=None)
    parser.add_argument("--requires-strategy-suitability-matrix", action="store_true")
    parser.add_argument("--output-root", default="data/paper_readiness")
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--reviewer-note",
        action="append",
        default=None,
        help="Reviewer note required before readiness can pass. Can be repeated.",
    )
    parser.add_argument(
        "--static-check-json",
        default=None,
        help="Optional existing static-check JSON to consume instead of running scan_paths.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = Path(args.config)
    _require_file(config_path, "config")
    config = load_json_config(config_path)
    strategy_file = _find_strategy_source(Path(args.strategy_path), args.strategy)
    _require_file(strategy_file, "strategy source")

    run_id = args.run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    inputs = PaperReadinessInputs(
        root_dir=ROOT_DIR,
        strategy=args.strategy,
        run_id=run_id,
        config_path=config_path,
        strategy_path=Path(args.strategy_path),
        historical_dir=Path(args.historical_dir),
        walk_forward_dir=Path(args.walk_forward_dir),
        training_dir=Path(args.training_dir),
        regime_scorecard_path=Path(args.regime_scorecard) if args.regime_scorecard else None,
        requires_regime_scorecard=args.requires_regime_scorecard,
        market_state_scorecard_path=Path(args.market_state_scorecard)
        if args.market_state_scorecard
        else None,
        requires_market_state_scorecard=args.requires_market_state_scorecard,
        strategy_suitability_matrix_path=Path(args.strategy_suitability_matrix)
        if args.strategy_suitability_matrix
        else None,
        requires_strategy_suitability_matrix=args.requires_strategy_suitability_matrix,
        output_root=Path(args.output_root),
        reviewer_notes=list(args.reviewer_note or []),
        command=[sys.executable, *sys.argv],
    )

    static_report = (
        _load_static_report(Path(args.static_check_json))
        if args.static_check_json
        else scan_paths([strategy_file])
    )

    readiness, candidate_artifacts, config_safety = evaluate_paper_readiness(
        inputs,
        static_report=static_report,
        config=config,
        strategy_file=strategy_file,
    )
    write_paper_readiness_artifacts(
        inputs=inputs,
        readiness=readiness,
        candidate_artifacts=candidate_artifacts,
        config_safety=config_safety,
    )

    print(json.dumps({"readiness": readiness["readiness"], "output_dir": str(inputs.output_dir)}, indent=2))
    print(f"Paper readiness artifacts written: {inputs.output_dir}")
    return 1 if readiness["readiness"] == "blocked" else 0


def _load_static_report(path: Path) -> SafetyReport:
    payload = load_json_file(path)
    findings = [
        Finding(
            path=str(item.get("path", "")),
            line=int(item.get("line", 0) or 0),
            rule=str(item.get("rule", "")),
            severity=str(item.get("severity", "")),
            message=str(item.get("message", "")),
        )
        for item in payload.get("findings", [])
        if isinstance(item, dict)
    ]
    return SafetyReport(
        ok=bool(payload.get("ok")),
        files_checked=int(payload.get("files_checked", 0) or 0),
        findings=findings,
    )


def _find_strategy_source(strategy_path: Path, strategy_name: str) -> Path:
    exact = strategy_path / f"{strategy_name}.py"
    if exact.exists():
        return exact
    if strategy_path.is_file():
        return strategy_path
    if not strategy_path.is_dir():
        return strategy_path

    for file_path in sorted(strategy_path.rglob("*.py")):
        try:
            tree = ast.parse(file_path.read_text(encoding="utf-8"), filename=str(file_path))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == strategy_name:
                return file_path
    return strategy_path


def _require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise SystemExit(f"{label} file not found: {path}")


if __name__ == "__main__":
    sys.exit(main())
