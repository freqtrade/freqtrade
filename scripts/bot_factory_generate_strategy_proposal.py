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

from freqtrade_ext.bot_factory.strategy_proposals import (
    StrategyProposalEvidenceInput,
    StrategyProposalInputs,
    build_strategy_proposal,
    write_strategy_proposal_artifacts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a Bot Factory strategy proposal Markdown artifact and "
            "sidecar metadata from explicit local inputs only. This command "
            "does not generate code, run backtests, start paper/live trading, "
            "call exchange order endpoints, promote candidates, or manage any "
            "bot process."
        )
    )
    parser.add_argument("--strategy-name", required=True)
    parser.add_argument("--strategy-type", required=True)
    parser.add_argument("--target-exchange", required=True)
    parser.add_argument("--target-symbol", action="append", required=True)
    parser.add_argument("--timeframe", required=True)
    parser.add_argument("--spot-or-futures", required=True, choices=["spot", "futures"])
    parser.add_argument("--long-short", default="long-only")
    parser.add_argument("--summary", required=True)
    parser.add_argument("--hypothesis", required=True)
    parser.add_argument("--market-condition", required=True)
    parser.add_argument("--entry-logic", required=True)
    parser.add_argument("--exit-logic", required=True)
    parser.add_argument("--risk-logic", required=True)
    parser.add_argument("--required-data", action="append", required=True)
    parser.add_argument("--parameters", action="append", required=True)
    parser.add_argument("--expected-failure-case", action="append", required=True)
    parser.add_argument("--backtest-plan", required=True)
    parser.add_argument("--rejection-condition", action="append", required=True)
    parser.add_argument("--generator-mode", default="rule_based", choices=["rule_based", "freqai", "hybrid_ml"])
    parser.add_argument("--thesis-id", default=None)
    parser.add_argument("--thesis-type", default=None)
    parser.add_argument("--thesis-statement", default=None)
    parser.add_argument("--falsification-criteria", default=None)
    parser.add_argument("--novelty-vs-previous", default=None)
    parser.add_argument("--evidence-ref", action="append", default=None)
    parser.add_argument(
        "--failure-taxonomy-code",
        action="append",
        default=None,
        choices=["FAIL_OVERFIT_WF_GAP", "FAIL_COST_SENSITIVE", "FAIL_REGIME_FRAGILE"],
    )
    parser.add_argument("--retry-budget-per-thesis", type=int, default=3)
    parser.add_argument("--thesis-retry-count", type=int, default=0)
    parser.add_argument("--parameter-only-retry-limit", type=int, default=1)
    parser.add_argument("--parameter-only-retry-count", type=int, default=0)
    parser.add_argument("--force-distinct-hypothesis-family", action="store_true")
    parser.add_argument(
        "--strategy-logic-variant",
        default=None,
        choices=["mean_reversion_pullback", "trend_continuation", "volatility_breakout"],
    )
    parser.add_argument("--feature", action="append", default=None)
    parser.add_argument("--target-definition", default=None)
    parser.add_argument("--label-horizon", type=int, default=None)
    parser.add_argument("--prediction-threshold", type=float, default=None)
    parser.add_argument("--rule-filter", action="append", default=None)
    parser.add_argument("--risk-policy", default="long_only_leverage_1")
    parser.add_argument("--reviewer-note", action="append", default=None)
    parser.add_argument("--output-root", default="registry/strategies/proposals")
    parser.add_argument("--created-by-agent", default="codex")
    parser.add_argument("--created-at", default=None)
    parser.add_argument("--ohlcv-quality-json", action="append", default=None)
    parser.add_argument("--previous-metrics-json", action="append", default=None)
    parser.add_argument("--walk-forward-metrics-json", action="append", default=None)
    parser.add_argument("--training-manifest-json", action="append", default=None)
    parser.add_argument("--reviewer-notes-path", action="append", default=None)
    parser.add_argument(
        "--evidence-path",
        action="append",
        default=None,
        help="Additional local evidence as LABEL=PATH. Can be repeated.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    created_at = args.created_at or datetime.now(UTC).replace(microsecond=0).isoformat()
    inputs = StrategyProposalInputs(
        root_dir=ROOT_DIR,
        strategy_name=args.strategy_name,
        strategy_type=args.strategy_type,
        target_exchange=args.target_exchange,
        target_symbols=args.target_symbol,
        timeframe=args.timeframe,
        spot_or_futures=args.spot_or_futures,
        long_short=args.long_short,
        summary=args.summary,
        hypothesis=args.hypothesis,
        market_condition=args.market_condition,
        entry_logic=args.entry_logic,
        exit_logic=args.exit_logic,
        risk_logic=args.risk_logic,
        required_data=list(args.required_data),
        parameters=list(args.parameters),
        expected_failure_cases=list(args.expected_failure_case),
        backtest_plan=args.backtest_plan,
        rejection_conditions=list(args.rejection_condition),
        generator_mode=args.generator_mode,
        thesis_id=args.thesis_id,
        thesis_type=args.thesis_type,
        thesis_statement=args.thesis_statement,
        falsification_criteria=args.falsification_criteria,
        novelty_vs_previous=args.novelty_vs_previous,
        evidence_refs=list(args.evidence_ref or []),
        failure_taxonomy_codes=list(args.failure_taxonomy_code or []),
        retry_budget_per_thesis=args.retry_budget_per_thesis,
        thesis_retry_count=args.thesis_retry_count,
        parameter_only_retry_limit=args.parameter_only_retry_limit,
        parameter_only_retry_count=args.parameter_only_retry_count,
        force_distinct_hypothesis_family=args.force_distinct_hypothesis_family,
        strategy_logic_variant=args.strategy_logic_variant,
        feature_list=list(args.feature or []),
        target_definition=args.target_definition,
        label_horizon=args.label_horizon,
        prediction_threshold=args.prediction_threshold,
        rule_filters=list(args.rule_filter or []),
        risk_policy=args.risk_policy,
        reviewer_notes=list(args.reviewer_note or []),
        evidence_paths=_evidence_inputs(args),
        output_root=Path(args.output_root),
        created_by_agent=args.created_by_agent,
        created_at=created_at,
        command=[sys.executable, *sys.argv],
    )
    artifacts = build_strategy_proposal(inputs)
    write_strategy_proposal_artifacts(artifacts)

    print(
        json.dumps(
            {
                "status": artifacts.metadata["status"],
                "proposal_path": str(artifacts.proposal_path),
                "metadata_path": str(artifacts.metadata_path),
                "code_generation_eligible": artifacts.metadata[
                    "code_generation_eligible"
                ],
                "generator_mode": artifacts.metadata["generator_mode"],
                "strategy_logic_variant": artifacts.metadata["strategy_logic_variant"],
                "thesis_id": artifacts.metadata["thesis_id"],
            },
            indent=2,
        )
    )
    print(f"Strategy proposal artifacts written: {artifacts.proposal_path.parent}")
    return 0 if artifacts.metadata["status"] == "accepted" else 1


def _evidence_inputs(args: argparse.Namespace) -> list[StrategyProposalEvidenceInput]:
    evidence: list[StrategyProposalEvidenceInput] = []
    for label, values in [
        ("ohlcv_quality", args.ohlcv_quality_json),
        ("previous_metrics", args.previous_metrics_json),
        ("walk_forward_metrics", args.walk_forward_metrics_json),
        ("training_manifest", args.training_manifest_json),
        ("reviewer_notes", args.reviewer_notes_path),
    ]:
        for value in values or []:
            evidence.append(StrategyProposalEvidenceInput(label=label, path=Path(value)))

    for index, value in enumerate(args.evidence_path or [], start=1):
        if "=" in value:
            label, path = value.split("=", 1)
        else:
            label, path = f"evidence_{index}", value
        evidence.append(
            StrategyProposalEvidenceInput(label=label.strip(), path=Path(path.strip()))
        )
    return evidence


if __name__ == "__main__":
    sys.exit(main())
