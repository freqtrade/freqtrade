from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Sequence

from freqtrade_ext.bot_factory.backtest_results import BacktestMetrics, evaluate_initial_gate
from freqtrade_ext.bot_factory.freqai_backtest import (
    candidate_freqai_identifier,
    sanitize_freqai_identifier,
)

RESEARCH_HANDOFF_KEYS = (
    "research_decision_question_handoff",
    "research_decision_novelty_handoff",
    "local_falsification_handoff",
    "structural_data_quality_handoff",
    "structural_data_capability_handoff",
)
PARAMETER_OPTIMIZATION_POLICY = "theory_fixed_parameters_no_freqtrade_hyperopt"


@dataclass(frozen=True)
class CandidateEvaluationInputs:
    root_dir: Path
    proposal_metadata_path: Path
    generated_metadata_path: Path
    candidate_id: str
    output_root: Path = Path("registry/strategies/candidates")
    index_path: Path = Path("registry/strategies/candidates/index.jsonl")
    config_path: Path | None = None
    strategy_path: Path | None = None
    ohlcv_parquet_paths: list[Path] | None = None
    static_check_path: Path | None = None
    freqai_validation_path: Path | None = None
    ohlcv_quality_path: Path | None = None
    funding_rate_quality_path: Path | None = None
    mark_price_quality_path: Path | None = None
    backtest_metrics_path: Path | None = None
    backtest_trades_path: Path | None = None
    backtest_report_path: Path | None = None
    walk_forward_metrics_path: Path | None = None
    walk_forward_report_path: Path | None = None
    training_manifest_path: Path | None = None
    training_report_path: Path | None = None
    reviewer_notes: list[str] | None = None
    execute_historical_chain: bool = False
    execution_run_id: str | None = None
    python_executable: str = sys.executable
    timeframe: str | None = None
    timerange: str | None = None
    pairs: list[str] | None = None
    walk_forward_windows: list[str] | None = None
    training_timerange: str | None = None
    freqai_identifier: str | None = None
    execution_output_root: Path = Path("registry/strategies/candidates/executions")
    backtest_output_root: Path = Path("data/backtests")
    freqai_output_root: Path = Path("data/freqai")
    walk_forward_output_root: Path = Path("data/walk_forward")
    training_output_root: Path = Path("data/freqai_training")
    command_runner: Callable[[Sequence[str], Path], Any] | None = field(
        default=None, repr=False, compare=False
    )


def evaluate_candidate(inputs: CandidateEvaluationInputs) -> dict[str, Any]:
    root = inputs.root_dir.resolve()
    proposal = _load_json(_resolve(inputs.proposal_metadata_path, root))
    generated = _load_json(_resolve(inputs.generated_metadata_path, root))

    generator_mode = str(generated.get("generator_mode") or proposal.get("generator_mode") or "rule_based").strip()
    ml_mode_required = generator_mode in {"freqai", "hybrid_ml"}
    strategy_name = str(generated.get("strategy_name") or proposal.get("strategy_name") or "")
    candidate_execution, executed_paths = _execute_historical_chain_if_requested(
        inputs,
        proposal=proposal,
        generated=generated,
        generator_mode=generator_mode,
        strategy_name=strategy_name,
        root=root,
    )
    static_check_path = inputs.static_check_path or executed_paths.get("static_check")
    freqai_validation_path = inputs.freqai_validation_path or executed_paths.get("freqai_validation")
    ohlcv_quality_path = inputs.ohlcv_quality_path or executed_paths.get("ohlcv_quality")
    funding_rate_quality_path = inputs.funding_rate_quality_path
    mark_price_quality_path = inputs.mark_price_quality_path
    backtest_metrics_path = inputs.backtest_metrics_path or executed_paths.get("backtest_metrics")
    backtest_trades_path = inputs.backtest_trades_path or executed_paths.get("backtest_trades")
    backtest_report_path = inputs.backtest_report_path or executed_paths.get("backtest_report")
    walk_forward_metrics_path = inputs.walk_forward_metrics_path or executed_paths.get("walk_forward_metrics")
    walk_forward_report_path = inputs.walk_forward_report_path or executed_paths.get("walk_forward_report")
    training_manifest_path = inputs.training_manifest_path or executed_paths.get("training_manifest")
    training_report_path = inputs.training_report_path or executed_paths.get("training_report")
    steps = [
        _step("static_strategy_check", static_check_path, root, key="ok", command_preview=["python", "scripts/bot_factory_static_check.py", _display_path(inputs.strategy_path) or "user_data/strategies"]),
        _step("freqai_feature_label_validation", freqai_validation_path, root, key="ok", required=ml_mode_required, command_preview=["python", "scripts/bot_factory_validate_freqai_strategy.py", _display_path(inputs.strategy_path) or "<strategy_path>"]),
        _step("ohlcv_quality_check", ohlcv_quality_path, root, key="ok", command_preview=["python", "scripts/bot_factory_check_ohlcv.py", "<ohlcv_parquet>", "--timeframe", str(proposal.get("timeframe") or generated.get("timeframe") or "")]),
        _step("funding_rate_quality_check", funding_rate_quality_path, root, key="ok", required=funding_rate_quality_path is not None, command_preview=["python", "scripts/bot_factory_check_funding_rate.py", "<funding_rate_parquet>", "--timeframe", "8h"]),
        _step("mark_price_quality_check", mark_price_quality_path, root, key="ok", required=mark_price_quality_path is not None, command_preview=["python", "scripts/bot_factory_check_mark_price.py", "<mark_price_parquet>", "--timeframe", "4h"]),
        _step("historical_backtest", backtest_metrics_path, root, key="recommendation", pass_values={"pass"}, command_preview=["python", "scripts/bot_factory_run_backtest.py", "--strategy", str(generated.get("strategy_name") or proposal.get("strategy_name") or "")]),
        _strategy_identity_step("historical_strategy_identity", backtest_metrics_path, root, expected_strategy=strategy_name, keys=["strategy_name", "strategy"]),
        _file_step("historical_trades_export", backtest_trades_path, root, command_preview=["python", "scripts/bot_factory_run_backtest.py", "--export", "trades"]),
        _file_step("historical_markdown_report", backtest_report_path, root, command_preview=["python", "scripts/bot_factory_generate_report.py", "<backtest-result-json>"]),
        _step("walk_forward", walk_forward_metrics_path, root, key="recommendation", pass_values={"pass"}, command_preview=["python", "scripts/bot_factory_run_walk_forward.py", "--strategy", str(generated.get("strategy_name") or proposal.get("strategy_name") or "")]),
        _strategy_identity_step("walk_forward_strategy_identity", walk_forward_metrics_path, root, expected_strategy=strategy_name, keys=["strategy"]),
        _file_step("walk_forward_markdown_report", walk_forward_report_path, root, command_preview=["python", "scripts/bot_factory_run_walk_forward.py", "--write-report"]),
        _step("training_factory", training_manifest_path, root, key="recommendation", pass_values={"pass"}, required=ml_mode_required, command_preview=["python", "scripts/bot_factory_run_freqai_training.py", "--strategy", str(generated.get("strategy_name") or proposal.get("strategy_name") or "")]),
        _strategy_identity_step("training_strategy_identity", training_manifest_path, root, expected_strategy=strategy_name, keys=["strategy"], required=ml_mode_required),
        _file_step("training_markdown_report", training_report_path, root, required=ml_mode_required, command_preview=["python", "scripts/bot_factory_run_freqai_training.py", "--write-report"]),
    ]

    checks = [s["check"] for s in steps]
    checks.append(
        _parameter_optimization_policy_check(
            generated,
            _resolve(inputs.generated_metadata_path, root),
            root,
        )
    )

    failures = [c for c in checks if c["status"] in {"fail", "missing"}]
    failure_codes = list(dict.fromkeys(proposal.get("failure_taxonomy_codes", []) + generated.get("failure_taxonomy_codes", [])))
    research_brief = _candidate_research_brief(proposal, generated)
    research_brief_path = _metadata_path(generated.get("research_brief_path"), root)
    research_check = _research_brief_check(research_brief, research_brief_path, root)
    checks.append(research_check)
    failures = [c for c in checks if c["status"] in {"fail", "missing"}]

    parameter_policy_failed = any(
        c["name"] == "generated_parameter_optimization_policy"
        and c["status"] == "fail"
        for c in failures
    )
    if parameter_policy_failed:
        recommendation = "reject"
        recommendation_rationale = (
            "Generated strategy exposes or fails to prove disabled Freqtrade "
            "parameter optimization; reject rather than evaluate a hyperopt surface."
        )
    elif proposal.get("code_generation_eligible") is False or generated.get("candidate_evaluation_eligible") is False:
        recommendation = "reject"
        recommendation_rationale = "Candidate is not eligible for evaluation based on proposal/generated metadata flags."
    elif not failures:
        recommendation = "pass"
        recommendation_rationale = "All required historical-safe checks passed for this generator mode."
    elif any(c["status"] == "missing" for c in failures):
        recommendation = "fail"
        recommendation_rationale = "Required artifacts are missing; regenerate/evaluate required steps before retry."
    elif failure_codes:
        recommendation = "retry"
        recommendation_rationale = "Checks failed with known taxonomy; retry is allowed with thesis-guided iteration input."
    else:
        recommendation = "fail"
        recommendation_rationale = "Checks failed without retry taxonomy guidance."

    manifest = {
        "generated_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "factory": "candidate_evaluation_pipeline",
        "candidate_id": inputs.candidate_id,
        "strategy_name": strategy_name,
        "proposal_metadata_path": _rel(_resolve(inputs.proposal_metadata_path, root), root),
        "generated_metadata_path": _rel(_resolve(inputs.generated_metadata_path, root), root),
        "input_paths": {
            "config": _maybe_rel(inputs.config_path, root),
            "strategy": _maybe_rel(inputs.strategy_path, root) or generated.get("generated_strategy_path"),
            "ohlcv_parquet": [_rel(_resolve(path, root), root) for path in inputs.ohlcv_parquet_paths or []],
            "funding_rate_quality": _maybe_rel(inputs.funding_rate_quality_path, root),
            "mark_price_quality": _maybe_rel(inputs.mark_price_quality_path, root),
        },
        "checks": checks,
        "evaluation_orchestration": {
            "generator_mode": generator_mode,
            "ml_mode_required": ml_mode_required,
            "steps": steps,
            "executed_by_pipeline": False,
            "note": "This pipeline aggregates existing local historical-safe artifacts only and records safe command previews.",
        },
        "candidate_execution": candidate_execution,
        "recommendation": recommendation,
        "recommendation_rationale": recommendation_rationale,
        "failure_taxonomy_codes": failure_codes,
        "research_brief": research_brief,
        "thesis": {
            "thesis_id": proposal.get("thesis_id") or generated.get("thesis_id"),
            "thesis_type": proposal.get("thesis_type") or generated.get("thesis_type"),
            "falsification_criteria": proposal.get("falsification_criteria") or generated.get("falsification_criteria"),
        },
        "next_candidate_input": {
            "thesis_id": proposal.get("thesis_id") or generated.get("thesis_id"),
            "failure_taxonomy_codes": failure_codes,
            "retry_budget_per_thesis": proposal.get("retry_budget_per_thesis"),
            "thesis_retry_count": proposal.get("thesis_retry_count"),
            "parameter_only_retry_count": proposal.get("parameter_only_retry_count"),
            "force_distinct_hypothesis_family": proposal.get("force_distinct_hypothesis_family", False),
            "thesis_statement": proposal.get("thesis_statement") or generated.get("thesis_statement"),
            "evidence_refs": proposal.get("evidence_refs") or generated.get("evidence_refs") or [],
            "research_references": research_brief.get("research_references", []),
            "blocked_next_actions": research_brief.get("blocked_next_actions", []),
        },
        "reviewer_notes": inputs.reviewer_notes or [],
        "safety_scope": {"historical_only": True, "live_trading": False, "process_control": False},
    }
    return manifest


def write_candidate_artifacts(manifest: dict[str, Any], *, root_dir: Path, output_root: Path, index_path: Path) -> tuple[Path, Path]:
    root = root_dir.resolve()
    strategy = _safe_path_component(str(manifest.get("strategy_name") or "unknown_strategy"))
    candidate_id = _safe_path_component(str(manifest["candidate_id"]))
    out_dir = _resolve(output_root, root) / strategy / candidate_id
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "candidate_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    idx_path = _resolve(index_path, root)
    idx_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "created_at": manifest["generated_at"],
        "candidate_id": candidate_id,
        "strategy_name": strategy,
        "recommendation": manifest["recommendation"],
        "status": manifest["recommendation"],
        "failure_taxonomy_codes": manifest.get("failure_taxonomy_codes", []),
        "thesis_id": manifest.get("thesis", {}).get("thesis_id"),
        "manifest_path": _rel(manifest_path, root),
        "recommendation_rationale": manifest.get("recommendation_rationale"),
        "key_metrics": _extract_metrics_summary(manifest),
        "research_reference_count": len(manifest.get("research_brief", {}).get("research_references", [])),
        "artifact_paths": _extract_artifact_paths(manifest),
        "candidate_record_path": _rel(out_dir / "candidate_record.json", root),
        "metrics_summary_path": _rel(out_dir / "metrics_summary.json", root),
        "artifact_paths_path": _rel(out_dir / "artifact_paths.json", root),
        "candidate_report_path": _rel(out_dir / "candidate_report.md", root),
    }

    candidate_record = {
        "candidate_id": candidate_id,
        "strategy_name": strategy,
        "recommendation": manifest["recommendation"],
        "failure_taxonomy_codes": manifest.get("failure_taxonomy_codes", []),
        "thesis": manifest.get("thesis", {}),
        "next_candidate_input": manifest.get("next_candidate_input", {}),
        "research_brief": manifest.get("research_brief", {}),
        "input_paths": manifest.get("input_paths", {}),
        "artifact_paths": _extract_artifact_paths(manifest),
        "metrics_summary": _extract_metrics_summary(manifest),
        "manifest_path": _rel(manifest_path, root),
        "recommendation_rationale": manifest.get("recommendation_rationale"),
    }
    (out_dir / "candidate_record.json").write_text(json.dumps(candidate_record, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "candidate_report.md").write_text(_render_candidate_report(manifest), encoding="utf-8")
    (out_dir / "metrics_summary.json").write_text(json.dumps(_extract_metrics_summary(manifest), indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "artifact_paths.json").write_text(json.dumps(_extract_artifact_paths(manifest), indent=2, ensure_ascii=False), encoding="utf-8")
    with idx_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return manifest_path, idx_path


def _execute_historical_chain_if_requested(
    inputs: CandidateEvaluationInputs,
    *,
    proposal: dict[str, Any],
    generated: dict[str, Any],
    generator_mode: str,
    strategy_name: str,
    root: Path,
) -> tuple[dict[str, Any], dict[str, Path]]:
    plan = _execution_plan(
        inputs,
        proposal=proposal,
        generated=generated,
        generator_mode=generator_mode,
        strategy_name=strategy_name,
        root=root,
    )
    if not inputs.execute_historical_chain:
        return plan, {}
    if plan["status"] == "blocked":
        return plan, {}

    runner = inputs.command_runner or _default_command_runner
    results: list[dict[str, Any]] = []
    for step in plan["steps"]:
        result = runner(step["command"], root)
        normalized = _normalized_run_result(result)
        log_dir = _resolve(Path(step["log_dir"]), root)
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / f"{step['name']}.stdout.log").write_text(
            normalized["stdout"], encoding="utf-8"
        )
        (log_dir / f"{step['name']}.stderr.log").write_text(
            normalized["stderr"], encoding="utf-8"
        )
        (log_dir / f"{step['name']}.command.txt").write_text(
            " ".join(step["command"]), encoding="utf-8"
        )
        results.append(
            {
                "name": step["name"],
                "returncode": normalized["returncode"],
                "stdout_log": _rel(log_dir / f"{step['name']}.stdout.log", root),
                "stderr_log": _rel(log_dir / f"{step['name']}.stderr.log", root),
                "command_log": _rel(log_dir / f"{step['name']}.command.txt", root),
            }
        )
        if normalized["returncode"] != 0:
            plan["status"] = "failed"
            plan["results"] = results
            return plan, _path_artifacts(plan, root)

    plan["status"] = "completed"
    plan["results"] = results
    return plan, _path_artifacts(plan, root)


def _execution_plan(
    inputs: CandidateEvaluationInputs,
    *,
    proposal: dict[str, Any],
    generated: dict[str, Any],
    generator_mode: str,
    strategy_name: str,
    root: Path,
) -> dict[str, Any]:
    run_id = _safe_path_component(inputs.execution_run_id or inputs.candidate_id)
    log_dir = _resolve(inputs.execution_output_root, root) / _safe_path_component(strategy_name) / run_id
    strategy_path = _candidate_strategy_path(inputs, generated, root)
    config_path = _maybe_rel(inputs.config_path, root)
    blockers: list[str] = []
    if inputs.execute_historical_chain and not inputs.config_path:
        blockers.append("config_required_for_execution")
    if inputs.execute_historical_chain and not inputs.timerange:
        blockers.append("timerange_required_for_historical_backtest_execution")
    if inputs.execute_historical_chain and not strategy_path:
        blockers.append("strategy_path_required_for_execution")

    ml_mode = generator_mode in {"freqai", "hybrid_ml"}
    freqai_identifier_value = _freqai_identifier_for_execution(
        inputs,
        generated=generated,
        strategy_name=strategy_name,
        ml_mode=ml_mode,
    )
    artifacts = _expected_execution_artifacts(
        inputs,
        strategy_name=strategy_name,
        run_id=run_id,
        ml_mode=ml_mode,
        root=root,
    )
    steps: list[dict[str, Any]] = []
    if not blockers:
        steps = _execution_steps(
            inputs,
            strategy_name=strategy_name,
            strategy_path=strategy_path,
            config_path=config_path,
            run_id=run_id,
            ml_mode=ml_mode,
            freqai_identifier=freqai_identifier_value,
            artifacts=artifacts,
            log_dir=log_dir,
            root=root,
        )
    return {
        "requested": bool(inputs.execute_historical_chain),
        "status": "blocked" if blockers else ("planned" if not inputs.execute_historical_chain else "ready"),
        "blockers": blockers,
        "run_id": run_id,
        "generator_mode": generator_mode,
        "freqai": {
            "identifier": freqai_identifier_value,
            "identifier_policy": (
                generated.get("freqai_identifier_policy")
                or ("candidate_specific" if ml_mode else "not_applicable")
            ),
            "expected_target_column": generated.get("freqai_expected_target_column"),
            "cache_policy": generated.get("freqai_cache_policy"),
        },
        "executed_by_pipeline": bool(inputs.execute_historical_chain),
        "steps": steps,
        "results": [],
        "artifact_paths": {key: _rel(path, root) for key, path in artifacts.items()},
        "safety_scope": {
            "historical_only": True,
            "uses_checked_wrappers_only": True,
            "freqtrade_trade_started": False,
            "paper_trading_started": False,
            "dry_run_trading_started": False,
            "live_trading": False,
            "exchange_order_placement": False,
            "process_control": False,
        },
    }


def _execution_steps(
    inputs: CandidateEvaluationInputs,
    *,
    strategy_name: str,
    strategy_path: str | None,
    config_path: str | None,
    run_id: str,
    ml_mode: bool,
    freqai_identifier: str | None,
    artifacts: dict[str, Path],
    log_dir: Path,
    root: Path,
) -> list[dict[str, Any]]:
    python = inputs.python_executable
    timeframe = inputs.timeframe or ""
    steps: list[dict[str, Any]] = [
        _execution_step(
            "static_strategy_check",
            [
                python,
                "scripts/bot_factory_static_check.py",
                strategy_path or "user_data/strategies",
                "--output",
                _rel(artifacts["static_check"], root),
            ],
            log_dir,
        )
    ]
    if ml_mode:
        steps.append(
            _execution_step(
                "freqai_feature_label_validation",
                [
                    python,
                    "scripts/bot_factory_validate_freqai_strategy.py",
                    strategy_path or "<strategy_path>",
                    "--output",
                    _rel(artifacts["freqai_validation"], root),
                ],
                log_dir,
            )
        )
    if inputs.ohlcv_parquet_paths:
        steps.append(
            _execution_step(
                "ohlcv_quality_check",
                [
                    python,
                    "scripts/bot_factory_check_ohlcv.py",
                    *[_maybe_rel(path, root) or str(path) for path in inputs.ohlcv_parquet_paths],
                    "--timeframe",
                    timeframe,
                    "--output",
                    _rel(artifacts["ohlcv_quality"], root),
                ],
                log_dir,
            )
        )

    historical_script = (
        "scripts/bot_factory_run_freqai_backtest.py"
        if ml_mode
        else "scripts/bot_factory_run_backtest.py"
    )
    historical_command = [
        python,
        historical_script,
        "--config",
        config_path or "<config>",
        "--strategy",
        strategy_name,
        "--strategy-path",
        strategy_path or "user_data/strategies",
        "--output-root",
        _rel(_resolve(inputs.freqai_output_root if ml_mode else inputs.backtest_output_root, root), root),
        "--run-id",
        f"{run_id}_historical",
        "--python",
        python,
        "--reviewer-note",
        "Candidate evaluation historical execution only; no paper or live promotion.",
    ]
    if ml_mode and freqai_identifier:
        historical_command.extend(["--freqai-identifier", freqai_identifier])
    if inputs.timerange:
        historical_command.extend(["--timerange", inputs.timerange])
    if timeframe:
        historical_command.extend(["--timeframe", timeframe])
    if inputs.pairs:
        historical_command.extend(["--pairs", *inputs.pairs])
    if ml_mode:
        for path in inputs.ohlcv_parquet_paths or []:
            historical_command.extend(["--ohlcv-file", _maybe_rel(path, root) or str(path)])
    steps.append(_execution_step("historical_backtest", historical_command, log_dir))

    if inputs.walk_forward_windows:
        wf_command = [
            python,
            "scripts/bot_factory_run_walk_forward.py",
            "--config",
            config_path or "<config>",
            "--strategy",
            strategy_name,
            "--strategy-path",
            strategy_path or "user_data/strategies",
            "--output-root",
            _rel(_resolve(inputs.walk_forward_output_root, root), root),
            "--run-id",
            f"{run_id}_walk_forward",
            "--python",
            python,
            "--runner-script",
            "scripts/bot_factory_run_freqai_backtest.py" if ml_mode else "scripts/bot_factory_run_backtest.py",
            "--reviewer-note",
            "Candidate evaluation walk-forward execution only; no paper or live promotion.",
        ]
        if ml_mode and freqai_identifier:
            wf_command.extend(["--freqai-identifier", freqai_identifier])
        for window in inputs.walk_forward_windows:
            wf_command.extend(["--window", window])
        if timeframe:
            wf_command.extend(["--timeframe", timeframe])
        if inputs.pairs:
            wf_command.extend(["--pairs", *inputs.pairs])
        if ml_mode:
            for path in inputs.ohlcv_parquet_paths or []:
                wf_command.extend(["--ohlcv-file", _maybe_rel(path, root) or str(path)])
        steps.append(_execution_step("walk_forward", wf_command, log_dir))

    if ml_mode and inputs.training_timerange:
        training_command = [
            python,
            "scripts/bot_factory_run_freqai_training.py",
            "--config",
            config_path or "<config>",
            "--strategy",
            strategy_name,
            "--strategy-path",
            strategy_path or "user_data/strategies",
            "--timerange",
            inputs.training_timerange,
            "--output-root",
            _rel(_resolve(inputs.training_output_root, root), root),
            "--run-id",
            f"{run_id}_training",
            "--python",
            python,
            "--reviewer-note",
            "Candidate evaluation training execution only; no paper or live promotion.",
        ]
        if freqai_identifier:
            training_command.extend(["--freqai-identifier", freqai_identifier])
        if timeframe:
            training_command.extend(["--timeframe", timeframe])
        if inputs.pairs:
            training_command.extend(["--pairs", *inputs.pairs])
        for path in inputs.ohlcv_parquet_paths or []:
            training_command.extend(["--ohlcv-file", _maybe_rel(path, root) or str(path)])
        steps.append(_execution_step("training_factory", training_command, log_dir))

    return steps


def _freqai_identifier_for_execution(
    inputs: CandidateEvaluationInputs,
    *,
    generated: dict[str, Any],
    strategy_name: str,
    ml_mode: bool,
) -> str | None:
    if not ml_mode:
        return None
    if inputs.freqai_identifier:
        return sanitize_freqai_identifier(inputs.freqai_identifier)
    generated_identifier = generated.get("freqai_identifier")
    if generated_identifier:
        return sanitize_freqai_identifier(str(generated_identifier))
    target_definition = str(generated.get("target_definition") or "future_return")
    generated_candidate_id = str(generated.get("candidate_id") or inputs.candidate_id)
    return candidate_freqai_identifier(
        strategy_name,
        generated_candidate_id,
        target_definition,
    )


def _execution_step(name: str, command: list[str], log_dir: Path) -> dict[str, Any]:
    return {
        "name": name,
        "command": command,
        "log_dir": str(log_dir),
    }


def _expected_execution_artifacts(
    inputs: CandidateEvaluationInputs,
    *,
    strategy_name: str,
    run_id: str,
    ml_mode: bool,
    root: Path,
) -> dict[str, Path]:
    execution_dir = _resolve(inputs.execution_output_root, root) / _safe_path_component(strategy_name) / run_id
    historical_root = _resolve(inputs.freqai_output_root if ml_mode else inputs.backtest_output_root, root)
    historical_dir = historical_root / strategy_name / f"{run_id}_historical"
    walk_dir = _resolve(inputs.walk_forward_output_root, root) / strategy_name / f"{run_id}_walk_forward"
    training_dir = _resolve(inputs.training_output_root, root) / strategy_name / f"{run_id}_training"
    artifacts = {
        "static_check": execution_dir / "static_check.json",
        "ohlcv_quality": execution_dir / "ohlcv_quality.json",
        "backtest_metrics": historical_dir / "metrics.json",
        "backtest_trades": historical_dir / "trades.csv",
        "backtest_report": historical_dir / "report.md",
        "walk_forward_metrics": walk_dir / "walk_forward_metrics.json",
        "walk_forward_report": walk_dir / "walk_forward_report.md",
        "training_manifest": training_dir / "training_manifest.json",
        "training_report": training_dir / "training_report.md",
    }
    if ml_mode:
        artifacts["freqai_validation"] = execution_dir / "freqai_validation.json"
    return artifacts


def _candidate_strategy_path(
    inputs: CandidateEvaluationInputs, generated: dict[str, Any], root: Path
) -> str | None:
    if inputs.strategy_path:
        return _maybe_rel(inputs.strategy_path, root)
    raw = generated.get("generated_strategy_path")
    if not raw:
        return None
    path = _resolve(Path(str(raw)), root)
    return _rel(path.parent if path.suffix == ".py" else path, root)


def _path_artifacts(plan: dict[str, Any], root: Path) -> dict[str, Path]:
    return {key: _resolve(Path(value), root) for key, value in plan.get("artifact_paths", {}).items()}


def _default_command_runner(command: Sequence[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(list(command), cwd=cwd, text=True, capture_output=True)


def _normalized_run_result(result: Any) -> dict[str, Any]:
    if isinstance(result, dict):
        return {
            "returncode": int(result.get("returncode", 0)),
            "stdout": str(result.get("stdout") or ""),
            "stderr": str(result.get("stderr") or ""),
        }
    return {
        "returncode": int(getattr(result, "returncode", 0)),
        "stdout": str(getattr(result, "stdout", "") or ""),
        "stderr": str(getattr(result, "stderr", "") or ""),
    }


def _candidate_research_brief(
    proposal: dict[str, Any], generated: dict[str, Any]
) -> dict[str, Any]:
    proposal_brief = (
        proposal.get("research_brief")
        if isinstance(proposal.get("research_brief"), dict)
        else {}
    )
    generated_brief = (
        generated.get("research_brief")
        if isinstance(generated.get("research_brief"), dict)
        else {}
    )
    research_references = _research_references(proposal, generated, generated_brief, proposal_brief)
    blocked_next_actions = _blocked_next_actions(
        generated,
        proposal,
        generated_brief,
        proposal_brief,
    )
    brief = {
        "thesis_id": (
            generated_brief.get("thesis_id")
            or proposal_brief.get("thesis_id")
            or proposal.get("thesis_id")
            or generated.get("thesis_id")
        ),
        "thesis_statement": (
            generated_brief.get("thesis_statement")
            or proposal_brief.get("thesis_statement")
            or proposal.get("thesis_statement")
            or generated.get("thesis_statement")
        ),
        "research_references": research_references,
        "evidence_refs": list(
            generated.get("evidence_refs")
            or proposal.get("evidence_refs")
            or generated_brief.get("evidence_refs")
            or proposal_brief.get("evidence_refs")
            or []
        ),
        "failure_taxonomy_codes": list(
            generated.get("failure_taxonomy_codes")
            or proposal.get("failure_taxonomy_codes")
            or generated_brief.get("failure_taxonomy_codes")
            or proposal_brief.get("failure_taxonomy_codes")
            or []
        ),
        "strategy_logic_variant": (
            generated.get("strategy_logic_variant")
            or proposal.get("strategy_logic_variant")
            or generated_brief.get("strategy_logic_variant")
            or proposal_brief.get("strategy_logic_variant")
        ),
        "novelty_vs_previous": (
            generated.get("novelty_vs_previous")
            or proposal.get("novelty_vs_previous")
            or generated_brief.get("novelty_vs_previous")
            or proposal_brief.get("novelty_vs_previous")
        ),
        "blocked_next_actions": blocked_next_actions,
        "research_handoff_summaries": _research_handoff_summaries(
            generated,
            proposal,
            generated_brief,
            proposal_brief,
        ),
    }
    brief.update(
        _research_handoffs(generated, proposal, generated_brief, proposal_brief)
    )
    return brief


def _research_references(
    proposal: dict[str, Any],
    generated: dict[str, Any],
    generated_brief: dict[str, Any],
    proposal_brief: dict[str, Any],
) -> list[dict[str, Any]]:
    for source in (generated, proposal, generated_brief, proposal_brief):
        refs = source.get("research_references")
        if isinstance(refs, list) and refs:
            return [ref for ref in refs if isinstance(ref, dict)]
    return []


def _research_handoffs(*sources: Any) -> dict[str, Any]:
    handoffs: dict[str, Any] = {}
    for key in RESEARCH_HANDOFF_KEYS:
        for source in sources:
            if not isinstance(source, dict):
                continue
            value = source.get(key)
            if isinstance(value, dict):
                handoffs[key] = value
                break
    return handoffs


def _research_handoff_summaries(*sources: Any) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    seen: set[str] = set()
    for source in sources:
        if not isinstance(source, dict):
            continue
        _extend_unique_handoff_summaries(
            summaries,
            seen,
            source.get("research_handoff_summaries", []),
        )
        for key in ("failure_synthesis_constraints", "research_decision_constraints"):
            raw_constraints = source.get(key, [])
            if not isinstance(raw_constraints, list):
                continue
            for item in raw_constraints:
                if isinstance(item, dict):
                    _extend_unique_handoff_summaries(
                        summaries,
                        seen,
                        item.get("research_handoff_summaries", []),
                    )
    return summaries


def _extend_unique_handoff_summaries(
    target: list[dict[str, Any]], seen: set[str], values: Any
) -> None:
    if not isinstance(values, list):
        return
    for value in values:
        if not isinstance(value, dict):
            continue
        copied = _copy_jsonish(value)
        key = json.dumps(copied, sort_keys=True, ensure_ascii=False)
        if key not in seen:
            seen.add(key)
            target.append(copied)


def _blocked_next_actions(*sources: Any) -> list[str]:
    actions: list[str] = []
    for source in sources:
        if not isinstance(source, dict):
            continue
        _extend_unique_strings(actions, source.get("blocked_next_actions", []))
        for key in ("failure_synthesis_constraints", "research_decision_constraints"):
            raw_constraints = source.get(key, [])
            if not isinstance(raw_constraints, list):
                continue
            for item in raw_constraints:
                if isinstance(item, dict):
                    _extend_unique_strings(actions, item.get("blocked_next_actions", []))
    return actions


def _extend_unique_strings(target: list[str], values: Any) -> None:
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, list):
        return
    for value in values:
        text = str(value).strip()
        if text and text not in target:
            target.append(text)


def _copy_jsonish(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _copy_jsonish(nested) for key, nested in value.items()}
    if isinstance(value, list):
        return [_copy_jsonish(item) for item in value]
    return value


def _research_brief_check(
    research_brief: dict[str, Any], research_brief_path: Path | None, root: Path
) -> dict[str, Any]:
    refs = research_brief.get("research_references", [])
    thesis_id = str(research_brief.get("thesis_id") or "").strip()
    path_status = "pass"
    rel_path = None
    if research_brief_path is not None:
        rel_path = _rel(research_brief_path, root)
        path_status = "pass" if research_brief_path.is_file() else "missing"
    refs_present = isinstance(refs, list) and bool(refs)
    refs_valid = refs_present and all(
        isinstance(ref, dict)
        and ref.get("reference_id")
        and ref.get("title")
        and ref.get("source")
        and ref.get("published_at")
        and ref.get("relevance")
        and thesis_id in _motivated_thesis_ids(ref)
        for ref in refs
    )
    if not refs_present or not thesis_id:
        status = "missing"
    elif path_status == "missing" or not refs_valid:
        status = "fail"
    else:
        status = "pass"
    return {
        "name": "research_brief",
        "status": status,
        "path": rel_path,
        "reference_count": len(refs) if isinstance(refs, list) else 0,
        "thesis_id": thesis_id or None,
    }


def _motivated_thesis_ids(reference: dict[str, Any]) -> set[str]:
    raw = reference.get("motivated_thesis_ids", [])
    if isinstance(raw, str):
        raw = [raw]
    return {str(item).strip() for item in raw if str(item).strip()}


def _parameter_optimization_policy_check(
    generated: dict[str, Any],
    generated_metadata_path: Path,
    root: Path,
) -> dict[str, Any]:
    applies = (
        generated.get("factory") == "strategy_code_generator"
        or generated.get("strategy_code_generated") is True
        or bool(generated.get("generated_strategy_path"))
    )
    if not applies:
        return {
            "name": "generated_parameter_optimization_policy",
            "status": "skipped",
            "path": _rel(generated_metadata_path, root),
            "required": False,
        }

    safety_scope = generated.get("safety_scope", {})
    if not isinstance(safety_scope, dict):
        safety_scope = {}
    generated_checks = generated.get("checks", [])
    if not isinstance(generated_checks, list):
        generated_checks = []
    generator_check_status = next(
        (
            str(check.get("status"))
            for check in generated_checks
            if isinstance(check, dict)
            and check.get("name") == "generated_code_freqtrade_hyperopt_disabled"
        ),
        None,
    )

    strategy_path = _metadata_path(generated.get("generated_strategy_path"), root)
    strategy_path_rel = _rel(strategy_path, root) if strategy_path else None
    strategy_file_present = strategy_path is not None and strategy_path.is_file()
    code_read_error: str | None = None
    code_contains_optimize_true: bool | None = None
    code_contains_optimize_false: bool | None = None
    if strategy_file_present and strategy_path is not None:
        try:
            strategy_code = strategy_path.read_text(encoding="utf-8")
            code_contains_optimize_true = "optimize=True" in strategy_code
            code_contains_optimize_false = "optimize=False" in strategy_code
        except OSError as exc:
            code_read_error = str(exc)
            strategy_file_present = False
        except UnicodeDecodeError as exc:
            code_read_error = str(exc)
            strategy_file_present = False

    metadata_policy_ok = (
        generated.get("parameter_optimization_enabled") is False
        and generated.get("parameter_optimization_policy") == PARAMETER_OPTIMIZATION_POLICY
    )
    safety_policy_ok = safety_scope.get("freqtrade_hyperopt_parameter_optimization") is False
    generator_check_ok = generator_check_status == "pass"
    code_policy_ok = (
        strategy_file_present
        and code_contains_optimize_true is False
        and code_contains_optimize_false is True
    )
    passed = (
        metadata_policy_ok
        and safety_policy_ok
        and generator_check_ok
        and code_policy_ok
    )
    return {
        "name": "generated_parameter_optimization_policy",
        "status": "pass" if passed else "fail",
        "path": _rel(generated_metadata_path, root),
        "strategy_path": strategy_path_rel,
        "parameter_optimization_enabled": generated.get("parameter_optimization_enabled"),
        "parameter_optimization_policy": generated.get("parameter_optimization_policy"),
        "expected_parameter_optimization_policy": PARAMETER_OPTIMIZATION_POLICY,
        "safety_scope_freqtrade_hyperopt_parameter_optimization": safety_scope.get(
            "freqtrade_hyperopt_parameter_optimization"
        ),
        "generated_code_freqtrade_hyperopt_disabled_status": generator_check_status,
        "strategy_file_present": strategy_file_present,
        "code_contains_optimize_true": code_contains_optimize_true,
        "code_contains_optimize_false": code_contains_optimize_false,
        "code_read_error": code_read_error,
    }


def _safe_path_component(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value.strip())
    cleaned = cleaned.strip("._")
    return cleaned or "unknown"


def _render_candidate_report(manifest: dict[str, Any]) -> str:
    lines = [
        "# Candidate Report",
        "",
        f"- candidate_id: {manifest.get('candidate_id')}",
        f"- strategy_name: {manifest.get('strategy_name')}",
        f"- recommendation: {manifest.get('recommendation')}",
        f"- rationale: {manifest.get('recommendation_rationale', '')}",
        "- paper_live_promotion: not authorized by this pipeline",
        "",
        "## Checks",
        "",
    ]
    for check in manifest.get("checks", []):
        lines.append(
            f"- {check.get('name')}: {check.get('status')} ({check.get('path')})"
        )
    research_refs = manifest.get("research_brief", {}).get("research_references", [])
    lines.extend(["", "## Research Brief", ""])
    if research_refs:
        for ref in research_refs:
            lines.append(
                f"- {ref.get('reference_id')}: {ref.get('title')} "
                f"({ref.get('published_at')})"
            )
    else:
        lines.append("- None.")
    lines.extend(["", "## Reviewer Notes", ""])
    notes = manifest.get("reviewer_notes") or []
    if notes:
        lines.extend(f"- {note}" for note in notes)
    else:
        lines.append("- None.")
    lines.append("")
    return "\n".join(lines)


def _check(name: str, path: Path | None, root: Path, *, key: str, pass_values: set[str] | None = None, required: bool = True) -> dict[str, Any]:
    if path is None:
        return {"name": name, "status": "missing" if required else "skipped", "path": None}
    resolved = _resolve(path, root)
    if not resolved.is_file():
        return {"name": name, "status": "missing" if required else "skipped", "path": _rel(resolved, root)}
    payload = _load_json(resolved)
    value = payload.get(key)
    if value is None:
        value = _derive_check_value(name, key, payload)
    passed = bool(value) if pass_values is None else str(value) in pass_values
    return {
        "name": name,
        "status": "pass" if passed else "fail",
        "path": _rel(resolved, root),
        "value": value,
        "payload_summary": _summarize_payload(name, payload),
    }


def _derive_check_value(name: str, key: str, payload: dict[str, Any]) -> Any:
    if name != "historical_backtest" or key != "recommendation":
        return None
    required_gate_fields = {"trade_count", "profit_factor", "max_drawdown_pct", "sortino"}
    if any(field not in payload for field in required_gate_fields):
        return None
    values = {field: payload.get(field) for field in BacktestMetrics.__dataclass_fields__}
    try:
        gate = evaluate_initial_gate(BacktestMetrics(**values))
    except Exception:
        return None
    return gate.get("recommendation")


def _file_check(name: str, path: Path | None, root: Path, *, required: bool = True) -> dict[str, Any]:
    if path is None:
        return {"name": name, "status": "missing" if required else "skipped", "path": None}
    resolved = _resolve(path, root)
    if not resolved.is_file():
        return {"name": name, "status": "missing" if required else "skipped", "path": _rel(resolved, root)}
    return {
        "name": name,
        "status": "pass",
        "path": _rel(resolved, root),
        "bytes": resolved.stat().st_size,
    }


def _strategy_identity_check(
    name: str,
    path: Path | None,
    root: Path,
    *,
    expected_strategy: str,
    keys: list[str],
    required: bool = True,
) -> dict[str, Any]:
    if path is None:
        return {"name": name, "status": "missing" if required else "skipped", "path": None}
    resolved = _resolve(path, root)
    if not resolved.is_file():
        return {"name": name, "status": "missing" if required else "skipped", "path": _rel(resolved, root)}
    payload = _load_json(resolved)
    observed = None
    for key in keys:
        if payload.get(key):
            observed = str(payload.get(key))
            break
    passed = bool(expected_strategy) and observed == expected_strategy
    return {
        "name": name,
        "status": "pass" if passed else "fail",
        "path": _rel(resolved, root),
        "expected_strategy": expected_strategy,
        "observed_strategy": observed,
    }


def _step(name: str, path: Path | None, root: Path, *, key: str, pass_values: set[str] | None = None, required: bool = True, command_preview: list[str] | None = None) -> dict[str, Any]:
    check = _check(name, path, root, key=key, pass_values=pass_values, required=required)
    return {"name": name, "check": check, "input_path": check.get("path"), "output_path": check.get("path"), "output_status": check.get("status"), "command_preview": command_preview or []}


def _file_step(name: str, path: Path | None, root: Path, *, required: bool = True, command_preview: list[str] | None = None) -> dict[str, Any]:
    check = _file_check(name, path, root, required=required)
    return {"name": name, "check": check, "input_path": check.get("path"), "output_path": check.get("path"), "output_status": check.get("status"), "command_preview": command_preview or []}


def _strategy_identity_step(
    name: str,
    path: Path | None,
    root: Path,
    *,
    expected_strategy: str,
    keys: list[str],
    required: bool = True,
) -> dict[str, Any]:
    check = _strategy_identity_check(
        name,
        path,
        root,
        expected_strategy=expected_strategy,
        keys=keys,
        required=required,
    )
    return {"name": name, "check": check, "input_path": check.get("path"), "output_path": check.get("path"), "output_status": check.get("status"), "command_preview": []}


def _extract_metrics_summary(manifest: dict[str, Any]) -> dict[str, Any]:
    checks = {item["name"]: item for item in manifest.get("checks", [])}
    return {
        "recommendation": manifest.get("recommendation"),
        "historical_backtest": checks.get("historical_backtest", {}).get("value"),
        "walk_forward": checks.get("walk_forward", {}).get("value"),
        "training_factory": checks.get("training_factory", {}).get("value"),
        "historical_summary": checks.get("historical_backtest", {}).get("payload_summary", {}),
        "walk_forward_summary": checks.get("walk_forward", {}).get("payload_summary", {}),
        "training_summary": checks.get("training_factory", {}).get("payload_summary", {}),
    }


def _extract_artifact_paths(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "proposal_metadata_path": manifest.get("proposal_metadata_path"),
        "generated_metadata_path": manifest.get("generated_metadata_path"),
        "input_paths": manifest.get("input_paths", {}),
        "check_artifact_paths": {item.get("name"): item.get("path") for item in manifest.get("checks", [])},
    }


def _resolve(path: Path, root: Path) -> Path:
    return (path if path.is_absolute() else root / path).resolve()


def _maybe_rel(path: Path | None, root: Path) -> str | None:
    if path is None:
        return None
    return _rel(_resolve(path, root), root)


def _metadata_path(value: Any, root: Path) -> Path | None:
    if not value:
        return None
    return _resolve(Path(str(value)), root)


def _display_path(path: Path | None) -> str | None:
    if path is None:
        return None
    return str(path)


def _rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root)).replace("\\", "/")
    except Exception:
        return str(path)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected object JSON: {path}")
    return payload


def _summarize_payload(name: str, payload: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "strategy_name",
        "strategy",
        "recommendation",
        "status",
        "total_return",
        "total_return_pct",
        "trade_count",
        "max_drawdown_pct",
        "profit_factor",
        "sortino",
    ]
    summary = {key: payload.get(key) for key in keys if key in payload}
    nested = payload.get("summary")
    if isinstance(nested, dict):
        summary["summary"] = {
            key: nested.get(key)
            for key in [
                "window_count",
                "pass_rate",
                "profitable_windows_ratio",
                "total_return",
                "total_return_pct",
                "max_drawdown_pct_any_window",
                "max_single_window_profit_dependency",
                "stage_count",
                "completed_stages",
                "failed_stages",
            ]
            if key in nested
        }
    if name == "training_factory" and isinstance(payload.get("stages"), list):
        summary["stage_recommendations"] = [
            stage.get("recommendation") for stage in payload["stages"] if isinstance(stage, dict)
        ]
    return summary
