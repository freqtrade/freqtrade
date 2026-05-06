from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence


FORBIDDEN_REVISION_RE = re.compile(
    r"(?i)\b("
    r"lookahead|future\s+(data|candle|close|return|price)|live[- ]only|"
    r"real[- ]time|realtime|account\s+balance|position\s+data|"
    r"create_order|order\s+endpoint|place\s+orders?|api[_ -]?key|secret|"
    r"password|token|credential|leverage\s+[2-9]|[2-9]x\s+leverage|"
    r"shorting|enter_short|exit_short|can_short|freqtrade\s+trade|"
    r"paper\s+trading|dry[- ]run\s+trading|live\s+trading|process\s+control"
    r")\b"
)
TIMERANGE_RE = re.compile(r"^(?P<start>\d{8})-(?P<end>\d{8})$")


@dataclass(frozen=True)
class CandidateIterationInputs:
    root_dir: Path
    candidate_manifest_path: Path
    proposal_metadata_path: Path
    generated_metadata_path: Path | None = None
    output_root: Path = Path("registry/strategies/reviews")
    revision_id: str | None = None
    reviewer_findings: Sequence[str] = field(default_factory=list)
    changed_assumptions: Sequence[str] = field(default_factory=list)
    changed_parameters: Sequence[str] = field(default_factory=list)
    changed_data_requirements: Sequence[str] = field(default_factory=list)
    unchanged_rejection_rules: Sequence[str] = field(default_factory=list)
    prior_timerange: str | None = None
    proposed_timerange: str | None = None
    max_parameter_changes: int = 4
    max_attempts_per_strategy_family: int = 5
    timeout_minutes: int = 60


def build_candidate_iteration_plan(inputs: CandidateIterationInputs) -> dict[str, Any]:
    root = inputs.root_dir.resolve()
    generated_at = datetime.now(UTC).replace(microsecond=0).isoformat()
    manifest_path = _resolve(inputs.candidate_manifest_path, root)
    proposal_path = _resolve(inputs.proposal_metadata_path, root)
    generated_path = _resolve(inputs.generated_metadata_path, root) if inputs.generated_metadata_path else None
    manifest = _load_json(manifest_path)
    proposal = _load_json(proposal_path)
    generated = _load_json(generated_path) if generated_path and generated_path.is_file() else {}

    checks = _iteration_checks(inputs, manifest)
    retry_budget = int(_next_input(manifest).get("retry_budget_per_thesis") or proposal.get("retry_budget_per_thesis") or 1)
    thesis_retry_count = int(_next_input(manifest).get("thesis_retry_count") or proposal.get("thesis_retry_count") or 0)
    parameter_retry_count = int(_next_input(manifest).get("parameter_only_retry_count") or proposal.get("parameter_only_retry_count") or 0)
    force_distinct = bool(_next_input(manifest).get("force_distinct_hypothesis_family") or proposal.get("force_distinct_hypothesis_family"))
    budget_exceeded = thesis_retry_count >= retry_budget and not force_distinct
    safety_blocked = any(check["status"] == "blocked" and check["category"] == "safety" for check in checks)
    hard_limit_blocked = any(check["status"] == "blocked" and check["category"] == "limit" for check in checks)
    if safety_blocked or hard_limit_blocked or budget_exceeded:
        action = "reject"
    elif any(check["status"] == "blocked" for check in checks):
        action = "blocked"
    else:
        action = "revise"

    revision_id = inputs.revision_id or _revision_id(generated_at)
    proposal_revision_input = _proposal_revision_input(
        proposal=proposal,
        generated=generated,
        manifest=manifest,
        inputs=inputs,
        revision_id=revision_id,
        action=action,
        thesis_retry_count=thesis_retry_count,
        parameter_retry_count=parameter_retry_count,
        force_distinct=force_distinct or budget_exceeded,
    )
    return {
        "generated_at": generated_at,
        "factory": "candidate_iteration_loop",
        "revision_id": revision_id,
        "candidate_id": manifest.get("candidate_id"),
        "strategy_name": manifest.get("strategy_name") or proposal.get("strategy_name"),
        "action": action,
        "checks": checks,
        "lineage": {
            "candidate_manifest_path": _rel(manifest_path, root),
            "proposal_metadata_path": _rel(proposal_path, root),
            "generated_metadata_path": _rel(generated_path, root) if generated_path else None,
            "candidate_recommendation": manifest.get("recommendation"),
            "failure_taxonomy_codes": manifest.get("failure_taxonomy_codes", []),
            "previous_thesis_id": _next_input(manifest).get("thesis_id") or proposal.get("thesis_id"),
        },
        "reviewer_findings_addressed": list(inputs.reviewer_findings),
        "changed_assumptions": list(inputs.changed_assumptions),
        "changed_parameters": list(inputs.changed_parameters),
        "changed_data_requirements": list(inputs.changed_data_requirements),
        "unchanged_rejection_rules": list(inputs.unchanged_rejection_rules),
        "proposal_revision_input": proposal_revision_input,
        "pre_evaluation_requirements": [
            "regenerate proposal metadata with unchanged safety scope",
            "generate strategy code and metadata",
            "run generated strategy static safety scan",
            "validate generated metadata hypothesis and retry budget fields",
            "run OHLCV quality checks before any historical backtest",
            "run historical backtest, walk-forward, and training checks as applicable",
        ],
        "evaluation_allowed_by_this_plan": False,
        "safety_scope": {
            "historical_only": True,
            "paper_trading_started": False,
            "dry_run_trading_started": False,
            "live_trading": False,
            "exchange_order_placement": False,
            "leverage_above_one": False,
            "shorting": False,
            "process_control": False,
        },
    }


def write_candidate_iteration_artifacts(
    plan: dict[str, Any], *, root_dir: Path, output_root: Path
) -> tuple[Path, Path, Path]:
    root = root_dir.resolve()
    strategy = _safe_path_component(str(plan.get("strategy_name") or "unknown_strategy"))
    candidate_id = _safe_path_component(str(plan.get("candidate_id") or "unknown_candidate"))
    revision_id = _safe_path_component(str(plan["revision_id"]))
    out_dir = _resolve(output_root, root) / strategy / candidate_id / revision_id
    out_dir.mkdir(parents=True, exist_ok=True)
    plan_path = out_dir / "iteration_plan.json"
    revision_input_path = out_dir / "proposal_revision_input.json"
    report_path = out_dir / "iteration_report.md"
    plan_path.write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")
    revision_input_path.write_text(
        json.dumps(plan["proposal_revision_input"], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    report_path.write_text(_render_report(plan), encoding="utf-8")
    return plan_path, revision_input_path, report_path


def _iteration_checks(inputs: CandidateIterationInputs, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    all_revision_text = "\n".join(
        [
            *inputs.reviewer_findings,
            *inputs.changed_assumptions,
            *inputs.changed_parameters,
            *inputs.changed_data_requirements,
        ]
    )
    walk_forward = _check_by_name(manifest, "walk_forward")
    parameter_only = (
        bool(inputs.changed_parameters)
        and not inputs.changed_assumptions
        and not inputs.changed_data_requirements
    )
    return [
        _check("reviewer_findings_present", bool(inputs.reviewer_findings), "input"),
        _check(
            "candidate_not_already_passed",
            manifest.get("recommendation") != "pass",
            "input",
            "Passing candidates should be ranked, not iterated.",
        ),
        _check(
            "revision_safety_scope_preserved",
            not FORBIDDEN_REVISION_RE.search(all_revision_text),
            "safety",
            "Revision text must not relax toward future/live/order/secret/leverage/short/process-control dependencies.",
        ),
        _check(
            "out_of_sample_walk_forward_evidence_present",
            walk_forward.get("status") not in {None, "missing", "skipped"},
            "overfit",
            "Iteration requires existing walk-forward evidence before changing the candidate.",
        ),
        _check(
            "timerange_not_narrowed_after_failure",
            not _timerange_narrowed(inputs.prior_timerange, inputs.proposed_timerange),
            "overfit",
            "Do not narrow timeranges after a failed candidate.",
        ),
        _check(
            "parameter_change_breadth_limited",
            len(inputs.changed_parameters) <= int(inputs.max_parameter_changes),
            "limit",
            "Parameter-search breadth exceeds the configured limit.",
        ),
        _check(
            "parameter_only_retry_not_default_path",
            not parameter_only,
            "limit",
            "Iteration must change assumptions, data requirements, or hypothesis family, not only parameters.",
        ),
        _check(
            "unchanged_rejection_rules_recorded",
            bool(inputs.unchanged_rejection_rules),
            "overfit",
            "Record unchanged rejection rules so revisions cannot move the goalposts.",
        ),
        _check(
            "timeout_limit_configured",
            0 < int(inputs.timeout_minutes) <= 24 * 60,
            "limit",
            "Timeout must be positive and no more than one day.",
        ),
        _check(
            "strategy_family_attempt_limit_configured",
            int(inputs.max_attempts_per_strategy_family) > 0,
            "limit",
            "max_attempts_per_strategy_family must be positive.",
        ),
    ]


def _proposal_revision_input(
    *,
    proposal: dict[str, Any],
    generated: dict[str, Any],
    manifest: dict[str, Any],
    inputs: CandidateIterationInputs,
    revision_id: str,
    action: str,
    thesis_retry_count: int,
    parameter_retry_count: int,
    force_distinct: bool,
) -> dict[str, Any]:
    parameter_only = (
        bool(inputs.changed_parameters)
        and not inputs.changed_assumptions
        and not inputs.changed_data_requirements
    )
    return {
        "revision_id": revision_id,
        "action": action,
        "source_candidate_id": manifest.get("candidate_id"),
        "strategy_name": proposal.get("strategy_name") or manifest.get("strategy_name"),
        "generator_mode": proposal.get("generator_mode") or generated.get("generator_mode") or "rule_based",
        "strategy_logic_variant": _next_logic_variant(proposal, manifest, force_distinct),
        "thesis_id": proposal.get("thesis_id") or _next_input(manifest).get("thesis_id"),
        "thesis_type": proposal.get("thesis_type") or manifest.get("thesis", {}).get("thesis_type"),
        "thesis_statement": proposal.get("thesis_statement") or _next_input(manifest).get("thesis_statement"),
        "falsification_criteria": proposal.get("falsification_criteria") or manifest.get("thesis", {}).get("falsification_criteria"),
        "novelty_vs_previous": "; ".join(inputs.changed_assumptions) or proposal.get("novelty_vs_previous"),
        "evidence_refs": list(dict.fromkeys((proposal.get("evidence_refs") or []) + [f"local:candidate_manifest:{manifest.get('candidate_id')}"])),
        "failure_taxonomy_codes": manifest.get("failure_taxonomy_codes", []),
        "retry_budget_per_thesis": proposal.get("retry_budget_per_thesis") or _next_input(manifest).get("retry_budget_per_thesis"),
        "thesis_retry_count": thesis_retry_count + (0 if force_distinct else 1),
        "parameter_only_retry_limit": proposal.get("parameter_only_retry_limit") or 1,
        "parameter_only_retry_count": parameter_retry_count + (1 if parameter_only else 0),
        "force_distinct_hypothesis_family": force_distinct,
        "changed_assumptions": list(inputs.changed_assumptions),
        "changed_parameters": list(inputs.changed_parameters),
        "changed_data_requirements": list(inputs.changed_data_requirements),
        "unchanged_rejection_rules": list(inputs.unchanged_rejection_rules),
        "reviewer_findings_addressed": list(inputs.reviewer_findings),
        "safety_scope": {
            "long_only": True,
            "historical_evaluation_only": True,
            "live_trading": False,
            "paper_trading_started": False,
            "exchange_order_placement": False,
            "leverage": 1.0,
            "shorting": False,
            "process_control": False,
        },
    }


def _next_logic_variant(
    proposal: dict[str, Any], manifest: dict[str, Any], force_distinct: bool
) -> str:
    current = str(proposal.get("strategy_logic_variant") or "mean_reversion_pullback")
    if not force_distinct:
        return current
    order = ["mean_reversion_pullback", "trend_continuation", "volatility_breakout"]
    if current not in order:
        return "trend_continuation"
    return order[(order.index(current) + 1) % len(order)]


def _check(
    name: str,
    passed: bool,
    category: str,
    message: str | None = None,
) -> dict[str, Any]:
    return {
        "name": name,
        "status": "pass" if passed else "blocked",
        "category": category,
        "message": message or name,
    }


def _check_by_name(manifest: dict[str, Any], name: str) -> dict[str, Any]:
    for check in manifest.get("checks", []):
        if check.get("name") == name:
            return check
    return {}


def _next_input(manifest: dict[str, Any]) -> dict[str, Any]:
    value = manifest.get("next_candidate_input")
    return value if isinstance(value, dict) else {}


def _timerange_narrowed(prior: str | None, proposed: str | None) -> bool:
    if not prior or not proposed:
        return False
    prior_days = _timerange_days(prior)
    proposed_days = _timerange_days(proposed)
    return prior_days is not None and proposed_days is not None and proposed_days < prior_days


def _timerange_days(value: str) -> int | None:
    match = TIMERANGE_RE.match(value.strip())
    if not match:
        return None
    start = datetime.strptime(match.group("start"), "%Y%m%d")
    end = datetime.strptime(match.group("end"), "%Y%m%d")
    return (end - start).days


def _render_report(plan: dict[str, Any]) -> str:
    lines = [
        "# Candidate Iteration Report",
        "",
        f"- revision_id: {plan.get('revision_id')}",
        f"- candidate_id: {plan.get('candidate_id')}",
        f"- action: {plan.get('action')}",
        "- evaluation_allowed_by_this_plan: false",
        "- paper_live_promotion: not authorized by this iteration plan",
        "",
        "## Checks",
        "",
    ]
    for check in plan.get("checks", []):
        lines.append(f"- {check.get('name')}: {check.get('status')}")
    lines.append("")
    return "\n".join(lines)


def _revision_id(generated_at: str) -> str:
    parsed = datetime.fromisoformat(generated_at.replace("Z", "+00:00"))
    return parsed.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")


def _safe_path_component(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value)
    return cleaned.strip("._") or "unknown"


def _resolve(path: Path | None, root: Path) -> Path:
    if path is None:
        raise ValueError("Cannot resolve a missing path")
    return (path if path.is_absolute() else root / path).resolve()


def _rel(path: Path | None, root: Path) -> str | None:
    if path is None:
        return None
    try:
        return str(path.relative_to(root)).replace("\\", "/")
    except ValueError:
        return str(path)


def _load_json(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected object JSON: {path}")
    return payload
