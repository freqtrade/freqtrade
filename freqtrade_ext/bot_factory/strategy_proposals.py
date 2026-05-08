from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence


STRATEGY_PROPOSAL_GENERATOR_VERSION = "strategy_proposal_generator_v1"
STRATEGY_PROPOSAL_NOTICE = (
    "Strategy proposal generation writes local Markdown and metadata artifacts "
    "only. It does not generate strategy code, run backtests, start paper or "
    "dry-run trading, call exchange order endpoints, promote candidates, or "
    "manage any bot process."
)

REQUIRED_PROPOSAL_SECTIONS = [
    "Metadata",
    "Summary",
    "Hypothesis",
    "Research References",
    "Market Condition",
    "Entry Logic",
    "Exit Logic",
    "Risk Logic",
    "Required Data",
    "Parameters",
    "Expected Failure Cases",
    "Backtest Plan",
    "Rejection Conditions",
]

ALLOWED_DATA_CLASSES = [
    "historical_ohlcv_closed_candles",
    "local_ohlcv_quality_json",
    "local_data_quality_json",
    "local_previous_metrics_json",
    "local_walk_forward_metrics_json",
    "local_training_manifest_json",
    "local_candidate_failure_synthesis_json",
    "local_research_decision_json",
    "local_edge_discovery_json",
    "local_reviewer_notes",
]
ALLOWED_GENERATOR_MODES = {"rule_based", "freqai", "hybrid_ml"}
ALLOWED_FAILURE_TAXONOMY_CODES = {
    "FAIL_OVERFIT_WF_GAP",
    "FAIL_COST_SENSITIVE",
    "FAIL_REGIME_FRAGILE",
}
ALLOWED_STRATEGY_LOGIC_VARIANTS = {
    "amihud_illiquidity_premium",
    "bipower_jump_decay",
    "calendar_turnover_seasonality",
    "crowding_unwind_reaccumulation",
    "cross_asset_cointegration_spread",
    "cross_asset_correlation_recovery",
    "cross_asset_lead_lag",
    "directional_change_overshoot",
    "downside_liquidity_shock_reversal",
    "entropy_regime_transition",
    "fractal_long_memory_regime",
    "funding_pressure_carry",
    "intraday_session_liquidity_reclaim",
    "liquidity_recovery_horizon",
    "market_beta_drawdown_carry",
    "mark_discount_reclaim_continuation",
    "mark_fair_value_momentum_lag",
    "mark_price_dislocation_reclaim",
    "mean_reversion_pullback",
    "microstructure_spread_reversion",
    "range_quarticity_vol_of_vol_state",
    "realized_skewness_tail_shape",
    "regime_state_reentry",
    "semivariance_asymmetry_regime",
    "signed_volume_imbalance_accumulation",
    "trend_continuation",
    "variance_ratio_regime_switch",
    "volatility_breakout",
}
MATERIAL_CAUSAL_CATEGORY_MIN_SHARE = 0.70
HIGH_RISK_CAUSAL_RESPONSE_MIN_RISK_SCORE = 80.0

_STRUCTURAL_DATA_RE = re.compile(
    r"(?i)\b(open[-_ ]?interest|long[-_ /]?short[-_ ]?(?:account[-_ ]?)?ratio|"
    r"account[-_ ]?ratio|liquidations?|order[-_ ]?book|orderbook|"
    r"market[-_ ]?depth|book[-_ ]?imbalance|depth[-_ ]?imbalance)\b"
)

_PRIVATE_ENV_RE = re.compile(
    r"(?i)(\$\{[^}]*?(KEY|SECRET|TOKEN|PASSWORD|PASSWD|UID|JWT)[^}]*?\}|"
    r"env:[A-Z_][A-Z0-9_]*|%[A-Z_]*(KEY|SECRET|TOKEN|PASSWORD|PASSWD|UID|JWT)[A-Z0-9_]*%)"
)
_SECRET_ASSIGNMENT_RE = re.compile(
    r"""(?ix)
    (?P<label>api[_ -]?key|secret|password|passwd|token|jwt|credential)
    (?P<sep>\s*[:=]\s*)
    (?P<quote>["'])?
    (?P<value>[A-Za-z0-9_./+=:-]{8,})
    (?P=quote)?
    """
)
_SECRET_PHRASE_RE = re.compile(
    r"""(?ix)
    (?P<label>api[_ -]?key|secret|password|passwd|token|jwt|credential)
    (?P<sep>\s+)
    (?P<value>[A-Za-z0-9_./+=:-]{12,})
    """
)
_FUTURE_DATA_RE = re.compile(
    r"(?i)\b(lookahead|future\s+(data|candle|close|return|price)|"
    r"next\s+(candle|bar|close|return|price)|tomorrow'?s?\s+(close|price))\b|"
    r"shift\s*\(\s*-\d+"
)
_LIVE_ONLY_DATA_RE = re.compile(
    r"(?i)\b(live[- ]only|live\s+data|real[- ]time|realtime|"
    r"websocket\s+only|streaming\s+only|current\s+open\s+candle|"
    r"unclosed\s+candle)\b"
)
_ACCOUNT_POSITION_RE = re.compile(
    r"(?i)\b(account\s+balance|wallet\s+balance|private\s+balance|"
    r"position\s+data|open\s+positions?|current\s+positions?|fills?)\b"
)
_ORDER_ENDPOINT_RE = re.compile(
    r"(?i)\b(create_order|private_post_order|fapiPrivatePostOrder|request_order|"
    r"order\s+endpoints?|exchange\s+order\s+endpoints?|place\s+orders?|"
    r"order\s+placement|requests\.post|httpx\.post)\b"
)
_CREDENTIAL_DEPENDENCY_RE = re.compile(
    r"(?i)\b(api[_ -]?keys?|secrets?|passwords?|tokens?|credentials?)\b"
)
_PROCESS_CONTROL_RE = re.compile(
    r"(?i)\b(freqtrade\s+trade|bot\s+startup|process\s+control|"
    r"paper\s+trading|dry[- ]run\s+trading|live\s+trading|canary\s+live|"
    r"start\s+(paper|bot|process)|stop\s+(paper|bot|process)|"
    r"poll\s+(paper|bot|process)|manage\s+(paper|bot|process))\b"
)
_SHORTING_RE = re.compile(
    r"(?i)\b(enter_short|exit_short|can_short|short\s+entry|short\s+exit|"
    r"short\s+signals?|short\s+trades?|shorting|go\s+short|allow\s+short)\b"
)
_NEGATION_PREFIX_RE = re.compile(
    r"(?i)(\bno\b|\bnot\b|\bnever\b|\bwithout\b|\bdo\s+not\b|"
    r"\bdoes\s+not\b|\bmust\s+not\b|\bdisable\b|\bdisabled\b)\W*$"
)
_DIVERSIFIED_BACKTEST_RE = re.compile(
    r"(?i)\b(walk[- ]forward|out[- ]of[- ]sample|multiple|multi[- ]window|"
    r"rolling|several|holdout|train/test|cross[- ]validation)\b"
)
_NARROW_BACKTEST_RE = re.compile(
    r"(?i)\b(only\s+one|single\s+narrow|one\s+narrow|single\s+backtest|"
    r"one\s+backtest|only\s+\d{8}\s*-\s*\d{8})\b"
)


@dataclass(frozen=True)
class StrategyProposalEvidenceInput:
    label: str
    path: Path


@dataclass(frozen=True)
class StrategyProposalResearchReference:
    reference_id: str
    title: str
    source: str
    relevance: str
    published_at: str | None = None
    motivated_thesis_ids: Sequence[str] = field(default_factory=list)


@dataclass(frozen=True)
class StrategyProposalInputs:
    root_dir: Path
    strategy_name: str
    strategy_type: str
    target_exchange: str
    target_symbols: Sequence[str]
    timeframe: str
    spot_or_futures: str
    long_short: str
    summary: str
    hypothesis: str
    market_condition: str
    entry_logic: str
    exit_logic: str
    risk_logic: str
    required_data: Sequence[str]
    parameters: Sequence[str]
    expected_failure_cases: Sequence[str]
    backtest_plan: str
    rejection_conditions: Sequence[str]
    generator_mode: str = "rule_based"
    thesis_id: str | None = None
    thesis_type: str | None = None
    thesis_statement: str | None = None
    falsification_criteria: str | None = None
    novelty_vs_previous: str | None = None
    evidence_refs: Sequence[str] = field(default_factory=list)
    research_references: Sequence[
        StrategyProposalResearchReference | dict[str, Any]
    ] = field(default_factory=list)
    failure_taxonomy_codes: Sequence[str] = field(default_factory=list)
    retry_budget_per_thesis: int = 3
    thesis_retry_count: int = 0
    parameter_only_retry_limit: int = 1
    parameter_only_retry_count: int = 0
    force_distinct_hypothesis_family: bool = False
    strategy_logic_variant: str | None = None
    feature_list: Sequence[str] = field(default_factory=list)
    target_definition: str | None = None
    label_horizon: int | None = None
    prediction_threshold: float | None = None
    rule_filters: Sequence[str] = field(default_factory=list)
    risk_policy: str = "long_only_leverage_1"
    reviewer_notes: Sequence[str] = field(default_factory=list)
    evidence_paths: Sequence[StrategyProposalEvidenceInput] = field(default_factory=list)
    output_root: Path = Path("registry/strategies/proposals")
    created_by_agent: str = "codex"
    created_at: str | None = None
    command: Sequence[str] = field(default_factory=list)


@dataclass(frozen=True)
class StrategyProposalCheck:
    name: str
    status: str
    severity: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StrategyProposalArtifacts:
    proposal_markdown: str
    metadata: dict[str, Any]
    proposal_path: Path
    metadata_path: Path


def build_strategy_proposal(inputs: StrategyProposalInputs) -> StrategyProposalArtifacts:
    created_at = inputs.created_at or datetime.now(UTC).replace(microsecond=0).isoformat()
    proposal_path, metadata_path = _proposal_paths(inputs, created_at)

    text_fields = _sanitized_text_fields(inputs)
    evidence, evidence_checks = _evidence_summary(inputs)
    checks: list[StrategyProposalCheck] = []
    checks.extend(_required_input_checks(inputs))
    checks.extend(_scope_checks(inputs))
    checks.extend(_hypothesis_candidate_checks(inputs, created_at))
    checks.extend(_failure_synthesis_novelty_checks(inputs, created_at))
    checks.extend(_research_decision_gate_checks(inputs, created_at))
    checks.extend(_edge_discovery_gate_checks(inputs, created_at))
    checks.extend(_forbidden_dependency_checks(inputs, text_fields))
    checks.extend(evidence_checks)

    status = "blocked" if any(check.status == "blocked" for check in checks) else "accepted"
    proposal_markdown = _render_proposal_markdown(
        inputs=inputs,
        text_fields=text_fields,
        created_at=created_at,
        status=status,
        evidence=evidence,
    )
    checks.extend(_required_section_checks(proposal_markdown))
    status = "blocked" if any(check.status == "blocked" for check in checks) else "accepted"

    if status == "blocked" and "- proposal_status: accepted" in proposal_markdown:
        proposal_markdown = proposal_markdown.replace(
            "- proposal_status: accepted", "- proposal_status: blocked", 1
        )

    proposal_hash = _sha256_text(proposal_markdown)
    metadata = _build_metadata(
        inputs=inputs,
        created_at=created_at,
        status=status,
        proposal_path=proposal_path,
        metadata_path=metadata_path,
        proposal_hash=proposal_hash,
        checks=checks,
        evidence=evidence,
        text_fields=text_fields,
    )
    return StrategyProposalArtifacts(
        proposal_markdown=proposal_markdown,
        metadata=metadata,
        proposal_path=proposal_path,
        metadata_path=metadata_path,
    )


def write_strategy_proposal_artifacts(artifacts: StrategyProposalArtifacts) -> None:
    artifacts.proposal_path.parent.mkdir(parents=True, exist_ok=True)
    artifacts.proposal_path.write_text(artifacts.proposal_markdown, encoding="utf-8")
    artifacts.metadata_path.write_text(
        json.dumps(artifacts.metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def load_strategy_proposal_metadata(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Strategy proposal metadata must contain an object: {path}")
    return payload


def _build_metadata(
    *,
    inputs: StrategyProposalInputs,
    created_at: str,
    status: str,
    proposal_path: Path,
    metadata_path: Path,
    proposal_hash: str,
    checks: Sequence[StrategyProposalCheck],
    evidence: Sequence[dict[str, Any]],
    text_fields: dict[str, Any],
) -> dict[str, Any]:
    blockers = [check for check in checks if check.status == "blocked"]
    input_paths: dict[str, Any] = {}
    for item in evidence:
        label = str(item["label"])
        input_paths.setdefault(label, []).append(item.get("path"))
    research_references = _research_references(inputs, created_at)
    evidence_refs = _evidence_refs(
        inputs,
        evidence,
        proposal_path,
        research_references=research_references,
    )
    failure_synthesis_constraints = _failure_synthesis_constraints(inputs, created_at)
    research_decision_constraints = _research_decision_constraints(inputs, created_at)
    edge_discovery_handoff = _edge_discovery_handoff(inputs, created_at)

    return {
        "generated_at": created_at,
        "phase": "strategy_generation",
        "factory": "strategy_proposal_generator",
        "generator_version": STRATEGY_PROPOSAL_GENERATOR_VERSION,
        "status": status,
        "proposal_status": status,
        "code_generation_eligible": status == "accepted",
        "strategy_name": inputs.strategy_name,
        "strategy_type": inputs.strategy_type,
        "target_exchange": inputs.target_exchange,
        "target_symbols": [symbol for symbol in inputs.target_symbols],
        "timeframe": inputs.timeframe,
        "spot_or_futures": inputs.spot_or_futures,
        "long_short": inputs.long_short,
        "created_at": created_at,
        "created_by_agent": inputs.created_by_agent,
        "proposal_path": _safe_relative_path(proposal_path, inputs.root_dir),
        "metadata_path": _safe_relative_path(metadata_path, inputs.root_dir),
        "proposal_content_hash": proposal_hash,
        "generator_mode": _generator_mode(inputs.generator_mode),
        "strategy_logic_variant": _strategy_logic_variant(inputs),
        "feature_list": _feature_list(inputs),
        "target_definition": _target_definition(inputs),
        "label_horizon": _label_horizon(inputs),
        "prediction_threshold": _prediction_threshold(inputs),
        "rule_filters": _rule_filters(inputs),
        "risk_policy": _sanitize_text(inputs.risk_policy),
        "parameter_overrides": _parameter_overrides(inputs),
        "thesis_id": _thesis_id(inputs, created_at),
        "thesis_type": _thesis_type(inputs),
        "thesis_statement": _thesis_statement(inputs),
        "falsification_criteria": _falsification_criteria(inputs),
        "novelty_vs_previous": _novelty_vs_previous(inputs, evidence),
        "evidence_refs": evidence_refs,
        "research_references": research_references,
        "failure_synthesis_constraints": failure_synthesis_constraints,
        "research_decision_constraints": research_decision_constraints,
        "edge_discovery_handoff": edge_discovery_handoff,
        "structural_data_requirement": _proposal_structural_data_requirement(inputs),
        "research_brief": {
            "thesis_id": _thesis_id(inputs, created_at),
            "thesis_statement": _thesis_statement(inputs),
            "research_references": research_references,
            "evidence_refs": evidence_refs,
            "failure_taxonomy_codes": _failure_taxonomy_codes(inputs),
            "strategy_logic_variant": _strategy_logic_variant(inputs),
            "novelty_vs_previous": _novelty_vs_previous(inputs, evidence),
            "blocked_next_actions": _blocked_next_actions_from_constraints(
                failure_synthesis_constraints,
                research_decision_constraints,
                [edge_discovery_handoff],
            ),
            "research_handoff_summaries": _research_handoff_summaries_from_constraints(
                research_decision_constraints
            ),
            "edge_discovery_handoff": edge_discovery_handoff,
            "generated_at": created_at,
        },
        "failure_taxonomy_codes": _failure_taxonomy_codes(inputs),
        "retry_budget_per_thesis": int(inputs.retry_budget_per_thesis),
        "thesis_retry_count": int(inputs.thesis_retry_count),
        "parameter_only_retry_limit": int(inputs.parameter_only_retry_limit),
        "parameter_only_retry_count": int(inputs.parameter_only_retry_count),
        "force_distinct_hypothesis_family": bool(inputs.force_distinct_hypothesis_family),
        "source_input_paths": input_paths,
        "source_input_hashes": {
            name: _sha256_text(_join_text(value)) for name, value in text_fields.items()
        },
        "allowed_data_classes": list(ALLOWED_DATA_CLASSES),
        "evidence": list(evidence),
        "rejected_or_blocked_evidence": [
            item for item in evidence if item.get("status") == "blocked"
        ],
        "checks": [check.to_dict() for check in checks],
        "blockers": [check.to_dict() for check in blockers],
        "rejection_reasons": [check.message for check in blockers],
        "safety_scope": {
            "command": "strategy proposal generation only",
            "long_only": _normalizes_to_long_only(inputs.long_short),
            "historical_evaluation_only": True,
            "live_data": False,
            "live_trading": False,
            "paper_trading_started": False,
            "dry_run_trading_started": False,
            "exchange_order_placement": False,
            "uses_api_keys_or_secrets": False,
            "metadata_contains_secrets": False,
            "leverage": 1.0,
            "leverage_above_one": False,
            "shorting": False,
            "process_control": False,
            "code_generation_started": False,
            "backtest_started": False,
            "local_artifacts_source_of_truth": True,
        },
        "command": [_sanitize_text(token) for token in inputs.command],
        "notice": STRATEGY_PROPOSAL_NOTICE,
    }


def _render_proposal_markdown(
    *,
    inputs: StrategyProposalInputs,
    text_fields: dict[str, Any],
    created_at: str,
    status: str,
    evidence: Sequence[dict[str, Any]],
) -> str:
    lines = [
        f"# Strategy Proposal: {inputs.strategy_name}",
        "",
        "## Metadata",
        "",
        f"- created_at: {created_at}",
        f"- created_by_agent: {inputs.created_by_agent}",
        f"- strategy_type: {inputs.strategy_type}",
        f"- target_exchange: {inputs.target_exchange}",
        f"- target_symbols: {', '.join(inputs.target_symbols)}",
        f"- timeframe: {inputs.timeframe}",
        f"- spot_or_futures: {inputs.spot_or_futures}",
        f"- long_short: {inputs.long_short}",
        f"- proposal_status: {status}",
        f"- generator_mode: {_generator_mode(inputs.generator_mode)}",
        f"- thesis_id: {_thesis_id(inputs, created_at)}",
        f"- thesis_type: {_thesis_type(inputs)}",
        f"- strategy_logic_variant: {_strategy_logic_variant(inputs)}",
        "- safety_scope: long-only, leverage=1.0, historical-evaluation-only, "
        "no live data, no order endpoints, no secrets, no process control",
        "",
    ]
    if evidence:
        lines.extend(["- source_evidence:"])
        for item in evidence:
            lines.append(
                f"  - {item['label']}: `{item.get('path')}` ({item.get('status')})"
            )
        lines.append("")

    research_references = _research_references(inputs, created_at)
    section_values: list[tuple[str, Any]] = [
        ("Summary", text_fields["summary"]),
        ("Hypothesis", text_fields["hypothesis"]),
        ("Research References", _research_reference_lines(research_references)),
        ("Market Condition", text_fields["market_condition"]),
        ("Entry Logic", text_fields["entry_logic"]),
        ("Exit Logic", text_fields["exit_logic"]),
        ("Risk Logic", text_fields["risk_logic"]),
        ("Required Data", text_fields["required_data"]),
        ("Parameters", text_fields["parameters"]),
        ("Expected Failure Cases", text_fields["expected_failure_cases"]),
        ("Backtest Plan", text_fields["backtest_plan"]),
        ("Rejection Conditions", text_fields["rejection_conditions"]),
    ]
    for heading, value in section_values:
        lines.extend([f"## {heading}", ""])
        if isinstance(value, list):
            lines.extend(_bullet_lines(value))
        else:
            lines.append(str(value).strip())
        lines.append("")

    if text_fields["reviewer_notes"]:
        lines.extend(["## Reviewer Notes", ""])
        lines.extend(_bullet_lines(text_fields["reviewer_notes"]))
        lines.append("")

    lines.extend(
        [
            "## Generation Boundary",
            "",
            f"- {STRATEGY_PROPOSAL_NOTICE}",
            "- This proposal is not eligible for strategy code generation unless "
            "the sidecar metadata status is `accepted`.",
            "- Local JSON, CSV, Markdown, and log artifacts remain the source of truth.",
            "",
        ]
    )
    return "\n".join(lines)


def _research_reference_lines(references: Sequence[dict[str, Any]]) -> list[str]:
    return [
        (
            f"{ref.get('reference_id')}: {ref.get('title')} "
            f"({ref.get('source')}, {ref.get('published_at')}) - "
            f"{ref.get('relevance')} Motivates: "
            f"{', '.join(ref.get('motivated_thesis_ids', []))}."
        )
        for ref in references
    ] or ["No structured research reference supplied."]


def _required_input_checks(inputs: StrategyProposalInputs) -> list[StrategyProposalCheck]:
    checks = [
        _check("strategy_name_present", bool(inputs.strategy_name.strip()), "blocker",
               "Strategy name is required."),
        _check("strategy_type_present", bool(inputs.strategy_type.strip()), "blocker",
               "Strategy type is required."),
        _check("target_exchange_present", bool(inputs.target_exchange.strip()), "blocker",
               "Target exchange is required."),
        _check("target_symbols_present", _non_empty_sequence(inputs.target_symbols), "blocker",
               "At least one target symbol is required."),
        _check("timeframe_present", bool(inputs.timeframe.strip()), "blocker",
               "Timeframe is required."),
        _check("summary_present", bool(inputs.summary.strip()), "blocker",
               "Summary is required."),
        _check("hypothesis_present", bool(inputs.hypothesis.strip()), "blocker",
               "Hypothesis is required."),
        _check("market_condition_present", bool(inputs.market_condition.strip()), "blocker",
               "Market condition is required."),
        _check("entry_logic_present", bool(inputs.entry_logic.strip()), "blocker",
               "Entry logic is required."),
        _check("exit_logic_present", bool(inputs.exit_logic.strip()), "blocker",
               "Exit logic is required."),
        _check("risk_logic_present", bool(inputs.risk_logic.strip()), "blocker",
               "Risk logic is required."),
        _check("required_data_present", _non_empty_sequence(inputs.required_data), "blocker",
               "Required data must be explicit."),
        _check("parameters_present", _non_empty_sequence(inputs.parameters), "blocker",
               "Parameters must be explicit."),
        _check(
            "expected_failure_cases_present",
            _non_empty_sequence(inputs.expected_failure_cases),
            "blocker",
            "Expected failure cases must be explicit.",
        ),
        _check("backtest_plan_present", bool(inputs.backtest_plan.strip()), "blocker",
               "Backtest plan is required."),
        _check(
            "rejection_conditions_present",
            _non_empty_sequence(inputs.rejection_conditions),
            "blocker",
            "Rejection conditions must be explicit.",
        ),
    ]
    return checks


def _scope_checks(inputs: StrategyProposalInputs) -> list[StrategyProposalCheck]:
    return [
        _check(
            "long_short_scope_long_only",
            _normalizes_to_long_only(inputs.long_short),
            "blocker",
            "Strategy proposal scope must be long-only with no shorting.",
            {"long_short": inputs.long_short},
        ),
        _check(
            "spot_or_futures_supported",
            inputs.spot_or_futures.strip().lower() in {"spot", "futures"},
            "blocker",
            "Spot/futures mode must be either spot or futures.",
            {"spot_or_futures": inputs.spot_or_futures},
        ),
    ]


def _hypothesis_candidate_checks(
    inputs: StrategyProposalInputs, created_at: str
) -> list[StrategyProposalCheck]:
    retry_budget = int(inputs.retry_budget_per_thesis)
    thesis_retry_count = int(inputs.thesis_retry_count)
    parameter_retry_limit = int(inputs.parameter_only_retry_limit)
    parameter_retry_count = int(inputs.parameter_only_retry_count)
    raw_generator_mode = str(inputs.generator_mode or "rule_based").strip().lower()
    raw_logic_variant = str(inputs.strategy_logic_variant or "").strip().lower()
    failure_codes = _failure_taxonomy_codes(inputs)
    raw_failure_codes = [str(code).strip() for code in inputs.failure_taxonomy_codes]
    research_references = _research_references(inputs, created_at)
    thesis_id = _thesis_id(inputs, created_at)
    research_refs_structured = all(
        all(ref.get(field_name) for field_name in ("reference_id", "title", "source"))
        for ref in research_references
    )
    research_refs_have_relevance = all(
        bool(str(ref.get("relevance") or "").strip()) for ref in research_references
    )
    research_refs_have_dates = all(
        bool(str(ref.get("published_at") or "").strip()) for ref in research_references
    )
    research_refs_motivate_thesis = all(
        thesis_id in {
            str(item).strip()
            for item in ref.get("motivated_thesis_ids", [])
            if str(item).strip()
        }
        for ref in research_references
    )
    return [
        _check(
            "generator_mode_supported",
            raw_generator_mode in ALLOWED_GENERATOR_MODES,
            "blocker",
            "Generator mode must be rule_based, freqai, or hybrid_ml.",
            {"generator_mode": inputs.generator_mode},
        ),
        _check(
            "strategy_logic_variant_supported",
            not raw_logic_variant or raw_logic_variant in ALLOWED_STRATEGY_LOGIC_VARIANTS,
            "blocker",
            "Strategy logic variant must be one of the supported hypothesis families.",
            {"strategy_logic_variant": inputs.strategy_logic_variant},
        ),
        _check(
            "thesis_retry_budget_configured",
            retry_budget > 0,
            "blocker",
            "retry_budget_per_thesis must be greater than zero.",
        ),
        _check(
            "thesis_retry_budget_not_exceeded",
            thesis_retry_count <= retry_budget,
            "blocker",
            "Thesis retry budget is already exceeded; switch to a distinct hypothesis family.",
        ),
        _check(
            "parameter_only_retry_limit_configured",
            parameter_retry_limit > 0,
            "blocker",
            "parameter_only_retry_limit must be greater than zero.",
        ),
        _check(
            "parameter_only_retry_guard",
            parameter_retry_count <= parameter_retry_limit,
            "blocker",
            "Parameter-only retry count exceeds the configured limit.",
        ),
        _check(
            "distinct_hypothesis_family_after_repeated_failure",
            bool(inputs.force_distinct_hypothesis_family) or thesis_retry_count <= 1,
            "blocker",
            "Repeated failures require force_distinct_hypothesis_family=true.",
        ),
        _check(
            "failure_taxonomy_codes_normalized",
            len(failure_codes) == len([code for code in raw_failure_codes if code]),
            "blocker",
            "Failure taxonomy codes must use normalized Bot Factory values.",
            {"allowed": sorted(ALLOWED_FAILURE_TAXONOMY_CODES)},
        ),
        _check(
            "research_references_present",
            bool(research_references),
            "blocker",
            "At least one structured theory or literature reference is required.",
        ),
        _check(
            "research_references_structured",
            bool(research_references) and research_refs_structured,
            "blocker",
            "Research references must include reference_id, title, and source.",
        ),
        _check(
            "research_references_have_relevance",
            bool(research_references) and research_refs_have_relevance,
            "blocker",
            "Research references must explain why they are relevant.",
        ),
        _check(
            "research_references_record_publication_date",
            bool(research_references) and research_refs_have_dates,
            "blocker",
            "Research references must record a publication date or version date.",
        ),
        _check(
            "research_references_motivate_current_thesis",
            bool(research_references) and research_refs_motivate_thesis,
            "blocker",
            "Research references must list the current thesis_id as motivated.",
            {"thesis_id": thesis_id},
        ),
    ]


def _failure_synthesis_novelty_checks(
    inputs: StrategyProposalInputs, created_at: str
) -> list[StrategyProposalCheck]:
    checks: list[StrategyProposalCheck] = []
    thesis_id = _thesis_id(inputs, created_at)
    current_family_tokens = _current_family_tokens(inputs)
    research_reference_count = len(_research_references(inputs, created_at))
    parameter_retry_count = int(inputs.parameter_only_retry_count)
    for index, item in enumerate(_failure_synthesis_evidence_inputs(inputs), start=1):
        path = _resolve_workspace_path(item.path, inputs.root_dir)
        relative_path = _safe_relative_path(path, inputs.root_dir)
        payload = _load_failure_synthesis_payload(path, inputs.root_dir)
        freshness = _failure_synthesis_freshness(
            failure_synthesis=payload,
            failure_synthesis_path=path,
            root_dir=inputs.root_dir,
        )
        brief = payload.get("next_research_brief", {}) if payload else {}
        failed_thesis_ids = {
            str(value).strip()
            for value in brief.get("failed_thesis_ids", [])
            if str(value).strip()
        }
        prior_families = {
            str(value).strip()
            for value in brief.get("prior_hypothesis_families_to_avoid_as_default", [])
            if str(value).strip()
        }
        repeated_families = sorted(prior_families & current_family_tokens)
        minimum_reference_count = int(brief.get("minimum_research_reference_count") or 0)
        requires_new_thesis = bool(brief.get("requires_new_thesis_id"))
        requires_new_references = bool(brief.get("requires_new_research_references"))
        parameter_only_retry_allowed = bool(brief.get("parameter_only_retry_allowed", True))
        checks.extend(
            [
                _check(
                    f"failure_synthesis_{index}_parseable",
                    payload is not None,
                    "blocker",
                    "Failure synthesis evidence must be a parseable local JSON object.",
                    {"path": relative_path},
                ),
                _check(
                    f"failure_synthesis_{index}_completed",
                    bool(payload) and payload.get("status") == "completed",
                    "blocker",
                    "Failure synthesis evidence must have status=completed.",
                    {"path": relative_path},
                ),
                _check(
                    f"failure_synthesis_{index}_is_latest",
                    not freshness["checked"] or freshness["is_latest"],
                    "blocker",
                    "Failure synthesis evidence must be the latest registry synthesis.",
                    {
                        "failure_synthesis_latest_checked": freshness["checked"],
                        "failure_synthesis_is_latest": freshness["is_latest"],
                        "latest_failure_synthesis_path": freshness["latest_path"],
                        "latest_failure_synthesis_id": freshness[
                            "latest_synthesis_id"
                        ],
                        "latest_failure_synthesis_generated_at": freshness[
                            "latest_generated_at"
                        ],
                    },
                ),
                _check(
                    f"failure_synthesis_{index}_blocks_parameter_only_retry",
                    parameter_only_retry_allowed or parameter_retry_count == 0,
                    "blocker",
                    "Failure synthesis forbids parameter-only retries; choose a new thesis mechanism.",
                    {
                        "parameter_only_retry_allowed": parameter_only_retry_allowed,
                        "parameter_only_retry_count": parameter_retry_count,
                    },
                ),
                _check(
                    f"failure_synthesis_{index}_requires_new_thesis_id",
                    not (requires_new_thesis and thesis_id in failed_thesis_ids),
                    "blocker",
                    "Failure synthesis requires a new thesis_id outside failed thesis IDs.",
                    {"thesis_id": thesis_id, "failed_thesis_ids": sorted(failed_thesis_ids)},
                ),
                _check(
                    f"failure_synthesis_{index}_requires_new_hypothesis_family",
                    not (requires_new_thesis and repeated_families),
                    "blocker",
                    "Failure synthesis requires a distinct hypothesis family outside failed families.",
                    {
                        "current_family_tokens": sorted(current_family_tokens),
                        "repeated_families": repeated_families,
                    },
                ),
                _check(
                    f"failure_synthesis_{index}_minimum_research_references",
                    not requires_new_references
                    or research_reference_count >= minimum_reference_count,
                    "blocker",
                    "Failure synthesis requires enough new structured research references.",
                    {
                        "research_reference_count": research_reference_count,
                        "minimum_research_reference_count": minimum_reference_count,
                    },
                ),
            ]
        )
    return checks


def _forbidden_dependency_checks(
    inputs: StrategyProposalInputs, text_fields: dict[str, Any]
) -> list[StrategyProposalCheck]:
    dependency_text = "\n".join(
        [
            text_fields["summary"],
            text_fields["hypothesis"],
            text_fields["market_condition"],
            text_fields["entry_logic"],
            text_fields["exit_logic"],
            text_fields["risk_logic"],
            _join_text(text_fields["required_data"]),
            _join_text(text_fields["parameters"]),
            _join_text(text_fields["expected_failure_cases"]),
            text_fields["backtest_plan"],
        ]
    )
    all_text = "\n".join(
        [
            dependency_text,
            _join_text(text_fields["rejection_conditions"]),
            _join_text(text_fields["reviewer_notes"]),
        ]
    )
    secret_findings = _secret_findings(all_text)
    private_env_findings = _private_env_findings(all_text)
    leverage_findings = _leverage_above_one_findings(dependency_text)

    return [
        _check(
            "no_future_data_dependency",
            not _non_negated_matches(_FUTURE_DATA_RE, dependency_text),
            "blocker",
            "Proposal must not depend on future data, lookahead, next-candle values, or negative shifts.",
        ),
        _check(
            "no_live_only_data_dependency",
            not _non_negated_matches(_LIVE_ONLY_DATA_RE, dependency_text),
            "blocker",
            "Proposal must not depend on live-only, real-time, streaming-only, or unclosed-candle data.",
        ),
        _check(
            "no_account_or_position_data_dependency",
            not _non_negated_matches(_ACCOUNT_POSITION_RE, dependency_text),
            "blocker",
            "Proposal must not depend on account, balance, fill, or position data.",
        ),
        _check(
            "no_order_endpoint_dependency",
            not _non_negated_matches(_ORDER_ENDPOINT_RE, dependency_text),
            "blocker",
            "Proposal must not depend on exchange order endpoints or direct order placement.",
        ),
        _check(
            "no_api_key_or_secret_dependency",
            not _non_negated_matches(_CREDENTIAL_DEPENDENCY_RE, dependency_text)
            and not secret_findings
            and not private_env_findings,
            "blocker",
            "Proposal must not depend on API keys, secrets, private environment values, or credentials.",
            {
                "secret_reference_count": len(secret_findings),
                "private_env_reference_count": len(private_env_findings),
            },
        ),
        _check(
            "no_leverage_above_one_dependency",
            not leverage_findings,
            "blocker",
            "Proposal must not use leverage above 1.0.",
            {"findings": leverage_findings},
        ),
        _check(
            "no_shorting_dependency",
            not _non_negated_matches(_SHORTING_RE, dependency_text),
            "blocker",
            "Proposal must not include shorting behavior.",
        ),
        _check(
            "no_paper_live_or_process_control_dependency",
            not _non_negated_matches(_PROCESS_CONTROL_RE, dependency_text),
            "blocker",
            "Proposal generation must not depend on paper/live startup or process control.",
        ),
        _check(
            "backtest_plan_requires_broader_validation",
            _backtest_plan_has_broader_validation(inputs.backtest_plan),
            "blocker",
            "Backtest plan must not depend on one narrow backtest period; include walk-forward or broader validation.",
        ),
    ]


def _evidence_summary(
    inputs: StrategyProposalInputs,
) -> tuple[list[dict[str, Any]], list[StrategyProposalCheck]]:
    evidence: list[dict[str, Any]] = []
    checks: list[StrategyProposalCheck] = []
    for index, item in enumerate(inputs.evidence_paths, start=1):
        label = _safe_label(item.label or f"evidence_{index}")
        path = _resolve_workspace_path(item.path, inputs.root_dir)
        within_workspace = _path_is_within_root(path, inputs.root_dir)
        exists = within_workspace and path.is_file()
        status = "accepted" if within_workspace and exists else "blocked"
        reasons: list[str] = []
        if not within_workspace:
            reasons.append("Path does not resolve inside the repository workspace.")
        if within_workspace and not exists:
            reasons.append("Path does not exist as a local file.")

        info: dict[str, Any] = {
            "label": label,
            "path": _safe_relative_path(path, inputs.root_dir),
            "status": status,
            "reasons": reasons,
        }
        checks.append(
            _check(
                f"evidence_{label}_within_workspace",
                within_workspace,
                "blocker",
                "Evidence path must resolve inside the repository workspace.",
                {"path": info["path"]},
            )
        )
        checks.append(
            _check(
                f"evidence_{label}_file_present",
                exists,
                "blocker",
                "Evidence path must exist as a local file.",
                {"path": info["path"]},
            )
        )
        if exists:
            stat = path.stat()
            info.update({"bytes": stat.st_size, "sha256": _sha256_file(path)})
            evidence_checks, blocked_reasons = _evidence_content_checks(label, path)
            checks.extend(evidence_checks)
            if blocked_reasons:
                info["status"] = "blocked"
                info["reasons"].extend(blocked_reasons)
        evidence.append(info)
    return evidence, checks


def _failure_synthesis_evidence_inputs(
    inputs: StrategyProposalInputs,
) -> list[StrategyProposalEvidenceInput]:
    return [
        item
        for item in inputs.evidence_paths
        if _safe_label(item.label) in {"candidate_failure_synthesis", "failure_synthesis"}
    ]


def _load_failure_synthesis_payload(path: Path, root_dir: Path) -> dict[str, Any] | None:
    if not _path_is_within_root(path, root_dir) or not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("factory") != "candidate_failure_synthesis":
        return None
    if not isinstance(payload.get("next_research_brief"), dict):
        return None
    return payload


def _failure_synthesis_freshness(
    *,
    failure_synthesis: dict[str, Any] | None,
    failure_synthesis_path: Path,
    root_dir: Path,
) -> dict[str, Any]:
    default_root = (root_dir / "registry" / "strategies" / "synthesis").resolve()
    current_path = failure_synthesis_path.resolve()
    try:
        current_path.relative_to(default_root)
    except ValueError:
        return _failure_synthesis_freshness_unchecked()
    if not default_root.is_dir():
        return _failure_synthesis_freshness_unchecked()

    candidates: list[tuple[datetime, Path, str]] = []
    for path in default_root.rglob("candidate_failure_synthesis.json"):
        payload = _load_failure_synthesis_payload(path, root_dir)
        if payload is None:
            continue
        generated_at = _parse_datetime(payload.get("generated_at"))
        if generated_at is None:
            continue
        candidates.append(
            (
                generated_at,
                path.resolve(),
                str(payload.get("synthesis_id") or path.parent.name),
            )
        )

    current_generated_at = _parse_datetime(
        failure_synthesis.get("generated_at")
        if isinstance(failure_synthesis, dict)
        else None
    )
    if not candidates or current_generated_at is None:
        return _failure_synthesis_freshness_unchecked()

    latest_generated_at, latest_path, latest_id = max(
        candidates,
        key=lambda item: (item[0], str(item[1])),
    )
    return {
        "checked": True,
        "is_latest": current_path == latest_path,
        "latest_path": _safe_relative_path(latest_path, root_dir),
        "latest_synthesis_id": latest_id,
        "latest_generated_at": latest_generated_at.isoformat(),
    }


def _failure_synthesis_freshness_unchecked() -> dict[str, Any]:
    return {
        "checked": False,
        "is_latest": True,
        "latest_path": None,
        "latest_synthesis_id": None,
        "latest_generated_at": None,
    }


def _parse_datetime(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _failure_synthesis_constraints(
    inputs: StrategyProposalInputs, created_at: str
) -> list[dict[str, Any]]:
    constraints: list[dict[str, Any]] = []
    thesis_id = _thesis_id(inputs, created_at)
    current_family_tokens = _current_family_tokens(inputs)
    for item in _failure_synthesis_evidence_inputs(inputs):
        path = _resolve_workspace_path(item.path, inputs.root_dir)
        payload = _load_failure_synthesis_payload(path, inputs.root_dir)
        freshness = _failure_synthesis_freshness(
            failure_synthesis=payload,
            failure_synthesis_path=path,
            root_dir=inputs.root_dir,
        )
        brief = payload.get("next_research_brief", {}) if payload else {}
        prior_families = [
            str(value)
            for value in brief.get("prior_hypothesis_families_to_avoid_as_default", [])
        ]
        failed_thesis_ids = [str(value) for value in brief.get("failed_thesis_ids", [])]
        constraints.append(
            {
                "path": _safe_relative_path(path, inputs.root_dir),
                "status": payload.get("status") if payload else "unavailable",
                "ranking_path": payload.get("ranking_path") if payload else None,
                "failure_synthesis_latest_checked": freshness["checked"],
                "failure_synthesis_is_latest": freshness["is_latest"],
                "latest_failure_synthesis_path": freshness["latest_path"],
                "latest_failure_synthesis_id": freshness["latest_synthesis_id"],
                "latest_failure_synthesis_generated_at": freshness[
                    "latest_generated_at"
                ],
                "requires_new_thesis_id": bool(brief.get("requires_new_thesis_id")),
                "requires_new_research_references": bool(
                    brief.get("requires_new_research_references")
                ),
                "minimum_research_reference_count": int(
                    brief.get("minimum_research_reference_count") or 0
                ),
                "parameter_only_retry_allowed": bool(
                    brief.get("parameter_only_retry_allowed", True)
                ),
                "current_thesis_id": thesis_id,
                "failed_thesis_id_match": thesis_id in set(failed_thesis_ids),
                "current_family_tokens": sorted(current_family_tokens),
                "prior_hypothesis_families_to_avoid_as_default": prior_families,
                "repeated_family_matches": sorted(set(prior_families) & current_family_tokens),
                "blocked_next_actions": list(brief.get("blocked_next_actions") or []),
            }
        )
    return constraints


def _research_decision_gate_checks(
    inputs: StrategyProposalInputs, created_at: str
) -> list[StrategyProposalCheck]:
    checks: list[StrategyProposalCheck] = []
    thesis_id = _thesis_id(inputs, created_at)
    decision_inputs = _research_decision_evidence_inputs(inputs)
    requires_decision = _failure_synthesis_requires_research_decision(inputs)
    failure_synthesis_ids = set(_failure_synthesis_ids(inputs))
    structural_data_requirement = _proposal_structural_data_requirement(inputs)
    checks.append(
        _check(
            "research_decision_required_for_failure_synthesis",
            not requires_decision or bool(decision_inputs),
            "blocker",
            "Failure synthesis requires an approved research decision before proposal generation.",
            {"research_decision_count": len(decision_inputs)},
        )
    )
    checks.append(
        _check(
            "research_decision_required_for_structural_data",
            not structural_data_requirement["required"] or bool(decision_inputs),
            "blocker",
            "Structural-data proposals require an approved research decision with passing local data quality evidence.",
            {
                "structural_terms": structural_data_requirement["terms"],
                "research_decision_count": len(decision_inputs),
            },
        )
    )
    for index, item in enumerate(decision_inputs, start=1):
        path = _resolve_workspace_path(item.path, inputs.root_dir)
        relative_path = _safe_relative_path(path, inputs.root_dir)
        payload = _load_research_decision_payload(path, inputs.root_dir)
        thesis = payload.get("thesis", {}) if payload else {}
        novelty = payload.get("novelty_assessment", {}) if payload else {}
        safety = payload.get("safety_scope", {}) if payload else {}
        causal_map = payload.get("causal_failure_map", {}) if payload else {}
        if not isinstance(causal_map, dict):
            causal_map = {}
        causal_source_synthesis_id = str(
            causal_map.get("source_synthesis_id") or ""
        ).strip()
        research_references = payload.get("research_references", []) if payload else []
        refs_motivate_thesis = _research_decision_refs_motivate_thesis(
            research_references, thesis_id
        )
        decision_repeated_failed_family_matches = list(
            novelty.get("repeated_failed_family_matches", []) if payload else []
        )
        decision_failed_thesis_id_match = bool(
            novelty.get("failed_thesis_id_match") if payload else False
        )
        decision_local_failed_thesis_id_match = bool(
            novelty.get("local_falsification_failed_thesis_id_match")
            if payload
            else False
        )
        decision_local_failed_mechanism_matches = list(
            novelty.get("local_falsification_failed_mechanism_class_matches", [])
            if payload
            else []
        )
        decision_failure_synthesis_latest_checked = bool(
            novelty.get("failure_synthesis_latest_checked") if payload else False
        )
        decision_failure_synthesis_is_latest = (
            novelty.get("failure_synthesis_is_latest") if payload else None
        )
        causal_quality = _research_decision_causal_quality(causal_map)
        selection_score = _research_decision_selection_score(payload)
        structural_quality = _research_decision_structural_quality(
            payload, inputs.root_dir
        )
        structural_capability = _research_decision_structural_capability_support(
            structural_data_requirement,
            structural_quality,
        )
        local_falsification_handoff = _research_decision_local_falsification_handoff(
            payload,
            causal_map,
        )
        checks.extend(
            [
                _check(
                    f"research_decision_{index}_parseable",
                    payload is not None,
                    "blocker",
                    "Research decision evidence must be a parseable local JSON object.",
                    {"path": relative_path},
                ),
                _check(
                    f"research_decision_{index}_approved_for_proposal_generation",
                    bool(payload)
                    and payload.get("status") == "approved_for_proposal_generation"
                    and payload.get("proposal_generation_allowed") is True,
                    "blocker",
                    "Research decision must explicitly approve proposal generation.",
                    {
                        "status": payload.get("status") if payload else None,
                        "proposal_generation_allowed": (
                            payload.get("proposal_generation_allowed") if payload else None
                        ),
                    },
                ),
                _check(
                    f"research_decision_{index}_matches_current_thesis_id",
                    bool(payload) and thesis.get("thesis_id") == thesis_id,
                    "blocker",
                    "Research decision thesis_id must match the proposal thesis_id.",
                    {
                        "proposal_thesis_id": thesis_id,
                        "decision_thesis_id": thesis.get("thesis_id") if payload else None,
                    },
                ),
                _check(
                    f"research_decision_{index}_references_motivate_current_thesis",
                    bool(payload) and refs_motivate_thesis,
                    "blocker",
                    "Research decision references must motivate the current thesis_id.",
                    {"thesis_id": thesis_id},
                ),
                _check(
                    f"research_decision_{index}_does_not_directly_allow_code_generation",
                    bool(payload) and payload.get("code_generation_allowed") is False,
                    "blocker",
                    "Research decision may approve proposal generation, not direct code generation.",
                ),
                _check(
                    f"research_decision_{index}_novelty_assessment_passed",
                    bool(payload)
                    and not decision_failed_thesis_id_match
                    and not decision_repeated_failed_family_matches,
                    "blocker",
                    "Research decision novelty assessment must not repeat failed thesis IDs or families.",
                    {
                        "failed_thesis_id_match": decision_failed_thesis_id_match,
                        "repeated_failed_family_matches": (
                            decision_repeated_failed_family_matches
                        ),
                    },
                ),
                _check(
                    f"research_decision_{index}_outside_failure_synthesis_local_rejections",
                    bool(payload)
                    and not decision_local_failed_thesis_id_match
                    and not decision_local_failed_mechanism_matches,
                    "blocker",
                    (
                        "Research decision must not repeat validated local "
                        "falsification rejection memory from the latest synthesis."
                    ),
                    {
                        "local_falsification_failed_thesis_id_match": (
                            decision_local_failed_thesis_id_match
                        ),
                        "local_falsification_failed_mechanism_class_matches": (
                            decision_local_failed_mechanism_matches
                        ),
                    },
                ),
                _check(
                    f"research_decision_{index}_historical_only_safety_scope",
                    bool(payload)
                    and safety.get("historical_only") is True
                    and safety.get("paper_trading_started") is False
                    and safety.get("dry_run_trading_started") is False
                    and safety.get("live_trading") is False
                    and safety.get("exchange_order_placement") is False
                    and safety.get("process_control") is False,
                    "blocker",
                    "Research decision safety scope must be local historical-only.",
                ),
                _check(
                    f"research_decision_{index}_failure_synthesis_was_latest",
                    bool(payload)
                    and (
                        not decision_failure_synthesis_latest_checked
                        or decision_failure_synthesis_is_latest is True
                    ),
                    "blocker",
                    "Research decision must be based on the latest failure synthesis.",
                    {
                        "failure_synthesis_latest_checked": (
                            decision_failure_synthesis_latest_checked
                        ),
                        "failure_synthesis_is_latest": (
                            decision_failure_synthesis_is_latest
                        ),
                        "latest_failure_synthesis_path": novelty.get(
                            "latest_failure_synthesis_path"
                        )
                        if payload
                        else None,
                        "latest_failure_synthesis_id": novelty.get(
                            "latest_failure_synthesis_id"
                        )
                        if payload
                        else None,
                    },
                ),
                _check(
                    f"research_decision_{index}_uses_causal_failure_map",
                    not requires_decision
                    or (
                        bool(payload)
                        and causal_map.get("used") is True
                        and causal_map.get("available") is True
                    ),
                    "blocker",
                    "Research decision must consume a causal failure map when failure synthesis requires a research decision.",
                    {
                        "causal_failure_map_used": (
                            causal_map.get("used") if isinstance(causal_map, dict) else None
                        ),
                        "causal_failure_map_available": (
                            causal_map.get("available") if isinstance(causal_map, dict) else None
                        ),
                    },
                ),
                _check(
                    f"research_decision_{index}_causal_map_matches_failure_synthesis",
                    not requires_decision
                    or (
                        bool(payload)
                        and causal_map.get("used") is True
                        and bool(causal_source_synthesis_id)
                        and causal_source_synthesis_id in failure_synthesis_ids
                    ),
                    "blocker",
                    "Research decision causal map must come from the supplied failure synthesis.",
                    {
                        "causal_source_synthesis_id": (
                            causal_source_synthesis_id or None
                        ),
                        "supplied_failure_synthesis_ids": sorted(failure_synthesis_ids),
                    },
                ),
                _check(
                    f"research_decision_{index}_causal_failure_responses_complete",
                    not requires_decision
                    or (
                        bool(payload)
                        and causal_map.get("used") is True
                        and not causal_quality["missing_response_categories"]
                    ),
                    "blocker",
                    "Research decision causal responses must cover required failure categories.",
                    {
                        "missing_response_categories": causal_quality[
                            "missing_response_categories"
                        ]
                    },
                ),
                _check(
                    f"research_decision_{index}_causal_required_categories_match_current_policy",
                    not requires_decision
                    or (
                        bool(payload)
                        and causal_map.get("used") is True
                        and causal_quality["current_policy_available"]
                        and not causal_quality["missing_current_required_categories"]
                        and not causal_quality["missing_current_response_categories"]
                    ),
                    "blocker",
                    "Research decision causal responses must cover the current dominant/material category policy.",
                    {
                        "expected_required_categories": causal_quality[
                            "expected_required_categories"
                        ],
                        "claimed_required_categories": causal_quality[
                            "claimed_required_categories"
                        ],
                        "response_categories": causal_quality["response_categories"],
                        "missing_current_required_categories": causal_quality[
                            "missing_current_required_categories"
                        ],
                        "missing_current_response_categories": causal_quality[
                            "missing_current_response_categories"
                        ],
                    },
                ),
                _check(
                    f"research_decision_{index}_causal_response_quality_passed",
                    not requires_decision
                    or (
                        bool(payload)
                        and causal_map.get("used") is True
                        and not causal_quality["weak_response_categories"]
                        and not causal_quality["category_evidence_gaps"]
                        and not causal_quality["parameter_only_response_categories"]
                    ),
                    "blocker",
                    "Research decision causal responses must pass quality checks.",
                    {
                        "weak_response_categories": causal_quality[
                            "weak_response_categories"
                        ],
                        "category_evidence_gaps": causal_quality[
                            "category_evidence_gaps"
                        ],
                        "parameter_only_response_categories": causal_quality[
                            "parameter_only_response_categories"
                        ],
                    },
                ),
                _check(
                    f"research_decision_{index}_research_question_responses_complete",
                    not requires_decision
                    or not causal_quality["requires_research_question_responses"]
                    or (
                        bool(payload)
                        and causal_map.get("used") is True
                        and causal_quality["required_research_questions"]
                        and not causal_quality[
                            "missing_research_question_response_indexes"
                        ]
                        and not causal_quality[
                            "weak_research_question_response_indexes"
                        ]
                    ),
                    "blocker",
                    "Research decision must answer required causal-map research questions.",
                    {
                        "required_research_questions": causal_quality[
                            "required_research_questions"
                        ],
                        "response_question_indexes": causal_quality[
                            "research_question_response_indexes"
                        ],
                        "reported_missing_question_indexes": causal_quality[
                            "reported_missing_research_question_response_indexes"
                        ],
                        "computed_missing_question_indexes": causal_quality[
                            "computed_missing_research_question_response_indexes"
                        ],
                        "missing_question_indexes": causal_quality[
                            "missing_research_question_response_indexes"
                        ],
                        "weak_question_indexes": causal_quality[
                            "weak_research_question_response_indexes"
                        ],
                    },
                ),
                _check(
                    f"research_decision_{index}_research_selection_score_passed",
                    not requires_decision
                    or (
                        bool(payload)
                        and selection_score["available"]
                        and selection_score["passes_minimum"]
                    ),
                    "blocker",
                    "Research decision must include a passing research selection score.",
                    {
                        "score": selection_score["score"],
                        "minimum_score_required": selection_score[
                            "minimum_score_required"
                        ],
                        "passes_minimum": selection_score["passes_minimum"],
                        "failed_components": selection_score["failed_components"],
                    },
                ),
                _check(
                    f"research_decision_{index}_risk_weights_cover_required_categories",
                    not requires_decision
                    or not causal_quality["causal_risk_weights_present"]
                    or not causal_quality["missing_required_risk_weight_categories"],
                    "blocker",
                    "Risk-weighted causal maps must include risk weights for every required category.",
                    {
                        "missing_required_risk_weight_categories": causal_quality[
                            "missing_required_risk_weight_categories"
                        ],
                    },
                ),
                _check(
                    f"research_decision_{index}_risk_weighted_selection_score_present",
                    not requires_decision
                    or not causal_quality["causal_risk_weights_present"]
                    or (
                        selection_score["version"] == "research_selection_score_v2"
                        and selection_score["weighted_causal_score_available"]
                    ),
                    "blocker",
                    "Research decisions using risk-weighted causal maps must include research_selection_score_v2 weighted causal score details.",
                    {
                        "score_version": selection_score["version"],
                        "weighted_causal_score_available": selection_score[
                            "weighted_causal_score_available"
                        ],
                        "weighted_response_score": selection_score[
                            "weighted_response_score"
                        ],
                        "unanswered_required_risk_weight": selection_score[
                            "unanswered_required_risk_weight"
                        ],
                    },
                ),
                _check(
                    f"research_decision_{index}_local_falsification_handoff_passed",
                    not local_falsification_handoff["required"]
                    or local_falsification_handoff["passed"],
                    "blocker",
                    "High-risk cost-sensitive research decisions must preserve a passing local falsification handoff before proposal generation.",
                    local_falsification_handoff,
                ),
                _check(
                    f"research_decision_{index}_structural_data_quality_report_present",
                    not structural_data_requirement["required"]
                    or structural_quality["valid"],
                    "blocker",
                    "Structural-data proposals require the research decision to carry passing local data quality report evidence.",
                    {
                        "structural_terms": structural_data_requirement["terms"],
                        "quality_report_paths": [
                            report["path"] for report in structural_quality["reports"]
                        ],
                        "quality_report_paths_exist": structural_quality[
                            "report_paths_exist"
                        ],
                        "local_data_quality_reports_valid_check_passed": (
                            structural_quality[
                                "local_data_quality_reports_valid_check_passed"
                            ]
                        ),
                        "structural_quality_check_passed": structural_quality[
                            "structural_quality_check_passed"
                        ],
                    },
                ),
                _check(
                    f"research_decision_{index}_structural_data_capability_report_present",
                    not structural_data_requirement["required"]
                    or structural_capability["capability_report_gate_passed"],
                    "blocker",
                    "Structural-data proposals require the research decision to carry passing structural capability report evidence.",
                    {
                        "structural_classes": structural_data_requirement["classes"],
                        "capability_report_paths": [
                            report["path"]
                            for report in structural_quality["capability_reports"]
                        ],
                        "capability_report_paths_exist": structural_quality[
                            "capability_report_paths_exist"
                        ],
                        "capability_reports_valid": structural_quality[
                            "capability_reports_valid"
                        ],
                        "capability_reports_valid_check_passed": structural_quality[
                            "capability_reports_valid_check_passed"
                        ],
                        "structural_capability_check_passed": structural_quality[
                            "structural_capability_check_passed"
                        ],
                    },
                ),
                _check(
                    f"research_decision_{index}_structural_data_capability_supports_required_classes",
                    not structural_data_requirement["required"]
                    or structural_capability["required_classes_supported"],
                    "blocker",
                    "Structural-data proposals require local research support for every required structural data class.",
                    structural_capability,
                ),
            ]
        )
    return checks


def _edge_discovery_gate_checks(
    inputs: StrategyProposalInputs, created_at: str
) -> list[StrategyProposalCheck]:
    handoff = _edge_discovery_handoff(inputs, created_at)
    return [
        _check(
            "edge_discovery_handoff_artifact_present",
            handoff["artifact_count"] > 0,
            "blocker",
            "Strategy proposals require a passing edge-discovery artifact before proposal generation.",
            handoff,
        ),
        _check(
            "edge_discovery_handoff_passed",
            handoff["passed"],
            "blocker",
            "Strategy proposals require positive post-cost Edge Discovery evidence for the current thesis.",
            handoff,
        ),
    ]


def _edge_discovery_handoff(
    inputs: StrategyProposalInputs, created_at: str
) -> dict[str, Any]:
    thesis_id = _thesis_id(inputs, created_at)
    artifacts: list[dict[str, Any]] = []
    for item in _edge_discovery_evidence_inputs(inputs):
        path = _resolve_workspace_path(item.path, inputs.root_dir)
        within_workspace = _path_is_within_root(path, inputs.root_dir)
        payload = _load_edge_discovery_payload(path, inputs.root_dir)
        safety = payload.get("safety_scope", {}) if payload else {}
        promotion_gate = payload.get("promotion_gate", {}) if payload else {}
        anti_parameter_search = (
            payload.get("anti_parameter_search", {}) if payload else {}
        )
        best_horizon = payload.get("best_horizon_by_net_edge") if payload else None
        best_horizon = best_horizon if isinstance(best_horizon, dict) else {}
        horizon_results = payload.get("horizon_results", []) if payload else []
        candidate = {
            "path": _safe_relative_path(path, inputs.root_dir),
            "within_workspace": within_workspace,
            "file_present": within_workspace and path.is_file(),
            "parseable": payload is not None,
            "factory": payload.get("factory") if payload else None,
            "factory_valid": bool(payload)
            and payload.get("factory") == "research_edge_discovery",
            "status": payload.get("status") if payload else None,
            "status_passed": bool(payload) and payload.get("status") == "passed",
            "proposal_generation_allowed": bool(payload)
            and (
                payload.get("proposal_generation_allowed") is True
                or promotion_gate.get("proposal_generation_allowed") is True
            ),
            "direct_strategy_codegen_blocked": bool(payload)
            and payload.get("strategy_codegen_allowed") is False
            and promotion_gate.get("strategy_codegen_allowed") is False,
            "proposal_thesis_id": thesis_id,
            "edge_thesis_id": payload.get("thesis_id") if payload else None,
            "thesis_id_match": bool(payload) and payload.get("thesis_id") == thesis_id,
            "safety_scope_valid": _edge_discovery_safety_scope_valid(safety),
            "anti_parameter_search_valid": bool(payload)
            and anti_parameter_search.get("valid") is True,
            "passing_horizon_count": _non_negative_int(
                payload.get("passing_horizon_count") if payload else None
            ),
            "net_edge_positive": _edge_discovery_net_edge_positive(
                best_horizon,
                horizon_results,
            ),
            "blocked_next_actions": _string_list(
                payload.get("blocked_next_actions", []) if payload else []
            ),
            "blocker_names": [
                str(check.get("name"))
                for check in payload.get("blockers", [])
                if isinstance(check, dict) and check.get("name")
            ]
            if payload
            else [],
        }
        candidate["passed"] = (
            candidate["within_workspace"]
            and candidate["file_present"]
            and candidate["parseable"]
            and candidate["factory_valid"]
            and candidate["status_passed"]
            and candidate["proposal_generation_allowed"]
            and candidate["direct_strategy_codegen_blocked"]
            and candidate["thesis_id_match"]
            and candidate["safety_scope_valid"]
            and candidate["anti_parameter_search_valid"]
            and candidate["passing_horizon_count"] > 0
            and candidate["net_edge_positive"]
            and not candidate["blocker_names"]
        )
        artifacts.append(candidate)
    passing = [artifact for artifact in artifacts if artifact["passed"]]
    return {
        "required": True,
        "passed": bool(passing),
        "artifact_count": len(artifacts),
        "parseable_artifact_count": sum(1 for item in artifacts if item["parseable"]),
        "matching_thesis_artifact_count": sum(
            1 for item in artifacts if item["thesis_id_match"]
        ),
        "passing_edge_artifact_count": len(passing),
        "paths_valid": bool(artifacts)
        and all(item["within_workspace"] and item["file_present"] for item in artifacts),
        "factory_valid": bool(artifacts)
        and all(item["factory_valid"] for item in artifacts),
        "safety_scope_valid": bool(artifacts)
        and all(item["safety_scope_valid"] for item in artifacts),
        "anti_parameter_search_valid": bool(artifacts)
        and all(item["anti_parameter_search_valid"] for item in artifacts),
        "direct_strategy_codegen_blocked": bool(artifacts)
        and all(item["direct_strategy_codegen_blocked"] for item in artifacts),
        "artifact_paths": [item["path"] for item in artifacts],
        "blocked_next_actions": _edge_discovery_blocked_next_actions(artifacts),
        "blocker_names": _edge_discovery_blocker_names(artifacts),
        "artifacts": artifacts,
    }


def _edge_discovery_evidence_inputs(
    inputs: StrategyProposalInputs,
) -> list[StrategyProposalEvidenceInput]:
    return [
        item
        for item in inputs.evidence_paths
        if _safe_label(item.label) in {"edge_discovery", "research_edge_discovery"}
    ]


def _load_edge_discovery_payload(path: Path, root_dir: Path) -> dict[str, Any] | None:
    if not _path_is_within_root(path, root_dir) or not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _edge_discovery_safety_scope_valid(safety_scope: Any) -> bool:
    if not isinstance(safety_scope, dict):
        return False
    unsafe_flags = (
        "future_data",
        "backtest_started",
        "strategy_code_generated",
        "paper_trading_started",
        "dry_run_trading_started",
        "live_trading",
        "exchange_order_placement",
        "shorting",
        "process_control",
    )
    leverage = _float_or_none(safety_scope.get("leverage"))
    return (
        safety_scope.get("historical_only") is True
        and safety_scope.get("local_artifacts_source_of_truth") is True
        and all(not bool(safety_scope.get(flag)) for flag in unsafe_flags)
        and (leverage is None or leverage <= 1.0)
    )


def _edge_discovery_net_edge_positive(
    best_horizon: dict[str, Any], horizon_results: Any
) -> bool:
    best_net_edge = _float_or_none(best_horizon.get("net_edge_bps"))
    if best_horizon.get("status") == "passed" and best_net_edge is not None:
        return best_net_edge > 0.0
    if not isinstance(horizon_results, list):
        return False
    for item in horizon_results:
        if not isinstance(item, dict) or item.get("status") != "passed":
            continue
        net_edge = _float_or_none(item.get("net_edge_bps"))
        if net_edge is not None and net_edge > 0.0:
            return True
    return False


def _edge_discovery_blocked_next_actions(
    artifacts: Sequence[dict[str, Any]]
) -> list[str]:
    if any(artifact.get("passed") is True for artifact in artifacts):
        return []
    actions: list[str] = []
    for artifact in artifacts:
        for action in _string_list(artifact.get("blocked_next_actions", [])):
            if action not in actions:
                actions.append(action)
    if not actions:
        actions.append("proposal_generation_without_passing_edge_discovery")
    return actions


def _edge_discovery_blocker_names(
    artifacts: Sequence[dict[str, Any]]
) -> list[str]:
    names: list[str] = []
    for artifact in artifacts:
        for name in _string_list(artifact.get("blocker_names", [])):
            if name not in names:
                names.append(name)
    if not artifacts:
        names.append("edge_discovery_artifact_missing")
    return names


def _failure_synthesis_requires_research_decision(inputs: StrategyProposalInputs) -> bool:
    for item in _failure_synthesis_evidence_inputs(inputs):
        path = _resolve_workspace_path(item.path, inputs.root_dir)
        payload = _load_failure_synthesis_payload(path, inputs.root_dir)
        brief = payload.get("next_research_brief", {}) if payload else {}
        if bool(brief.get("requires_new_thesis_id")) or bool(
            brief.get("requires_new_research_references")
        ):
            return True
    return False


def _failure_synthesis_ids(inputs: StrategyProposalInputs) -> list[str]:
    synthesis_ids: list[str] = []
    for item in _failure_synthesis_evidence_inputs(inputs):
        path = _resolve_workspace_path(item.path, inputs.root_dir)
        payload = _load_failure_synthesis_payload(path, inputs.root_dir)
        synthesis_id = str(payload.get("synthesis_id") or "").strip() if payload else ""
        if synthesis_id:
            synthesis_ids.append(synthesis_id)
    return synthesis_ids


def _research_decision_evidence_inputs(
    inputs: StrategyProposalInputs,
) -> list[StrategyProposalEvidenceInput]:
    return [
        item
        for item in inputs.evidence_paths
        if _safe_label(item.label)
        in {"research_decision", "research_selection_decision", "research_selection_gate"}
    ]


def _load_research_decision_payload(path: Path, root_dir: Path) -> dict[str, Any] | None:
    if not _path_is_within_root(path, root_dir) or not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("factory") != "research_selection_gate":
        return None
    if not isinstance(payload.get("thesis"), dict):
        return None
    return payload


def _research_decision_refs_motivate_thesis(
    references: Any, thesis_id: str
) -> bool:
    if not isinstance(references, list) or not references:
        return False
    for item in references:
        if not isinstance(item, dict):
            return False
        motivated = item.get("motivated_thesis_ids", [])
        if isinstance(motivated, str):
            motivated = [motivated]
        if thesis_id not in {str(value).strip() for value in motivated}:
            return False
    return True


def _research_decision_causal_quality(causal_map: Any) -> dict[str, Any]:
    if not isinstance(causal_map, dict):
        causal_map = {}
    claimed_required = _causal_category_list(
        causal_map.get("required_categories_to_address", [])
    )
    response_categories = _causal_category_list(causal_map.get("response_categories", []))
    expected_required = _expected_research_decision_required_categories(causal_map)
    required_questions = [
        _sanitize_text(str(question)).strip()
        for question in causal_map.get("required_research_questions", []) or []
        if str(question).strip()
    ]
    response_question_indexes = _positive_int_list(
        causal_map.get("research_question_response_indexes", [])
    )
    reported_missing_question_indexes = _positive_int_list(
        causal_map.get("missing_research_question_response_indexes", [])
    )
    computed_missing_question_indexes = [
        index
        for index in range(1, len(required_questions) + 1)
        if index not in response_question_indexes
    ]
    missing_question_indexes = list(
        dict.fromkeys(
            [*reported_missing_question_indexes, *computed_missing_question_indexes]
        )
    )
    return {
        "claimed_required_categories": claimed_required,
        "response_categories": response_categories,
        "expected_required_categories": expected_required,
        "current_policy_available": bool(expected_required),
        "missing_current_required_categories": [
            category for category in expected_required if category not in claimed_required
        ],
        "missing_current_response_categories": [
            category for category in expected_required if category not in response_categories
        ],
        "missing_response_categories": list(
            causal_map.get("missing_response_categories", []) or []
        ),
        "weak_response_categories": list(
            causal_map.get("weak_response_categories", []) or []
        ),
        "category_evidence_gaps": list(causal_map.get("category_evidence_gaps", []) or []),
        "parameter_only_response_categories": list(
            causal_map.get("parameter_only_response_categories", []) or []
        ),
        "requires_research_question_responses": bool(
            causal_map.get("requires_research_question_responses")
        ),
        "required_research_questions": required_questions,
        "research_question_response_indexes": response_question_indexes,
        "reported_missing_research_question_response_indexes": (
            reported_missing_question_indexes
        ),
        "computed_missing_research_question_response_indexes": (
            computed_missing_question_indexes
        ),
        "missing_research_question_response_indexes": missing_question_indexes,
        "weak_research_question_response_indexes": _positive_int_list(
            causal_map.get("weak_research_question_response_indexes", [])
        ),
        "causal_risk_weights_present": bool(
            _research_decision_causal_risk_weight_categories(causal_map)
        ),
        "missing_required_risk_weight_categories": [
            category
            for category in expected_required
            if category
            not in _research_decision_causal_risk_weight_categories(causal_map)
        ],
    }


def _research_decision_local_falsification_handoff(
    payload: Any,
    causal_map: dict[str, Any],
) -> dict[str, Any]:
    local_falsification = (
        payload.get("local_falsification_evidence", {})
        if isinstance(payload, dict)
        else {}
    )
    if not isinstance(local_falsification, dict):
        local_falsification = {}
    artifacts = local_falsification.get("artifacts", [])
    if not isinstance(artifacts, list):
        artifacts = []
    artifact_dicts = [item for item in artifacts if isinstance(item, dict)]
    blocker_names = _research_decision_local_falsification_blocker_names(payload)
    required = (
        local_falsification.get("high_risk_cost_evidence_required") is True
        or _research_decision_requires_high_risk_cost_falsification(causal_map)
    )
    artifact_count = max(
        _non_negative_int(local_falsification.get("artifact_count")),
        len(artifact_dicts),
    )
    parseable_count = max(
        _non_negative_int(local_falsification.get("parseable_artifact_count")),
        sum(1 for item in artifact_dicts if item.get("parseable") is True),
    )
    matching_count = max(
        _non_negative_int(
            local_falsification.get("matching_thesis_artifact_count")
        ),
        sum(1 for item in artifact_dicts if item.get("thesis_matches") is True),
    )
    passing_count = max(
        _non_negative_int(
            local_falsification.get("passing_cost_edge_artifact_count")
        ),
        sum(1 for item in artifact_dicts if item.get("cost_edge_passes") is True),
    )
    paths_valid = bool(artifact_dicts) and all(
        item.get("within_workspace") is True
        and item.get("exists") is True
        and item.get("parseable") is True
        for item in artifact_dicts
    )
    factory_valid = bool(artifact_dicts) and all(
        item.get("factory_valid") is True for item in artifact_dicts
    )
    safety_scope_valid = bool(artifact_dicts) and all(
        item.get("safety_scope_valid") is True for item in artifact_dicts
    )
    event_source_valid = bool(artifact_dicts) and all(
        item.get("event_source_valid") is True for item in artifact_dicts
    )
    context_alignment_valid = bool(artifact_dicts) and all(
        item.get("event_source_context_alignment_valid") is True
        for item in artifact_dicts
    )
    failure_synthesis_guard_valid = bool(artifact_dicts) and all(
        item.get("event_source_failure_synthesis_guard_valid") is True
        for item in artifact_dicts
    )
    required_checks_pass = (
        artifact_count > 0
        and parseable_count == artifact_count
        and paths_valid
        and factory_valid
        and safety_scope_valid
        and event_source_valid
        and context_alignment_valid
        and failure_synthesis_guard_valid
        and matching_count > 0
        and passing_count > 0
        and not blocker_names
    )
    return {
        "required": required,
        "passed": (not required) or required_checks_pass,
        "artifact_count": artifact_count,
        "parseable_artifact_count": parseable_count,
        "matching_thesis_artifact_count": matching_count,
        "passing_cost_edge_artifact_count": passing_count,
        "paths_valid": paths_valid,
        "factory_valid": factory_valid,
        "safety_scope_valid": safety_scope_valid,
        "event_source_valid": event_source_valid,
        "event_source_context_alignment_valid": context_alignment_valid,
        "event_source_failure_synthesis_guard_valid": failure_synthesis_guard_valid,
        "blocker_names": blocker_names,
        "artifact_paths": [
            str(item.get("path"))
            for item in artifact_dicts
            if str(item.get("path") or "").strip()
        ],
        "failure_reasons": list(
            local_falsification.get("failures", [])
            if isinstance(local_falsification.get("failures"), list)
            else []
        ),
    }


def _research_decision_requires_high_risk_cost_falsification(
    causal_map: dict[str, Any],
) -> bool:
    for item in causal_map.get("causal_risk_weights", []) or []:
        if not isinstance(item, dict):
            continue
        if _safe_label(str(item.get("category", ""))) != "cost_sensitive_mechanism":
            continue
        risk_score = _float_or_none(item.get("risk_score")) or 0.0
        if risk_score >= HIGH_RISK_CAUSAL_RESPONSE_MIN_RISK_SCORE:
            return True
    return False


def _research_decision_local_falsification_blocker_names(payload: Any) -> list[str]:
    if not isinstance(payload, dict):
        return []
    blockers = payload.get("blockers", [])
    if not isinstance(blockers, list):
        return []
    names: list[str] = []
    for item in blockers:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        if name.startswith("local_falsification_"):
            names.append(name)
    return names


def _non_negative_int(value: Any) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _positive_int_list(value: Any) -> list[int]:
    if not isinstance(value, list):
        return []
    indexes: list[int] = []
    for item in value:
        try:
            index = int(item)
        except (TypeError, ValueError):
            continue
        if index > 0 and index not in indexes:
            indexes.append(index)
    return indexes


def _research_decision_selection_score(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {
            "available": False,
            "score": None,
            "minimum_score_required": None,
            "passes_minimum": False,
            "failed_components": [],
            "version": None,
            "weighted_causal_score_available": False,
            "weighted_response_score": None,
            "unanswered_required_risk_weight": None,
        }
    score_payload = payload.get("research_selection_score", {})
    if not isinstance(score_payload, dict):
        return {
            "available": False,
            "score": None,
            "minimum_score_required": None,
            "passes_minimum": False,
            "failed_components": [],
            "version": None,
            "weighted_causal_score_available": False,
            "weighted_response_score": None,
            "unanswered_required_risk_weight": None,
        }
    score = _float_or_none(score_payload.get("score"))
    minimum = _float_or_none(score_payload.get("minimum_score_required"))
    if minimum is None:
        causal_map = payload.get("causal_failure_map", {})
        if isinstance(causal_map, dict):
            minimum = _float_or_none(
                causal_map.get("minimum_research_selection_score")
            )
    computed_passes = score is not None and (minimum is None or score >= minimum)
    explicit_passes = score_payload.get("passes_minimum") is True
    weighted_causal = _research_decision_weighted_causal_score(score_payload)
    return {
        "available": score is not None,
        "version": str(score_payload.get("version") or ""),
        "score": score,
        "minimum_score_required": minimum,
        "passes_minimum": computed_passes and explicit_passes,
        "failed_components": list(score_payload.get("failed_components", []) or []),
        "weighted_causal_score_available": weighted_causal["available"],
        "weighted_response_score": weighted_causal["weighted_response_score"],
        "unanswered_required_risk_weight": weighted_causal[
            "unanswered_required_risk_weight"
        ],
    }


def _research_decision_weighted_causal_score(score_payload: dict[str, Any]) -> dict[str, Any]:
    component = {}
    for item in score_payload.get("components", []) or []:
        if isinstance(item, dict) and item.get("name") == "causal_failure_response_quality":
            component = item
            break
    details = component.get("details", {}) if isinstance(component, dict) else {}
    if not isinstance(details, dict):
        details = {}
    weighted_response_score = _float_or_none(details.get("weighted_response_score"))
    unanswered_required_risk_weight = _float_or_none(
        details.get("unanswered_required_risk_weight")
    )
    category_scores = details.get("category_scores", [])
    available = (
        weighted_response_score is not None
        and unanswered_required_risk_weight is not None
        and isinstance(category_scores, list)
        and bool(category_scores)
    )
    return {
        "available": available,
        "weighted_response_score": weighted_response_score,
        "unanswered_required_risk_weight": unanswered_required_risk_weight,
    }


def _expected_research_decision_required_categories(causal_map: dict[str, Any]) -> list[str]:
    dominant_categories = _dominant_research_decision_categories(causal_map)
    required = [item["category"] for item in dominant_categories[:3]]
    threshold = _material_research_decision_category_threshold(causal_map)
    if threshold is not None:
        for item in dominant_categories[3:]:
            category_count = _float_or_none(item.get("candidate_count"))
            if category_count is not None and category_count >= threshold:
                required.append(item["category"])
    return list(dict.fromkeys(required))


def _research_decision_causal_risk_weight_categories(
    causal_map: dict[str, Any],
) -> set[str]:
    categories: set[str] = set()
    for item in causal_map.get("causal_risk_weights", []) or []:
        if not isinstance(item, dict):
            continue
        category = _safe_label(str(item.get("category", "")))
        if category:
            categories.add(category)
    return categories


def _dominant_research_decision_categories(causal_map: dict[str, Any]) -> list[dict[str, Any]]:
    dominant: list[dict[str, Any]] = []
    for item in causal_map.get("dominant_failure_categories", []) or []:
        if not isinstance(item, dict):
            continue
        category = _safe_label(str(item.get("category", "")))
        if not category:
            continue
        dominant.append(
            {
                "category": category,
                "candidate_count": item.get("candidate_count"),
            }
        )
    return dominant


def _material_research_decision_category_threshold(
    causal_map: dict[str, Any],
) -> float | None:
    candidate_count = _float_or_none(causal_map.get("candidate_count"))
    if candidate_count is None or candidate_count <= 0:
        return None
    share = _float_or_none(causal_map.get("material_category_min_share"))
    if share is None:
        share = MATERIAL_CAUSAL_CATEGORY_MIN_SHARE
    return candidate_count * share


def _causal_category_list(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    return [_safe_label(str(value)) for value in values if _safe_label(str(value))]


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _proposal_structural_data_requirement(inputs: StrategyProposalInputs) -> dict[str, Any]:
    text = " ".join(
        [
            str(inputs.strategy_type),
            str(inputs.thesis_type or ""),
            str(inputs.thesis_statement or ""),
            str(inputs.hypothesis),
            str(inputs.market_condition),
            str(inputs.entry_logic),
            str(inputs.backtest_plan),
            " ".join(str(item) for item in inputs.required_data),
            " ".join(str(item) for item in inputs.rejection_conditions),
            " ".join(str(item) for item in inputs.feature_list),
            " ".join(str(item) for item in inputs.rule_filters),
        ]
    )
    terms = list(
        dict.fromkeys(
            match.group(0).lower() for match in _STRUCTURAL_DATA_RE.finditer(text)
        )
    )
    return {
        "required": bool(terms),
        "terms": terms,
        "classes": _structural_data_classes(terms),
    }


def _structural_data_classes(terms: Sequence[str]) -> list[str]:
    classes: set[str] = set()
    for term in terms:
        normalized = term.lower().replace("-", " ").replace("_", " ")
        if "open" in normalized and "interest" in normalized:
            classes.add("open_interest")
        if (
            "long" in normalized
            and "short" in normalized
            and "ratio" in normalized
        ) or "account ratio" in normalized:
            classes.add("long_short_ratio")
        if "liquidation" in normalized:
            classes.add("liquidation")
        if (
            "order book" in normalized
            or "orderbook" in normalized
            or "market depth" in normalized
            or "book imbalance" in normalized
            or "depth imbalance" in normalized
        ):
            classes.add("order_book")
    return sorted(classes)


def _research_decision_structural_quality(
    payload: Any, root_dir: Path
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {
            "valid": False,
            "reports": [],
            "report_paths_exist": False,
            "local_data_quality_reports_valid_check_passed": False,
            "structural_quality_check_passed": False,
            "capability_reports": [],
            "capability_report_paths_exist": False,
            "capability_reports_valid": False,
            "capability_reports_valid_check_passed": False,
            "structural_capability_check_passed": False,
            "structural_capability_support_check_passed": False,
            "capability_usable_classes": [],
            "unsupported_required_classes": [],
        }
    thesis = payload.get("thesis", {})
    if not isinstance(thesis, dict):
        thesis = {}
    reports: list[dict[str, Any]] = []
    for raw_path in _string_list(thesis.get("local_data_quality_report_paths", [])):
        path = _resolve_workspace_path(Path(raw_path), root_dir)
        within_workspace = _path_is_within_root(path, root_dir)
        reports.append(
            {
                "path": _safe_relative_path(path, root_dir),
                "within_workspace": within_workspace,
                "exists": within_workspace and path.is_file(),
            }
        )
    capability_reports = _research_decision_structural_capability_reports(
        thesis,
        root_dir,
    )
    local_reports_valid = _payload_check_passed(
        payload, "local_data_quality_reports_valid"
    )
    structural_quality_passed = _payload_check_passed(
        payload, "structural_data_quality_report_present"
    )
    capability_reports_valid_check_passed = _payload_check_passed(
        payload, "structural_data_capability_reports_valid"
    )
    structural_capability_passed = _payload_check_passed(
        payload, "structural_data_capability_report_present"
    )
    structural_capability_support_passed = _payload_check_passed(
        payload, "structural_data_capability_supports_required_classes"
    )
    report_paths_exist = bool(reports) and all(
        report["within_workspace"] and report["exists"] for report in reports
    )
    capability_report_paths_exist = bool(capability_reports) and all(
        report["within_workspace"] and report["exists"] for report in capability_reports
    )
    capability_reports_valid = capability_report_paths_exist and all(
        report["parseable"] and report["factory_valid"] for report in capability_reports
    )
    capability_usable_classes = sorted(
        {
            item
            for report in capability_reports
            for item in report["local_research_usable"]
        }
    )
    return {
        "valid": report_paths_exist
        and local_reports_valid
        and structural_quality_passed,
        "reports": reports,
        "report_paths_exist": report_paths_exist,
        "local_data_quality_reports_valid_check_passed": local_reports_valid,
        "structural_quality_check_passed": structural_quality_passed,
        "capability_reports": capability_reports,
        "capability_report_paths_exist": capability_report_paths_exist,
        "capability_reports_valid": capability_reports_valid,
        "capability_reports_valid_check_passed": capability_reports_valid_check_passed,
        "structural_capability_check_passed": structural_capability_passed,
        "structural_capability_support_check_passed": (
            structural_capability_support_passed
        ),
        "capability_usable_classes": capability_usable_classes,
    }


def _research_decision_structural_capability_reports(
    thesis: dict[str, Any], root_dir: Path
) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    for raw_path in _string_list(
        thesis.get("structural_data_capability_report_paths", [])
    ):
        path = _resolve_workspace_path(Path(raw_path), root_dir)
        within_workspace = _path_is_within_root(path, root_dir)
        payload = _load_json_payload(path) if within_workspace and path.is_file() else None
        guidance = payload.get("proposal_guidance", {}) if isinstance(payload, dict) else {}
        if not isinstance(guidance, dict):
            guidance = {}
        reports.append(
            {
                "path": _safe_relative_path(path, root_dir),
                "within_workspace": within_workspace,
                "exists": within_workspace and path.is_file(),
                "parseable": isinstance(payload, dict),
                "factory_valid": bool(
                    isinstance(payload, dict)
                    and payload.get("factory") == "structural_data_capability_report"
                ),
                "local_research_usable": _string_list(
                    guidance.get("local_research_usable", [])
                ),
                "blocked_without_new_data": _string_list(
                    guidance.get("blocked_without_new_data", [])
                ),
                "must_not_codegen": _string_list(guidance.get("must_not_codegen", [])),
            }
        )
    return reports


def _research_decision_structural_capability_support(
    structural_data_requirement: dict[str, Any],
    structural_quality: dict[str, Any],
) -> dict[str, Any]:
    required_classes = set(structural_data_requirement.get("classes") or [])
    usable_classes = set(structural_quality.get("capability_usable_classes") or [])
    unsupported_required_classes = sorted(required_classes - usable_classes)
    capability_report_gate_passed = bool(
        structural_quality.get("capability_report_paths_exist")
        and structural_quality.get("capability_reports_valid")
        and structural_quality.get("capability_reports_valid_check_passed")
        and structural_quality.get("structural_capability_check_passed")
    )
    required_classes_supported = bool(
        capability_report_gate_passed
        and structural_quality.get("structural_capability_support_check_passed")
        and not unsupported_required_classes
    )
    return {
        "required_classes": sorted(required_classes),
        "usable_classes": sorted(usable_classes),
        "unsupported_required_classes": unsupported_required_classes,
        "capability_report_gate_passed": capability_report_gate_passed,
        "required_classes_supported": required_classes_supported,
        "structural_capability_support_check_passed": bool(
            structural_quality.get("structural_capability_support_check_passed")
        ),
    }


def _load_json_payload(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _payload_check_passed(payload: dict[str, Any], name: str) -> bool:
    for item in payload.get("checks", []) or []:
        if not isinstance(item, dict):
            continue
        if item.get("name") == name and item.get("status") in {"pass", "passed"}:
            return True
    return False


def _string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _blocked_next_actions_from_constraints(
    *constraint_groups: Sequence[dict[str, Any]]
) -> list[str]:
    actions: list[str] = []
    for group in constraint_groups:
        for item in group:
            if not isinstance(item, dict):
                continue
            for action in _string_list(item.get("blocked_next_actions", [])):
                if action not in actions:
                    actions.append(action)
    return actions


def _research_handoff_summaries_from_constraints(
    constraints: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    seen: set[str] = set()
    for constraint in constraints:
        if not isinstance(constraint, dict):
            continue
        for raw in constraint.get("research_handoff_summaries", []) or []:
            if not isinstance(raw, dict):
                continue
            copied = _copy_jsonish(raw)
            key = json.dumps(copied, sort_keys=True, ensure_ascii=False)
            if key not in seen:
                seen.add(key)
                summaries.append(copied)
    return summaries


def _research_handoff_summaries_from_causal_map(
    causal_map: dict[str, Any]
) -> list[dict[str, Any]]:
    if not isinstance(causal_map, dict):
        return []
    return _research_handoff_summaries_from_constraints([causal_map])


def _copy_jsonish(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _copy_jsonish(nested) for key, nested in value.items()}
    if isinstance(value, list):
        return [_copy_jsonish(item) for item in value]
    return value


def _research_decision_constraints(
    inputs: StrategyProposalInputs, created_at: str
) -> list[dict[str, Any]]:
    constraints: list[dict[str, Any]] = []
    thesis_id = _thesis_id(inputs, created_at)
    structural_data_requirement = _proposal_structural_data_requirement(inputs)
    failure_synthesis_ids = set(_failure_synthesis_ids(inputs))
    for item in _research_decision_evidence_inputs(inputs):
        path = _resolve_workspace_path(item.path, inputs.root_dir)
        payload = _load_research_decision_payload(path, inputs.root_dir)
        thesis = payload.get("thesis", {}) if payload else {}
        novelty = payload.get("novelty_assessment", {}) if payload else {}
        causal_map = payload.get("causal_failure_map", {}) if payload else {}
        if not isinstance(causal_map, dict):
            causal_map = {}
        causal_source_synthesis_id = str(
            causal_map.get("source_synthesis_id") or ""
        ).strip()
        causal_quality = _research_decision_causal_quality(causal_map)
        selection_score = _research_decision_selection_score(payload)
        structural_quality = _research_decision_structural_quality(
            payload, inputs.root_dir
        )
        structural_capability = _research_decision_structural_capability_support(
            structural_data_requirement,
            structural_quality,
        )
        local_falsification_handoff = _research_decision_local_falsification_handoff(
            payload,
            causal_map,
        )
        constraints.append(
            {
                "path": _safe_relative_path(path, inputs.root_dir),
                "status": payload.get("status") if payload else "unavailable",
                "decision_id": payload.get("decision_id") if payload else None,
                "proposal_generation_allowed": (
                    payload.get("proposal_generation_allowed") if payload else None
                ),
                "code_generation_allowed": (
                    payload.get("code_generation_allowed") if payload else None
                ),
                "proposal_thesis_id": thesis_id,
                "decision_thesis_id": thesis.get("thesis_id") if payload else None,
                "thesis_id_match": bool(payload) and thesis.get("thesis_id") == thesis_id,
                "repeated_failed_family_matches": list(
                    novelty.get("repeated_failed_family_matches", []) if payload else []
                ),
                "failed_thesis_id_match": (
                    novelty.get("failed_thesis_id_match") if payload else None
                ),
                "local_falsification_failed_thesis_ids": list(
                    novelty.get("local_falsification_failed_thesis_ids", [])
                    if payload
                    else []
                ),
                "local_falsification_failed_thesis_id_match": (
                    novelty.get("local_falsification_failed_thesis_id_match")
                    if payload
                    else None
                ),
                "local_falsification_failed_mechanism_tokens": list(
                    novelty.get("local_falsification_failed_mechanism_tokens", [])
                    if payload
                    else []
                ),
                "local_falsification_failed_mechanism_class_matches": list(
                    novelty.get(
                        "local_falsification_failed_mechanism_class_matches", []
                    )
                    if payload
                    else []
                ),
                "failure_synthesis_latest_checked": (
                    novelty.get("failure_synthesis_latest_checked") if payload else None
                ),
                "failure_synthesis_is_latest": (
                    novelty.get("failure_synthesis_is_latest") if payload else None
                ),
                "latest_failure_synthesis_path": (
                    novelty.get("latest_failure_synthesis_path") if payload else None
                ),
                "latest_failure_synthesis_id": (
                    novelty.get("latest_failure_synthesis_id") if payload else None
                ),
                "latest_failure_synthesis_generated_at": (
                    novelty.get("latest_failure_synthesis_generated_at")
                    if payload
                    else None
                ),
                "causal_failure_map_used": (
                    causal_map.get("used") if isinstance(causal_map, dict) else None
                ),
                "causal_failure_map_available": (
                    causal_map.get("available") if isinstance(causal_map, dict) else None
                ),
                "causal_source_synthesis_id": causal_source_synthesis_id or None,
                "supplied_failure_synthesis_ids": sorted(failure_synthesis_ids),
                "causal_map_matches_failure_synthesis": (
                    bool(causal_source_synthesis_id)
                    and causal_source_synthesis_id in failure_synthesis_ids
                ),
                "causal_required_categories_to_address": list(
                    causal_map.get("required_categories_to_address", [])
                    if isinstance(causal_map, dict)
                    else []
                ),
                "blocked_next_actions": _string_list(
                    causal_map.get("blocked_next_actions", [])
                    if isinstance(causal_map, dict)
                    else []
                ),
                "research_handoff_summaries": (
                    _research_handoff_summaries_from_causal_map(causal_map)
                ),
                "causal_expected_required_categories": causal_quality[
                    "expected_required_categories"
                ],
                "causal_response_categories": causal_quality["response_categories"],
                "causal_current_policy_available": causal_quality[
                    "current_policy_available"
                ],
                "causal_risk_weights_present": causal_quality[
                    "causal_risk_weights_present"
                ],
                "missing_required_risk_weight_categories": causal_quality[
                    "missing_required_risk_weight_categories"
                ],
                "missing_current_required_categories": causal_quality[
                    "missing_current_required_categories"
                ],
                "missing_current_response_categories": causal_quality[
                    "missing_current_response_categories"
                ],
                "missing_response_categories": causal_quality[
                    "missing_response_categories"
                ],
                "weak_response_categories": causal_quality["weak_response_categories"],
                "category_evidence_gaps": causal_quality["category_evidence_gaps"],
                "parameter_only_response_categories": causal_quality[
                    "parameter_only_response_categories"
                ],
                "requires_research_question_responses": causal_quality[
                    "requires_research_question_responses"
                ],
                "required_research_questions": causal_quality[
                    "required_research_questions"
                ],
                "research_question_response_indexes": causal_quality[
                    "research_question_response_indexes"
                ],
                "reported_missing_research_question_response_indexes": causal_quality[
                    "reported_missing_research_question_response_indexes"
                ],
                "computed_missing_research_question_response_indexes": causal_quality[
                    "computed_missing_research_question_response_indexes"
                ],
                "missing_research_question_response_indexes": causal_quality[
                    "missing_research_question_response_indexes"
                ],
                "weak_research_question_response_indexes": causal_quality[
                    "weak_research_question_response_indexes"
                ],
                "research_selection_score_available": selection_score["available"],
                "research_selection_score_version": selection_score["version"],
                "research_selection_score": selection_score["score"],
                "minimum_research_selection_score": selection_score[
                    "minimum_score_required"
                ],
                "research_selection_score_passes_minimum": selection_score[
                    "passes_minimum"
                ],
                "research_selection_failed_components": selection_score[
                    "failed_components"
                ],
                "weighted_causal_score_available": selection_score[
                    "weighted_causal_score_available"
                ],
                "weighted_response_score": selection_score["weighted_response_score"],
                "unanswered_required_risk_weight": selection_score[
                    "unanswered_required_risk_weight"
                ],
                "local_falsification_handoff_required": (
                    local_falsification_handoff["required"]
                ),
                "local_falsification_handoff_passed": (
                    local_falsification_handoff["passed"]
                ),
                "local_falsification_artifact_count": (
                    local_falsification_handoff["artifact_count"]
                ),
                "local_falsification_parseable_artifact_count": (
                    local_falsification_handoff["parseable_artifact_count"]
                ),
                "local_falsification_matching_thesis_artifact_count": (
                    local_falsification_handoff["matching_thesis_artifact_count"]
                ),
                "local_falsification_passing_cost_edge_artifact_count": (
                    local_falsification_handoff[
                        "passing_cost_edge_artifact_count"
                    ]
                ),
                "local_falsification_paths_valid": (
                    local_falsification_handoff["paths_valid"]
                ),
                "local_falsification_factory_valid": (
                    local_falsification_handoff["factory_valid"]
                ),
                "local_falsification_safety_scope_valid": (
                    local_falsification_handoff["safety_scope_valid"]
                ),
                "local_falsification_event_source_valid": (
                    local_falsification_handoff["event_source_valid"]
                ),
                "local_falsification_event_source_context_alignment_valid": (
                    local_falsification_handoff[
                        "event_source_context_alignment_valid"
                    ]
                ),
                "local_falsification_event_source_failure_synthesis_guard_valid": (
                    local_falsification_handoff[
                        "event_source_failure_synthesis_guard_valid"
                    ]
                ),
                "local_falsification_artifact_paths": (
                    local_falsification_handoff["artifact_paths"]
                ),
                "local_falsification_blocker_names": (
                    local_falsification_handoff["blocker_names"]
                ),
                "proposal_structural_data_required": structural_data_requirement[
                    "required"
                ],
                "proposal_structural_data_terms": structural_data_requirement["terms"],
                "proposal_structural_data_classes": structural_data_requirement[
                    "classes"
                ],
                "structural_data_quality_report_paths": [
                    report["path"] for report in structural_quality["reports"]
                ],
                "structural_data_quality_reports_exist": structural_quality[
                    "report_paths_exist"
                ],
                "structural_data_quality_reports_valid_check_passed": (
                    structural_quality[
                        "local_data_quality_reports_valid_check_passed"
                    ]
                ),
                "structural_data_quality_check_passed": structural_quality[
                    "structural_quality_check_passed"
                ],
                "structural_data_quality_report_gate_passed": structural_quality[
                    "valid"
                ],
                "structural_data_capability_report_paths": [
                    report["path"]
                    for report in structural_quality["capability_reports"]
                ],
                "structural_data_capability_reports_exist": structural_quality[
                    "capability_report_paths_exist"
                ],
                "structural_data_capability_reports_valid": structural_quality[
                    "capability_reports_valid"
                ],
                "structural_data_capability_reports_valid_check_passed": (
                    structural_quality["capability_reports_valid_check_passed"]
                ),
                "structural_data_capability_check_passed": structural_quality[
                    "structural_capability_check_passed"
                ],
                "structural_data_capability_support_check_passed": (
                    structural_quality[
                        "structural_capability_support_check_passed"
                    ]
                ),
                "structural_data_capability_usable_classes": structural_capability[
                    "usable_classes"
                ],
                "structural_data_capability_unsupported_required_classes": (
                    structural_capability["unsupported_required_classes"]
                ),
                "structural_data_capability_report_gate_passed": (
                    structural_capability["capability_report_gate_passed"]
                ),
                "structural_data_capability_required_classes_supported": (
                    structural_capability["required_classes_supported"]
                ),
                "blocker_names": [
                    str(check.get("name"))
                    for check in payload.get("blockers", [])
                    if isinstance(check, dict) and check.get("name")
                ]
                if payload
                else [],
            }
        )
    return constraints


def _evidence_content_checks(label: str, path: Path) -> tuple[list[StrategyProposalCheck], list[str]]:
    checks: list[StrategyProposalCheck] = []
    reasons: list[str] = []
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return checks, reasons

    if path.suffix.lower() == ".json":
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            checks.append(
                _check(
                    f"evidence_{label}_json_parseable",
                    False,
                    "blocker",
                    "JSON evidence must be parseable.",
                )
            )
            reasons.append("JSON evidence is not parseable.")
            return checks, reasons
        scan_text = json.dumps(payload, ensure_ascii=False)
    else:
        scan_text = text

    secret_findings = _secret_findings(scan_text)
    private_env_findings = _private_env_findings(scan_text)
    checks.append(
        _check(
            f"evidence_{label}_no_secret_values",
            not secret_findings,
            "blocker",
            "Evidence metadata must not contain non-empty API keys, secrets, tokens, or passwords.",
            {"secret_reference_count": len(secret_findings)},
        )
    )
    checks.append(
        _check(
            f"evidence_{label}_no_private_env_references",
            not private_env_findings,
            "blocker",
            "Evidence metadata must not contain private environment variable references.",
            {"private_env_reference_count": len(private_env_findings)},
        )
    )
    if secret_findings:
        reasons.append("Evidence contains credential-like values.")
    if private_env_findings:
        reasons.append("Evidence contains private environment references.")
    return checks, reasons


def _required_section_checks(markdown: str) -> list[StrategyProposalCheck]:
    return [
        _check(
            f"markdown_section_{_safe_label(section)}_present",
            f"## {section}" in markdown,
            "blocker",
            f"Generated proposal must include the {section} section.",
        )
        for section in REQUIRED_PROPOSAL_SECTIONS
    ]


def _sanitized_text_fields(inputs: StrategyProposalInputs) -> dict[str, Any]:
    return {
        "summary": _sanitize_text(inputs.summary),
        "hypothesis": _sanitize_text(inputs.hypothesis),
        "market_condition": _sanitize_text(inputs.market_condition),
        "entry_logic": _sanitize_text(inputs.entry_logic),
        "exit_logic": _sanitize_text(inputs.exit_logic),
        "risk_logic": _sanitize_text(inputs.risk_logic),
        "required_data": [_sanitize_text(item) for item in inputs.required_data],
        "parameters": [_sanitize_text(item) for item in inputs.parameters],
        "expected_failure_cases": [
            _sanitize_text(item) for item in inputs.expected_failure_cases
        ],
        "backtest_plan": _sanitize_text(inputs.backtest_plan),
        "rejection_conditions": [_sanitize_text(item) for item in inputs.rejection_conditions],
        "reviewer_notes": [_sanitize_text(item) for item in inputs.reviewer_notes],
    }


def _proposal_paths(inputs: StrategyProposalInputs, created_at: str) -> tuple[Path, Path]:
    output_root = _resolve_workspace_path(inputs.output_root, inputs.root_dir)
    stem = f"{_timestamp_slug(created_at)}_{_safe_filename(inputs.strategy_name)}"
    proposal_path = output_root / f"{stem}.md"
    return proposal_path, output_root / f"{stem}.metadata.json"


def _timestamp_slug(created_at: str) -> str:
    try:
        parsed = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    except ValueError:
        parsed = datetime.now(UTC)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")


def _safe_filename(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_")
    return token or "strategy_proposal"


def _safe_label(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_").lower()
    return token or "evidence"


def _sanitize_text(text: str) -> str:
    sanitized = _SECRET_ASSIGNMENT_RE.sub(
        lambda match: f"{match.group('label')}{match.group('sep')}[REDACTED]",
        str(text),
    )
    sanitized = _SECRET_PHRASE_RE.sub(
        lambda match: f"{match.group('label')}{match.group('sep')}[REDACTED]",
        sanitized,
    )
    return _PRIVATE_ENV_RE.sub("[REDACTED_ENV]", sanitized)


def _normalizes_to_long_only(value: str) -> bool:
    normalized = re.sub(r"[^a-z]+", "-", value.strip().lower()).strip("-")
    return normalized in {"long", "long-only", "longonly"}


def _generator_mode(value: str) -> str:
    mode = str(value or "rule_based").strip().lower()
    return mode if mode in ALLOWED_GENERATOR_MODES else "rule_based"


def _strategy_logic_variant(inputs: StrategyProposalInputs) -> str:
    explicit = str(inputs.strategy_logic_variant or "").strip().lower()
    if explicit in ALLOWED_STRATEGY_LOGIC_VARIANTS:
        return explicit
    thesis_type = _thesis_type(inputs)
    failure_codes = set(_failure_taxonomy_codes(inputs))
    if thesis_type in {
        "amihud_illiquidity",
        "amihud_illiquidity_premium",
        "illiquidity_premium",
        "price_impact_illiquidity",
        "turnover_illiquidity",
    }:
        return "amihud_illiquidity_premium"
    if thesis_type in {
        "bipower_jump_decay",
        "continuous_variance_decay",
        "jump_continuous_variance_decay",
        "post_jump_variance_decay",
        "realized_bipower_jump_decay",
        "realized_multipower_jump_decay",
    }:
        return "bipower_jump_decay"
    if thesis_type in {
        "directional_change",
        "directional_change_event_time",
        "directional_change_overshoot",
        "event_time_overshoot",
        "intrinsic_time_overshoot",
        "overshoot_continuation_reversal",
    }:
        return "directional_change_overshoot"
    if thesis_type in {
        "ohlc_quarticity_volatility_state_transition",
        "ohlc_range_quarticity",
        "quarticity_vol_of_vol_state",
        "range_quarticity_state_decay",
        "range_quarticity_vol_of_vol",
        "range_quarticity_vol_of_vol_state",
    }:
        return "range_quarticity_vol_of_vol_state"
    if thesis_type in {
        "account_ratio_reaccumulation",
        "crowding_unwind",
        "crowding_unwind_reaccumulation",
        "long_short_reaccumulation",
        "open_interest_unwind_reaccumulation",
        "positioning_unwind_reaccumulation",
    }:
        return "crowding_unwind_reaccumulation"
    if thesis_type in {
        "btc_eth_correlation_recovery",
        "correlation_breakdown_recovery",
        "cross_asset_correlation",
        "cross_asset_correlation_recovery",
        "dynamic_correlation_recovery",
    }:
        return "cross_asset_correlation_recovery"
    if thesis_type in {
        "btc_eth_cointegration",
        "cointegrated_spread_reversion",
        "cross_asset_cointegration",
        "crypto_pair_cointegration",
        "statistical_arbitrage_spread",
    }:
        return "cross_asset_cointegration_spread"
    if thesis_type in {
        "btc_eth_lead_lag",
        "cross_asset_lead_lag",
        "cross_asset_spillover",
        "eth_btc_lead_lag",
        "inter_crypto_lead_lag",
    }:
        return "cross_asset_lead_lag"
    if thesis_type in {
        "calendar_anomaly",
        "calendar_liquidity_seasonality",
        "calendar_turnover",
        "calendar_turnover_seasonality",
        "day_of_week_effect",
        "day_of_week_turnover",
        "time_of_week_turnover",
        "weekend_liquidity_seasonality",
    }:
        return "calendar_turnover_seasonality"
    if thesis_type in {
        "entropy_regime",
        "entropy_regime_transition",
        "information_entropy_regime",
        "range_efficiency_entropy",
    }:
        return "entropy_regime_transition"
    if thesis_type in {
        "autocorrelation_regime",
        "random_walk_deviation",
        "return_autocorrelation_regime",
        "variance_ratio_regime",
        "variance_ratio_regime_switch",
    }:
        return "variance_ratio_regime_switch"
    if thesis_type in {
        "fractal_long_memory",
        "fractal_market_regime",
        "hurst_persistence",
        "long_memory_regime",
    }:
        return "fractal_long_memory_regime"
    if thesis_type in {
        "closed_candle_liquidity_resilience",
        "closed_candle_liquidity_resilience_recovery",
        "liquidity_normalization_recovery",
        "liquidity_recovery_horizon",
        "post_stress_liquidity_recovery",
    }:
        return "liquidity_recovery_horizon"
    if thesis_type in {
        "funding_carry",
        "funding_pressure",
        "funding_pressure_carry",
        "perpetual_funding",
        "perpetual_funding_pressure",
    }:
        return "funding_pressure_carry"
    if thesis_type in {
        "crypto_beta_risk_premium",
        "drawdown_controlled_beta",
        "market_beta_carry",
        "market_beta_drawdown_carry",
        "risk_budget_beta_carry",
    }:
        return "market_beta_drawdown_carry"
    if thesis_type in {
        "fair_value_mark_momentum_lag",
        "mark_fair_value_momentum_lag",
        "mark_momentum_lag",
        "mark_price_momentum_lag",
        "perpetual_mark_momentum_lag",
    }:
        return "mark_fair_value_momentum_lag"
    if thesis_type in {
        "fair_price_dislocation",
        "last_mark_dislocation",
        "mark_discount_reclaim",
        "mark_discount_reclaim_continuation",
        "mark_price_dislocation",
        "mark_price_dislocation_reclaim",
        "perpetual_mark_dislocation",
        "perpetual_mark_reclaim",
    }:
        if thesis_type in {"mark_discount_reclaim", "mark_discount_reclaim_continuation"}:
            return "mark_discount_reclaim_continuation"
        return "mark_price_dislocation_reclaim"
    if thesis_type in {
        "bid_ask_spread_reversion",
        "corwin_schultz_spread",
        "microstructure_noise_reversion",
        "microstructure_spread",
        "microstructure_spread_reversion",
        "roll_spread_reversion",
    }:
        return "microstructure_spread_reversion"
    if thesis_type in {
        "bull_bear_state_reentry",
        "hidden_markov_proxy",
        "regime_state_reentry",
        "regime_switching_state",
        "state_dependent_drift",
    }:
        return "regime_state_reentry"
    if thesis_type in {
        "higher_moment_tail_shape",
        "realized_skewness",
        "realized_skewness_tail",
        "skewness_kurtosis",
        "tail_shape_moments",
    }:
        return "realized_skewness_tail_shape"
    if thesis_type in {
        "good_bad_volatility",
        "realized_semivariance",
        "semivariance_asymmetry",
        "semivariance_regime",
        "upside_downside_volatility",
    }:
        return "semivariance_asymmetry_regime"
    if thesis_type in {
        "order_flow_imbalance",
        "signed_volume_accumulation",
        "signed_volume_imbalance",
        "volume_imbalance_accumulation",
    }:
        return "signed_volume_imbalance_accumulation"
    if thesis_type in {"trend", "momentum", "trend_following", "trend_continuation"}:
        return "trend_continuation"
    if "FAIL_REGIME_FRAGILE" in failure_codes:
        return "volatility_breakout"
    if "FAIL_COST_SENSITIVE" in failure_codes:
        return "trend_continuation"
    return "mean_reversion_pullback"


def _current_family_tokens(inputs: StrategyProposalInputs) -> set[str]:
    tokens = {
        _safe_label(_thesis_type(inputs)),
        _safe_label(inputs.strategy_type),
        _safe_label(_strategy_logic_variant(inputs)),
    }
    if _generator_mode(inputs.generator_mode) == "hybrid_ml":
        tokens.add("hybrid_ml_return_filter")
    variant = _strategy_logic_variant(inputs)
    if variant == "amihud_illiquidity_premium":
        tokens.update(
            {
                "amihud_illiquidity",
                "amihud_illiquidity_premium",
                "illiquidity_premium",
                "price_impact_illiquidity",
                "turnover_illiquidity",
            }
        )
    if variant == "bipower_jump_decay":
        tokens.update(
            {
                "bipower_jump_decay",
                "continuous_variance_decay",
                "jump_continuous_variance_decay",
                "post_jump_variance_decay",
                "realized_bipower_jump_decay",
                "realized_multipower_jump_decay",
            }
        )
    if variant == "crowding_unwind_reaccumulation":
        tokens.update(
            {
                "account_ratio_reaccumulation",
                "crowding_unwind",
                "crowding_unwind_reaccumulation",
                "long_short_reaccumulation",
                "open_interest_unwind_reaccumulation",
                "positioning_unwind_reaccumulation",
            }
        )
    if variant == "directional_change_overshoot":
        tokens.update(
            {
                "directional_change",
                "directional_change_event_time",
                "directional_change_overshoot",
                "event_time_overshoot",
                "intrinsic_time_overshoot",
                "overshoot_continuation_reversal",
            }
        )
    if variant == "range_quarticity_vol_of_vol_state":
        tokens.update(
            {
                "ohlc_quarticity_volatility_state_transition",
                "ohlc_range_quarticity",
                "quarticity_vol_of_vol_state",
                "range_quarticity_state_decay",
                "range_quarticity_vol_of_vol",
                "range_quarticity_vol_of_vol_state",
            }
        )
    if variant == "calendar_turnover_seasonality":
        tokens.update(
            {
                "calendar_anomaly",
                "calendar_liquidity_seasonality",
                "calendar_turnover",
                "calendar_turnover_seasonality",
                "day_of_week_effect",
                "day_of_week_turnover",
                "time_of_week_turnover",
                "weekend_liquidity_seasonality",
            }
        )
    if variant == "cross_asset_lead_lag":
        tokens.update(
            {
                "btc_eth_lead_lag",
                "cross_asset_lead_lag",
                "cross_asset_spillover",
                "eth_btc_lead_lag",
                "inter_crypto_lead_lag",
            }
        )
    if variant == "cross_asset_cointegration_spread":
        tokens.update(
            {
                "btc_eth_cointegration",
                "cointegrated_spread_reversion",
                "cross_asset_cointegration",
                "crypto_pair_cointegration",
                "statistical_arbitrage_spread",
            }
        )
    if variant == "cross_asset_correlation_recovery":
        tokens.update(
            {
                "btc_eth_correlation_recovery",
                "correlation_breakdown_recovery",
                "cross_asset_correlation",
                "cross_asset_correlation_recovery",
                "dynamic_correlation_recovery",
            }
        )
    if variant == "mean_reversion_pullback":
        tokens.update({"mean_reversion", "liquidity_mean_reversion"})
    if variant == "signed_volume_imbalance_accumulation":
        tokens.update({"signed_volume_imbalance", "signed_volume_accumulation"})
    if variant == "intraday_session_liquidity_reclaim":
        tokens.update({"intraday_session_liquidity", "session_liquidity"})
    if variant == "liquidity_recovery_horizon":
        tokens.update(
            {
                "closed_candle_liquidity_resilience",
                "closed_candle_liquidity_resilience_recovery",
                "liquidity_normalization_recovery",
                "liquidity_recovery_horizon",
                "post_stress_liquidity_recovery",
            }
        )
    if variant == "downside_liquidity_shock_reversal":
        tokens.update({"downside_liquidity_shock", "downside_liquidity_shock_reversal"})
    if variant == "entropy_regime_transition":
        tokens.update(
            {
                "entropy_regime",
                "information_entropy_regime",
                "range_efficiency_entropy",
            }
        )
    if variant == "fractal_long_memory_regime":
        tokens.update(
            {
                "fractal_long_memory",
                "fractal_market_regime",
                "hurst_persistence",
                "long_memory_regime",
            }
        )
    if variant == "variance_ratio_regime_switch":
        tokens.update(
            {
                "autocorrelation_regime",
                "random_walk_deviation",
                "return_autocorrelation_regime",
                "variance_ratio_regime",
                "variance_ratio_regime_switch",
            }
        )
    if variant == "funding_pressure_carry":
        tokens.update(
            {
                "funding_carry",
                "funding_pressure",
                "perpetual_funding",
                "perpetual_funding_pressure",
            }
        )
    if variant == "market_beta_drawdown_carry":
        tokens.update(
            {
                "crypto_beta_risk_premium",
                "drawdown_controlled_beta",
                "market_beta_carry",
                "market_beta_drawdown_carry",
                "risk_budget_beta_carry",
            }
        )
    if variant == "mark_price_dislocation_reclaim":
        tokens.update(
            {
                "fair_price_dislocation",
                "last_mark_dislocation",
                "mark_price_dislocation",
                "mark_price_dislocation_reclaim",
                "perpetual_mark_dislocation",
                "perpetual_mark_reclaim",
            }
        )
    if variant == "mark_fair_value_momentum_lag":
        tokens.update(
            {
                "fair_value_mark_momentum_lag",
                "mark_fair_value_momentum_lag",
                "mark_momentum_lag",
                "mark_price_momentum_lag",
                "perpetual_mark_momentum_lag",
            }
        )
    if variant == "microstructure_spread_reversion":
        tokens.update(
            {
                "bid_ask_spread_reversion",
                "corwin_schultz_spread",
                "microstructure_noise_reversion",
                "microstructure_spread",
                "microstructure_spread_reversion",
                "roll_spread_reversion",
            }
        )
    if variant == "regime_state_reentry":
        tokens.update(
            {
                "bull_bear_state_reentry",
                "hidden_markov_proxy",
                "regime_state_reentry",
                "regime_switching_state",
                "state_dependent_drift",
            }
        )
    if variant == "realized_skewness_tail_shape":
        tokens.update(
            {
                "higher_moment_tail_shape",
                "realized_skewness",
                "realized_skewness_tail",
                "skewness_kurtosis",
                "tail_shape_moments",
            }
        )
    if variant == "semivariance_asymmetry_regime":
        tokens.update(
            {
                "good_bad_volatility",
                "realized_semivariance",
                "semivariance_asymmetry",
                "semivariance_regime",
                "upside_downside_volatility",
            }
        )
    return {token for token in tokens if token}


def _feature_list(inputs: StrategyProposalInputs) -> list[str]:
    features = [str(item).strip() for item in inputs.feature_list if str(item).strip()]
    if features:
        return features
    variant = _strategy_logic_variant(inputs)
    if variant == "amihud_illiquidity_premium":
        return [
            "amihud_illiquidity",
            "amihud_illiquidity_mean",
            "amihud_illiquidity_delta",
            "dollar_volume",
            "illiquidity_drift",
            "range_pct",
            "rolling_mid",
        ]
    if variant == "bipower_jump_decay":
        return [
            "log_return",
            "realized_variance_fast",
            "bipower_variation",
            "jump_variation",
            "jump_variation_ratio",
            "continuous_variance_decay",
            "positive_jump_event",
            "post_jump_drift",
            "jump_follow_through",
            "rolling_mid",
        ]
    if variant == "crowding_unwind_reaccumulation":
        return [
            "open_interest",
            "open_interest_delta_pct_288",
            "long_short_ratio",
            "long_short_ratio_zscore_864",
            "sma_distance_bps_144",
            "volume_zscore_288",
            "rolling_mid",
            "range_pct",
            "range_pct_mean",
        ]
    if variant == "directional_change_overshoot":
        return [
            "directional_change_state",
            "directional_change_extreme",
            "directional_change_event_age",
            "overshoot_return",
            "overshoot_length",
            "overshoot_ratio",
            "event_time_trend",
            "adverse_reversal_distance",
            "turnover_proxy",
            "rolling_mid",
        ]
    if variant == "range_quarticity_vol_of_vol_state":
        return [
            "log_return",
            "ohlc_range",
            "range_return",
            "range_quarticity_proxy",
            "range_quarticity_mean",
            "range_vol_of_vol_state",
            "range_state_decay",
            "range_stress_ratio",
            "participation_recovery",
            "turnover_proxy",
            "rolling_mid",
        ]
    if variant == "calendar_turnover_seasonality":
        return [
            "weekday",
            "hour_utc",
            "calendar_turnover_ratio",
            "calendar_turnover_ratio_mean",
            "weekend_turnover_baseline",
            "weekday_turnover_baseline",
            "calendar_drift",
        ]
    if variant == "cross_asset_lead_lag":
        return [
            "eth_log_return",
            "eth_lead_return",
            "eth_lead_return_mean",
            "btc_log_return",
            "eth_btc_return_spread",
            "eth_btc_spread_mean",
            "cross_asset_drift",
        ]
    if variant == "cross_asset_cointegration_spread":
        return [
            "eth_close",
            "btc_eth_log_ratio",
            "btc_eth_ratio_mean",
            "btc_eth_ratio_zscore",
            "btc_eth_ratio_zscore_delta",
            "eth_regime_drift",
            "range_pct",
            "rolling_mid",
        ]
    if variant == "cross_asset_correlation_recovery":
        return [
            "eth_close",
            "btc_log_return",
            "eth_log_return",
            "btc_eth_return_corr",
            "btc_eth_corr_mean",
            "btc_eth_corr_delta",
            "btc_eth_relative_return",
            "btc_eth_relative_return_mean",
            "eth_regime_drift",
            "range_pct",
            "rolling_mid",
        ]
    if variant == "downside_liquidity_shock_reversal":
        return ["lookback_return", "rsi_washout", "atr_normalized_drop", "volume_regime", "local_low_reclaim"]
    if variant == "entropy_regime_transition":
        return [
            "direction_entropy",
            "direction_entropy_baseline",
            "range_efficiency",
            "range_efficiency_mean",
            "entropy_drift",
            "rolling_mid",
        ]
    if variant == "fractal_long_memory_regime":
        return [
            "log_return",
            "hurst_proxy",
            "fractal_efficiency",
            "fractal_efficiency_mean",
            "fractal_drift",
            "rolling_mid",
        ]
    if variant == "variance_ratio_regime_switch":
        return [
            "log_return",
            "variance_ratio",
            "variance_ratio_mean",
            "variance_ratio_delta",
            "return_autocorr",
            "autocorr_mean",
            "regime_drift",
            "normalized_regime_return",
            "range_pct",
        ]
    if variant == "funding_pressure_carry":
        return [
            "funding_rate",
            "funding_rate_mean",
            "funding_rate_abs_mean",
            "funding_pressure",
            "funding_pressure_delta",
            "rolling_mid",
        ]
    if variant == "market_beta_drawdown_carry":
        return [
            "log_return",
            "realized_volatility",
            "realized_volatility_mean",
            "market_beta_high",
            "market_beta_drawdown",
            "market_beta_drift",
            "rolling_mid",
            "volume_mean",
        ]
    if variant == "mark_price_dislocation_reclaim":
        return [
            "mark_close",
            "mark_log_return",
            "mark_price_gap",
            "mark_price_gap_delta",
            "mark_price_gap_mean",
            "mark_price_gap_abs_mean",
            "mark_price_trend",
            "rolling_mid",
            "range_pct",
            "range_pct_mean",
            "volume_mean",
        ]
    if variant == "mark_fair_value_momentum_lag":
        return [
            "mark_close",
            "mark_price_return_bps",
            "traded_lag_return_bps",
            "range_pct",
            "volume_zscore",
        ]
    if variant == "mark_discount_reclaim_continuation":
        return [
            "mark_close",
            "mark_price_gap",
            "mark_price_gap_delta_6",
            "return_3",
            "volume_mean",
        ]
    if variant == "microstructure_spread_reversion":
        return [
            "log_return",
            "roll_spread_proxy",
            "roll_spread_mean",
            "roll_spread_delta",
            "hl_spread_proxy",
            "hl_spread_mean",
            "microstructure_noise_ratio",
            "rolling_mid",
            "range_pct",
            "range_pct_mean",
            "volume_mean",
        ]
    if variant == "regime_state_reentry":
        return [
            "log_return",
            "regime_return_fast",
            "regime_return_slow",
            "regime_negative_frequency",
            "regime_negative_frequency_mean",
            "regime_volatility",
            "regime_volatility_mean",
            "regime_drawdown",
            "regime_trendline",
            "rolling_mid",
            "volume_mean",
        ]
    if variant == "realized_skewness_tail_shape":
        return [
            "realized_skewness",
            "realized_skewness_mean",
            "realized_kurtosis",
            "realized_kurtosis_mean",
            "max_return",
            "max_return_mean",
            "tail_shape_drift",
        ]
    if variant == "semivariance_asymmetry_regime":
        return [
            "upside_semivariance",
            "downside_semivariance",
            "downside_semivariance_mean",
            "semivariance_balance",
            "semivariance_drift",
            "range_pct",
        ]
    if variant == "intraday_session_liquidity_reclaim":
        return ["hour_utc", "weekday", "session_vwap", "session_vwap_distance", "volume_mean", "atr_regime"]
    if variant == "liquidity_recovery_horizon":
        return [
            "liquidity_stress_recent",
            "liquidity_recovery_score",
            "liquidity_recovery_anchor",
            "volume_recovery_ratio",
            "amihud_illiquidity_ratio",
            "range_recovery_ratio",
            "recovery_horizon_return",
            "rolling_mid",
        ]
    if variant == "signed_volume_imbalance_accumulation":
        return [
            "signed_volume",
            "signed_volume_imbalance",
            "close_location_value",
            "close_location_mean",
            "rolling_mid",
            "range_pct",
        ]
    if variant == "trend_continuation":
        return ["ema_fast", "ema_slow", "rsi", "volume_mean", "atr"]
    if variant == "volatility_breakout":
        return ["rolling_high", "rolling_low", "atr", "volume_mean", "close_range"]
    return ["rsi", "ema_fast", "ema_slow", "volume_mean", "atr"]


def _target_definition(inputs: StrategyProposalInputs) -> str | None:
    if inputs.target_definition:
        return _sanitize_text(inputs.target_definition)
    if _generator_mode(inputs.generator_mode) in {"freqai", "hybrid_ml"}:
        return "future_return"
    return None


def _label_horizon(inputs: StrategyProposalInputs) -> int | None:
    if inputs.label_horizon is not None:
        return int(inputs.label_horizon)
    if _generator_mode(inputs.generator_mode) in {"freqai", "hybrid_ml"}:
        return 12
    return None


def _prediction_threshold(inputs: StrategyProposalInputs) -> float | None:
    if inputs.prediction_threshold is not None:
        return float(inputs.prediction_threshold)
    if _generator_mode(inputs.generator_mode) == "hybrid_ml":
        return 0.005
    if _generator_mode(inputs.generator_mode) == "freqai":
        return 0.0
    return None


def _rule_filters(inputs: StrategyProposalInputs) -> list[str]:
    filters = [str(item).strip() for item in inputs.rule_filters if str(item).strip()]
    if filters:
        return filters
    variant = _strategy_logic_variant(inputs)
    if variant == "amihud_illiquidity_premium":
        return [
            "price_impact_premium",
            "illiquidity_releasing",
            "not_extreme_impact",
            "price_resilience",
            "positive_illiquidity_drift",
            "controlled_range",
            "volume_floor",
        ]
    if variant == "bipower_jump_decay":
        return [
            "positive_jump_detected",
            "jump_dominates_continuous_variance",
            "continuous_variance_decaying",
            "post_jump_drift_positive",
            "not_overextended_after_jump",
            "volume_positive",
        ]
    if variant == "crowding_unwind_reaccumulation":
        return [
            "open_interest_unwinding",
            "short_account_reaccumulation",
            "price_above_sma",
            "volume_participation_floor",
            "price_resilience",
            "controlled_range",
            "not_overheated",
        ]
    if variant == "directional_change_overshoot":
        return [
            "directional_change_confirmed",
            "overshoot_persisted",
            "event_time_trend_positive",
            "adverse_reversal_absent",
            "turnover_controlled",
            "volume_positive",
        ]
    if variant == "range_quarticity_vol_of_vol_state":
        return [
            "range_quarticity_state_decay",
            "post_stress_stabilization",
            "participation_present",
            "range_not_reexpanding",
            "positive_stabilization_drift",
            "turnover_controlled",
            "volume_positive",
        ]
    if variant == "calendar_turnover_seasonality":
        return [
            "calendar_risk_window",
            "weekend_discount_context",
            "turnover_recovery",
            "positive_calendar_drift",
            "midline_hold",
            "controlled_range",
            "not_breakout_chase",
        ]
    if variant == "cross_asset_lead_lag":
        return [
            "eth_positive_lead",
            "btc_lag_discount",
            "spread_not_extreme",
            "btc_resilience",
            "positive_cross_asset_drift",
            "controlled_range",
            "volume_filter",
        ]
    if variant == "cross_asset_cointegration_spread":
        return [
            "btc_discount_to_eth",
            "spread_reversion_turn",
            "eth_market_support",
            "btc_resilience",
            "cointegration_spread_not_extreme",
            "controlled_range",
            "volume_filter",
        ]
    if variant == "cross_asset_correlation_recovery":
        return [
            "correlation_breakdown",
            "correlation_recovery",
            "btc_relative_recovery",
            "eth_market_support",
            "btc_resilience",
            "controlled_range",
            "volume_filter",
        ]
    if variant == "downside_liquidity_shock_reversal":
        return ["downside_shock", "rsi_washout_recovery", "quiet_volume", "local_low_reclaim"]
    if variant == "entropy_regime_transition":
        return [
            "low_directional_entropy",
            "efficiency_expanding",
            "positive_entropy_drift",
            "midline_hold",
            "range_not_extended",
            "volume_filter",
        ]
    if variant == "fractal_long_memory_regime":
        return [
            "persistent_memory_regime",
            "efficient_path",
            "positive_fractal_drift",
            "midline_hold",
            "not_range_extension",
            "volume_filter",
        ]
    if variant == "variance_ratio_regime_switch":
        return [
            "variance_ratio_expansion",
            "positive_autocorr_regime",
            "positive_regime_drift",
            "controlled_regime_return",
            "midline_resilience",
            "controlled_range",
            "volume_filter",
        ]
    if variant == "funding_pressure_carry":
        return [
            "negative_funding_pressure",
            "funding_pressure_releasing",
            "price_resilience",
            "not_positive_crowding",
            "controlled_range",
            "volume_filter",
        ]
    if variant == "market_beta_drawdown_carry":
        return [
            "moderate_drawdown",
            "volatility_budget",
            "positive_candle_reentry",
            "beta_resilience",
            "participation_floor",
            "not_overheated",
        ]
    if variant == "mark_price_dislocation_reclaim":
        return [
            "mark_discount_pressure",
            "mark_gap_reclaiming",
            "mark_price_support",
            "discount_not_extreme",
            "price_resilience",
            "controlled_range",
            "participation_floor",
        ]
    if variant == "mark_discount_reclaim_continuation":
        return [
            "mark_discount_pressure",
            "six_candle_discount_reclaim",
            "short_return_nonnegative",
        ]
    if variant == "mark_fair_value_momentum_lag":
        return [
            "mark_fair_value_momentum",
            "traded_price_lag",
            "range_budget",
            "participation_floor",
            "event_cooldown",
        ]
    if variant == "microstructure_spread_reversion":
        return [
            "spread_pressure",
            "spread_compressing",
            "hl_spread_normalizing",
            "price_resilience",
            "positive_recovery",
            "controlled_range",
            "participation_floor",
        ]
    if variant == "regime_state_reentry":
        return [
            "positive_regime_drift",
            "state_stability",
            "volatility_state_budget",
            "trendline_support",
            "closed_candle_reentry",
            "drawdown_state_intact",
            "participation_floor",
            "not_overheated",
        ]
    if variant == "realized_skewness_tail_shape":
        return [
            "low_realized_skewness",
            "kurtosis_risk_premium",
            "lottery_tail_cooling",
            "positive_tail_shape_drift",
            "midline_hold",
            "controlled_range",
            "volume_filter",
        ]
    if variant == "semivariance_asymmetry_regime":
        return [
            "good_volatility_dominance",
            "bad_volatility_decay",
            "positive_semivariance_drift",
            "midline_hold",
            "controlled_range",
            "not_range_extension",
            "volume_filter",
        ]
    if variant == "intraday_session_liquidity_reclaim":
        return ["session_window", "weekday_liquidity", "vwap_reclaim", "volume_filter", "controlled_atr"]
    if variant == "liquidity_recovery_horizon":
        return [
            "recent_liquidity_stress",
            "liquidity_normalizing",
            "participation_recovered",
            "below_recovery_anchor",
            "recovery_turn",
            "controlled_cost_proxy",
        ]
    if variant == "signed_volume_imbalance_accumulation":
        return [
            "positive_signed_imbalance",
            "close_location_accumulation",
            "mid_reclaim",
            "not_breakout_chase",
            "controlled_range",
            "volume_filter",
        ]
    if variant == "trend_continuation":
        return ["trend_filter", "volume_filter", "atr_floor"]
    if variant == "volatility_breakout":
        return ["breakout_filter", "volume_filter", "atr_expansion_filter"]
    return ["pullback_filter", "trend_filter", "volume_filter"]


def _thesis_id(inputs: StrategyProposalInputs, created_at: str) -> str:
    if inputs.thesis_id and inputs.thesis_id.strip():
        return _sanitize_text(inputs.thesis_id).strip()
    type_token = _safe_label(_thesis_type(inputs)).upper()
    return f"THESIS-{type_token}-{_timestamp_slug(created_at)}"


def _thesis_type(inputs: StrategyProposalInputs) -> str:
    return _sanitize_text(inputs.thesis_type or inputs.strategy_type).strip()


def _thesis_statement(inputs: StrategyProposalInputs) -> str:
    return _sanitize_text(inputs.thesis_statement or inputs.hypothesis).strip()


def _falsification_criteria(inputs: StrategyProposalInputs) -> str:
    if inputs.falsification_criteria and inputs.falsification_criteria.strip():
        return _sanitize_text(inputs.falsification_criteria).strip()
    return "; ".join(_sanitize_text(item).strip() for item in inputs.rejection_conditions if str(item).strip())


def _parameter_overrides(inputs: StrategyProposalInputs) -> dict[str, int | float]:
    overrides: dict[str, int | float] = {}
    pattern = re.compile(
        r"\b(?P<key>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?P<value>[-+]?\d+(?:\.\d+)?)\b"
    )
    for item in inputs.parameters:
        for match in pattern.finditer(str(item)):
            key = match.group("key").strip().lower()
            try:
                raw_value = float(match.group("value"))
            except ValueError:
                continue
            if raw_value != raw_value:
                continue
            if raw_value.is_integer():
                overrides[key] = int(raw_value)
            else:
                overrides[key] = raw_value
    return overrides


def _novelty_vs_previous(
    inputs: StrategyProposalInputs, evidence: Sequence[dict[str, Any]]
) -> str:
    if inputs.novelty_vs_previous and inputs.novelty_vs_previous.strip():
        return _sanitize_text(inputs.novelty_vs_previous).strip()
    if evidence:
        return "Uses local evidence references to vary hypothesis, features, labels, or filters from prior candidates."
    return "Initial hypothesis-family candidate; future revisions must describe changed assumptions or filters."


def _evidence_refs(
    inputs: StrategyProposalInputs,
    evidence: Sequence[dict[str, Any]],
    proposal_path: Path,
    *,
    research_references: Sequence[dict[str, Any]] | None = None,
) -> list[str]:
    refs = [_sanitize_text(item).strip() for item in inputs.evidence_refs if str(item).strip()]
    refs.extend(
        f"research:{ref['reference_id']}"
        for ref in research_references or []
        if ref.get("reference_id")
    )
    refs.extend(
        f"local:{item['label']}:{item.get('path')}"
        for item in evidence
        if item.get("status") == "accepted"
    )
    if not refs:
        refs.append(f"local:proposal:{_safe_relative_path(proposal_path, inputs.root_dir)}")
    return list(dict.fromkeys(refs))


def _research_references(
    inputs: StrategyProposalInputs, created_at: str
) -> list[dict[str, Any]]:
    thesis_id = _thesis_id(inputs, created_at)
    references: list[dict[str, Any]] = []
    for raw in inputs.research_references:
        if isinstance(raw, StrategyProposalResearchReference):
            payload = asdict(raw)
        elif isinstance(raw, dict):
            payload = dict(raw)
        else:
            payload = {}
        raw_motivated = payload.get("motivated_thesis_ids", [])
        if isinstance(raw_motivated, str):
            raw_motivated = [raw_motivated]
        motivated = [
            _sanitize_text(item).strip()
            for item in raw_motivated
            if str(item).strip()
        ]
        if not motivated:
            motivated = [thesis_id]
        references.append(
            {
                "reference_id": _sanitize_text(payload.get("reference_id", "")).strip(),
                "title": _sanitize_text(payload.get("title", "")).strip(),
                "source": _sanitize_text(payload.get("source", "")).strip(),
                "published_at": _sanitize_text(payload.get("published_at", "")).strip()
                or None,
                "relevance": _sanitize_text(payload.get("relevance", "")).strip(),
                "motivated_thesis_ids": list(dict.fromkeys(motivated)),
            }
        )
    return references


def _failure_taxonomy_codes(inputs: StrategyProposalInputs) -> list[str]:
    codes = [str(code).strip() for code in inputs.failure_taxonomy_codes if str(code).strip()]
    return [code for code in codes if code in ALLOWED_FAILURE_TAXONOMY_CODES]


def _backtest_plan_has_broader_validation(backtest_plan: str) -> bool:
    text = backtest_plan.strip()
    return bool(_DIVERSIFIED_BACKTEST_RE.search(text)) and not bool(
        _NARROW_BACKTEST_RE.search(text)
    )


def _non_negated_matches(pattern: re.Pattern[str], text: str) -> list[str]:
    findings: list[str] = []
    for match in pattern.finditer(text):
        prefix = text[max(0, match.start() - 24) : match.start()]
        if _NEGATION_PREFIX_RE.search(prefix):
            continue
        findings.append(match.group(0))
    return findings


def _secret_findings(text: str) -> list[str]:
    findings = [match.group("label") for match in _SECRET_ASSIGNMENT_RE.finditer(text)]
    findings.extend(match.group("label") for match in _SECRET_PHRASE_RE.finditer(text))
    return findings


def _private_env_findings(text: str) -> list[str]:
    return [match.group(0) for match in _PRIVATE_ENV_RE.finditer(text)]


def _leverage_above_one_findings(text: str) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    patterns = [
        re.compile(r"(?i)\bleverage\D{0,24}(?P<number>\d+(?:\.\d+)?)\b"),
        re.compile(r"(?i)\b(?P<number>\d+(?:\.\d+)?)\s*x\s+leverage\b"),
    ]
    for pattern in patterns:
        for match in pattern.finditer(text):
            prefix = text[max(0, match.start() - 24) : match.start()]
            if _NEGATION_PREFIX_RE.search(prefix):
                continue
            number = float(match.group("number"))
            if number > 1.0:
                findings.append({"match": match.group(0), "number": number})
    return findings


def _check(
    name: str,
    passed: bool,
    severity: str,
    message: str,
    details: dict[str, Any] | None = None,
) -> StrategyProposalCheck:
    return StrategyProposalCheck(
        name=name,
        status="pass" if passed else "blocked",
        severity=severity,
        message=message,
        details=details or {},
    )


def _bullet_lines(values: Sequence[str]) -> list[str]:
    lines = [f"- {str(value).strip()}" for value in values if str(value).strip()]
    return lines or ["- Not supplied."]


def _non_empty_sequence(values: Sequence[str]) -> bool:
    return any(str(value).strip() for value in values)


def _join_text(value: Any) -> str:
    if isinstance(value, list):
        return "\n".join(str(item) for item in value)
    return str(value)


def _resolve_workspace_path(path: Path, root_dir: Path) -> Path:
    return path if path.is_absolute() else root_dir / path


def _path_is_within_root(path: Path, root_dir: Path) -> bool:
    try:
        path.resolve().relative_to(root_dir.resolve())
        return True
    except ValueError:
        return False


def _safe_relative_path(path: Path, root_dir: Path) -> str:
    try:
        return str(path.resolve().relative_to(root_dir.resolve()))
    except ValueError:
        return path.name


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
