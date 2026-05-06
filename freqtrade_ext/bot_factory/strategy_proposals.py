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
    "local_previous_metrics_json",
    "local_walk_forward_metrics_json",
    "local_training_manifest_json",
    "local_reviewer_notes",
]
ALLOWED_GENERATOR_MODES = {"rule_based", "freqai", "hybrid_ml"}
ALLOWED_FAILURE_TAXONOMY_CODES = {
    "FAIL_OVERFIT_WF_GAP",
    "FAIL_COST_SENSITIVE",
    "FAIL_REGIME_FRAGILE",
}
ALLOWED_STRATEGY_LOGIC_VARIANTS = {
    "mean_reversion_pullback",
    "trend_continuation",
    "volatility_breakout",
}

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
    checks.extend(_hypothesis_candidate_checks(inputs))
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
        "thesis_id": _thesis_id(inputs, created_at),
        "thesis_type": _thesis_type(inputs),
        "thesis_statement": _thesis_statement(inputs),
        "falsification_criteria": _falsification_criteria(inputs),
        "novelty_vs_previous": _novelty_vs_previous(inputs, evidence),
        "evidence_refs": _evidence_refs(inputs, evidence, proposal_path),
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

    section_values: list[tuple[str, Any]] = [
        ("Summary", text_fields["summary"]),
        ("Hypothesis", text_fields["hypothesis"]),
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


def _hypothesis_candidate_checks(inputs: StrategyProposalInputs) -> list[StrategyProposalCheck]:
    retry_budget = int(inputs.retry_budget_per_thesis)
    thesis_retry_count = int(inputs.thesis_retry_count)
    parameter_retry_limit = int(inputs.parameter_only_retry_limit)
    parameter_retry_count = int(inputs.parameter_only_retry_count)
    raw_generator_mode = str(inputs.generator_mode or "rule_based").strip().lower()
    raw_logic_variant = str(inputs.strategy_logic_variant or "").strip().lower()
    failure_codes = _failure_taxonomy_codes(inputs)
    raw_failure_codes = [str(code).strip() for code in inputs.failure_taxonomy_codes]
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
    ]


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
    if thesis_type in {"trend", "momentum", "trend_following", "trend_continuation"}:
        return "trend_continuation"
    if "FAIL_REGIME_FRAGILE" in failure_codes:
        return "volatility_breakout"
    if "FAIL_COST_SENSITIVE" in failure_codes:
        return "trend_continuation"
    return "mean_reversion_pullback"


def _feature_list(inputs: StrategyProposalInputs) -> list[str]:
    features = [str(item).strip() for item in inputs.feature_list if str(item).strip()]
    if features:
        return features
    variant = _strategy_logic_variant(inputs)
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
) -> list[str]:
    refs = [_sanitize_text(item).strip() for item in inputs.evidence_refs if str(item).strip()]
    refs.extend(
        f"local:{item['label']}:{item.get('path')}"
        for item in evidence
        if item.get("status") == "accepted"
    )
    if not refs:
        refs.append(f"local:proposal:{_safe_relative_path(proposal_path, inputs.root_dir)}")
    return list(dict.fromkeys(refs))


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
