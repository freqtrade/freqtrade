from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

from freqtrade_ext.bot_factory.strategy_proposals import (
    StrategyProposalResearchReference,
)


RESEARCH_SELECTION_GATE_VERSION = "research_selection_gate_v1"
RESEARCH_SELECTION_SCORE_VERSION = "research_selection_score_v2"
RESEARCH_SELECTION_NOTICE = (
    "Research selection writes local JSON and Markdown decision artifacts only. "
    "It does not generate strategy code, run backtests, start paper or dry-run "
    "trading, call exchange order endpoints, promote candidates, or manage any "
    "bot process."
)

_PRIVATE_ENV_RE = re.compile(
    r"(?i)(\$\{[^}]*?(KEY|SECRET|TOKEN|PASSWORD|PASSWD|UID|JWT)[^}]*?\}|"
    r"env:[A-Z_][A-Z0-9_]*|%[A-Z_]*(KEY|SECRET|TOKEN|PASSWORD|PASSWD|UID|JWT)"
    r"[A-Z0-9_]*%)"
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
    r"(?i)\b(live[- ]only|live\s+data|real[- ]time|realtime|websocket\s+only|"
    r"streaming\s+only|current\s+open\s+candle|unclosed\s+candle)\b"
)
_ACCOUNT_POSITION_RE = re.compile(
    r"(?i)\b(account\s+balance|wallet\s+balance|private\s+balance|position\s+data|"
    r"open\s+positions?|current\s+positions?|fills?)\b"
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
    r"(?i)\b(freqtrade\s+trade|bot\s+startup|process\s+control|paper\s+trading|"
    r"dry[- ]run\s+trading|live\s+trading|canary\s+live|"
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
_LOCAL_HISTORICAL_RE = re.compile(
    r"(?i)\b(local|historical|closed[- ]candle|closed\s+candle|ohlcv|"
    r"walk[- ]forward|out[- ]of[- ]sample)\b"
)
_STRUCTURAL_DATA_RE = re.compile(
    r"(?i)\b(open[-_ ]?interest|long[-_ /]?short[-_ ]?(?:account[-_ ]?)?ratio|"
    r"account[-_ ]?ratio|liquidations?|order[-_ ]?book|orderbook|"
    r"market[-_ ]?depth|book[-_ ]?imbalance|depth[-_ ]?imbalance)\b"
)
_PARAMETER_TUNING_RE = re.compile(
    r"(?i)\b(parameter|threshold|hyperopt|grid\s*search|optimi[sz](?:e|ation)|"
    r"tune|retune|loosen|tighten|lookback|window\s+length|roi|stoploss|"
    r"trailing)\b"
)
_RESEARCH_MECHANISM_SUBSTANCE_RE = re.compile(
    r"(?i)\b(mechanism|market|regime|state|segment|condition|microstructure|"
    r"liquidity|spread|funding|basis|order[- ]flow|volume|volatility|"
    r"skewness|semivariance|entropy|cointegration|lead[- ]lag|calendar|"
    r"seasonality|risk[- ]premium|edge|expectancy|costs?|fees?|slippage|"
    r"turnover|historical|closed[- ]candle|ohlcv|walk[- ]forward|"
    r"out[- ]of[- ]sample|falsif|reject|evidence|entry|signal)\b"
)
_CAUSAL_RESPONSE_SUBSTANCE_RE = re.compile(
    r"(?i)\b(mechanism|regime|state|segment|condition|walk[- ]forward|"
    r"out[- ]of[- ]sample|historical|evidence|falsif|reject|pass|fail|"
    r"costs?|fees?|slippage|turnover|edge|expectancy|entry|market|"
    r"liquidity|spread|window|split)\b"
)
_RESEARCH_QUESTION_CALENDAR_EVIDENCE_RE = re.compile(
    r"(?i)\b(calendar[-_ ]?window|calendar_window|"
    r"profitable_calendar_windows_ratio|calendar_window_summaries|"
    r"quarterly|quarter)\b"
)
_RESEARCH_QUESTION_CALENDAR_RESPONSE_RE = re.compile(
    r"(?i)\b(calendar|calendar[-_ ]?window|calendar_window|"
    r"profitable_calendar_windows_ratio|calendar_window_summaries|"
    r"quarterly|quarter|regime[-_ ]?window)\b"
)
_CAUSAL_RESPONSE_MIN_WORDS = 10
_MATERIAL_CAUSAL_CATEGORY_MIN_SHARE = 0.70
_DEFAULT_RESEARCH_SELECTION_MIN_SCORE = 80.0
_HIGH_RISK_CAUSAL_RESPONSE_MIN_RISK_SCORE = 80.0
_LOCAL_FALSIFICATION_MIN_SAMPLE_COUNT = 20
_LOCAL_FALSIFICATION_MIN_DATA_SPAN_DAYS = 180.0
_CONTEXT_MERGE_SEMANTICS = "closed_context_candle_availability_v1"
_QUANTIFIED_COST_RE = re.compile(
    r"(?i)(\b\d+(?:\.\d+)?\s*(?:bps?|basis\s+points?|%|percent)\b|"
    r"\b(?:fees?|slippage|turnover|spread)\s*(?:<=|>=|<|>|=|under|over|below|above)?\s*\d)"
)
_CAUSAL_RESPONSE_CATEGORY_REQUIREMENTS: dict[str, tuple[tuple[str, re.Pattern[str]], ...]] = {
    "regime_fragile_mechanism": (
        ("regime_terms", re.compile(r"(?i)\b(regime|state|segment|condition)\b")),
        ("evidence_terms", re.compile(r"(?i)\b(evidence|historical|falsif|reject|fail)\b")),
    ),
    "walk_forward_fragility": (
        (
            "walk_forward_terms",
            re.compile(r"(?i)\b(walk[- ]forward|out[- ]of[- ]sample|window|split)\b"),
        ),
        ("evidence_terms", re.compile(r"(?i)\b(pass|fail|reject|evidence|support)\b")),
    ),
    "cost_sensitive_mechanism": (
        (
            "cost_terms",
            re.compile(r"(?i)\b(costs?|fees?|slippage|turnover|spread|drag)\b"),
        ),
        ("edge_terms", re.compile(r"(?i)\b(edge|expectancy|negative|dominate|reject)\b")),
    ),
    "entry_exists_negative_edge": (
        ("entry_terms", re.compile(r"(?i)\b(entry|entries|signal|setup)\b")),
        ("edge_terms", re.compile(r"(?i)\b(edge|expectancy|negative|loss|reject)\b")),
    ),
    "no_profitable_walk_forward_windows": (
        (
            "walk_forward_terms",
            re.compile(r"(?i)\b(walk[- ]forward|out[- ]of[- ]sample|window|split)\b"),
        ),
        (
            "profitability_terms",
            re.compile(r"(?i)\b(profit|profitable|return|expectancy|loss|negative|reject)\b"),
        ),
    ),
    "overfit_or_window_dependency": (
        (
            "window_terms",
            re.compile(r"(?i)\b(window|split|walk[- ]forward|out[- ]of[- ]sample)\b"),
        ),
        ("dependency_terms", re.compile(r"(?i)\b(overfit|dependency|fragile|reject|fail)\b")),
    ),
    "thesis_rejected_after_entries": (
        ("thesis_terms", re.compile(r"(?i)\b(thesis|mechanism|hypothesis)\b")),
        ("entry_terms", re.compile(r"(?i)\b(entry|entries|signal|setup)\b")),
        ("reject_terms", re.compile(r"(?i)\b(reject|fail|negative|loss|expectancy)\b")),
    ),
    "zero_trade_or_signal_sparsity": (
        ("signal_terms", re.compile(r"(?i)\b(signal|entry|entries|setup|gate)\b")),
        ("coverage_terms", re.compile(r"(?i)\b(sparse|coverage|rows|count|reject)\b")),
    ),
}
_HIGH_RISK_CAUSAL_RESPONSE_REQUIREMENTS: dict[
    str, tuple[tuple[str, re.Pattern[str]], ...]
] = {
    "cost_sensitive_mechanism": (
        ("quantified_cost_terms", _QUANTIFIED_COST_RE),
    ),
}
_DEFAULT_CAUSAL_RESPONSE_REQUIREMENTS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("mechanism_terms", re.compile(r"(?i)\b(mechanism|market|edge|regime|signal)\b")),
    ("evidence_terms", re.compile(r"(?i)\b(evidence|historical|falsif|reject|fail)\b")),
)

_FAMILY_ALIASES: dict[str, set[str]] = {
    "amihud_illiquidity_premium": {
        "amihud_illiquidity",
        "illiquidity_premium",
        "price_impact_illiquidity",
        "turnover_illiquidity",
    },
    "calendar_turnover_seasonality": {
        "calendar_anomaly",
        "calendar_liquidity_seasonality",
        "calendar_turnover",
        "day_of_week_effect",
        "day_of_week_turnover",
        "time_of_week_turnover",
        "weekend_liquidity_seasonality",
    },
    "cross_asset_cointegration_spread": {
        "btc_eth_cointegration",
        "cointegrated_spread_reversion",
        "cross_asset_cointegration",
        "crypto_pair_cointegration",
        "statistical_arbitrage_spread",
    },
    "cross_asset_correlation_recovery": {
        "btc_eth_correlation_recovery",
        "correlation_breakdown_recovery",
        "cross_asset_correlation",
        "dynamic_correlation_recovery",
    },
    "cross_asset_lead_lag": {
        "btc_eth_lead_lag",
        "cross_asset_spillover",
        "eth_btc_lead_lag",
        "inter_crypto_lead_lag",
    },
    "downside_liquidity_shock_reversal": {
        "downside_liquidity_shock",
    },
    "entropy_regime_transition": {
        "entropy_regime",
        "information_entropy_regime",
        "range_efficiency_entropy",
    },
    "fractal_long_memory_regime": {
        "fractal_long_memory",
        "fractal_market_regime",
        "hurst_persistence",
        "long_memory_regime",
    },
    "funding_pressure_carry": {
        "funding_carry",
        "funding_pressure",
        "perpetual_funding",
        "perpetual_funding_pressure",
    },
    "hybrid_ml_return_filter": {
        "freqai_return_filter",
        "hybrid_ml",
        "ml_return_filter",
    },
    "market_beta_drawdown_carry": {
        "crypto_beta_risk_premium",
        "drawdown_controlled_beta",
        "market_beta_carry",
        "risk_budget_beta_carry",
    },
    "mark_price_dislocation_reclaim": {
        "fair_price_dislocation",
        "last_mark_dislocation",
        "mark_price_dislocation",
        "perpetual_mark_dislocation",
        "perpetual_mark_reclaim",
    },
    "mean_reversion_pullback": {
        "liquidity_mean_reversion",
        "mean_reversion",
    },
    "microstructure_spread_reversion": {
        "bid_ask_spread_reversion",
        "corwin_schultz_spread",
        "microstructure_noise_reversion",
        "microstructure_spread",
        "roll_spread_reversion",
    },
    "realized_skewness_tail_shape": {
        "higher_moment_tail_shape",
        "realized_skewness",
        "realized_skewness_tail",
        "skewness_kurtosis",
        "tail_shape_moments",
    },
    "regime_state_reentry": {
        "bull_bear_state_reentry",
        "hidden_markov_proxy",
        "regime_switching_state",
        "state_dependent_drift",
    },
    "semivariance_asymmetry_regime": {
        "good_bad_volatility",
        "realized_semivariance",
        "semivariance_asymmetry",
        "semivariance_regime",
        "upside_downside_volatility",
    },
    "signed_volume_imbalance_accumulation": {
        "order_flow_imbalance",
        "signed_volume_accumulation",
        "signed_volume_imbalance",
        "volume_imbalance_accumulation",
    },
    "trend_continuation": {
        "momentum",
        "trend",
        "trend_following",
    },
    "variance_ratio_regime_switch": {
        "autocorrelation_regime",
        "random_walk_deviation",
        "return_autocorrelation_regime",
        "variance_ratio_regime",
    },
    "volatility_breakout": {
        "range_breakout",
        "volatility_expansion",
    },
}


@dataclass(frozen=True)
class ResearchSelectionInputs:
    root_dir: Path
    failure_synthesis_path: Path
    thesis_id: str
    thesis_family: str
    mechanism_class: str
    thesis_statement: str
    mechanism_summary: str
    novelty_rationale: str
    required_data: Sequence[str]
    edge_rationale: str
    transaction_cost_exposure: str
    falsification_plan: str
    stop_conditions: Sequence[str]
    research_references: Sequence[
        StrategyProposalResearchReference | dict[str, Any]
    ] = field(default_factory=list)
    local_data_paths: Sequence[Path] = field(default_factory=list)
    local_data_quality_report_paths: Sequence[Path] = field(default_factory=list)
    structural_data_capability_report_paths: Sequence[Path] = field(default_factory=list)
    causal_failure_map_path: Path | None = None
    causal_failure_responses: Sequence[str] = field(default_factory=list)
    research_question_responses: Sequence[str] = field(default_factory=list)
    local_falsification_paths: Sequence[Path] = field(default_factory=list)
    prior_local_falsification_paths: Sequence[Path] = field(default_factory=list)
    output_root: Path = Path("registry/strategies/research_decisions")
    decision_id: str | None = None
    reviewer_notes: Sequence[str] = field(default_factory=list)
    created_by_agent: str = "codex"
    created_at: str | None = None
    command: Sequence[str] = field(default_factory=list)


@dataclass(frozen=True)
class ResearchSelectionCheck:
    name: str
    status: str
    severity: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def select_research_thesis(inputs: ResearchSelectionInputs) -> dict[str, Any]:
    root = inputs.root_dir.resolve()
    created_at = inputs.created_at or datetime.now(UTC).replace(microsecond=0).isoformat()
    decision_id = inputs.decision_id or _decision_id(created_at, inputs.thesis_id)
    failure_synthesis_path = _resolve_inside(inputs.failure_synthesis_path, root)
    failure_synthesis, failure_synthesis_error = _load_json_object(failure_synthesis_path)
    causal_failure_map_path = (
        _resolve_inside(inputs.causal_failure_map_path, root)
        if inputs.causal_failure_map_path is not None
        else None
    )
    causal_failure_map, causal_failure_map_error = (
        _load_json_object(causal_failure_map_path)
        if causal_failure_map_path is not None
        else (None, None)
    )
    text_fields = _sanitized_text_fields(inputs)
    research_references = _research_references(inputs)
    local_data_paths = _local_data_paths(inputs, root)
    local_data_quality_reports = _local_data_quality_reports(inputs, root)
    structural_data_capability_reports = _structural_data_capability_reports(inputs, root)
    causal_failure_responses = _causal_failure_responses(inputs)
    research_question_responses = _research_question_responses(inputs)
    research_quality = _research_quality(inputs, text_fields=text_fields)
    causal_failure_constraints = _causal_failure_map_constraints(
        causal_failure_map=causal_failure_map,
        causal_failure_map_path=causal_failure_map_path,
        causal_failure_map_error=causal_failure_map_error,
        causal_failure_responses=causal_failure_responses,
        research_question_responses=research_question_responses,
        failure_synthesis=failure_synthesis,
        root=root,
    )
    local_falsification_evidence = _local_falsification_evidence(
        inputs,
        root,
        thesis_id=_sanitize_text(inputs.thesis_id).strip(),
        causal_failure_constraints=causal_failure_constraints,
    )
    prior_local_falsification_rejections = _prior_local_falsification_rejections(
        inputs,
        root,
    )
    failure_constraints = _failure_synthesis_constraints(
        failure_synthesis=failure_synthesis,
        failure_synthesis_path=failure_synthesis_path,
        root=root,
        inputs=inputs,
        research_reference_count=len(research_references),
    )
    research_selection_score = _research_selection_score(
        inputs=inputs,
        failure_constraints=failure_constraints,
        causal_failure_constraints=causal_failure_constraints,
        research_references=research_references,
        local_data_paths=local_data_paths,
        research_quality=research_quality,
    )
    checks = _build_checks(
        inputs=inputs,
        root=root,
        failure_synthesis=failure_synthesis,
        failure_synthesis_error=failure_synthesis_error,
        failure_constraints=failure_constraints,
        causal_failure_constraints=causal_failure_constraints,
        research_references=research_references,
        local_data_paths=local_data_paths,
        local_data_quality_reports=local_data_quality_reports,
        structural_data_capability_reports=structural_data_capability_reports,
        local_falsification_evidence=local_falsification_evidence,
        prior_local_falsification_rejections=prior_local_falsification_rejections,
        research_quality=research_quality,
        research_selection_score=research_selection_score,
    )
    checks_dicts = [check.to_dict() for check in checks]
    blockers = [check for check in checks_dicts if check["status"] == "blocked"]
    deferrals = [check for check in checks_dicts if check["status"] == "deferred"]
    decision = _decision_status(blockers=blockers, deferrals=deferrals)

    return {
        "generated_at": created_at,
        "factory": "research_selection_gate",
        "gate_version": RESEARCH_SELECTION_GATE_VERSION,
        "notice": RESEARCH_SELECTION_NOTICE,
        "decision_id": decision_id,
        "status": decision,
        "decision": decision,
        "proposal_generation_allowed": decision == "approved_for_proposal_generation",
        "strategy_code_generation_allowed": False,
        "code_generation_allowed": False,
        "code_generation_permission": (
            "deferred_until_accepted_strategy_proposal_and_supported_generator"
            if decision == "approved_for_proposal_generation"
            else "blocked_or_deferred_by_research_selection"
        ),
        "created_by_agent": _sanitize_text(inputs.created_by_agent),
        "command": list(inputs.command),
        "failure_synthesis": _failure_synthesis_summary(
            failure_synthesis,
            path=failure_synthesis_path,
            root=root,
            error=failure_synthesis_error,
        ),
        "causal_failure_map": _causal_failure_map_summary(
            causal_failure_map,
            path=causal_failure_map_path,
            root=root,
            error=causal_failure_map_error,
            constraints=causal_failure_constraints,
        ),
        "causal_failure_responses": causal_failure_responses,
        "research_question_responses": research_question_responses,
        "local_falsification_evidence": local_falsification_evidence,
        "prior_local_falsification_rejections": prior_local_falsification_rejections,
        "thesis": {
            "thesis_id": _sanitize_text(inputs.thesis_id).strip(),
            "thesis_family": _sanitize_text(inputs.thesis_family).strip(),
            "mechanism_class": _sanitize_text(inputs.mechanism_class).strip(),
            "thesis_statement": text_fields["thesis_statement"],
            "mechanism_summary": text_fields["mechanism_summary"],
            "novelty_rationale": text_fields["novelty_rationale"],
            "edge_rationale": text_fields["edge_rationale"],
            "transaction_cost_exposure": text_fields["transaction_cost_exposure"],
            "required_data": text_fields["required_data"],
            "local_data_paths": [item["path"] for item in local_data_paths],
            "local_data_quality_report_paths": [
                item["path"] for item in local_data_quality_reports
            ],
            "structural_data_capability_report_paths": [
                item["path"] for item in structural_data_capability_reports
            ],
            "local_falsification_paths": [
                item["path"] for item in local_falsification_evidence["artifacts"]
            ],
            "prior_local_falsification_paths": [
                item["path"] for item in prior_local_falsification_rejections["artifacts"]
            ],
            "falsification_plan": text_fields["falsification_plan"],
            "stop_conditions": text_fields["stop_conditions"],
        },
        "structural_data_capability_reports": structural_data_capability_reports,
        "novelty_assessment": {
            "failure_synthesis_latest_checked": failure_constraints[
                "failure_synthesis_latest_checked"
            ],
            "failure_synthesis_is_latest": failure_constraints[
                "failure_synthesis_is_latest"
            ],
            "latest_failure_synthesis_path": failure_constraints[
                "latest_failure_synthesis_path"
            ],
            "latest_failure_synthesis_id": failure_constraints[
                "latest_failure_synthesis_id"
            ],
            "current_family_tokens": failure_constraints["current_family_tokens"],
            "failed_family_tokens": failure_constraints["failed_family_tokens"],
            "repeated_failed_family_matches": failure_constraints[
                "repeated_failed_family_matches"
            ],
            "failed_thesis_id_match": failure_constraints["failed_thesis_id_match"],
            "local_falsification_failed_thesis_ids": failure_constraints[
                "local_falsification_failed_thesis_ids"
            ],
            "local_falsification_failed_thesis_id_match": failure_constraints[
                "local_falsification_failed_thesis_id_match"
            ],
            "local_falsification_failed_mechanism_tokens": failure_constraints[
                "local_falsification_failed_mechanism_tokens"
            ],
            "local_falsification_failed_mechanism_class_matches": (
                failure_constraints[
                    "local_falsification_failed_mechanism_class_matches"
                ]
            ),
            "edge_discovery_failed_thesis_ids": failure_constraints[
                "edge_discovery_failed_thesis_ids"
            ],
            "edge_discovery_failed_thesis_id_match": failure_constraints[
                "edge_discovery_failed_thesis_id_match"
            ],
            "edge_discovery_failed_mechanism_tokens": failure_constraints[
                "edge_discovery_failed_mechanism_tokens"
            ],
            "edge_discovery_failed_mechanism_class_matches": failure_constraints[
                "edge_discovery_failed_mechanism_class_matches"
            ],
            "minimum_research_reference_count": failure_constraints[
                "minimum_research_reference_count"
            ],
        },
        "research_quality": research_quality,
        "research_selection_score": research_selection_score,
        "research_references": research_references,
        "checks": checks_dicts,
        "blockers": blockers,
        "deferrals": deferrals,
        "reviewer_notes": [_sanitize_text(note) for note in inputs.reviewer_notes],
        "safety_scope": {
            "historical_only": True,
            "backtest_started": False,
            "strategy_code_generated": False,
            "paper_trading_started": False,
            "dry_run_trading_started": False,
            "live_trading": False,
            "exchange_order_placement": False,
            "shorting": False,
            "leverage": 1.0,
            "process_control": False,
            "promotion_authorized_by_this_command": False,
            "local_artifacts_source_of_truth": True,
        },
    }


def write_research_selection_artifacts(
    decision: dict[str, Any], *, root_dir: Path, output_root: Path
) -> tuple[Path, Path]:
    root = root_dir.resolve()
    decision_id = _safe_path_component(str(decision.get("decision_id") or "research_decision"))
    out_dir = _resolve_inside(output_root, root) / decision_id
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "research_decision.json"
    report_path = out_dir / "research_decision_report.md"
    json_path.write_text(json.dumps(decision, indent=2, ensure_ascii=False), encoding="utf-8")
    report_path.write_text(_render_report(decision), encoding="utf-8")
    return json_path, report_path


def _build_checks(
    *,
    inputs: ResearchSelectionInputs,
    root: Path,
    failure_synthesis: dict[str, Any] | None,
    failure_synthesis_error: str | None,
    failure_constraints: dict[str, Any],
    causal_failure_constraints: dict[str, Any],
    research_references: Sequence[dict[str, Any]],
    local_data_paths: Sequence[dict[str, Any]],
    local_data_quality_reports: Sequence[dict[str, Any]],
    structural_data_capability_reports: Sequence[dict[str, Any]],
    local_falsification_evidence: dict[str, Any],
    prior_local_falsification_rejections: dict[str, Any],
    research_quality: dict[str, Any],
    research_selection_score: dict[str, Any],
) -> list[ResearchSelectionCheck]:
    refs_have_required_fields = bool(research_references) and all(
        ref["reference_id"] and ref["title"] and ref["source"] for ref in research_references
    )
    refs_have_relevance = bool(research_references) and all(
        ref["relevance"] for ref in research_references
    )
    refs_have_dates = bool(research_references) and all(
        ref["published_at"] for ref in research_references
    )
    thesis_id = _sanitize_text(inputs.thesis_id).strip()
    refs_motivate_thesis = bool(research_references) and all(
        thesis_id in ref["motivated_thesis_ids"] for ref in research_references
    )
    min_reference_count = failure_constraints["minimum_research_reference_count"]
    failure_brief = (
        failure_synthesis.get("next_research_brief", {})
        if isinstance(failure_synthesis, dict)
        else {}
    )
    aggregate = (
        failure_synthesis.get("aggregate_failure_summary", {})
        if isinstance(failure_synthesis, dict)
        else {}
    )
    paper_or_live_allowed = bool(failure_brief.get("paper_or_live_promotion_allowed"))
    paper_ready_count = int(aggregate.get("paper_ready_count") or 0)
    repeated_families = failure_constraints["repeated_failed_family_matches"]
    text_for_dependency_scan = _raw_dependency_text(inputs, research_references)
    secret_findings = _secret_findings(text_for_dependency_scan)
    private_env_findings = _private_env_findings(text_for_dependency_scan)
    leverage_findings = _leverage_above_one_findings(text_for_dependency_scan)
    structural_data_requirement = _structural_data_requirement(inputs)
    local_data_quality_reports_ok = bool(local_data_quality_reports) and all(
        report["within_workspace"]
        and report["exists"]
        and report["parseable"]
        and report["ok"]
        for report in local_data_quality_reports
    )
    structural_capability = _structural_data_capability_support(
        structural_data_requirement,
        structural_data_capability_reports,
    )

    checks = [
        _check(
            "failure_synthesis_file_present",
            inputs.failure_synthesis_path and failure_constraints["failure_synthesis_file_present"],
            "blocker",
            "A local candidate failure synthesis JSON is required before selecting a new thesis.",
            {"path": failure_constraints["failure_synthesis_path"]},
        ),
        _check(
            "failure_synthesis_parseable",
            failure_synthesis is not None and not failure_synthesis_error,
            "blocker",
            "Failure synthesis must be a parseable JSON object.",
            {"error": failure_synthesis_error},
        ),
        _check(
            "failure_synthesis_factory_valid",
            bool(failure_synthesis) and failure_synthesis.get("factory") == "candidate_failure_synthesis",
            "blocker",
            "Failure synthesis must come from the Bot Factory candidate failure synthesis.",
        ),
        _check(
            "failure_synthesis_is_latest",
            not failure_constraints["failure_synthesis_latest_checked"]
            or failure_constraints["failure_synthesis_is_latest"],
            "blocker",
            "Research selection must use the latest local candidate failure synthesis.",
            {
                "supplied_path": failure_constraints["failure_synthesis_path"],
                "latest_path": failure_constraints["latest_failure_synthesis_path"],
                "latest_synthesis_id": failure_constraints["latest_failure_synthesis_id"],
            },
        ),
        _check(
            "failure_synthesis_has_next_research_brief",
            bool(failure_brief),
            "blocker",
            "Failure synthesis must include next_research_brief constraints.",
        ),
        _check(
            "no_paper_ready_candidates_to_promote",
            paper_ready_count == 0 and not paper_or_live_allowed,
            "blocker",
            "Do not open a new research branch when current evidence permits promotion.",
            {
                "paper_ready_count": paper_ready_count,
                "paper_or_live_promotion_allowed": paper_or_live_allowed,
            },
        ),
        _check("thesis_id_present", bool(thesis_id), "blocker", "Thesis ID is required."),
        _check(
            "thesis_id_outside_failed_thesis_ids",
            not failure_constraints["failed_thesis_id_match"],
            "blocker",
            "Thesis ID must be outside failed thesis IDs from the latest synthesis.",
            {"thesis_id": thesis_id},
        ),
        _check(
            "thesis_family_present",
            bool(str(inputs.thesis_family).strip()),
            "blocker",
            "Thesis family is required.",
        ),
        _check(
            "mechanism_class_present",
            bool(str(inputs.mechanism_class).strip()),
            "blocker",
            "Mechanism class is required.",
        ),
        _check(
            "thesis_family_outside_failed_families",
            not repeated_families,
            "blocker",
            "Thesis family must be outside failed hypothesis families by default.",
            {"repeated_failed_family_matches": repeated_families},
        ),
        _check(
            "research_thesis_not_previously_rejected_by_local_falsification",
            prior_local_falsification_rejections["matching_rejection_count"] == 0,
            "blocker",
            "Thesis ID or mechanism class was already rejected by local falsification evidence.",
            {
                "matching_rejections": prior_local_falsification_rejections[
                    "matching_rejections"
                ],
            },
        ),
        _check(
            "research_thesis_outside_failure_synthesis_local_rejections",
            not failure_constraints["local_falsification_failed_thesis_id_match"]
            and not failure_constraints[
                "local_falsification_failed_mechanism_class_matches"
            ],
            "blocker",
            (
                "Thesis ID or mechanism class is already present in validated "
                "local falsification rejection memory from the latest synthesis."
            ),
            {
                "local_falsification_failed_thesis_id_match": failure_constraints[
                    "local_falsification_failed_thesis_id_match"
                ],
                "local_falsification_failed_mechanism_class_matches": (
                    failure_constraints[
                        "local_falsification_failed_mechanism_class_matches"
                    ]
                ),
            },
        ),
        _check(
            "research_thesis_outside_failure_synthesis_edge_rejections",
            not failure_constraints["edge_discovery_failed_thesis_id_match"]
            and not failure_constraints[
                "edge_discovery_failed_mechanism_class_matches"
            ],
            "blocker",
            (
                "Thesis ID or mechanism class is already present in validated "
                "edge discovery rejection memory from the latest synthesis."
            ),
            {
                "edge_discovery_failed_thesis_id_match": failure_constraints[
                    "edge_discovery_failed_thesis_id_match"
                ],
                "edge_discovery_failed_mechanism_class_matches": failure_constraints[
                    "edge_discovery_failed_mechanism_class_matches"
                ],
            },
        ),
        _check(
            "thesis_statement_present",
            bool(str(inputs.thesis_statement).strip()),
            "blocker",
            "Thesis statement is required.",
        ),
        _check(
            "mechanism_summary_present",
            bool(str(inputs.mechanism_summary).strip()),
            "blocker",
            "Mechanism summary is required.",
        ),
        _check(
            "novelty_rationale_present",
            bool(str(inputs.novelty_rationale).strip()),
            "blocker",
            "Novelty rationale versus prior failures is required.",
        ),
        _check(
            "required_data_present",
            _non_empty_sequence(inputs.required_data),
            "blocker",
            "Required data must be explicit.",
        ),
        _check(
            "local_data_paths_present",
            bool(local_data_paths),
            "warning",
            "At least one local data artifact should be supplied before proposal generation.",
            status_if_false="deferred",
        ),
        _check(
            "local_data_paths_within_workspace",
            all(path_info["within_workspace"] for path_info in local_data_paths),
            "blocker",
            "Local data paths must resolve inside the repository workspace.",
            {"paths": [path_info["path"] for path_info in local_data_paths]},
        ),
        _check(
            "local_data_paths_exist",
            bool(local_data_paths) and all(path_info["exists"] for path_info in local_data_paths),
            "warning",
            "Local data paths must exist to falsify the thesis before proposal generation.",
            {"paths": local_data_paths},
            status_if_false="deferred",
        ),
        _check(
            "local_data_quality_reports_valid",
            not local_data_quality_reports
            or all(
                report["within_workspace"]
                and report["exists"]
                and report["parseable"]
                and report["ok"]
                for report in local_data_quality_reports
            ),
            "blocker",
            "Supplied local data quality reports must exist, parse, and pass.",
            {"reports": local_data_quality_reports},
        ),
        _check(
            "structural_data_quality_report_present",
            not structural_data_requirement["required"] or local_data_quality_reports_ok,
            "blocker",
            "Structural data theses require a passing local data quality report before research selection.",
            {
                "structural_terms": structural_data_requirement["terms"],
                "quality_report_count": len(local_data_quality_reports),
                "quality_reports_ok": local_data_quality_reports_ok,
            },
        ),
        _check(
            "structural_data_capability_reports_valid",
            not structural_data_capability_reports
            or all(report["valid"] for report in structural_data_capability_reports),
            "blocker",
            "Supplied structural data capability reports must exist, parse, and come from the Bot Factory capability reporter.",
            {"reports": structural_data_capability_reports},
        ),
        _check(
            "structural_data_capability_report_present",
            not structural_data_requirement["required"]
            or structural_capability["capability_reports_ok"],
            "blocker",
            "Structural data theses require a passing structural data capability report before research selection.",
            {
                "required_classes": structural_data_requirement["classes"],
                "capability_report_count": len(structural_data_capability_reports),
                "capability_reports_ok": structural_capability["capability_reports_ok"],
            },
        ),
        _check(
            "structural_data_capability_supports_required_classes",
            not structural_data_requirement["required"]
            or not structural_capability["unsupported_required_classes"],
            "blocker",
            "Structural data thesis requires local research support for every required structural data class.",
            structural_capability,
        ),
        _check(
            "edge_rationale_present",
            bool(str(inputs.edge_rationale).strip()),
            "blocker",
            "Expected edge source must be explicit.",
        ),
        _check(
            "transaction_cost_exposure_present",
            bool(str(inputs.transaction_cost_exposure).strip()),
            "blocker",
            "Transaction-cost exposure must be explicit.",
        ),
        _check(
            "falsification_plan_present",
            bool(str(inputs.falsification_plan).strip()),
            "blocker",
            "Falsification plan is required.",
        ),
        _check(
            "falsification_plan_uses_local_historical_data",
            bool(_LOCAL_HISTORICAL_RE.search(str(inputs.falsification_plan))),
            "warning",
            "Falsification must be possible with local historical closed-candle artifacts.",
            status_if_false="deferred",
        ),
        _check(
            "stop_conditions_present",
            _non_empty_sequence(inputs.stop_conditions),
            "blocker",
            "Stop conditions before code generation must be explicit.",
        ),
        _check(
            "research_thesis_not_parameter_only",
            not research_quality["parameter_only_fields"],
            "blocker",
            "Research thesis must describe a market mechanism and falsification path, not parameter/threshold tuning alone.",
            {"parameter_only_fields": research_quality["parameter_only_fields"]},
        ),
        _check(
            "minimum_research_references",
            len(research_references) >= min_reference_count,
            "blocker",
            "Failure synthesis requires enough structured research references.",
            {
                "research_reference_count": len(research_references),
                "minimum_research_reference_count": min_reference_count,
            },
        ),
        _check(
            "research_references_structured",
            refs_have_required_fields,
            "blocker",
            "Research references must include reference_id, title, and source.",
        ),
        _check(
            "research_references_have_relevance",
            refs_have_relevance,
            "blocker",
            "Research references must explain relevance to the proposed mechanism.",
        ),
        _check(
            "research_references_record_publication_date",
            refs_have_dates,
            "blocker",
            "Research references must record a publication or version date.",
        ),
        _check(
            "research_references_motivate_current_thesis",
            refs_motivate_thesis,
            "blocker",
            "Research references must list the current thesis_id as motivated.",
            {"thesis_id": thesis_id},
        ),
        _check(
            "no_future_data_dependency",
            not _non_negated_matches(_FUTURE_DATA_RE, text_for_dependency_scan),
            "blocker",
            "Research selection must not depend on future data, lookahead, or negative shifts.",
        ),
        _check(
            "no_live_only_data_dependency",
            not _non_negated_matches(_LIVE_ONLY_DATA_RE, text_for_dependency_scan),
            "blocker",
            "Research selection must not depend on live-only or unclosed-candle data.",
        ),
        _check(
            "no_account_or_position_data_dependency",
            not _non_negated_matches(_ACCOUNT_POSITION_RE, text_for_dependency_scan),
            "blocker",
            "Research selection must not depend on account, balance, fill, or position data.",
        ),
        _check(
            "no_order_endpoint_dependency",
            not _non_negated_matches(_ORDER_ENDPOINT_RE, text_for_dependency_scan),
            "blocker",
            "Research selection must not depend on exchange order endpoints.",
        ),
        _check(
            "no_api_key_or_secret_dependency",
            not _non_negated_matches(_CREDENTIAL_DEPENDENCY_RE, text_for_dependency_scan)
            and not secret_findings
            and not private_env_findings,
            "blocker",
            "Research selection must not depend on API keys, secrets, or private env values.",
            {
                "secret_reference_count": len(secret_findings),
                "private_env_reference_count": len(private_env_findings),
            },
        ),
        _check(
            "no_leverage_above_one_dependency",
            not leverage_findings,
            "blocker",
            "Research selection must not use leverage above 1.0.",
            {"findings": leverage_findings},
        ),
        _check(
            "no_shorting_dependency",
            not _non_negated_matches(_SHORTING_RE, text_for_dependency_scan),
            "blocker",
            "Research selection must not include shorting behavior.",
        ),
        _check(
            "no_paper_live_or_process_control_dependency",
            not _non_negated_matches(_PROCESS_CONTROL_RE, text_for_dependency_scan),
            "blocker",
            "Research selection must not depend on paper/live startup or process control.",
        ),
    ]
    if local_falsification_evidence["high_risk_cost_evidence_required"]:
        checks.extend(
            [
                _check(
                    "local_falsification_cost_evidence_present",
                    bool(local_falsification_evidence["artifacts"]),
                    "blocker",
                    "High-risk cost-sensitive research must supply a local falsification JSON artifact.",
                    {
                        "required_category": "cost_sensitive_mechanism",
                        "required_risk_score": _HIGH_RISK_CAUSAL_RESPONSE_MIN_RISK_SCORE,
                    },
                ),
                _check(
                    "local_falsification_cost_evidence_paths_valid",
                    all(
                        item["within_workspace"] and item["exists"] and item["parseable"]
                        for item in local_falsification_evidence["artifacts"]
                    ),
                    "blocker",
                    "Local falsification artifacts must exist inside the workspace and parse as JSON objects.",
                    {"artifacts": local_falsification_evidence["artifacts"]},
                ),
                _check(
                    "local_falsification_cost_evidence_factory_valid",
                    all(
                        item["factory_valid"]
                        for item in local_falsification_evidence["artifacts"]
                    ),
                    "blocker",
                    "Local falsification evidence must come from the Bot Factory local falsification artifact generator.",
                    {"artifacts": local_falsification_evidence["artifacts"]},
                ),
                _check(
                    "local_falsification_cost_evidence_safety_scope_valid",
                    all(
                        item["safety_scope_valid"]
                        for item in local_falsification_evidence["artifacts"]
                    ),
                    "blocker",
                    "Local falsification evidence must preserve historical-only safety scope.",
                    {"artifacts": local_falsification_evidence["artifacts"]},
                ),
                _check(
                    "local_falsification_cost_evidence_event_source_valid",
                    all(
                        item["event_source_valid"]
                        for item in local_falsification_evidence["artifacts"]
                    ),
                    "blocker",
                    "Local falsification evidence must be linked to a Bot Factory local event source.",
                    {"artifacts": local_falsification_evidence["artifacts"]},
                ),
                _check(
                    "local_falsification_cost_evidence_event_source_context_alignment_valid",
                    all(
                        item["event_source_context_alignment_valid"]
                        for item in local_falsification_evidence["artifacts"]
                    ),
                    "blocker",
                    (
                        "Local falsification event sources that use funding or mark "
                        "context must prove closed-candle context alignment."
                    ),
                    {"artifacts": local_falsification_evidence["artifacts"]},
                ),
                _check(
                    "local_falsification_cost_evidence_event_source_failure_synthesis_guarded",
                    all(
                        item["event_source_failure_synthesis_guard_valid"]
                        for item in local_falsification_evidence["artifacts"]
                    ),
                    "blocker",
                    "Local falsification evidence must come from an event source that consumed failure synthesis and did not repeat a failed thesis or family.",
                    {"artifacts": local_falsification_evidence["artifacts"]},
                ),
                _check(
                    "local_falsification_cost_evidence_thesis_matches",
                    local_falsification_evidence["matching_thesis_artifact_count"] > 0,
                    "blocker",
                    "Local falsification evidence must be scoped to the current thesis ID.",
                    {
                        "thesis_id": thesis_id,
                        "matching_artifact_count": local_falsification_evidence[
                            "matching_thesis_artifact_count"
                        ],
                    },
                ),
                _check(
                    "local_falsification_cost_edge_exceeds_costs",
                    local_falsification_evidence["passing_cost_edge_artifact_count"] > 0,
                    "blocker",
                    "Local falsification evidence must show expected edge exceeding all-in cost over sufficient sample and data span.",
                    {
                        "passing_artifact_count": local_falsification_evidence[
                            "passing_cost_edge_artifact_count"
                        ],
                        "minimum_sample_count": local_falsification_evidence[
                            "minimum_sample_count"
                        ],
                        "minimum_data_span_days": local_falsification_evidence[
                            "minimum_data_span_days"
                        ],
                        "failures": local_falsification_evidence["failures"],
                    },
                ),
            ]
        )
    if causal_failure_constraints["path"] is not None:
        checks.extend(
            [
                _check(
                    "causal_failure_map_file_present",
                    causal_failure_constraints["file_present"],
                    "blocker",
                    "Causal failure map path was supplied but the file is missing.",
                    {"path": causal_failure_constraints["path"]},
                ),
                _check(
                    "causal_failure_map_parseable",
                    causal_failure_constraints["parseable"],
                    "blocker",
                    "Causal failure map must be a parseable JSON object.",
                    {"error": causal_failure_constraints["error"]},
                ),
                _check(
                    "causal_failure_map_factory_valid",
                    causal_failure_constraints["factory_valid"],
                    "blocker",
                    "Causal failure map must come from the Bot Factory candidate failure map.",
                    {"factory": causal_failure_constraints["factory"]},
                ),
                _check(
                    "causal_failure_map_completed",
                    causal_failure_constraints["completed"],
                    "blocker",
                    "Causal failure map must have completed status before thesis selection uses it.",
                    {"status": causal_failure_constraints["status"]},
                ),
                _check(
                    "causal_failure_map_matches_failure_synthesis",
                    causal_failure_constraints["source_synthesis_matches"],
                    "blocker",
                    "Causal failure map must be built from the same failure synthesis JSON.",
                    {
                        "map_source_synthesis_id": causal_failure_constraints[
                            "source_synthesis_id"
                        ],
                        "failure_synthesis_id": causal_failure_constraints[
                            "failure_synthesis_id"
                        ],
                    },
                ),
                _check(
                    "causal_failure_map_requires_research_decision",
                    causal_failure_constraints["requires_research_decision"],
                    "blocker",
                    "Causal failure map guidance must require a research decision before proposals.",
                ),
                _check(
                    "causal_failure_map_has_required_categories",
                    bool(causal_failure_constraints["required_categories"]),
                    "blocker",
                    "Causal failure map must expose dominant failure categories to address.",
                ),
                _check(
                    "causal_failure_responses_cover_required_categories",
                    not causal_failure_constraints["missing_response_categories"],
                    "blocker",
                    "Research thesis must explicitly answer the dominant causal failure categories.",
                    {
                        "required_categories": causal_failure_constraints[
                            "required_categories"
                        ],
                        "missing_categories": causal_failure_constraints[
                            "missing_response_categories"
                        ],
                    },
                ),
                _check(
                    "causal_failure_responses_are_substantive",
                    not causal_failure_constraints["weak_response_categories"],
                    "blocker",
                    "Causal failure responses must be substantive, not one-line placeholders.",
                    {
                        "minimum_word_count": _CAUSAL_RESPONSE_MIN_WORDS,
                        "weak_categories": causal_failure_constraints[
                            "weak_response_categories"
                        ],
                    },
                ),
                _check(
                    "causal_failure_responses_address_category_evidence",
                    not causal_failure_constraints["category_evidence_gaps"],
                    "blocker",
                    "Causal failure responses must address category-specific evidence.",
                    {
                        "category_evidence_gaps": causal_failure_constraints[
                            "category_evidence_gaps"
                        ]
                    },
                ),
                _check(
                    "causal_failure_responses_not_parameter_only",
                    not causal_failure_constraints["parameter_only_response_categories"],
                    "blocker",
                    "Causal failure responses must not rely on parameter or threshold tuning alone.",
                    {
                        "parameter_only_categories": causal_failure_constraints[
                            "parameter_only_response_categories"
                        ]
                    },
                ),
                _check(
                    "research_question_responses_cover_required_questions",
                    not causal_failure_constraints[
                        "requires_research_question_responses"
                    ]
                    or not causal_failure_constraints[
                        "missing_research_question_response_indexes"
                    ],
                    "blocker",
                    "Research thesis must explicitly answer the failure map's required research questions.",
                    {
                        "required_research_questions": causal_failure_constraints[
                            "required_research_questions"
                        ],
                        "missing_question_indexes": causal_failure_constraints[
                            "missing_research_question_response_indexes"
                        ],
                    },
                ),
                _check(
                    "research_question_responses_are_substantive",
                    not causal_failure_constraints[
                        "requires_research_question_responses"
                    ]
                    or not causal_failure_constraints[
                        "weak_research_question_response_indexes"
                    ],
                    "blocker",
                    "Research question responses must be substantive, not placeholders.",
                    {
                        "minimum_word_count": _CAUSAL_RESPONSE_MIN_WORDS,
                        "weak_question_indexes": causal_failure_constraints[
                            "weak_research_question_response_indexes"
                        ],
                    },
                ),
                _check(
                    "research_selection_score_meets_minimum",
                    bool(research_selection_score["passes_minimum"]),
                    "blocker",
                    "Research thesis must meet the current failure-map selection score before proposal generation.",
                    {
                        "score": research_selection_score["score"],
                        "minimum_score": research_selection_score[
                            "minimum_score_required"
                        ],
                        "failed_components": research_selection_score[
                            "failed_components"
                        ],
                    },
                ),
            ]
        )
    return checks


def _failure_synthesis_constraints(
    *,
    failure_synthesis: dict[str, Any] | None,
    failure_synthesis_path: Path,
    root: Path,
    inputs: ResearchSelectionInputs,
    research_reference_count: int,
) -> dict[str, Any]:
    aggregate = (
        failure_synthesis.get("aggregate_failure_summary", {})
        if isinstance(failure_synthesis, dict)
        else {}
    )
    brief = (
        failure_synthesis.get("next_research_brief", {})
        if isinstance(failure_synthesis, dict)
        else {}
    )
    failed_thesis_ids = {
        str(value).strip()
        for value in aggregate.get("thesis_ids_tried", [])
        if str(value).strip()
    }
    failed_thesis_ids.update(
        str(value).strip() for value in brief.get("failed_thesis_ids", []) if str(value).strip()
    )
    local_failed_thesis_ids = {
        str(value).strip()
        for value in aggregate.get("local_falsification_failed_thesis_ids", []) or []
        if str(value).strip()
    }
    failed_thesis_ids.update(local_failed_thesis_ids)
    edge_failed_thesis_ids = {
        str(value).strip()
        for value in aggregate.get("edge_discovery_failed_thesis_ids", []) or []
        if str(value).strip()
    }
    failed_thesis_ids.update(edge_failed_thesis_ids)
    prior_family_tokens = _failed_family_tokens(aggregate=aggregate, brief=brief)
    local_failed_mechanism_tokens = _family_tokens_from_raw_values(
        aggregate.get("local_falsification_failed_mechanism_classes", []) or []
    )
    prior_family_tokens.update(local_failed_mechanism_tokens)
    edge_failed_mechanism_tokens = _family_tokens_from_raw_values(
        aggregate.get("edge_discovery_failed_mechanism_classes", []) or []
    )
    prior_family_tokens.update(edge_failed_mechanism_tokens)
    current_family_tokens = _current_family_tokens(inputs)
    minimum_reference_count = _minimum_reference_count(brief)
    local_failed_mechanism_matches = sorted(
        current_family_tokens & local_failed_mechanism_tokens
    )
    edge_failed_mechanism_matches = sorted(
        current_family_tokens & edge_failed_mechanism_tokens
    )
    thesis_id = _sanitize_text(inputs.thesis_id).strip()
    freshness = _failure_synthesis_freshness(
        failure_synthesis=failure_synthesis,
        failure_synthesis_path=failure_synthesis_path,
        root=root,
    )
    return {
        "failure_synthesis_path": _rel(failure_synthesis_path, root),
        "failure_synthesis_file_present": failure_synthesis_path.is_file(),
        "failure_synthesis_latest_checked": freshness["checked"],
        "failure_synthesis_is_latest": freshness["is_latest"],
        "latest_failure_synthesis_path": freshness["latest_path"],
        "latest_failure_synthesis_id": freshness["latest_synthesis_id"],
        "latest_failure_synthesis_generated_at": freshness["latest_generated_at"],
        "failed_thesis_ids": sorted(failed_thesis_ids),
        "failed_family_tokens": sorted(prior_family_tokens),
        "current_family_tokens": sorted(current_family_tokens),
        "repeated_failed_family_matches": sorted(current_family_tokens & prior_family_tokens),
        "failed_thesis_id_match": thesis_id in failed_thesis_ids,
        "local_falsification_failed_thesis_ids": sorted(local_failed_thesis_ids),
        "local_falsification_failed_thesis_id_match": (
            thesis_id in local_failed_thesis_ids
        ),
        "local_falsification_failed_mechanism_tokens": sorted(
            local_failed_mechanism_tokens
        ),
        "local_falsification_failed_mechanism_class_matches": (
            local_failed_mechanism_matches
        ),
        "edge_discovery_failed_thesis_ids": sorted(edge_failed_thesis_ids),
        "edge_discovery_failed_thesis_id_match": (
            thesis_id in edge_failed_thesis_ids
        ),
        "edge_discovery_failed_mechanism_tokens": sorted(
            edge_failed_mechanism_tokens
        ),
        "edge_discovery_failed_mechanism_class_matches": (
            edge_failed_mechanism_matches
        ),
        "requires_new_thesis_id": bool(brief.get("requires_new_thesis_id", True)),
        "requires_new_research_references": bool(
            brief.get("requires_new_research_references", True)
        ),
        "minimum_research_reference_count": minimum_reference_count,
        "research_reference_count": research_reference_count,
    }


def _failure_synthesis_freshness(
    *,
    failure_synthesis: dict[str, Any] | None,
    failure_synthesis_path: Path,
    root: Path,
) -> dict[str, Any]:
    default_root = (root / "registry" / "strategies" / "synthesis").resolve()
    current_path = failure_synthesis_path.resolve()
    try:
        current_path.relative_to(default_root)
    except ValueError:
        return _failure_synthesis_freshness_unchecked()
    if not default_root.is_dir():
        return _failure_synthesis_freshness_unchecked()

    candidates: list[tuple[datetime, Path, str]] = []
    for path in default_root.rglob("candidate_failure_synthesis.json"):
        payload, _error = _load_json_object(path)
        if payload is None:
            continue
        if payload.get("factory") != "candidate_failure_synthesis":
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
        "latest_path": _rel(latest_path, root),
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


def _failed_family_tokens(*, aggregate: dict[str, Any], brief: dict[str, Any]) -> set[str]:
    raw_values = list(aggregate.get("hypothesis_families_tried", []) or [])
    raw_values.extend(brief.get("prior_hypothesis_families_to_avoid_as_default", []) or [])
    return _family_tokens_from_raw_values(raw_values)


def _family_tokens_from_raw_values(raw_values: Sequence[Any]) -> set[str]:
    tokens: set[str] = set()
    for value in raw_values:
        token = _safe_label(str(value))
        if token:
            tokens.add(token)
            tokens.update(_FAMILY_ALIASES.get(token, set()))
            for family, aliases in _FAMILY_ALIASES.items():
                if token in aliases:
                    tokens.add(family)
                    tokens.update(aliases)
    return tokens


def _current_family_tokens(inputs: ResearchSelectionInputs) -> set[str]:
    raw_values = [
        inputs.thesis_family,
        inputs.mechanism_class,
    ]
    tokens = {_safe_label(str(value)) for value in raw_values if str(value).strip()}
    expanded = set(tokens)
    for token in list(tokens):
        expanded.update(_FAMILY_ALIASES.get(token, set()))
        for family, aliases in _FAMILY_ALIASES.items():
            if token in aliases:
                expanded.add(family)
                expanded.update(aliases)
    return {token for token in expanded if token}


def _minimum_reference_count(brief: dict[str, Any]) -> int:
    try:
        minimum = int(brief.get("minimum_research_reference_count") or 2)
    except (TypeError, ValueError):
        minimum = 2
    return max(1, minimum)


def _research_references(inputs: ResearchSelectionInputs) -> list[dict[str, Any]]:
    references: list[dict[str, Any]] = []
    for raw in inputs.research_references:
        if isinstance(raw, StrategyProposalResearchReference):
            payload = asdict(raw)
        elif isinstance(raw, dict):
            payload = dict(raw)
        else:
            payload = {}
        motivated_raw = payload.get("motivated_thesis_ids", [])
        if isinstance(motivated_raw, str):
            motivated_raw = [motivated_raw]
        motivated = [
            _sanitize_text(str(item)).strip()
            for item in motivated_raw
            if str(item).strip()
        ]
        references.append(
            {
                "reference_id": _sanitize_text(str(payload.get("reference_id", ""))).strip(),
                "title": _sanitize_text(str(payload.get("title", ""))).strip(),
                "source": _sanitize_text(str(payload.get("source", ""))).strip(),
                "published_at": _sanitize_text(str(payload.get("published_at", ""))).strip()
                or None,
                "relevance": _sanitize_text(str(payload.get("relevance", ""))).strip(),
                "motivated_thesis_ids": list(dict.fromkeys(motivated)),
            }
        )
    return references


def _local_data_paths(inputs: ResearchSelectionInputs, root: Path) -> list[dict[str, Any]]:
    paths: list[dict[str, Any]] = []
    for raw_path in inputs.local_data_paths:
        path = Path(raw_path)
        resolved = path if path.is_absolute() else root / path
        try:
            resolved = resolved.resolve()
            resolved.relative_to(root)
            within_workspace = True
        except ValueError:
            within_workspace = False
        paths.append(
            {
                "path": _rel(resolved, root) if within_workspace else str(path),
                "exists": within_workspace and resolved.is_file(),
                "within_workspace": within_workspace,
            }
        )
    return paths


def _local_data_quality_reports(
    inputs: ResearchSelectionInputs, root: Path
) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    for raw_path in inputs.local_data_quality_report_paths:
        path = Path(raw_path)
        resolved = path if path.is_absolute() else root / path
        within_workspace = False
        payload: dict[str, Any] | None = None
        error: str | None = None
        try:
            resolved = resolved.resolve()
            resolved.relative_to(root)
            within_workspace = True
        except ValueError:
            error = "path_outside_workspace"
        exists = within_workspace and resolved.is_file()
        if exists:
            payload, error = _load_json_object(resolved)
        reports.append(
            {
                "path": _rel(resolved, root) if within_workspace else str(path),
                "exists": exists,
                "within_workspace": within_workspace,
                "parseable": payload is not None and not error,
                "ok": bool(payload and payload.get("ok") is True),
                "report_count": len(payload.get("reports", []))
                if isinstance(payload, dict) and isinstance(payload.get("reports"), list)
                else 0,
                "error": error,
            }
        )
    return reports


def _structural_data_capability_reports(
    inputs: ResearchSelectionInputs, root: Path
) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    for raw_path in inputs.structural_data_capability_report_paths:
        path = Path(raw_path)
        resolved = path if path.is_absolute() else root / path
        within_workspace = False
        payload: dict[str, Any] | None = None
        error: str | None = None
        try:
            resolved = resolved.resolve()
            resolved.relative_to(root)
            within_workspace = True
        except ValueError:
            error = "path_outside_workspace"
        exists = within_workspace and resolved.is_file()
        if exists:
            payload, error = _load_json_object(resolved)
        factory_valid = bool(
            isinstance(payload, dict)
            and payload.get("factory") == "structural_data_capability_report"
        )
        raw_guidance = payload.get("proposal_guidance", {}) if isinstance(payload, dict) else {}
        guidance = raw_guidance if isinstance(raw_guidance, dict) else {}
        reports.append(
            {
                "path": _rel(resolved, root) if within_workspace else str(path),
                "exists": exists,
                "within_workspace": within_workspace,
                "parseable": payload is not None and not error,
                "factory_valid": factory_valid,
                "valid": within_workspace
                and exists
                and payload is not None
                and not error
                and factory_valid,
                "local_research_usable": sorted(
                    str(item) for item in guidance.get("local_research_usable", [])
                )
                if isinstance(guidance.get("local_research_usable"), list)
                else [],
                "blocked_without_new_data": sorted(
                    str(item) for item in guidance.get("blocked_without_new_data", [])
                )
                if isinstance(guidance.get("blocked_without_new_data"), list)
                else [],
                "must_not_codegen": sorted(
                    str(item) for item in guidance.get("must_not_codegen", [])
                )
                if isinstance(guidance.get("must_not_codegen"), list)
                else [],
                "error": error,
            }
        )
    return reports


def _structural_data_requirement(inputs: ResearchSelectionInputs) -> dict[str, Any]:
    text = " ".join(
        [
            str(inputs.thesis_family),
            str(inputs.mechanism_class),
            str(inputs.thesis_statement),
            str(inputs.mechanism_summary),
            str(inputs.falsification_plan),
            " ".join(str(item) for item in inputs.required_data),
        ]
    )
    terms = sorted({match.group(0).lower().replace("_", " ") for match in _STRUCTURAL_DATA_RE.finditer(text)})
    classes = _structural_data_classes(terms)
    return {"required": bool(terms), "terms": terms, "classes": classes}


def _structural_data_classes(terms: Sequence[str]) -> list[str]:
    classes: set[str] = set()
    for term in terms:
        normalized = term.replace("-", " ").replace("_", " ")
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


def _structural_data_capability_support(
    structural_data_requirement: dict[str, Any],
    structural_data_capability_reports: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    required_classes = set(structural_data_requirement.get("classes") or [])
    valid_reports = [report for report in structural_data_capability_reports if report["valid"]]
    usable_classes = {
        item
        for report in valid_reports
        for item in report.get("local_research_usable", [])
    }
    blocked_without_new_data = {
        item
        for report in valid_reports
        for item in report.get("blocked_without_new_data", [])
    }
    must_not_codegen = {
        item for report in valid_reports for item in report.get("must_not_codegen", [])
    }
    unsupported_required_classes = sorted(required_classes - usable_classes)
    return {
        "required_classes": sorted(required_classes),
        "usable_classes": sorted(usable_classes),
        "blocked_without_new_data": sorted(blocked_without_new_data),
        "must_not_codegen": sorted(must_not_codegen),
        "unsupported_required_classes": unsupported_required_classes,
        "capability_report_count": len(structural_data_capability_reports),
        "valid_capability_report_count": len(valid_reports),
        "capability_reports_ok": bool(valid_reports)
        and len(valid_reports) == len(structural_data_capability_reports),
    }


def _local_falsification_evidence(
    inputs: ResearchSelectionInputs,
    root: Path,
    *,
    thesis_id: str,
    causal_failure_constraints: dict[str, Any],
) -> dict[str, Any]:
    high_risk_cost_required = _requires_high_risk_cost_falsification(
        causal_failure_constraints
    )
    artifacts = [
        _local_falsification_artifact(path, root=root, thesis_id=thesis_id)
        for path in inputs.local_falsification_paths
    ]
    return {
        "high_risk_cost_evidence_required": high_risk_cost_required,
        "minimum_sample_count": _LOCAL_FALSIFICATION_MIN_SAMPLE_COUNT,
        "minimum_data_span_days": _LOCAL_FALSIFICATION_MIN_DATA_SPAN_DAYS,
        "artifact_count": len(artifacts),
        "parseable_artifact_count": sum(1 for item in artifacts if item["parseable"]),
        "matching_thesis_artifact_count": sum(
            1 for item in artifacts if item["thesis_matches"]
        ),
        "passing_cost_edge_artifact_count": sum(
            1 for item in artifacts if item["cost_edge_passes"]
        ),
        "artifacts": artifacts,
        "failures": [
            {
                "path": item["path"],
                "reasons": item["failure_reasons"],
            }
            for item in artifacts
            if item["failure_reasons"]
        ],
    }


def _prior_local_falsification_rejections(
    inputs: ResearchSelectionInputs,
    root: Path,
) -> dict[str, Any]:
    thesis_id = _sanitize_text(inputs.thesis_id).strip()
    mechanism_class = _safe_label(str(inputs.mechanism_class))
    artifacts: list[dict[str, Any]] = []
    for raw_path in inputs.prior_local_falsification_paths:
        artifact = _local_falsification_artifact(raw_path, root=root, thesis_id=thesis_id)
        status = _safe_label(str(artifact.get("status") or ""))
        artifact_mechanism = _safe_label(str(artifact.get("mechanism_class") or ""))
        mechanism_matches = bool(mechanism_class and artifact_mechanism == mechanism_class)
        rejection_valid = bool(
            artifact["within_workspace"]
            and artifact["exists"]
            and artifact["parseable"]
            and artifact["factory_valid"]
            and artifact["safety_scope_valid"]
            and status in {"blocked", "failed", "rejected"}
        )
        artifact["rejection_valid"] = rejection_valid
        artifact["mechanism_matches"] = mechanism_matches
        artifact["matches_current_thesis_or_mechanism"] = bool(
            rejection_valid and (artifact["thesis_matches"] or mechanism_matches)
        )
        artifacts.append(artifact)

    matching = [
        {
            "path": item["path"],
            "thesis_id": item["thesis_id"],
            "mechanism_class": item["mechanism_class"],
            "status": item["status"],
            "net_edge_bps": item["net_edge_bps"],
            "failure_reasons": item["failure_reasons"],
        }
        for item in artifacts
        if item["matches_current_thesis_or_mechanism"]
    ]
    return {
        "artifact_count": len(artifacts),
        "valid_rejection_count": sum(1 for item in artifacts if item["rejection_valid"]),
        "matching_rejection_count": len(matching),
        "matching_rejections": matching,
        "artifacts": artifacts,
    }


def _requires_high_risk_cost_falsification(
    causal_failure_constraints: dict[str, Any],
) -> bool:
    for item in causal_failure_constraints.get("causal_risk_weights", []) or []:
        if _safe_label(str(item.get("category", ""))) != "cost_sensitive_mechanism":
            continue
        risk_score = _float_or_none(item.get("risk_score")) or 0.0
        if risk_score >= _HIGH_RISK_CAUSAL_RESPONSE_MIN_RISK_SCORE:
            return True
    return False


def _local_falsification_artifact(
    raw_path: Path, *, root: Path, thesis_id: str
) -> dict[str, Any]:
    path = Path(raw_path)
    resolved = path if path.is_absolute() else root / path
    within_workspace = False
    try:
        resolved = resolved.resolve()
        resolved.relative_to(root)
        within_workspace = True
    except ValueError:
        pass

    path_label = _rel(resolved, root) if within_workspace else str(path)
    exists = within_workspace and resolved.is_file()
    payload: dict[str, Any] | None = None
    error = None
    if exists:
        payload, error = _load_json_object(resolved)
    elif within_workspace:
        error = "file_not_found"
    else:
        error = "outside_workspace"

    evidence = _local_falsification_cost_payload(payload)
    factory = _safe_label(str((payload or {}).get("factory") or ""))
    factory_valid = factory == "research_local_falsification"
    safety_scope = (payload or {}).get("safety_scope") if isinstance(payload, dict) else None
    safety_scope_valid = _local_falsification_safety_scope_valid(safety_scope)
    event_source = evidence.get("event_source") or (payload or {}).get("event_source") or {}
    event_source_context_alignment_valid = (
        _local_falsification_event_source_context_alignment_valid(event_source)
    )
    event_source_valid = _local_falsification_event_source_valid(event_source)
    event_source_failure_synthesis_guard_valid = (
        _local_falsification_event_source_failure_synthesis_guard_valid(event_source)
    )
    artifact_thesis_id = _sanitize_text(
        evidence.get("thesis_id") or (payload or {}).get("thesis_id") or ""
    ).strip()
    mechanism_class = _sanitize_text(
        evidence.get("mechanism_class") or (payload or {}).get("mechanism_class") or ""
    ).strip()
    thesis_matches = bool(artifact_thesis_id) and artifact_thesis_id == thesis_id
    expected_edge_bps = _float_or_none(evidence.get("expected_edge_bps"))
    all_in_cost_bps = _float_or_none(evidence.get("all_in_cost_bps"))
    net_edge_bps = _float_or_none(evidence.get("net_edge_bps"))
    if net_edge_bps is None and expected_edge_bps is not None and all_in_cost_bps is not None:
        net_edge_bps = expected_edge_bps - all_in_cost_bps
    sample_count = _int_or_none(evidence.get("sample_count"))
    status = _safe_label(str(evidence.get("status") or (payload or {}).get("status") or ""))
    status_allows = status in {"", "pass", "passed", "approved", "completed"}
    numeric_edge_present = expected_edge_bps is not None and all_in_cost_bps is not None
    sample_sufficient = (
        sample_count is not None and sample_count >= _LOCAL_FALSIFICATION_MIN_SAMPLE_COUNT
    )
    data_span_days = _float_or_none(evidence.get("data_span_days"))
    data_span_sufficient = (
        data_span_days is not None
        and data_span_days >= _LOCAL_FALSIFICATION_MIN_DATA_SPAN_DAYS
    )
    edge_exceeds_cost = bool(
        numeric_edge_present and net_edge_bps is not None and net_edge_bps > 0.0
    )
    parseable = payload is not None and error is None
    cost_edge_passes = (
        parseable
        and factory_valid
        and safety_scope_valid
        and event_source_valid
        and event_source_failure_synthesis_guard_valid
        and thesis_matches
        and status_allows
        and numeric_edge_present
        and sample_sufficient
        and data_span_sufficient
        and edge_exceeds_cost
    )
    failure_reasons: list[str] = []
    if not within_workspace:
        failure_reasons.append("outside_workspace")
    if not exists:
        failure_reasons.append("file_not_found")
    if exists and not parseable:
        failure_reasons.append("not_parseable_json_object")
    if parseable and not factory_valid:
        failure_reasons.append("factory_invalid")
    if parseable and not safety_scope_valid:
        failure_reasons.append("safety_scope_invalid")
    if parseable and not event_source_valid:
        failure_reasons.append("event_source_invalid")
    if parseable and not event_source_context_alignment_valid:
        failure_reasons.append("event_source_context_alignment_missing_or_invalid")
    if parseable and not event_source_failure_synthesis_guard_valid:
        failure_reasons.append("event_source_failure_synthesis_guard_missing_or_failed")
    if parseable and not thesis_matches:
        failure_reasons.append("thesis_id_mismatch")
    if parseable and not status_allows:
        failure_reasons.append("status_not_passed")
    if parseable and not numeric_edge_present:
        failure_reasons.append("missing_expected_edge_or_cost_bps")
    if parseable and numeric_edge_present and not edge_exceeds_cost:
        failure_reasons.append("edge_does_not_exceed_cost")
    if parseable and not sample_sufficient:
        failure_reasons.append("insufficient_sample_count")
    if parseable and not data_span_sufficient:
        failure_reasons.append("insufficient_data_span")

    return {
        "path": path_label,
        "exists": exists,
        "within_workspace": within_workspace,
        "parseable": parseable,
        "error": error,
        "factory": factory or None,
        "factory_valid": factory_valid,
        "safety_scope_valid": safety_scope_valid,
        "event_source_valid": event_source_valid,
        "event_source_context_alignment_valid": event_source_context_alignment_valid,
        "event_source_failure_synthesis_guard_valid": (
            event_source_failure_synthesis_guard_valid
        ),
        "event_source": _local_falsification_event_source_summary(event_source),
        "thesis_id": artifact_thesis_id,
        "mechanism_class": mechanism_class,
        "thesis_matches": thesis_matches,
        "status": status or None,
        "expected_edge_bps": expected_edge_bps,
        "all_in_cost_bps": all_in_cost_bps,
        "net_edge_bps": None if net_edge_bps is None else round(net_edge_bps, 6),
        "sample_count": sample_count,
        "sample_sufficient": sample_sufficient,
        "data_span_days": data_span_days,
        "minimum_data_span_days": _LOCAL_FALSIFICATION_MIN_DATA_SPAN_DAYS,
        "data_span_sufficient": data_span_sufficient,
        "cost_edge_passes": cost_edge_passes,
        "failure_reasons": failure_reasons,
    }


def _local_falsification_cost_payload(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    nested = payload.get("cost_edge_evidence")
    if isinstance(nested, dict):
        merged = dict(payload)
        merged.update(nested)
        return merged
    return payload


def _local_falsification_safety_scope_valid(safety_scope: Any) -> bool:
    if not isinstance(safety_scope, dict):
        return False
    unsafe_flags = (
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
        and all(not bool(safety_scope.get(flag)) for flag in unsafe_flags)
        and (leverage is None or leverage <= 1.0)
    )


def _local_falsification_event_source_valid(event_source: Any) -> bool:
    if not isinstance(event_source, dict):
        return False
    return (
        event_source.get("factory_valid") is True
        and event_source.get("status_completed") is True
        and event_source.get("thesis_matches") is True
        and event_source.get("event_path_matches") is True
        and event_source.get("ohlcv_path_matches") is True
        and event_source.get("safety_scope_valid") is True
        and _local_falsification_event_source_context_alignment_valid(event_source)
    )


def _local_falsification_event_source_context_alignment_valid(event_source: Any) -> bool:
    if not isinstance(event_source, dict):
        return False
    if event_source.get("closed_context_candle_alignment_valid") is True:
        return True
    if event_source.get("context_features_used") is False:
        return True
    required_contexts = event_source.get("required_contexts")
    if isinstance(required_contexts, list) and not required_contexts:
        return True
    return (
        event_source.get("context_features_used") is True
        and event_source.get("context_merge_semantics") == _CONTEXT_MERGE_SEMANTICS
        and event_source.get("closed_context_candle_alignment_valid") is True
    )


def _local_falsification_event_source_failure_synthesis_guard_valid(
    event_source: Any,
) -> bool:
    if not isinstance(event_source, dict):
        return False
    if event_source.get("failure_synthesis_guard_valid") is True:
        return True

    nested = event_source.get("failure_synthesis_summary")
    if isinstance(nested, dict):
        used = nested.get("used")
        parseable = nested.get("parseable")
        allow_failed = nested.get("allow_failed_thesis_or_family")
        thesis_repeats = nested.get("thesis_repeats_failed_synthesis")
        mechanism_repeats = nested.get("mechanism_repeats_failed_synthesis")
    else:
        used = event_source.get("failure_synthesis_used")
        parseable = event_source.get("failure_synthesis_parseable")
        allow_failed = event_source.get("failure_synthesis_allow_failed_thesis_or_family")
        thesis_repeats = event_source.get("failure_synthesis_thesis_repeats")
        mechanism_repeats = event_source.get("failure_synthesis_mechanism_repeats")

    return (
        used is True
        and parseable is True
        and allow_failed is not True
        and thesis_repeats is not True
        and mechanism_repeats is not True
    )


def _local_falsification_event_source_summary(event_source: Any) -> dict[str, Any]:
    if not isinstance(event_source, dict):
        return {"valid": False}
    return {
        "valid": _local_falsification_event_source_valid(event_source),
        "context_alignment_valid": (
            _local_falsification_event_source_context_alignment_valid(event_source)
        ),
        "failure_synthesis_guard_valid": (
            _local_falsification_event_source_failure_synthesis_guard_valid(event_source)
        ),
        "path": event_source.get("path"),
        "factory": event_source.get("factory"),
        "status": event_source.get("status"),
        "thesis_id": event_source.get("thesis_id"),
        "events_csv_path": event_source.get("events_csv_path"),
        "source_ohlcv_path": event_source.get("source_ohlcv_path"),
        "event_count": event_source.get("event_count"),
        "context_features_used": event_source.get("context_features_used"),
        "required_contexts": event_source.get("required_contexts"),
        "context_merge_semantics": event_source.get("context_merge_semantics"),
        "closed_context_candle_alignment_valid": event_source.get(
            "closed_context_candle_alignment_valid"
        ),
        "failure_synthesis_used": event_source.get("failure_synthesis_used"),
        "failure_synthesis_parseable": event_source.get("failure_synthesis_parseable"),
        "failure_synthesis_path": event_source.get("failure_synthesis_path"),
        "failure_synthesis_failed_thesis_id_count": event_source.get(
            "failure_synthesis_failed_thesis_id_count"
        ),
        "failure_synthesis_failed_family_count": event_source.get(
            "failure_synthesis_failed_family_count"
        ),
        "failure_synthesis_thesis_repeats": event_source.get(
            "failure_synthesis_thesis_repeats"
        ),
        "failure_synthesis_mechanism_repeats": event_source.get(
            "failure_synthesis_mechanism_repeats"
        ),
        "failure_synthesis_allow_failed_thesis_or_family": event_source.get(
            "failure_synthesis_allow_failed_thesis_or_family"
        ),
    }


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _failure_synthesis_summary(
    failure_synthesis: dict[str, Any] | None,
    *,
    path: Path,
    root: Path,
    error: str | None,
) -> dict[str, Any]:
    if not isinstance(failure_synthesis, dict):
        return {
            "path": _rel(path, root),
            "available": False,
            "error": error,
        }
    aggregate = failure_synthesis.get("aggregate_failure_summary", {})
    brief = failure_synthesis.get("next_research_brief", {})
    return {
        "path": _rel(path, root),
        "available": True,
        "synthesis_id": failure_synthesis.get("synthesis_id"),
        "status": failure_synthesis.get("status"),
        "candidate_count": failure_synthesis.get("candidate_count"),
        "paper_ready_count": aggregate.get("paper_ready_count"),
        "negative_return_count": len(aggregate.get("negative_return_candidate_ids", []) or []),
        "walk_forward_failed_count": len(
            aggregate.get("walk_forward_failed_candidate_ids", []) or []
        ),
        "edge_discovery_rejection_count": aggregate.get(
            "edge_discovery_rejection_count"
        ),
        "parameter_only_retry_allowed": brief.get("parameter_only_retry_allowed"),
        "paper_or_live_promotion_allowed": brief.get("paper_or_live_promotion_allowed"),
        "requires_new_thesis_id": brief.get("requires_new_thesis_id"),
        "requires_new_research_references": brief.get("requires_new_research_references"),
    }


def _causal_failure_map_constraints(
    *,
    causal_failure_map: dict[str, Any] | None,
    causal_failure_map_path: Path | None,
    causal_failure_map_error: str | None,
    causal_failure_responses: Sequence[dict[str, Any]],
    research_question_responses: Sequence[dict[str, Any]],
    failure_synthesis: dict[str, Any] | None,
    root: Path,
) -> dict[str, Any]:
    if causal_failure_map_path is None:
        return {
            "path": None,
            "file_present": False,
            "parseable": False,
            "error": None,
            "factory": None,
            "factory_valid": False,
            "status": None,
            "completed": False,
            "map_id": None,
            "source_synthesis_id": None,
            "failure_synthesis_id": None,
            "source_synthesis_matches": False,
            "candidate_count": None,
            "category_count": None,
            "requires_research_decision": False,
            "dominant_failure_categories": [],
            "causal_risk_weights": [],
            "required_categories": [],
            "response_categories": [],
            "missing_response_categories": [],
            "weak_response_categories": [],
            "category_evidence_gaps": [],
            "parameter_only_response_categories": [],
            "response_quality_by_category": {},
            "required_research_questions": [],
            "validated_local_falsification_rejections": [],
            "validated_edge_discovery_rejections": [],
            "research_handoff_summaries": [],
            "blocked_next_actions": [],
            "requires_research_question_responses": False,
            "research_question_response_indexes": [],
            "missing_research_question_response_indexes": [],
            "weak_research_question_response_indexes": [],
            "research_question_response_quality_by_index": {},
            "minimum_research_selection_score": None,
            "research_selection_rubric": [],
        }

    guidance = (
        causal_failure_map.get("research_selection_guidance", {})
        if isinstance(causal_failure_map, dict)
        else {}
    )
    category_summary = (
        causal_failure_map.get("causal_failure_categories", {})
        if isinstance(causal_failure_map, dict)
        else {}
    )
    source_synthesis_id = (
        causal_failure_map.get("source_synthesis_id")
        if isinstance(causal_failure_map, dict)
        else None
    )
    failure_synthesis_id = (
        failure_synthesis.get("synthesis_id")
        if isinstance(failure_synthesis, dict)
        else None
    )
    candidate_count = (
        causal_failure_map.get("candidate_count")
        if isinstance(causal_failure_map, dict)
        else None
    )
    required_categories = _required_causal_failure_categories(
        guidance,
        candidate_count=candidate_count,
    )
    causal_risk_weights = _causal_risk_weights_from_guidance(
        guidance,
        required_categories=required_categories,
    )
    response_categories = sorted(
        {
            response["category"]
            for response in causal_failure_responses
            if response.get("category") and response.get("response")
        }
    )
    response_quality = _causal_failure_response_quality(
        required_categories=required_categories,
        causal_failure_responses=causal_failure_responses,
        causal_risk_weights=causal_risk_weights,
    )
    required_questions = [
        _sanitize_text(question)
        for question in guidance.get("required_research_questions", []) or []
        if str(question).strip()
    ]
    validated_local_rejections = [
        dict(item)
        for item in guidance.get("validated_local_falsification_rejections", []) or []
        if isinstance(item, dict)
    ]
    validated_edge_rejections = [
        dict(item)
        for item in guidance.get("validated_edge_discovery_rejections", []) or []
        if isinstance(item, dict)
    ]
    research_handoff_summaries = [
        dict(item)
        for item in guidance.get("research_handoff_summaries", []) or []
        if isinstance(item, dict)
    ]
    blocked_next_actions = [
        str(item).strip()
        for item in guidance.get("blocked_next_actions", []) or []
        if str(item).strip()
    ]
    question_quality = _research_question_response_quality(
        required_questions=required_questions,
        research_question_responses=research_question_responses,
    )
    return {
        "path": _rel(causal_failure_map_path, root),
        "file_present": causal_failure_map_path.is_file(),
        "parseable": causal_failure_map is not None and not causal_failure_map_error,
        "error": causal_failure_map_error,
        "factory": causal_failure_map.get("factory")
        if isinstance(causal_failure_map, dict)
        else None,
        "factory_valid": bool(causal_failure_map)
        and causal_failure_map.get("factory") == "candidate_failure_map",
        "status": causal_failure_map.get("status")
        if isinstance(causal_failure_map, dict)
        else None,
        "completed": bool(causal_failure_map)
        and causal_failure_map.get("status") == "completed",
        "map_id": causal_failure_map.get("map_id")
        if isinstance(causal_failure_map, dict)
        else None,
        "source_synthesis_id": source_synthesis_id,
        "failure_synthesis_id": failure_synthesis_id,
        "source_synthesis_matches": bool(source_synthesis_id)
        and bool(failure_synthesis_id)
        and source_synthesis_id == failure_synthesis_id,
        "candidate_count": candidate_count,
        "category_count": category_summary.get("category_count"),
        "material_category_min_share": _MATERIAL_CAUSAL_CATEGORY_MIN_SHARE,
        "requires_research_decision": bool(
            guidance.get("requires_research_decision_before_proposal")
        ),
        "dominant_failure_categories": _dominant_causal_failure_categories(guidance),
        "causal_risk_weights": causal_risk_weights,
        "required_categories": required_categories,
        "response_categories": response_categories,
        "missing_response_categories": [
            category for category in required_categories if category not in response_categories
        ],
        "weak_response_categories": response_quality["weak_response_categories"],
        "category_evidence_gaps": response_quality["category_evidence_gaps"],
        "parameter_only_response_categories": response_quality[
            "parameter_only_response_categories"
        ],
        "response_quality_by_category": response_quality["by_category"],
        "required_research_questions": required_questions,
        "validated_local_falsification_rejections": validated_local_rejections,
        "validated_edge_discovery_rejections": validated_edge_rejections,
        "research_handoff_summaries": research_handoff_summaries,
        "blocked_next_actions": list(dict.fromkeys(blocked_next_actions)),
        "requires_research_question_responses": bool(
            guidance.get("requires_research_question_responses")
        ),
        "research_question_response_indexes": question_quality["response_indexes"],
        "missing_research_question_response_indexes": question_quality[
            "missing_response_indexes"
        ],
        "weak_research_question_response_indexes": question_quality[
            "weak_response_indexes"
        ],
        "research_question_response_quality_by_index": question_quality["by_index"],
        "minimum_research_selection_score": _research_selection_minimum_score(guidance),
        "research_selection_rubric": [
            item
            for item in guidance.get("research_selection_rubric", []) or []
            if isinstance(item, dict)
        ],
    }


def _causal_failure_map_summary(
    causal_failure_map: dict[str, Any] | None,
    *,
    path: Path | None,
    root: Path,
    error: str | None,
    constraints: dict[str, Any],
) -> dict[str, Any]:
    if path is None:
        return {
            "used": False,
            "available": False,
            "required_categories_to_address": [],
            "causal_risk_weights": [],
            "missing_response_categories": [],
            "weak_response_categories": [],
            "category_evidence_gaps": [],
            "parameter_only_response_categories": [],
            "requires_research_question_responses": False,
            "required_research_questions": [],
            "validated_local_falsification_rejections": [],
            "validated_edge_discovery_rejections": [],
            "research_handoff_summaries": [],
            "blocked_next_actions": [],
            "research_question_response_indexes": [],
            "missing_research_question_response_indexes": [],
            "weak_research_question_response_indexes": [],
            "minimum_research_selection_score": None,
        }
    if not isinstance(causal_failure_map, dict):
        return {
            "used": True,
            "path": _rel(path, root),
            "available": False,
            "error": error,
            "required_categories_to_address": constraints["required_categories"],
            "causal_risk_weights": constraints["causal_risk_weights"],
            "missing_response_categories": constraints["missing_response_categories"],
            "weak_response_categories": constraints["weak_response_categories"],
            "category_evidence_gaps": constraints["category_evidence_gaps"],
            "parameter_only_response_categories": constraints[
                "parameter_only_response_categories"
            ],
            "requires_research_question_responses": constraints[
                "requires_research_question_responses"
            ],
            "required_research_questions": constraints["required_research_questions"],
            "validated_local_falsification_rejections": constraints[
                "validated_local_falsification_rejections"
            ],
            "validated_edge_discovery_rejections": constraints[
                "validated_edge_discovery_rejections"
            ],
            "research_handoff_summaries": constraints["research_handoff_summaries"],
            "blocked_next_actions": constraints["blocked_next_actions"],
            "research_question_response_indexes": constraints[
                "research_question_response_indexes"
            ],
            "missing_research_question_response_indexes": constraints[
                "missing_research_question_response_indexes"
            ],
            "weak_research_question_response_indexes": constraints[
                "weak_research_question_response_indexes"
            ],
            "minimum_research_selection_score": constraints[
                "minimum_research_selection_score"
            ],
        }
    return {
        "used": True,
        "path": _rel(path, root),
        "available": True,
        "map_id": causal_failure_map.get("map_id"),
        "status": causal_failure_map.get("status"),
        "source_synthesis_id": causal_failure_map.get("source_synthesis_id"),
        "candidate_count": causal_failure_map.get("candidate_count"),
        "category_count": constraints["category_count"],
        "material_category_min_share": constraints["material_category_min_share"],
        "requires_research_decision_before_proposal": constraints[
            "requires_research_decision"
        ],
        "dominant_failure_categories": constraints["dominant_failure_categories"],
        "causal_risk_weights": constraints["causal_risk_weights"],
        "required_categories_to_address": constraints["required_categories"],
        "response_categories": constraints["response_categories"],
        "missing_response_categories": constraints["missing_response_categories"],
        "weak_response_categories": constraints["weak_response_categories"],
        "category_evidence_gaps": constraints["category_evidence_gaps"],
        "parameter_only_response_categories": constraints[
            "parameter_only_response_categories"
        ],
        "response_quality_by_category": constraints["response_quality_by_category"],
        "required_research_questions": constraints["required_research_questions"],
        "validated_local_falsification_rejections": constraints[
            "validated_local_falsification_rejections"
        ],
        "validated_edge_discovery_rejections": constraints[
            "validated_edge_discovery_rejections"
        ],
        "research_handoff_summaries": constraints["research_handoff_summaries"],
        "blocked_next_actions": constraints["blocked_next_actions"],
        "requires_research_question_responses": constraints[
            "requires_research_question_responses"
        ],
        "research_question_response_indexes": constraints[
            "research_question_response_indexes"
        ],
        "missing_research_question_response_indexes": constraints[
            "missing_research_question_response_indexes"
        ],
        "weak_research_question_response_indexes": constraints[
            "weak_research_question_response_indexes"
        ],
        "research_question_response_quality_by_index": constraints[
            "research_question_response_quality_by_index"
        ],
        "minimum_research_selection_score": constraints[
            "minimum_research_selection_score"
        ],
        "research_selection_rubric": constraints["research_selection_rubric"],
    }


def _research_selection_score(
    *,
    inputs: ResearchSelectionInputs,
    failure_constraints: dict[str, Any],
    causal_failure_constraints: dict[str, Any],
    research_references: Sequence[dict[str, Any]],
    local_data_paths: Sequence[dict[str, Any]],
    research_quality: dict[str, Any],
) -> dict[str, Any]:
    thesis_id = _sanitize_text(inputs.thesis_id).strip()
    components: list[dict[str, Any]] = []

    def add_component(
        name: str,
        *,
        max_points: float,
        awarded_points: float,
        passed: bool,
        reason: str,
        details: dict[str, Any] | None = None,
        applicable: bool = True,
    ) -> None:
        components.append(
            {
                "name": name,
                "max_points": float(max_points),
                "awarded_points": float(max(0.0, min(max_points, awarded_points))),
                "passed": bool(passed),
                "applicable": bool(applicable),
                "reason": reason,
                "details": details or {},
            }
        )

    repeated_families = failure_constraints["repeated_failed_family_matches"]
    novelty_passed = (
        not failure_constraints["failed_thesis_id_match"] and not repeated_families
    )
    add_component(
        "novelty_against_failure_set",
        max_points=20.0,
        awarded_points=20.0 if novelty_passed else 0.0,
        passed=novelty_passed,
        reason="Thesis ID and family must be outside the latest failed set.",
        details={
            "failed_thesis_id_match": failure_constraints["failed_thesis_id_match"],
            "repeated_failed_family_matches": repeated_families,
            "local_falsification_failed_thesis_id_match": failure_constraints[
                "local_falsification_failed_thesis_id_match"
            ],
            "local_falsification_failed_mechanism_class_matches": (
                failure_constraints[
                    "local_falsification_failed_mechanism_class_matches"
                ]
            ),
            "edge_discovery_failed_thesis_id_match": failure_constraints[
                "edge_discovery_failed_thesis_id_match"
            ],
            "edge_discovery_failed_mechanism_class_matches": failure_constraints[
                "edge_discovery_failed_mechanism_class_matches"
            ],
        },
    )

    min_reference_count = int(failure_constraints["minimum_research_reference_count"] or 0)
    refs_have_required_fields = bool(research_references) and all(
        ref["reference_id"] and ref["title"] and ref["source"] for ref in research_references
    )
    refs_have_relevance = bool(research_references) and all(
        ref["relevance"] for ref in research_references
    )
    refs_have_dates = bool(research_references) and all(
        ref["published_at"] for ref in research_references
    )
    refs_motivate_thesis = bool(research_references) and all(
        thesis_id in ref["motivated_thesis_ids"] for ref in research_references
    )
    reference_count_points = (
        7.0
        if len(research_references) >= min_reference_count
        else 7.0 * (len(research_references) / max(1, min_reference_count))
    )
    reference_quality_points = sum(
        2.0
        for passed in (
            refs_have_required_fields,
            refs_have_relevance,
            refs_have_dates,
            refs_motivate_thesis,
        )
        if passed
    )
    references_passed = (
        len(research_references) >= min_reference_count
        and refs_have_required_fields
        and refs_have_relevance
        and refs_have_dates
        and refs_motivate_thesis
    )
    add_component(
        "structured_research_references",
        max_points=15.0,
        awarded_points=reference_count_points + reference_quality_points,
        passed=references_passed,
        reason="References must be numerous enough, structured, dated, relevant, and thesis-mapped.",
        details={
            "research_reference_count": len(research_references),
            "minimum_research_reference_count": min_reference_count,
            "refs_have_required_fields": refs_have_required_fields,
            "refs_have_relevance": refs_have_relevance,
            "refs_have_dates": refs_have_dates,
            "refs_motivate_thesis": refs_motivate_thesis,
        },
    )

    local_paths_exist = bool(local_data_paths) and all(
        item["within_workspace"] and item["exists"] for item in local_data_paths
    )
    falsification_uses_local_history = bool(
        _LOCAL_HISTORICAL_RE.search(str(inputs.falsification_plan))
    )
    local_points = 0.0
    if local_paths_exist:
        local_points += 9.0
    if falsification_uses_local_history:
        local_points += 6.0
    add_component(
        "local_historical_falsification",
        max_points=15.0,
        awarded_points=local_points,
        passed=local_paths_exist and falsification_uses_local_history,
        reason="A research thesis should be falsifiable with named local closed-candle artifacts.",
        details={
            "local_data_path_count": len(local_data_paths),
            "local_paths_exist": local_paths_exist,
            "falsification_uses_local_history": falsification_uses_local_history,
        },
    )

    causal_map_used = causal_failure_constraints["path"] is not None
    causal_weighted_score = _weighted_causal_response_score(causal_failure_constraints)
    causal_points = causal_weighted_score["weighted_response_score"] if causal_map_used else 0.0
    causal_passed = causal_map_used and causal_points == 30.0
    add_component(
        "causal_failure_response_quality",
        max_points=30.0,
        awarded_points=causal_points,
        passed=causal_passed,
        applicable=causal_map_used,
        reason="Responses must cover and materially answer the current causal failure categories.",
        details={
            "causal_failure_map_used": causal_map_used,
            "required_categories": causal_failure_constraints["required_categories"],
            "missing_response_categories": causal_failure_constraints[
                "missing_response_categories"
            ],
            "weak_response_categories": causal_failure_constraints[
                "weak_response_categories"
            ],
            "category_evidence_gaps": causal_failure_constraints[
                "category_evidence_gaps"
            ],
            "parameter_only_response_categories": causal_failure_constraints[
                "parameter_only_response_categories"
            ],
            "causal_risk_weights": causal_failure_constraints["causal_risk_weights"],
            "weighted_response_score": causal_weighted_score[
                "weighted_response_score"
            ],
            "weighted_response_ratio": causal_weighted_score[
                "weighted_response_ratio"
            ],
            "total_required_risk_weight": causal_weighted_score[
                "total_required_risk_weight"
            ],
            "unanswered_required_risk_weight": causal_weighted_score[
                "unanswered_required_risk_weight"
            ],
            "category_scores": causal_weighted_score["category_scores"],
        },
    )

    thesis_text = " ".join(
        [
            str(inputs.thesis_statement),
            str(inputs.mechanism_summary),
            str(inputs.novelty_rationale),
            str(inputs.edge_rationale),
            str(inputs.falsification_plan),
            " ".join(str(item) for item in inputs.stop_conditions),
        ]
    )
    mechanism_points = 0.0
    if _RESEARCH_MECHANISM_SUBSTANCE_RE.search(thesis_text):
        mechanism_points += 8.0
    if not research_quality["parameter_only_fields"]:
        mechanism_points += 6.0
    if falsification_uses_local_history:
        mechanism_points += 6.0
    add_component(
        "mechanism_and_falsification_substance",
        max_points=20.0,
        awarded_points=mechanism_points,
        passed=mechanism_points == 20.0,
        reason="Core thesis fields must describe a market mechanism and local falsification path.",
        details={
            "parameter_only_field_names": research_quality[
                "parameter_only_field_names"
            ],
            "falsification_uses_local_history": falsification_uses_local_history,
        },
    )

    maximum_score = sum(component["max_points"] for component in components)
    raw_score = sum(component["awarded_points"] for component in components)
    score = round(100.0 * raw_score / maximum_score, 2) if maximum_score else 0.0
    minimum_score = causal_failure_constraints.get("minimum_research_selection_score")
    passes_minimum = minimum_score is None or score >= float(minimum_score)
    return {
        "version": RESEARCH_SELECTION_SCORE_VERSION,
        "score": score,
        "maximum_score": 100.0,
        "minimum_score_required": minimum_score,
        "passes_minimum": passes_minimum,
        "components": components,
        "failed_components": [
            component["name"]
            for component in components
            if component["applicable"] and not component["passed"]
        ],
    }


def _dominant_causal_failure_categories(guidance: dict[str, Any]) -> list[dict[str, Any]]:
    dominant: list[dict[str, Any]] = []
    for item in guidance.get("dominant_failure_categories", []) or []:
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


def _causal_risk_weights_from_guidance(
    guidance: dict[str, Any], *, required_categories: Sequence[str]
) -> list[dict[str, Any]]:
    required_set = set(required_categories)
    weights: list[dict[str, Any]] = []
    for item in guidance.get("causal_risk_weights", []) or []:
        if not isinstance(item, dict):
            continue
        category = _safe_label(str(item.get("category", "")))
        if not category:
            continue
        risk_score = _float_or_none(item.get("risk_score"))
        candidate_share = _float_or_none(item.get("candidate_share"))
        severity = _float_or_none(item.get("severity_multiplier"))
        weights.append(
            {
                "category": category,
                "candidate_count": item.get("candidate_count"),
                "candidate_share": 0.0 if candidate_share is None else candidate_share,
                "severity_multiplier": 1.0 if severity is None else severity,
                "risk_score": 1.0 if risk_score is None else max(0.0, risk_score),
                "required_for_next_research": bool(
                    item.get("required_for_next_research")
                )
                or category in required_set,
                "response_focus": [
                    _sanitize_text(str(value))
                    for value in item.get("response_focus", []) or []
                    if str(value).strip()
                ],
            }
        )
    if weights:
        present_categories = {str(item.get("category") or "") for item in weights}
        for category in required_categories:
            if category not in present_categories:
                weights.append(
                    {
                        "category": category,
                        "candidate_count": None,
                        "candidate_share": None,
                        "severity_multiplier": 1.0,
                        "risk_score": 1.0,
                        "required_for_next_research": True,
                        "response_focus": [],
                    }
                )
        return sorted(
            weights,
            key=lambda item: (
                -float(item.get("risk_score") or 0.0),
                str(item.get("category") or ""),
            ),
        )
    return [
        {
            "category": category,
            "candidate_count": None,
            "candidate_share": None,
            "severity_multiplier": 1.0,
            "risk_score": 1.0,
            "required_for_next_research": True,
            "response_focus": [],
        }
        for category in required_categories
    ]


def _required_causal_failure_categories(
    guidance: dict[str, Any],
    *,
    candidate_count: Any,
) -> list[str]:
    dominant_categories = _dominant_causal_failure_categories(guidance)
    required = [item["category"] for item in dominant_categories[:3]]
    material_threshold = _material_causal_category_threshold(candidate_count)
    if material_threshold is not None:
        for item in dominant_categories[3:]:
            category = item["category"]
            category_count = _float_or_none(item.get("candidate_count"))
            if category_count is not None and category_count >= material_threshold:
                required.append(category)
    return list(dict.fromkeys(required))


def _research_selection_minimum_score(guidance: dict[str, Any]) -> float:
    configured = _float_or_none(guidance.get("minimum_research_selection_score"))
    if configured is None:
        return _DEFAULT_RESEARCH_SELECTION_MIN_SCORE
    return max(0.0, min(100.0, configured))


def _material_causal_category_threshold(candidate_count: Any) -> float | None:
    value = _float_or_none(candidate_count)
    if value is None or value <= 0:
        return None
    return value * _MATERIAL_CAUSAL_CATEGORY_MIN_SHARE


def _weighted_causal_response_score(
    causal_failure_constraints: dict[str, Any]
) -> dict[str, Any]:
    required_categories = list(causal_failure_constraints["required_categories"])
    required_set = set(required_categories)
    missing_categories = set(causal_failure_constraints["missing_response_categories"])
    weak_categories = set(causal_failure_constraints["weak_response_categories"])
    parameter_only_categories = set(
        causal_failure_constraints["parameter_only_response_categories"]
    )
    evidence_gap_categories = {
        str(item.get("category"))
        for item in causal_failure_constraints["category_evidence_gaps"]
        if isinstance(item, dict) and item.get("category")
    }
    raw_weights = [
        item
        for item in causal_failure_constraints.get("causal_risk_weights", [])
        if item.get("category") in required_set
    ]
    if not raw_weights:
        raw_weights = [
            {"category": category, "risk_score": 1.0}
            for category in required_categories
        ]

    category_scores: list[dict[str, Any]] = []
    total_weight = 0.0
    awarded_weight = 0.0
    for item in raw_weights:
        category = str(item.get("category") or "")
        risk_weight = _float_or_none(item.get("risk_score")) or 0.0
        risk_weight = max(0.0, risk_weight)
        total_weight += risk_weight
        quality_ratio = 1.0
        missing_reasons: list[str] = []
        if category in missing_categories:
            quality_ratio = 0.0
            missing_reasons.append("missing_response")
        else:
            if category in weak_categories:
                quality_ratio -= 0.35
                missing_reasons.append("weak_response")
            if category in evidence_gap_categories:
                quality_ratio -= 0.35
                missing_reasons.append("missing_category_evidence")
            if category in parameter_only_categories:
                quality_ratio -= 0.50
                missing_reasons.append("parameter_only_response")
            quality_ratio = max(0.0, quality_ratio)
        weighted_points = risk_weight * quality_ratio
        awarded_weight += weighted_points
        category_scores.append(
            {
                "category": category,
                "risk_weight": risk_weight,
                "quality_ratio": round(quality_ratio, 4),
                "weighted_points": round(weighted_points, 4),
                "required_for_next_research": True,
                "missing_reasons": missing_reasons,
            }
        )

    weighted_ratio = awarded_weight / total_weight if total_weight else 0.0
    unanswered_risk = total_weight - awarded_weight
    return {
        "weighted_response_ratio": round(weighted_ratio, 4),
        "weighted_response_score": round(30.0 * weighted_ratio, 2),
        "total_required_risk_weight": round(total_weight, 4),
        "unanswered_required_risk_weight": round(unanswered_risk, 4),
        "category_scores": category_scores,
    }


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _causal_failure_responses(inputs: ResearchSelectionInputs) -> list[dict[str, Any]]:
    responses: list[dict[str, Any]] = []
    for raw in inputs.causal_failure_responses:
        text = _sanitize_text(str(raw)).strip()
        if not text:
            continue
        category, response = _split_causal_failure_response(text)
        responses.append(
            {
                "category": _safe_label(category),
                "response": response.strip(),
                "raw": text,
            }
        )
    return responses


def _research_question_responses(inputs: ResearchSelectionInputs) -> list[dict[str, Any]]:
    responses: list[dict[str, Any]] = []
    for position, raw in enumerate(inputs.research_question_responses, start=1):
        text = _sanitize_text(str(raw)).strip()
        if not text:
            continue
        key, response = _split_research_question_response(text)
        question_index = None
        if key.isdigit():
            question_index = int(key)
        responses.append(
            {
                "question_index": question_index,
                "question": "" if question_index is not None else key.strip(),
                "response": response.strip(),
                "position": position,
            }
        )
    return responses


def _research_question_response_quality(
    *,
    required_questions: Sequence[str],
    research_question_responses: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    by_index: dict[int, dict[str, Any]] = {}
    normalized_required = {
        _normalize_question_text(question): index
        for index, question in enumerate(required_questions, start=1)
    }
    for response in research_question_responses:
        question_index = response.get("question_index")
        if question_index is None:
            question_index = normalized_required.get(
                _normalize_question_text(str(response.get("question", "")))
            )
        if not isinstance(question_index, int):
            continue
        text = str(response.get("response", "")).strip()
        word_count = _causal_response_word_count(text)
        question = (
            str(required_questions[question_index - 1])
            if 1 <= question_index <= len(required_questions)
            else ""
        )
        missing_groups = _missing_research_question_response_requirement_groups(
            question=question,
            response=text,
        )
        by_index[question_index] = {
            "question_index": question_index,
            "word_count": word_count,
            "minimum_word_count": _CAUSAL_RESPONSE_MIN_WORDS,
            "missing_requirement_groups": missing_groups,
            "weak": word_count < _CAUSAL_RESPONSE_MIN_WORDS or bool(missing_groups),
        }
    required_indexes = list(range(1, len(required_questions) + 1))
    response_indexes = sorted(index for index in by_index if index in required_indexes)
    return {
        "response_indexes": response_indexes,
        "missing_response_indexes": [
            index for index in required_indexes if index not in response_indexes
        ],
        "weak_response_indexes": [
            index
            for index in response_indexes
            if by_index[index].get("weak")
        ],
        "by_index": {str(index): by_index[index] for index in sorted(by_index)},
    }


def _missing_research_question_response_requirement_groups(
    *, question: str, response: str
) -> list[str]:
    missing: list[str] = []
    if _RESEARCH_QUESTION_CALENDAR_EVIDENCE_RE.search(question) and not (
        _RESEARCH_QUESTION_CALENDAR_RESPONSE_RE.search(response)
    ):
        missing.append("calendar_window_evidence")
    return missing


def _split_research_question_response(text: str) -> tuple[str, str]:
    if "=" in text:
        key, response = text.split("=", 1)
        return key.strip(), response.strip()
    if ":" in text:
        key, response = text.split(":", 1)
        return key.strip(), response.strip()
    return "", text


def _normalize_question_text(text: str) -> str:
    return re.sub(r"\W+", " ", text).strip().lower()


def _causal_failure_response_quality(
    *,
    required_categories: Sequence[str],
    causal_failure_responses: Sequence[dict[str, Any]],
    causal_risk_weights: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    responses_by_category: dict[str, list[str]] = {}
    for response in causal_failure_responses:
        category = str(response.get("category", "")).strip()
        text = str(response.get("response", "")).strip()
        if category and text:
            responses_by_category.setdefault(category, []).append(text)

    weak_categories: list[str] = []
    category_evidence_gaps: list[dict[str, Any]] = []
    parameter_only_categories: list[str] = []
    by_category: dict[str, dict[str, Any]] = {}
    risk_score_by_category = _causal_risk_score_by_category(causal_risk_weights)

    for category in required_categories:
        text = " ".join(responses_by_category.get(category, []))
        word_count = _causal_response_word_count(text)
        missing_groups = [
            *_missing_causal_response_requirement_groups(category, text),
            *_missing_high_risk_causal_response_requirement_groups(
                category,
                text,
                risk_score=risk_score_by_category.get(category, 0.0),
            ),
        ]
        parameter_only = _causal_response_is_parameter_only(text)
        by_category[category] = {
            "word_count": word_count,
            "minimum_word_count": _CAUSAL_RESPONSE_MIN_WORDS,
            "risk_score": risk_score_by_category.get(category),
            "high_risk_minimum_score": _HIGH_RISK_CAUSAL_RESPONSE_MIN_RISK_SCORE,
            "missing_requirement_groups": missing_groups,
            "parameter_only": parameter_only,
        }
        if text and word_count < _CAUSAL_RESPONSE_MIN_WORDS:
            weak_categories.append(category)
        if text and missing_groups:
            category_evidence_gaps.append(
                {
                    "category": category,
                    "missing_requirement_groups": missing_groups,
                }
            )
        if text and parameter_only:
            parameter_only_categories.append(category)

    return {
        "weak_response_categories": weak_categories,
        "category_evidence_gaps": category_evidence_gaps,
        "parameter_only_response_categories": parameter_only_categories,
        "by_category": by_category,
    }


def _missing_causal_response_requirement_groups(category: str, text: str) -> list[str]:
    requirements = _CAUSAL_RESPONSE_CATEGORY_REQUIREMENTS.get(
        category,
        _DEFAULT_CAUSAL_RESPONSE_REQUIREMENTS,
    )
    return [name for name, pattern in requirements if not pattern.search(text)]


def _missing_high_risk_causal_response_requirement_groups(
    category: str, text: str, *, risk_score: float
) -> list[str]:
    if risk_score < _HIGH_RISK_CAUSAL_RESPONSE_MIN_RISK_SCORE:
        return []
    requirements = _HIGH_RISK_CAUSAL_RESPONSE_REQUIREMENTS.get(category, ())
    return [name for name, pattern in requirements if not pattern.search(text)]


def _causal_risk_score_by_category(
    causal_risk_weights: Sequence[dict[str, Any]]
) -> dict[str, float]:
    scores: dict[str, float] = {}
    for item in causal_risk_weights:
        if not isinstance(item, dict):
            continue
        category = _safe_label(str(item.get("category", "")))
        if not category:
            continue
        risk_score = _float_or_none(item.get("risk_score")) or 0.0
        scores[category] = max(scores.get(category, 0.0), risk_score)
    return scores


def _causal_response_is_parameter_only(text: str) -> bool:
    return bool(_PARAMETER_TUNING_RE.search(text)) and not bool(
        _CAUSAL_RESPONSE_SUBSTANCE_RE.search(text)
    )


def _causal_response_word_count(text: str) -> int:
    ascii_tokens = re.findall(r"[A-Za-z0-9]+(?:[-_][A-Za-z0-9]+)?", text)
    non_ascii_chars = sum(1 for char in text if ord(char) > 127 and not char.isspace())
    return len(ascii_tokens) + non_ascii_chars // 2


def _split_causal_failure_response(text: str) -> tuple[str, str]:
    for separator in ("=", ":"):
        if separator in text:
            category, response = text.split(separator, 1)
            return category.strip(), response.strip()
    return "", text.strip()


def _research_quality(inputs: ResearchSelectionInputs, *, text_fields: dict[str, Any]) -> dict[str, Any]:
    parameter_only_fields = _parameter_only_research_fields(text_fields)
    return {
        "parameter_only_research_allowed": False,
        "parameter_only_fields": parameter_only_fields,
        "parameter_only_field_names": [item["field"] for item in parameter_only_fields],
        "parameter_only_claim_count": len(parameter_only_fields),
        "mechanism_substance_required": True,
        "quality_inputs": [
            "thesis_statement",
            "mechanism_summary",
            "novelty_rationale",
            "edge_rationale",
            "falsification_plan",
            "stop_conditions",
        ],
        "reviewer_notes_scanned": bool(inputs.reviewer_notes),
    }


def _parameter_only_research_fields(text_fields: dict[str, Any]) -> list[dict[str, Any]]:
    field_values: list[tuple[str, Any]] = [
        ("thesis_statement", text_fields.get("thesis_statement", "")),
        ("mechanism_summary", text_fields.get("mechanism_summary", "")),
        ("novelty_rationale", text_fields.get("novelty_rationale", "")),
        ("edge_rationale", text_fields.get("edge_rationale", "")),
        ("falsification_plan", text_fields.get("falsification_plan", "")),
        ("stop_conditions", text_fields.get("stop_conditions", [])),
    ]
    findings: list[dict[str, Any]] = []
    for field_name, raw_value in field_values:
        if isinstance(raw_value, list):
            for index, item in enumerate(raw_value, start=1):
                _append_parameter_only_finding(
                    findings,
                    field_name=f"{field_name}[{index}]",
                    text=str(item),
                )
        else:
            _append_parameter_only_finding(
                findings,
                field_name=field_name,
                text=str(raw_value),
            )
    return findings


def _append_parameter_only_finding(
    findings: list[dict[str, Any]], *, field_name: str, text: str
) -> None:
    if not _research_text_is_parameter_only(text):
        return
    findings.append(
        {
            "field": field_name,
            "excerpt": _text_excerpt(text),
            "parameter_terms": _non_negated_matches(_PARAMETER_TUNING_RE, text),
        }
    )


def _research_text_is_parameter_only(text: str) -> bool:
    return bool(_non_negated_matches(_PARAMETER_TUNING_RE, text)) and not bool(
        _RESEARCH_MECHANISM_SUBSTANCE_RE.search(text)
    )


def _text_excerpt(text: str, limit: int = 160) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return f"{compact[:limit].rstrip()}..."


def _sanitized_text_fields(inputs: ResearchSelectionInputs) -> dict[str, Any]:
    return {
        "thesis_statement": _sanitize_text(inputs.thesis_statement),
        "mechanism_summary": _sanitize_text(inputs.mechanism_summary),
        "novelty_rationale": _sanitize_text(inputs.novelty_rationale),
        "required_data": [_sanitize_text(item) for item in inputs.required_data],
        "edge_rationale": _sanitize_text(inputs.edge_rationale),
        "transaction_cost_exposure": _sanitize_text(inputs.transaction_cost_exposure),
        "falsification_plan": _sanitize_text(inputs.falsification_plan),
        "stop_conditions": [_sanitize_text(item) for item in inputs.stop_conditions],
    }


def _raw_dependency_text(
    inputs: ResearchSelectionInputs, research_references: Sequence[dict[str, Any]]
) -> str:
    fields: list[Any] = [
        inputs.thesis_id,
        inputs.thesis_family,
        inputs.mechanism_class,
        inputs.thesis_statement,
        inputs.mechanism_summary,
        inputs.novelty_rationale,
        *inputs.required_data,
        inputs.edge_rationale,
        inputs.transaction_cost_exposure,
        inputs.falsification_plan,
        *inputs.stop_conditions,
        *inputs.causal_failure_responses,
        *inputs.research_question_responses,
        *inputs.reviewer_notes,
    ]
    for ref in research_references:
        fields.extend(
            [
                ref.get("reference_id", ""),
                ref.get("title", ""),
                ref.get("source", ""),
                ref.get("relevance", ""),
            ]
        )
    return "\n".join(str(field) for field in fields)


def _render_report(decision: dict[str, Any]) -> str:
    thesis = decision.get("thesis", {})
    novelty = decision.get("novelty_assessment", {})
    failure_map = decision.get("causal_failure_map", {})
    quality = decision.get("research_quality", {})
    selection_score = decision.get("research_selection_score", {})
    lines = [
        "# Research Selection Decision",
        "",
        f"- decision_id: {decision.get('decision_id')}",
        f"- status: {decision.get('status')}",
        f"- proposal_generation_allowed: {decision.get('proposal_generation_allowed')}",
        f"- code_generation_allowed: {decision.get('code_generation_allowed')}",
        f"- thesis_id: {thesis.get('thesis_id')}",
        f"- thesis_family: {thesis.get('thesis_family')}",
        f"- mechanism_class: {thesis.get('mechanism_class')}",
        "",
        "## Novelty Assessment",
        "",
        "- repeated_failed_family_matches: "
        + ", ".join(novelty.get("repeated_failed_family_matches", []) or ["None"]),
        f"- failed_thesis_id_match: {novelty.get('failed_thesis_id_match')}",
        "- local_falsification_failed_mechanism_class_matches: "
        + ", ".join(
            novelty.get("local_falsification_failed_mechanism_class_matches", [])
            or ["None"]
        ),
        "- local_falsification_failed_thesis_ids: "
        + ", ".join(
            novelty.get("local_falsification_failed_thesis_ids", []) or ["None"]
        ),
        "- edge_discovery_failed_mechanism_class_matches: "
        + ", ".join(
            novelty.get("edge_discovery_failed_mechanism_class_matches", [])
            or ["None"]
        ),
        "- edge_discovery_failed_thesis_ids: "
        + ", ".join(novelty.get("edge_discovery_failed_thesis_ids", []) or ["None"]),
        f"- minimum_research_reference_count: {novelty.get('minimum_research_reference_count')}",
        "",
        "## Required Data",
        "",
    ]
    lines.extend(_bullet_lines(thesis.get("required_data", [])))
    lines.extend(["", "## Local Data Paths", ""])
    lines.extend(_bullet_lines(thesis.get("local_data_paths", [])))
    lines.extend(["", "## Local Data Quality Reports", ""])
    lines.extend(_bullet_lines(thesis.get("local_data_quality_report_paths", [])))
    lines.extend(["", "## Structural Data Capability Reports", ""])
    lines.extend(_bullet_lines(thesis.get("structural_data_capability_report_paths", [])))
    local_falsification = decision.get("local_falsification_evidence", {})
    if local_falsification:
        lines.extend(["", "## Local Falsification Evidence", ""])
        lines.extend(
            [
                "- high_risk_cost_evidence_required: "
                f"{local_falsification.get('high_risk_cost_evidence_required')}",
                f"- artifact_count: {local_falsification.get('artifact_count')}",
                "- passing_cost_edge_artifact_count: "
                f"{local_falsification.get('passing_cost_edge_artifact_count')}",
                "- minimum_data_span_days: "
                f"{local_falsification.get('minimum_data_span_days')}",
            ]
        )
        for item in local_falsification.get("artifacts", []) or []:
            lines.append(
                f"- {item.get('path')}: cost_edge_passes="
                f"{item.get('cost_edge_passes')}, net_edge_bps="
                f"{item.get('net_edge_bps')}, sample_count={item.get('sample_count')}, "
                f"data_span_days={item.get('data_span_days')}, "
                f"event_source_valid={item.get('event_source_valid')}, "
                "event_source_context_alignment_valid="
                f"{item.get('event_source_context_alignment_valid')}, "
                "event_source_failure_synthesis_guard_valid="
                f"{item.get('event_source_failure_synthesis_guard_valid')}"
            )
    prior_rejections = decision.get("prior_local_falsification_rejections", {})
    if prior_rejections:
        lines.extend(["", "## Prior Local Falsification Rejections", ""])
        lines.extend(
            [
                f"- artifact_count: {prior_rejections.get('artifact_count')}",
                "- matching_rejection_count: "
                f"{prior_rejections.get('matching_rejection_count')}",
            ]
        )
        for item in prior_rejections.get("matching_rejections", []) or []:
            lines.append(
                f"- {item.get('path')}: thesis_id={item.get('thesis_id')}, "
                f"mechanism_class={item.get('mechanism_class')}, "
                f"status={item.get('status')}, net_edge_bps={item.get('net_edge_bps')}"
            )
    lines.extend(["", "## Stop Conditions", ""])
    lines.extend(_bullet_lines(thesis.get("stop_conditions", [])))
    lines.extend(["", "## Research Quality", ""])
    lines.extend(
        [
            f"- parameter_only_research_allowed: {quality.get('parameter_only_research_allowed')}",
            "- parameter_only_field_names: "
            + ", ".join(quality.get("parameter_only_field_names", []) or ["None"]),
            f"- mechanism_substance_required: {quality.get('mechanism_substance_required')}",
        ]
    )
    if selection_score:
        lines.extend(
            [
                "",
                "## Research Selection Score",
                "",
                f"- score: {selection_score.get('score')}",
                f"- minimum_score_required: {selection_score.get('minimum_score_required')}",
                f"- passes_minimum: {selection_score.get('passes_minimum')}",
                "- failed_components: "
                + ", ".join(selection_score.get("failed_components", []) or ["None"]),
            ]
        )
        for component in selection_score.get("components", []) or []:
            lines.append(
                f"- {component.get('name')}: "
                f"{component.get('awarded_points')}/{component.get('max_points')} "
                f"passed={component.get('passed')}"
            )
    if failure_map.get("used"):
        lines.extend(["", "## Causal Failure Map", ""])
        lines.extend(
            [
                f"- map_id: {failure_map.get('map_id')}",
                f"- source_synthesis_id: {failure_map.get('source_synthesis_id')}",
                "- required_categories_to_address: "
                + ", ".join(
                    failure_map.get("required_categories_to_address", []) or ["None"]
                ),
                "- missing_response_categories: "
                + ", ".join(
                    failure_map.get("missing_response_categories", []) or ["None"]
                ),
                "- weak_response_categories: "
                + ", ".join(failure_map.get("weak_response_categories", []) or ["None"]),
                "- parameter_only_response_categories: "
                + ", ".join(
                    failure_map.get("parameter_only_response_categories", []) or ["None"]
                ),
                "- requires_research_question_responses: "
                f"{failure_map.get('requires_research_question_responses')}",
                "- missing_research_question_response_indexes: "
                + ", ".join(
                    str(item)
                    for item in (
                        failure_map.get(
                            "missing_research_question_response_indexes", []
                        )
                        or ["None"]
                    )
                ),
                "- weak_research_question_response_indexes: "
                + ", ".join(
                    str(item)
                    for item in (
                        failure_map.get("weak_research_question_response_indexes", [])
                        or ["None"]
                    )
                ),
            ]
        )
        risk_weights = failure_map.get("causal_risk_weights", []) or []
        lines.extend(["", "### Causal Risk Weights", ""])
        if risk_weights:
            for item in risk_weights:
                lines.append(
                    f"- {item.get('category')}: risk_score={item.get('risk_score')}, "
                    f"required_for_next_research={item.get('required_for_next_research')}"
                )
        else:
            lines.append("- None.")
        local_rejections = (
            failure_map.get("validated_local_falsification_rejections", []) or []
        )
        lines.extend(["", "### Validated Local Falsification Rejections", ""])
        if local_rejections:
            for item in local_rejections:
                lines.append(
                    "- "
                    f"{item.get('thesis_id')} / {item.get('mechanism_class')}: "
                    f"net_edge_bps={item.get('net_edge_bps')}, "
                    "profitable_windows_ratio="
                    f"{item.get('profitable_windows_ratio')}, "
                    "profitable_calendar_windows_ratio="
                    f"{item.get('profitable_calendar_windows_ratio')}"
                )
        else:
            lines.append("- None.")
        edge_rejections = failure_map.get("validated_edge_discovery_rejections", []) or []
        lines.extend(["", "### Validated Edge Discovery Rejections", ""])
        if edge_rejections:
            for item in edge_rejections:
                lines.append(
                    "- "
                    f"{item.get('thesis_id')} / {item.get('mechanism_class')}: "
                    f"best_hold_candles={item.get('best_hold_candles')}, "
                    f"net_edge_bps={item.get('net_edge_bps')}, "
                    f"passing_horizon_count={item.get('passing_horizon_count')}"
                )
        else:
            lines.append("- None.")
    lines.extend(["", "## Blockers", ""])
    lines.extend(
        [
            f"- {item.get('name')}: {item.get('message')}"
            for item in decision.get("blockers", [])
        ]
        or ["- None."]
    )
    lines.extend(["", "## Deferrals", ""])
    lines.extend(
        [
            f"- {item.get('name')}: {item.get('message')}"
            for item in decision.get("deferrals", [])
        ]
        or ["- None."]
    )
    lines.extend(["", "## Safety Scope", ""])
    safety = decision.get("safety_scope", {})
    lines.extend(
        [
            f"- paper_trading_started: {safety.get('paper_trading_started')}",
            f"- dry_run_trading_started: {safety.get('dry_run_trading_started')}",
            f"- live_trading: {safety.get('live_trading')}",
            f"- exchange_order_placement: {safety.get('exchange_order_placement')}",
            f"- process_control: {safety.get('process_control')}",
        ]
    )
    lines.append("")
    return "\n".join(lines)


def _decision_status(
    *, blockers: Sequence[dict[str, Any]], deferrals: Sequence[dict[str, Any]]
) -> str:
    if blockers:
        return "blocked"
    if deferrals:
        return "deferred"
    return "approved_for_proposal_generation"


def _check(
    name: str,
    passed: bool,
    severity: str,
    message: str,
    details: dict[str, Any] | None = None,
    *,
    status_if_false: str = "blocked",
) -> ResearchSelectionCheck:
    return ResearchSelectionCheck(
        name=name,
        status="pass" if passed else status_if_false,
        severity=severity,
        message=message,
        details=details or {},
    )


def _load_json_object(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    if not path.is_file():
        return None, "file_not_found"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, str(exc)
    if not isinstance(payload, dict):
        return None, "json_root_not_object"
    return payload, None


def _sanitize_text(text: Any) -> str:
    sanitized = _SECRET_ASSIGNMENT_RE.sub(
        lambda match: f"{match.group('label')}{match.group('sep')}[REDACTED]",
        str(text),
    )
    sanitized = _SECRET_PHRASE_RE.sub(
        lambda match: f"{match.group('label')}{match.group('sep')}[REDACTED]",
        sanitized,
    )
    return _PRIVATE_ENV_RE.sub("[REDACTED_ENV]", sanitized)


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


def _non_empty_sequence(values: Sequence[str]) -> bool:
    return any(str(value).strip() for value in values)


def _bullet_lines(values: Sequence[Any]) -> list[str]:
    lines = [f"- {str(value).strip()}" for value in values if str(value).strip()]
    return lines or ["- Not supplied."]


def _decision_id(created_at: str, thesis_id: str) -> str:
    try:
        parsed = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    except ValueError:
        parsed = datetime.now(UTC)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    timestamp = parsed.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}_{_safe_path_component(thesis_id) or 'research_thesis'}"


def _safe_label(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9_]+", "_", value.strip().lower()).strip("_")
    return token


def _safe_path_component(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9_]+", "_", value.strip()).strip("_")
    return token or "research_decision"


def _resolve_inside(path: Path, root: Path) -> Path:
    resolved = (path if path.is_absolute() else root / path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Path must resolve inside the workspace: {path}") from exc
    return resolved


def _rel(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root))
    except ValueError:
        return str(path)
