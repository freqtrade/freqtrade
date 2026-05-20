from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence


CURRENT_OBSERVATION_SOURCE_TYPES = {
    "backtest",
    "walk_forward",
    "local_shadow_replay",
}
FUTURE_OBSERVATION_SOURCE_TYPES = {
    "future_paper",
    "future_dry_run",
    "future_paper_observation",
    "future_dry_run_observation",
}
MARKET_REGIMES = {
    "trend_up",
    "trend_down",
    "range",
    "high_volatility",
    "low_volatility",
    "liquidity_stress",
    "post_spike_reversion",
    "mixed",
    "unknown",
}
ELIGIBILITY_OUTCOMES = {
    "GLOBAL_SELECTOR_ELIGIBLE",
    "REGIME_SCOPED_SELECTOR_ELIGIBLE",
    "SHADOW_ONLY",
    "NO_TRADE_POLICY",
    "QUARANTINE",
    "REJECT",
    "INSUFFICIENT_EVIDENCE",
}
REQUIRED_OBSERVATION_FIELDS = (
    "observation_id",
    "created_at",
    "source_type",
    "strategy_id",
    "strategy_version",
    "candidate_id",
    "signal_version",
    "risk_policy_version",
    "pair",
    "timeframe",
    "window_start",
    "window_end",
    "market_regime",
    "regime_classifier_version",
    "baseline_id",
    "cost_model_id",
    "normal_cost_bps",
    "stress_cost_bps",
    "trade_count",
    "exposure_ratio",
    "gross_return",
    "net_return_normal_cost",
    "net_return_stress_cost",
    "max_drawdown",
    "downside_deviation",
    "win_rate",
    "profit_factor",
    "no_trade_reason",
    "no_trade_opportunity_cost",
    "data_quality_flags",
    "reason_codes",
)
EVIDENCE_VERSION_FIELDS = (
    "strategy_version",
    "signal_version",
    "activation_regime_scope",
    "risk_policy_version",
    "regime_classifier_version",
    "cost_model_id",
)


@dataclass(frozen=True)
class RegimePromotionThresholds:
    min_sample_days: float = 30.0
    min_window_count: int = 2
    min_trade_count: int = 10
    min_walk_forward_pass_rate: float = 0.6
    min_lower_confidence_bound: float = 0.0
    max_pair_concentration: float = 0.8
    max_calendar_concentration: float = 0.5
    max_drawdown: float = 10.0
    min_global_regime_count: int = 2


@dataclass(frozen=True)
class RegimeStrategyContract:
    strategy_version: str
    signal_version: str
    risk_policy_version: str
    regime_classifier_version: str
    cost_model_id: str
    intended_regimes: Sequence[str]
    excluded_regimes: Sequence[str]
    activation_conditions: Sequence[str]
    no_trade_conditions: Sequence[str]
    regime_shift_stop_conditions: Sequence[str]
    required_features: Sequence[str]
    minimum_evidence: dict[str, Any] = field(default_factory=dict)
    maximum_drawdown_by_regime: dict[str, float] = field(default_factory=dict)
    cost_sensitivity_limits: dict[str, Any] = field(default_factory=dict)
    cooldown_after_regime_change: int = 0
    allowed_pairs: Sequence[str] = field(default_factory=list)
    allowed_timeframes: Sequence[str] = field(default_factory=list)

    @property
    def activation_regime_scope(self) -> str:
        return ",".join(sorted(str(item) for item in self.intended_regimes))


def observation_ledger_schema() -> dict[str, Any]:
    return {
        "factory": "regime_observation_ledger_schema",
        "schema_version": "regime_observation_ledger_v1",
        "required_fields": list(REQUIRED_OBSERVATION_FIELDS),
        "current_source_types": sorted(CURRENT_OBSERVATION_SOURCE_TYPES),
        "future_source_types_blocked_by_default": sorted(FUTURE_OBSERVATION_SOURCE_TYPES),
        "market_regimes": sorted(MARKET_REGIMES),
        "local_artifacts_only": True,
        "process_control": False,
    }


def regime_fitness_scorecard_schema() -> dict[str, Any]:
    return {
        "factory": "regime_fitness_scorecard_schema",
        "schema_version": "regime_fitness_scorecard_v1",
        "eligibility_outcomes": sorted(ELIGIBILITY_OUTCOMES),
        "evidence_version_fields": list(EVIDENCE_VERSION_FIELDS),
        "requires_scorecard_before_phase3_readiness": True,
        "raw_aggregate_pnl_promotion_allowed": False,
    }


def regime_strategy_contract_schema() -> dict[str, Any]:
    return {
        "factory": "regime_strategy_contract_schema",
        "schema_version": "regime_strategy_contract_v1",
        "required_fields": [
            "strategy_version",
            "signal_version",
            "risk_policy_version",
            "regime_classifier_version",
            "cost_model_id",
            "intended_regimes",
            "excluded_regimes",
            "activation_conditions",
            "no_trade_conditions",
            "regime_shift_stop_conditions",
            "required_features",
            "minimum_evidence",
            "maximum_drawdown_by_regime",
            "cost_sensitivity_limits",
            "cooldown_after_regime_change",
            "allowed_pairs",
            "allowed_timeframes",
        ],
        "market_regimes": sorted(MARKET_REGIMES),
    }


def validate_observation_record(
    observation: dict[str, Any], *, allow_future_sources: bool = False
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    missing = [field for field in REQUIRED_OBSERVATION_FIELDS if field not in observation]
    checks.append(_check("required_fields_present", not missing, {"missing_fields": missing}))

    source_type = str(observation.get("source_type") or "")
    allowed_sources = set(CURRENT_OBSERVATION_SOURCE_TYPES)
    if allow_future_sources:
        allowed_sources |= FUTURE_OBSERVATION_SOURCE_TYPES
    checks.append(
        _check(
            "source_type_current_scope_allowed",
            source_type in allowed_sources and (allow_future_sources or source_type not in FUTURE_OBSERVATION_SOURCE_TYPES),
            {
                "source_type": source_type,
                "current_scope_allowed": sorted(CURRENT_OBSERVATION_SOURCE_TYPES),
                "future_sources_blocked": sorted(FUTURE_OBSERVATION_SOURCE_TYPES),
            },
        )
    )

    regime = str(observation.get("market_regime") or "")
    checks.append(
        _check(
            "market_regime_predeclared",
            regime in MARKET_REGIMES,
            {"market_regime": regime, "allowed_regimes": sorted(MARKET_REGIMES)},
        )
    )

    checks.append(
        _check(
            "required_numeric_fields_valid",
            not _invalid_numeric_fields(observation),
            {"invalid_numeric_fields": _invalid_numeric_fields(observation)},
        )
    )
    checks.append(
        _check(
            "list_fields_valid",
            isinstance(observation.get("data_quality_flags", []), list)
            and isinstance(observation.get("reason_codes", []), list),
            {
                "data_quality_flags_type": type(observation.get("data_quality_flags")).__name__,
                "reason_codes_type": type(observation.get("reason_codes")).__name__,
            },
        )
    )
    ok = all(item["passed"] for item in checks)
    return {
        "factory": "regime_observation_validation",
        "schema_version": "regime_observation_ledger_v1",
        "ok": ok,
        "checks": checks,
        "safety_scope": _safety_scope(),
    }


def validate_strategy_contract(contract: RegimeStrategyContract) -> dict[str, Any]:
    intended = [str(item) for item in contract.intended_regimes]
    excluded = [str(item) for item in contract.excluded_regimes]
    checks = [
        _check("intended_regimes_present", bool(intended), {"intended_regimes": intended}),
        _check(
            "intended_regimes_predeclared",
            all(item in MARKET_REGIMES for item in intended),
            {"invalid_regimes": [item for item in intended if item not in MARKET_REGIMES]},
        ),
        _check(
            "excluded_regimes_predeclared",
            all(item in MARKET_REGIMES for item in excluded),
            {"invalid_regimes": [item for item in excluded if item not in MARKET_REGIMES]},
        ),
        _check(
            "regime_action_separated_from_label",
            "trend_down_or_avoid_long" not in intended + excluded,
            {
                "reason": "`trend_down` is a regime; `avoid_long` belongs in no_trade_conditions.",
            },
        ),
        _check(
            "no_trade_conditions_present_for_excluded_regimes",
            bool(contract.no_trade_conditions) or not excluded,
            {"excluded_regimes": excluded},
        ),
        _check(
            "regime_shift_stop_conditions_present",
            bool(contract.regime_shift_stop_conditions),
            {"regime_shift_stop_conditions": list(contract.regime_shift_stop_conditions)},
        ),
        _check(
            "allowed_pairs_present",
            bool(contract.allowed_pairs),
            {"allowed_pairs": list(contract.allowed_pairs)},
        ),
        _check(
            "allowed_timeframes_present",
            bool(contract.allowed_timeframes),
            {"allowed_timeframes": list(contract.allowed_timeframes)},
        ),
    ]
    ok = all(item["passed"] for item in checks)
    return {
        "factory": "regime_strategy_contract_validation",
        "schema_version": "regime_strategy_contract_v1",
        "ok": ok,
        "checks": checks,
        "contract": asdict(contract),
        "evidence_unit": evidence_unit(contract),
        "safety_scope": _safety_scope(),
    }


def evidence_unit(contract: RegimeStrategyContract) -> dict[str, str]:
    return {
        "strategy_version": contract.strategy_version,
        "signal_version": contract.signal_version,
        "activation_regime_scope": contract.activation_regime_scope,
        "risk_policy_version": contract.risk_policy_version,
        "regime_classifier_version": contract.regime_classifier_version,
        "cost_model_id": contract.cost_model_id,
    }


def build_observation_ledger(
    observations: Sequence[dict[str, Any]],
    *,
    ledger_id: str,
    reviewer_notes: Sequence[str] = (),
    allow_future_sources: bool = False,
) -> dict[str, Any]:
    validations = [
        validate_observation_record(item, allow_future_sources=allow_future_sources)
        for item in observations
    ]
    return {
        "factory": "regime_observation_ledger",
        "schema_version": "regime_observation_ledger_v1",
        "ledger_id": ledger_id,
        "created_at": _utc_now(),
        "observation_count": len(observations),
        "observations": list(observations),
        "validations": validations,
        "ok": all(item["ok"] for item in validations),
        "reviewer_notes": list(reviewer_notes),
        "safety_scope": _safety_scope(),
    }


def build_regime_fitness_scorecard(
    candidate_observations: Sequence[dict[str, Any]],
    *,
    contract: RegimeStrategyContract,
    baseline_observations: Sequence[dict[str, Any]] = (),
    thresholds: RegimePromotionThresholds | None = None,
    scorecard_id: str | None = None,
    reviewer_notes: Sequence[str] = (),
) -> dict[str, Any]:
    thresholds = thresholds or RegimePromotionThresholds()
    contract_validation = validate_strategy_contract(contract)
    observation_validations = [
        validate_observation_record(item) for item in candidate_observations
    ] + [validate_observation_record(item) for item in baseline_observations]

    regime_rows = [
        _regime_row(regime, candidate_observations, baseline_observations, contract, thresholds)
        for regime in sorted(set(contract.intended_regimes) | set(contract.excluded_regimes))
    ]
    intended_rows = [row for row in regime_rows if row["market_regime"] in contract.intended_regimes]
    scoped_rows = [
        row
        for row in intended_rows
        if row["decision"] == "REGIME_SCOPED_SELECTOR_ELIGIBLE"
    ]
    rejected_rows = [row for row in regime_rows if row["decision"] == "REJECT"]
    insufficient_rows = [row for row in intended_rows if row["decision"] == "INSUFFICIENT_EVIDENCE"]
    blocked_excluded = [
        row
        for row in regime_rows
        if row["market_regime"] in contract.excluded_regimes
        and row["decision"] in {"REJECT", "NO_TRADE_POLICY", "INSUFFICIENT_EVIDENCE"}
    ]

    if not contract_validation["ok"] or any(not item["ok"] for item in observation_validations):
        decision = "REJECT"
        reason_codes = ["schema_or_contract_validation_failed"]
    elif (
        len(scoped_rows) >= thresholds.min_global_regime_count
        and len(scoped_rows) == len(intended_rows)
        and not insufficient_rows
        and not rejected_rows
        and not contract.excluded_regimes
    ):
        decision = "GLOBAL_SELECTOR_ELIGIBLE"
        reason_codes = ["all_intended_regimes_passed_without_exclusions"]
    elif scoped_rows and not rejected_rows:
        decision = "REGIME_SCOPED_SELECTOR_ELIGIBLE"
        reason_codes = ["one_or_more_intended_regimes_passed"]
        if blocked_excluded:
            reason_codes.append("excluded_regimes_require_no_trade_or_blocking")
    elif insufficient_rows and not rejected_rows:
        decision = "INSUFFICIENT_EVIDENCE"
        reason_codes = ["one_or_more_intended_regimes_under_sampled"]
    elif rejected_rows and scoped_rows:
        decision = "SHADOW_ONLY"
        reason_codes = ["unsafe_regime_prevents_global_or_scoped_activation_without_guard"]
    else:
        decision = "REJECT"
        reason_codes = ["no_regime_has_eligible_scorecard"]

    return {
        "factory": "regime_fitness_scorecard",
        "schema_version": "regime_fitness_scorecard_v1",
        "scorecard_id": scorecard_id or _scorecard_id(),
        "created_at": _utc_now(),
        "decision": decision,
        "eligible_regimes": [
            row["market_regime"]
            for row in regime_rows
            if row["decision"] == "REGIME_SCOPED_SELECTOR_ELIGIBLE"
        ],
        "blocked_regimes": [
            row["market_regime"]
            for row in regime_rows
            if row["decision"] in {"REJECT", "NO_TRADE_POLICY", "INSUFFICIENT_EVIDENCE"}
        ],
        "reason_codes": reason_codes,
        "evidence_unit": evidence_unit(contract),
        "thresholds": asdict(thresholds),
        "contract_validation": contract_validation,
        "observation_validations": observation_validations,
        "scorecard_by_regime": regime_rows,
        "raw_aggregate_pnl_promotion_allowed": False,
        "phase3_readiness_bypassed": False,
        "phase3_readiness_required_after_scorecard": True,
        "reviewer_notes": list(reviewer_notes),
        "safety_scope": _safety_scope(),
    }


def render_regime_scorecard_report(scorecard: dict[str, Any]) -> str:
    lines = [
        "# Regime Fitness Scorecard",
        "",
        f"- Decision: `{scorecard.get('decision')}`",
        f"- Eligible regimes: {', '.join(scorecard.get('eligible_regimes', [])) or 'none'}",
        f"- Blocked regimes: {', '.join(scorecard.get('blocked_regimes', [])) or 'none'}",
        f"- Raw aggregate PnL promotion allowed: `{scorecard.get('raw_aggregate_pnl_promotion_allowed')}`",
        f"- Phase 3 readiness required after scorecard: `{scorecard.get('phase3_readiness_required_after_scorecard')}`",
        "",
        "## Scorecard by Regime",
        "",
        "| regime | decision | windows | trades | normal pnl | stress pnl | baseline delta | reason codes |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in scorecard.get("scorecard_by_regime", []):
        lines.append(
            "| {regime} | {decision} | {windows} | {trades} | {normal:.6f} | {stress:.6f} | {delta:.6f} | {reasons} |".format(
                regime=row.get("market_regime"),
                decision=row.get("decision"),
                windows=row.get("window_count", 0),
                trades=row.get("trade_count", 0),
                normal=row.get("net_pnl_normal_cost") or 0.0,
                stress=row.get("net_pnl_stress_cost") or 0.0,
                delta=row.get("baseline_delta_normal_cost") or 0.0,
                reasons=", ".join(row.get("reason_codes", [])),
            )
        )
    lines.extend(
        [
            "",
            "## Safety Boundary Confirmation",
            "",
            "- No paper, dry-run, live, exchange order, secret, leverage, shorting, or process-control action is authorized by this scorecard.",
            "- Passing this scorecard is necessary but not sufficient for any future Phase 3 paper readiness chain.",
            "",
        ]
    )
    return "\n".join(lines)


def write_regime_scorecard_artifacts(
    scorecard: dict[str, Any], *, root_dir: Path, output_root: Path
) -> tuple[Path, Path]:
    import json

    root = root_dir.resolve()
    out_dir = _resolve(output_root, root) / _safe_component(str(scorecard["scorecard_id"]))
    out_dir.mkdir(parents=True, exist_ok=True)
    scorecard_path = out_dir / "regime_fitness_scorecard.json"
    report_path = out_dir / "regime_fitness_scorecard_report.md"
    scorecard_path.write_text(json.dumps(scorecard, indent=2, ensure_ascii=False), encoding="utf-8")
    report_path.write_text(render_regime_scorecard_report(scorecard), encoding="utf-8")
    return scorecard_path, report_path


def _regime_row(
    regime: str,
    candidate_observations: Sequence[dict[str, Any]],
    baseline_observations: Sequence[dict[str, Any]],
    contract: RegimeStrategyContract,
    thresholds: RegimePromotionThresholds,
) -> dict[str, Any]:
    rows = [item for item in candidate_observations if item.get("market_regime") == regime]
    baselines = [item for item in baseline_observations if item.get("market_regime") == regime]
    sample_days = _sample_days(rows)
    window_count = len(rows)
    trade_count = sum(int(_number(item.get("trade_count")) or 0) for item in rows)
    net_normal = sum(float(_number(item.get("net_return_normal_cost")) or 0.0) for item in rows)
    net_stress = sum(float(_number(item.get("net_return_stress_cost")) or 0.0) for item in rows)
    gross = sum(float(_number(item.get("gross_return")) or 0.0) for item in rows)
    max_drawdown = max((float(_number(item.get("max_drawdown")) or 0.0) for item in rows), default=0.0)
    downside = max(
        (float(_number(item.get("downside_deviation")) or 0.0) for item in rows),
        default=0.0,
    )
    baseline_normal = sum(
        float(_number(item.get("net_return_normal_cost")) or 0.0) for item in baselines
    )
    baseline_stress = sum(
        float(_number(item.get("net_return_stress_cost")) or 0.0) for item in baselines
    )
    baseline_delta_normal = net_normal - baseline_normal
    baseline_delta_stress = net_stress - baseline_stress
    lower_confidence_bound = min(
        (
            float(_number(item.get("lower_confidence_bound")) or 0.0)
            for item in rows
            if item.get("lower_confidence_bound") is not None
        ),
        default=(net_normal / window_count - downside if window_count else 0.0),
    )
    pair_concentration = _max_share([str(item.get("pair")) for item in rows])
    calendar_concentration = _max_share([str(item.get("window_start", ""))[:7] for item in rows])
    data_quality_pass = not any(item.get("data_quality_flags") for item in rows)
    walk_forward_pass_rate = _pass_rate(rows)
    no_trade_opportunity_cost = sum(
        float(_number(item.get("no_trade_opportunity_cost")) or 0.0) for item in rows
    )
    reason_codes: list[str] = []
    regime_max_drawdown = contract.maximum_drawdown_by_regime.get(regime, thresholds.max_drawdown)

    if regime in contract.excluded_regimes:
        if contract.no_trade_conditions:
            decision = "NO_TRADE_POLICY"
            reason_codes.append("excluded_regime_requires_no_trade")
        else:
            decision = "REJECT"
            reason_codes.append("excluded_regime_without_no_trade_conditions")
    elif window_count < thresholds.min_window_count:
        decision = "INSUFFICIENT_EVIDENCE"
        reason_codes.append("minimum_window_count_not_met")
    elif sample_days < thresholds.min_sample_days:
        decision = "INSUFFICIENT_EVIDENCE"
        reason_codes.append("minimum_sample_days_not_met")
    elif trade_count < thresholds.min_trade_count:
        decision = "INSUFFICIENT_EVIDENCE"
        reason_codes.append("minimum_trade_count_not_met")
    elif not data_quality_pass:
        decision = "INSUFFICIENT_EVIDENCE"
        reason_codes.append("data_quality_flags_present")
    elif max_drawdown > regime_max_drawdown:
        decision = "REJECT"
        reason_codes.append("max_drawdown_exceeded")
    elif pair_concentration > thresholds.max_pair_concentration:
        decision = "INSUFFICIENT_EVIDENCE"
        reason_codes.append("pair_concentration_too_high")
    elif calendar_concentration > thresholds.max_calendar_concentration:
        decision = "INSUFFICIENT_EVIDENCE"
        reason_codes.append("calendar_concentration_too_high")
    elif walk_forward_pass_rate < thresholds.min_walk_forward_pass_rate:
        decision = "SHADOW_ONLY"
        reason_codes.append("walk_forward_pass_rate_below_threshold")
    elif lower_confidence_bound <= thresholds.min_lower_confidence_bound:
        decision = "SHADOW_ONLY"
        reason_codes.append("lower_confidence_bound_not_positive")
    elif net_normal <= 0 or net_stress <= 0:
        decision = "SHADOW_ONLY"
        reason_codes.append("net_return_not_positive_after_costs")
    elif baseline_delta_normal <= 0 or baseline_delta_stress <= 0:
        decision = "SHADOW_ONLY"
        reason_codes.append("baseline_delta_not_positive_after_costs")
    else:
        decision = "REGIME_SCOPED_SELECTOR_ELIGIBLE"
        reason_codes.append("regime_scorecard_passed")

    if not rows and regime in contract.excluded_regimes:
        reason_codes.append("no_candidate_observations_in_excluded_regime")

    return {
        "market_regime": regime,
        "decision": decision,
        "reason_codes": reason_codes,
        "sample_days": round(sample_days, 6),
        "window_count": window_count,
        "trade_count": trade_count,
        "exposure_ratio": _mean([_number(item.get("exposure_ratio")) for item in rows]),
        "expectancy": _mean([_number(item.get("net_return_normal_cost")) for item in rows]),
        "profit_factor": _mean([_number(item.get("profit_factor")) for item in rows]),
        "win_rate": _mean([_number(item.get("win_rate")) for item in rows]),
        "max_drawdown": max_drawdown,
        "downside_deviation": downside,
        "gross_return": gross,
        "net_pnl_normal_cost": net_normal,
        "net_pnl_stress_cost": net_stress,
        "baseline_delta_normal_cost": baseline_delta_normal,
        "baseline_delta_stress_cost": baseline_delta_stress,
        "incumbent_delta": None,
        "no_trade_opportunity_cost": no_trade_opportunity_cost,
        "confidence_interval": None,
        "lower_confidence_bound": lower_confidence_bound,
        "walk_forward_pass_rate": walk_forward_pass_rate,
        "pair_concentration": pair_concentration,
        "calendar_concentration": calendar_concentration,
        "data_quality_pass": data_quality_pass,
    }


def _sample_days(rows: Sequence[dict[str, Any]]) -> float:
    total = 0.0
    for item in rows:
        start = _parse_datetime(item.get("window_start"))
        end = _parse_datetime(item.get("window_end"))
        if start and end and end > start:
            total += (end - start).total_seconds() / 86400.0
    return total


def _pass_rate(rows: Sequence[dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    passes = 0
    for item in rows:
        if item.get("gate_recommendation") == "pass" or "pass" in item.get("reason_codes", []):
            passes += 1
        elif float(_number(item.get("net_return_normal_cost")) or 0.0) > 0 and float(
            _number(item.get("net_return_stress_cost")) or 0.0
        ) > 0:
            passes += 1
    return passes / len(rows)


def _invalid_numeric_fields(observation: dict[str, Any]) -> list[str]:
    fields = [
        "normal_cost_bps",
        "stress_cost_bps",
        "trade_count",
        "exposure_ratio",
        "gross_return",
        "net_return_normal_cost",
        "net_return_stress_cost",
        "max_drawdown",
        "downside_deviation",
        "win_rate",
        "profit_factor",
        "no_trade_opportunity_cost",
    ]
    invalid = []
    for field in fields:
        if field in observation and _number(observation.get(field)) is None:
            invalid.append(field)
    return invalid


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _mean(values: Sequence[float | None]) -> float | None:
    valid = [float(value) for value in values if value is not None]
    if not valid:
        return None
    return sum(valid) / len(valid)


def _max_share(values: Sequence[str]) -> float:
    clean = [value for value in values if value]
    if not clean:
        return 0.0
    counts = {value: clean.count(value) for value in set(clean)}
    return max(counts.values()) / len(clean)


def _parse_datetime(value: Any) -> datetime | None:
    if not value:
        return None
    text = str(value)
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        try:
            return datetime.strptime(text, "%Y%m%d").replace(tzinfo=UTC)
        except ValueError:
            return None


def _check(name: str, passed: bool, details: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "details": details or {}}


def _safety_scope() -> dict[str, bool]:
    return {
        "local_artifacts_only": True,
        "paper_trading_started": False,
        "dry_run_trading_started": False,
        "live_trading_started": False,
        "exchange_order_placement": False,
        "uses_api_keys_or_secrets": False,
        "leverage_above_1": False,
        "shorting": False,
        "process_control": False,
        "promotion_authorized_by_this_command": False,
        "phase3_readiness_bypassed": False,
    }


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _scorecard_id() -> str:
    return "regime_scorecard_" + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _resolve(path: Path, root: Path) -> Path:
    resolved = path if path.is_absolute() else root / path
    return resolved.resolve()


def _safe_component(value: str) -> str:
    clean = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value)
    return clean.strip("._") or "regime_scorecard"
