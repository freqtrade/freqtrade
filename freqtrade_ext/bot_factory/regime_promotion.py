from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

from freqtrade_ext.bot_factory.candidate_identity import (
    compare_candidate_identities,
    extract_candidate_identity,
    build_strategy_candidate_identity,
    validate_candidate_identity,
)
from freqtrade_ext.bot_factory.feature_quality import feature_quality_passes_thresholds
from freqtrade_ext.bot_factory.gate_semantics import gate_semantics_payload


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
    "candidate_identity",
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
OPTIONAL_STATE_OBSERVATION_FIELDS = (
    "state_id",
    "horizon_profile_id",
    "state_encoder_version",
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


@dataclass(frozen=True)
class RegimeStrategyLogicSpec:
    logic_id: str
    strategy_id: str
    strategy_class_name: str
    strategy_source_path: str
    strategy_version: str
    signal_version: str
    intended_regimes: Sequence[str]
    excluded_regimes: Sequence[str]
    entry_conditions: Sequence[str]
    exit_conditions: Sequence[str]
    no_trade_conditions: Sequence[str]
    required_features: Sequence[str]
    risk_policy_version: str
    regime_classifier_version: str
    cost_model_id: str
    allowed_pairs: Sequence[str]
    allowed_timeframes: Sequence[str]
    identity_created_at: str = "2026-05-21T00:00:00+09:00"
    source_artifacts: dict[str, str] = field(default_factory=dict)
    reviewer_notes: Sequence[str] = field(default_factory=list)


@dataclass(frozen=True)
class RuntimeRegimeSnapshot:
    current_regime: str
    pair: str
    timeframe: str
    regime_classifier_version: str
    data_quality_pass: bool
    available_features: Sequence[str]
    regime_confidence: float = 1.0
    feature_quality_report: dict[str, Any] | None = None
    production_assumption: bool = False
    process_control_allowed: bool = False
    paper_or_dry_run_process_running: bool = False


@dataclass(frozen=True)
class RuntimeSelectorState:
    last_selected_candidate_id: str | None = None
    last_selected_regime: str | None = None
    observations_since_switch: int = 0
    last_switch_reason: str | None = None


def observation_ledger_schema() -> dict[str, Any]:
    return {
        "factory": "regime_observation_ledger_schema",
        "schema_version": "regime_observation_ledger_v1",
        "required_fields": list(REQUIRED_OBSERVATION_FIELDS),
        "optional_state_fields": list(OPTIONAL_STATE_OBSERVATION_FIELDS),
        "candidate_identity_schema_version": "strategy_candidate_identity_v1",
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
        "phase3_readiness_required_after_scorecard": True,
        "promotion_authorized_by_this_command": False,
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


def strong_uptrend_momentum_logic_spec(
    *,
    strategy_id: str = "long_only_strong_uptrend_momentum",
    strategy_version: str = "strong_uptrend_momentum_v1",
    signal_version: str = "strong_uptrend_signal_v1",
    risk_policy_version: str = "long_only_risk_v1",
    regime_classifier_version: str = "regime_classifier_v1",
    cost_model_id: str = "cost_model_v1",
    allowed_pairs: Sequence[str] = ("BTC/USDT:USDT", "ETH/USDT:USDT"),
    allowed_timeframes: Sequence[str] = ("5m",),
    strategy_class_name: str = "DonchianTrendBullStrategy",
    strategy_source_path: str = "user_data/strategies/DonchianTrendBullStrategy.py",
) -> RegimeStrategyLogicSpec:
    return RegimeStrategyLogicSpec(
        logic_id="strong_uptrend_momentum_v1",
        strategy_id=strategy_id,
        strategy_class_name=strategy_class_name,
        strategy_source_path=strategy_source_path,
        strategy_version=strategy_version,
        signal_version=signal_version,
        intended_regimes=("trend_up",),
        excluded_regimes=("trend_down", "range", "high_volatility", "liquidity_stress", "unknown"),
        entry_conditions=(
            "closed-candle regime classifier emits trend_up",
            "short moving average slope remains positive",
            "close remains above medium moving average",
            "range efficiency confirms directional follow-through",
            "volume/liquidity proxy is not degraded",
        ),
        exit_conditions=(
            "closed-candle regime leaves trend_up",
            "close loses medium moving average",
            "range efficiency collapses into mixed/range behavior",
            "data quality or required feature check fails",
        ),
        no_trade_conditions=(
            "market_regime is not trend_up",
            "high_volatility, liquidity_stress, or unknown regime is active",
            "required regime, trend, volume, or cost features are missing",
            "scorecard eligibility is not regime-scoped or global",
        ),
        required_features=(
            "close",
            "volume",
            "moving_average_slope",
            "range_efficiency",
            "regime_label",
            "cost_model",
        ),
        risk_policy_version=risk_policy_version,
        regime_classifier_version=regime_classifier_version,
        cost_model_id=cost_model_id,
        allowed_pairs=tuple(allowed_pairs),
        allowed_timeframes=tuple(allowed_timeframes),
        source_artifacts={
            "strategy_source": strategy_source_path,
            "historical_backtest_metrics": (
                "data/backtests/DonchianTrendBullStrategy/"
                "historical_uptrend_20240202_20240305_v3/metrics.json"
            ),
            "historical_backtest_trades": (
                "data/backtests/DonchianTrendBullStrategy/"
                "historical_uptrend_20240202_20240305_v3/trades.csv"
            ),
            "selector_replay": (
                "data/regime_selector_replays/"
                "historical_uptrend_20240202_20240304/selector_replay.json"
            ),
        },
        reviewer_notes=(
            "Local selector-eligibility logic only; no paper, dry-run, live, or order process.",
        ),
    )


def downtrend_defensive_rebound_logic_spec(
    *,
    strategy_id: str = "long_only_downtrend_defensive_rebound",
    strategy_version: str = "downtrend_defensive_rebound_v1",
    signal_version: str = "downtrend_rebound_signal_v1",
    risk_policy_version: str = "long_only_risk_v1",
    regime_classifier_version: str = "regime_classifier_v1",
    cost_model_id: str = "cost_model_v1",
    allowed_pairs: Sequence[str] = ("BTC/USDT:USDT", "ETH/USDT:USDT"),
    allowed_timeframes: Sequence[str] = ("5m",),
) -> RegimeStrategyLogicSpec:
    return RegimeStrategyLogicSpec(
        logic_id="downtrend_defensive_rebound_v1",
        strategy_id=strategy_id,
        strategy_class_name="RegimeLogicSpecOnly",
        strategy_source_path="freqtrade_ext/bot_factory/regime_promotion.py",
        strategy_version=strategy_version,
        signal_version=signal_version,
        intended_regimes=("trend_down",),
        excluded_regimes=("trend_up", "range", "high_volatility", "liquidity_stress", "unknown"),
        entry_conditions=(
            "closed-candle regime classifier emits trend_down",
            "downside exhaustion proxy confirms selling pressure is stretched",
            "close reclaims short moving average after an oversold flush",
            "range efficiency confirms a controlled rebound rather than free fall",
            "volume/liquidity proxy is not degraded",
        ),
        exit_conditions=(
            "closed-candle regime leaves trend_down",
            "rebound reclaim fails and close loses the short moving average",
            "high_volatility, liquidity_stress, or unknown regime becomes active",
            "data quality or required feature check fails",
        ),
        no_trade_conditions=(
            "market_regime is not trend_down",
            "shorting is required for the thesis",
            "high_volatility, liquidity_stress, or unknown regime is active",
            "required regime, exhaustion, reclaim, volume, or cost features are missing",
            "scorecard eligibility is not regime-scoped or global",
        ),
        required_features=(
            "close",
            "volume",
            "moving_average_slope",
            "range_efficiency",
            "downside_exhaustion",
            "reclaim_confirmation",
            "regime_label",
            "cost_model",
        ),
        risk_policy_version=risk_policy_version,
        regime_classifier_version=regime_classifier_version,
        cost_model_id=cost_model_id,
        allowed_pairs=tuple(allowed_pairs),
        allowed_timeframes=tuple(allowed_timeframes),
        reviewer_notes=(
            "Local selector-eligibility logic only; no paper, dry-run, live, or order process.",
            "Long-only defensive rebound logic; shorting is explicitly out of scope.",
        ),
    )


def range_mean_reversion_logic_spec(
    *,
    strategy_id: str = "long_only_range_mean_reversion",
    strategy_version: str = "range_mean_reversion_v1",
    signal_version: str = "range_reversion_signal_v1",
    risk_policy_version: str = "long_only_risk_v1",
    regime_classifier_version: str = "regime_classifier_v1",
    cost_model_id: str = "cost_model_v1",
    allowed_pairs: Sequence[str] = ("BTC/USDT:USDT", "ETH/USDT:USDT"),
    allowed_timeframes: Sequence[str] = ("5m",),
) -> RegimeStrategyLogicSpec:
    return RegimeStrategyLogicSpec(
        logic_id="range_mean_reversion_v1",
        strategy_id=strategy_id,
        strategy_class_name="RegimeLogicSpecOnly",
        strategy_source_path="freqtrade_ext/bot_factory/regime_promotion.py",
        strategy_version=strategy_version,
        signal_version=signal_version,
        intended_regimes=("range",),
        excluded_regimes=("trend_up", "trend_down", "high_volatility", "liquidity_stress", "unknown"),
        entry_conditions=(
            "closed-candle regime classifier emits range",
            "price tests the lower range band without a confirmed downside breakout",
            "mean-reversion oscillator confirms stretched position inside the box",
            "range width is sufficient to clear normal and stress cost assumptions",
            "volume/liquidity proxy is not degraded",
        ),
        exit_conditions=(
            "price reverts to the range midpoint or upper range band",
            "closed-candle regime leaves range",
            "directional range efficiency indicates breakout risk",
            "data quality or required feature check fails",
        ),
        no_trade_conditions=(
            "market_regime is not range",
            "trend_up, trend_down, high_volatility, liquidity_stress, or unknown regime is active",
            "range band, oscillator, volume, or cost features are missing",
            "range width is too narrow after cost assumptions",
            "scorecard eligibility is not regime-scoped or global",
        ),
        required_features=(
            "close",
            "volume",
            "range_band_position",
            "range_width",
            "mean_reversion_oscillator",
            "regime_label",
            "cost_model",
        ),
        risk_policy_version=risk_policy_version,
        regime_classifier_version=regime_classifier_version,
        cost_model_id=cost_model_id,
        allowed_pairs=tuple(allowed_pairs),
        allowed_timeframes=tuple(allowed_timeframes),
        reviewer_notes=(
            "Local selector-eligibility logic only; no paper, dry-run, live, or order process.",
        ),
    )


def contract_from_logic_spec(
    logic: RegimeStrategyLogicSpec,
    *,
    minimum_evidence: dict[str, Any] | None = None,
    maximum_drawdown_by_regime: dict[str, float] | None = None,
    cost_sensitivity_limits: dict[str, Any] | None = None,
    cooldown_after_regime_change: int = 3,
) -> RegimeStrategyContract:
    return RegimeStrategyContract(
        strategy_version=logic.strategy_version,
        signal_version=logic.signal_version,
        risk_policy_version=logic.risk_policy_version,
        regime_classifier_version=logic.regime_classifier_version,
        cost_model_id=logic.cost_model_id,
        intended_regimes=tuple(logic.intended_regimes),
        excluded_regimes=tuple(logic.excluded_regimes),
        activation_conditions=tuple(logic.entry_conditions),
        no_trade_conditions=tuple(logic.no_trade_conditions),
        regime_shift_stop_conditions=tuple(logic.exit_conditions),
        required_features=tuple(logic.required_features),
        minimum_evidence=minimum_evidence or {"min_window_count": 2, "min_trade_count": 10},
        maximum_drawdown_by_regime=maximum_drawdown_by_regime
        or {str(regime): 8.0 for regime in logic.intended_regimes},
        cost_sensitivity_limits=cost_sensitivity_limits
        or {"normal_cost_bps_max": 10.0, "stress_cost_bps_max": 20.0},
        cooldown_after_regime_change=cooldown_after_regime_change,
        allowed_pairs=tuple(logic.allowed_pairs),
        allowed_timeframes=tuple(logic.allowed_timeframes),
    )


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
    state_fields_present = [
        field
        for field in OPTIONAL_STATE_OBSERVATION_FIELDS
        if observation.get(field) not in (None, "")
    ]
    if state_fields_present:
        missing_state_fields = [
            field
            for field in OPTIONAL_STATE_OBSERVATION_FIELDS
            if observation.get(field) in (None, "")
        ]
        checks.append(
            _check(
                "state_observation_fields_complete",
                not missing_state_fields,
                {
                    "present_fields": state_fields_present,
                    "missing_fields": missing_state_fields,
                },
            )
        )
        checks.append(
            _check(
                "state_observation_no_future_data",
                observation.get("future_data_used") is False,
                {"future_data_used": observation.get("future_data_used")},
            )
        )
    else:
        checks.append(
            _check(
                "state_observation_fields_not_supplied",
                True,
                {"optional_state_fields": list(OPTIONAL_STATE_OBSERVATION_FIELDS)},
            )
        )
    identity_validation = validate_candidate_identity(observation)
    checks.append(
        _check(
            "candidate_identity_valid",
            identity_validation["ok"],
            {"identity_checks": identity_validation["checks"]},
        )
    )
    if identity_validation["ok"]:
        identity = identity_validation["candidate_identity"]
        checks.extend(_observation_identity_checks(observation, identity))
    ok = all(item["passed"] for item in checks)
    return {
        "factory": "regime_observation_validation",
        "schema_version": "regime_observation_ledger_v1",
        "ok": ok,
        "checks": checks,
        "candidate_identity": identity_validation.get("candidate_identity"),
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


def candidate_identity_from_logic_spec(
    logic: RegimeStrategyLogicSpec,
    *,
    candidate_id: str,
    created_at: str | None = None,
    source_artifacts: dict[str, Any] | None = None,
) -> dict[str, Any]:
    artifacts = dict(logic.source_artifacts)
    artifacts.update(source_artifacts or {})
    return build_strategy_candidate_identity(
        candidate_id=candidate_id,
        strategy_id=logic.strategy_id,
        strategy_class_name=logic.strategy_class_name,
        strategy_source_path=logic.strategy_source_path,
        strategy_version=logic.strategy_version,
        signal_version=logic.signal_version,
        risk_policy_version=logic.risk_policy_version,
        regime_classifier_version=logic.regime_classifier_version,
        cost_model_id=logic.cost_model_id,
        allowed_pairs=logic.allowed_pairs,
        allowed_timeframes=logic.allowed_timeframes,
        created_at=created_at or logic.identity_created_at,
        source_artifacts=artifacts,
    )


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
        "candidate_identities": _unique_candidate_identities(observations),
        "validations": validations,
        "ok": all(item["ok"] for item in validations),
        "reviewer_notes": list(reviewer_notes),
        "safety_scope": _safety_scope(),
    }


def build_shadow_observation_leaderboards(
    observations: Sequence[dict[str, Any]],
    *,
    leaderboard_id: str = "shadow_observation_leaderboard",
) -> dict[str, Any]:
    accepted = []
    rejected = []
    for observation in observations:
        validation = validate_observation_record(observation)
        if validation["ok"]:
            accepted.append(observation)
        else:
            rejected.append({"observation": observation, "validation": validation})
    buckets = {
        "long_term_historical_evidence": [
            item for item in accepted if item.get("source_type") in {"backtest", "walk_forward"}
        ],
        "current_regime_evidence": [
            item for item in accepted if item.get("market_regime") not in {"unknown", "mixed"}
        ],
        "recent_observation_evidence": [
            item for item in accepted if item.get("source_type") == "local_shadow_replay"
        ],
        "data_quality_confidence": sorted(
            [
                {
                    "observation_id": item.get("observation_id"),
                    "data_quality_pass": not bool(item.get("data_quality_flags")),
                    "market_regime": item.get("market_regime"),
                    "source_type": item.get("source_type"),
                }
                for item in accepted
            ],
            key=lambda item: str(item.get("observation_id")),
        ),
    }
    return {
        "factory": "shadow_observation_leaderboard",
        "schema_version": "shadow_observation_leaderboard_v1",
        "leaderboard_id": leaderboard_id,
        "created_at": _utc_now(),
        "accepted_source_types": sorted(CURRENT_OBSERVATION_SOURCE_TYPES),
        "future_source_types_rejected": sorted(FUTURE_OBSERVATION_SOURCE_TYPES),
        "accepted_count": len(accepted),
        "rejected_count": len(rejected),
        "leaderboards": buckets,
        "rejected_observations": rejected,
        "historical_readiness_override_allowed": False,
        "parallel_observations_direct_promotion_allowed": False,
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
    candidate_identity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    thresholds = thresholds or RegimePromotionThresholds()
    contract_validation = validate_strategy_contract(contract)
    observation_validations = [
        validate_observation_record(item) for item in candidate_observations
    ] + [validate_observation_record(item) for item in baseline_observations]
    identity_lineage_validation = _scorecard_identity_lineage_validation(
        candidate_observations,
        baseline_observations,
        contract=contract,
        expected_identity=candidate_identity,
    )
    scorecard_identity = identity_lineage_validation.get("candidate_identity")

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

    if (
        not contract_validation["ok"]
        or any(not item["ok"] for item in observation_validations)
        or not identity_lineage_validation["ok"]
    ):
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
        "candidate_identity": scorecard_identity,
        "identity_lineage_validation": identity_lineage_validation,
        "thresholds": asdict(thresholds),
        "contract_validation": contract_validation,
        "observation_validations": observation_validations,
        "scorecard_by_regime": regime_rows,
        "raw_aggregate_pnl_promotion_allowed": False,
        "phase3_readiness_bypassed": False,
        "phase3_readiness_required_after_scorecard": True,
        "promotion_authorized_by_this_command": False,
        "manual_review_only": False,
        "gate_semantics": gate_semantics_payload(decision),
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
        f"- Promotion authorized by this command: `{scorecard.get('promotion_authorized_by_this_command')}`",
        "",
        "## Gate Semantics",
        "",
        "- Permits: local selector simulation only inside declared evidence scope.",
        "- Does not permit: paper trading, dry-run trading, live trading, process control, or exchange order placement.",
        "- Next required gate: `paper_readiness.pass`.",
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


def evaluate_runtime_strategy_selection(
    *,
    runtime: RuntimeRegimeSnapshot,
    candidates: Sequence[dict[str, Any]],
    selector_id: str = "regime_selector_v1",
    reviewer_notes: Sequence[str] = (),
    selector_state: RuntimeSelectorState | None = None,
    min_confidence_by_regime: dict[str, float] | None = None,
    cooldown_observations: int = 0,
    hysteresis_margin: float = 0.0,
) -> dict[str, Any]:
    selector_state = selector_state or RuntimeSelectorState()
    min_confidence_by_regime = min_confidence_by_regime or {}
    min_confidence = float(min_confidence_by_regime.get(runtime.current_regime, 0.0))
    preflight_blocks: list[str] = []
    if runtime.current_regime == "unknown":
        preflight_blocks.append("runtime_regime_unknown")
    if runtime.regime_confidence < min_confidence:
        preflight_blocks.append("runtime_regime_confidence_below_threshold")
    if (
        selector_state.last_selected_regime
        and selector_state.last_selected_regime != runtime.current_regime
        and selector_state.observations_since_switch < cooldown_observations
    ):
        preflight_blocks.append("runtime_regime_change_cooldown_active")
    evaluated = [
        _runtime_candidate_decision(runtime=runtime, candidate=candidate)
        for candidate in candidates
    ]
    selectable = [] if preflight_blocks else [item for item in evaluated if item["selectable"]]
    ranked_selectable = sorted(
        selectable,
        key=lambda item: (
            item.get("selector_score", 0.0),
            -(item["scorecard_summary"].get("max_drawdown", 0.0) or 0.0),
            str(item["candidate_id"]),
        ),
        reverse=True,
    )
    selected = ranked_selectable[0] if ranked_selectable else None
    previous = next(
        (
            item
            for item in ranked_selectable
            if item.get("candidate_id") == selector_state.last_selected_candidate_id
        ),
        None,
    )
    if (
        selected
        and previous
        and selected["candidate_id"] != previous["candidate_id"]
        and selected.get("selector_score", 0.0) - previous.get("selector_score", 0.0)
        < hysteresis_margin
    ):
        selected = previous
        preflight_blocks.append("selector_hysteresis_kept_previous_candidate")
    action = "select" if selected else "no_trade"
    reason_codes = (
        ["selected_regime_scoped_candidate", "selected_highest_stress_adjusted_candidate"]
        if selected
        else (preflight_blocks or ["no_candidate_matched_runtime_regime"])
    )
    if runtime.production_assumption:
        reason_codes.append("production_running_assumption_only_no_process_control")
    next_state = RuntimeSelectorState(
        last_selected_candidate_id=selected["candidate_id"] if selected else None,
        last_selected_regime=runtime.current_regime if selected else selector_state.last_selected_regime,
        observations_since_switch=(
            selector_state.observations_since_switch + 1
            if selected and selected["candidate_id"] == selector_state.last_selected_candidate_id
            else 0
        ),
        last_switch_reason=";".join(reason_codes),
    )
    return {
        "factory": "runtime_regime_strategy_selector",
        "schema_version": "runtime_regime_selector_v1",
        "selector_id": selector_id,
        "created_at": _utc_now(),
        "runtime": asdict(runtime),
        "action": action,
        "selected_candidate_id": selected["candidate_id"] if selected else None,
        "selected_strategy_id": selected["strategy_id"] if selected else None,
        "selected_logic_id": selected["logic_id"] if selected else None,
        "would_select_in_production_assumption": bool(selected),
        "reason_codes": reason_codes,
        "evaluated_candidates": evaluated,
        "selector_state": asdict(selector_state),
        "next_selector_state": asdict(next_state),
        "reviewer_notes": list(reviewer_notes),
        "safety_scope": _safety_scope(),
    }


def selection_candidate_from_scorecard(
    *,
    logic: RegimeStrategyLogicSpec,
    scorecard: dict[str, Any],
    candidate_id: str,
) -> dict[str, Any]:
    if (
        scorecard.get("diagnostic_only") is True
        or scorecard.get("evidence_eligibility") == "diagnostic_only"
        or scorecard.get("proxy_evidence") is True
        or scorecard.get("relaxed_thresholds_used") is True
        or scorecard.get("selector_candidate_creation_allowed") is False
    ):
        raise ValueError(
            "Diagnostic-only, proxy, relaxed-threshold, or selector-disallowed "
            "scorecards cannot become selector candidates."
        )
    if scorecard.get("manual_review_only") is True or scorecard.get("factory") != "regime_fitness_scorecard":
        raise ValueError(
            "Only deterministic regime_fitness_scorecard artifacts may become selector candidates."
        )
    if scorecard.get("decision") not in {
        "GLOBAL_SELECTOR_ELIGIBLE",
        "REGIME_SCOPED_SELECTOR_ELIGIBLE",
    }:
        raise ValueError(
            "Only selector-eligible regime scorecards may become selector candidates."
        )
    scorecard_identity = extract_candidate_identity(scorecard)
    expected_identity = candidate_identity_from_logic_spec(logic, candidate_id=candidate_id)
    identity_comparison = compare_candidate_identities(
        expected_identity,
        scorecard_identity,
        observed_label="scorecard",
    )
    if not identity_comparison["ok"]:
        raise ValueError(
            "Scorecard candidate identity does not match selector logic: "
            f"{identity_comparison['mismatches']}"
        )
    return {
        "candidate_id": candidate_id,
        "strategy_id": logic.strategy_id,
        "logic_id": logic.logic_id,
        "candidate_identity": scorecard_identity,
        "identity_comparison": identity_comparison,
        "strategy_version": logic.strategy_version,
        "signal_version": logic.signal_version,
        "risk_policy_version": logic.risk_policy_version,
        "regime_classifier_version": logic.regime_classifier_version,
        "cost_model_id": logic.cost_model_id,
        "allowed_pairs": list(logic.allowed_pairs),
        "allowed_timeframes": list(logic.allowed_timeframes),
        "required_features": list(logic.required_features),
        "feature_quality_thresholds": {"min_classifier_confidence": 0.6},
        "eligible_regimes": list(scorecard.get("eligible_regimes", [])),
        "blocked_regimes": list(scorecard.get("blocked_regimes", [])),
        "scorecard_decision": scorecard.get("decision"),
        "scorecard": scorecard,
    }


def _observation_identity_checks(
    observation: dict[str, Any], identity: dict[str, Any]
) -> list[dict[str, Any]]:
    field_map = {
        "candidate_id": "candidate_id",
        "strategy_id": "strategy_id",
        "strategy_version": "strategy_version",
        "signal_version": "signal_version",
        "risk_policy_version": "risk_policy_version",
        "regime_classifier_version": "regime_classifier_version",
        "cost_model_id": "cost_model_id",
    }
    checks = [
        _check(
            f"candidate_identity_{field}_matches_row",
            str(observation.get(row_field) or "") == str(identity.get(field) or ""),
            {
                "row_value": observation.get(row_field),
                "identity_value": identity.get(field),
            },
        )
        for row_field, field in field_map.items()
    ]
    checks.append(
        _check(
            "candidate_identity_pair_allows_row",
            str(observation.get("pair") or "") in set(identity.get("allowed_pairs", [])),
            {
                "pair": observation.get("pair"),
                "identity_allowed_pairs": identity.get("allowed_pairs", []),
            },
        )
    )
    checks.append(
        _check(
            "candidate_identity_timeframe_allows_row",
            str(observation.get("timeframe") or "") in set(identity.get("allowed_timeframes", [])),
            {
                "timeframe": observation.get("timeframe"),
                "identity_allowed_timeframes": identity.get("allowed_timeframes", []),
            },
        )
    )
    return checks


def _scorecard_identity_lineage_validation(
    candidate_observations: Sequence[dict[str, Any]],
    baseline_observations: Sequence[dict[str, Any]],
    *,
    contract: RegimeStrategyContract,
    expected_identity: dict[str, Any] | None,
) -> dict[str, Any]:
    observed_identities = [
        identity
        for identity in (extract_candidate_identity(item) for item in candidate_observations)
        if identity is not None
    ]
    baseline_identities = [
        identity
        for identity in (extract_candidate_identity(item) for item in baseline_observations)
        if identity is not None
    ]
    reference_identity = (
        extract_candidate_identity(expected_identity)
        or (observed_identities[0] if observed_identities else None)
    )
    checks = [
        _check(
            "scorecard_candidate_identity_present",
            reference_identity is not None,
            {"candidate_observation_count": len(candidate_observations)},
        )
    ]
    comparisons: list[dict[str, Any]] = []
    if reference_identity is not None:
        for index, identity in enumerate(observed_identities):
            comparison = compare_candidate_identities(
                reference_identity,
                identity,
                observed_label=f"candidate_observation_{index}",
            )
            comparisons.append(comparison)
        for index, identity in enumerate(baseline_identities):
            comparison = compare_candidate_identities(
                reference_identity,
                identity,
                observed_label=f"baseline_observation_{index}",
            )
            comparisons.append(comparison)
        checks.extend(_contract_identity_checks(contract, reference_identity))
    checks.append(
        _check(
            "scorecard_observation_identities_match",
            all(item["ok"] for item in comparisons),
            {"comparisons": comparisons},
        )
    )
    ok = all(item["passed"] for item in checks)
    return {
        "factory": "regime_scorecard_identity_lineage_validation",
        "schema_version": "strategy_candidate_identity_v1",
        "ok": ok,
        "candidate_identity": reference_identity,
        "checks": checks,
        "comparisons": comparisons,
    }


def _contract_identity_checks(
    contract: RegimeStrategyContract, identity: dict[str, Any]
) -> list[dict[str, Any]]:
    return [
        _check(
            "identity_strategy_version_matches_contract",
            identity.get("strategy_version") == contract.strategy_version,
            {
                "identity_strategy_version": identity.get("strategy_version"),
                "contract_strategy_version": contract.strategy_version,
            },
        ),
        _check(
            "identity_signal_version_matches_contract",
            identity.get("signal_version") == contract.signal_version,
            {
                "identity_signal_version": identity.get("signal_version"),
                "contract_signal_version": contract.signal_version,
            },
        ),
        _check(
            "identity_risk_policy_version_matches_contract",
            identity.get("risk_policy_version") == contract.risk_policy_version,
            {
                "identity_risk_policy_version": identity.get("risk_policy_version"),
                "contract_risk_policy_version": contract.risk_policy_version,
            },
        ),
        _check(
            "identity_regime_classifier_version_matches_contract",
            identity.get("regime_classifier_version") == contract.regime_classifier_version,
            {
                "identity_regime_classifier_version": identity.get("regime_classifier_version"),
                "contract_regime_classifier_version": contract.regime_classifier_version,
            },
        ),
        _check(
            "identity_cost_model_id_matches_contract",
            identity.get("cost_model_id") == contract.cost_model_id,
            {
                "identity_cost_model_id": identity.get("cost_model_id"),
                "contract_cost_model_id": contract.cost_model_id,
            },
        ),
        _check(
            "identity_pairs_cover_contract_pairs",
            set(contract.allowed_pairs).issubset(set(identity.get("allowed_pairs", []))),
            {
                "identity_allowed_pairs": identity.get("allowed_pairs", []),
                "contract_allowed_pairs": list(contract.allowed_pairs),
            },
        ),
        _check(
            "identity_timeframes_cover_contract_timeframes",
            set(contract.allowed_timeframes).issubset(set(identity.get("allowed_timeframes", []))),
            {
                "identity_allowed_timeframes": identity.get("allowed_timeframes", []),
                "contract_allowed_timeframes": list(contract.allowed_timeframes),
            },
        ),
    ]


def _unique_candidate_identities(observations: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for observation in observations:
        identity = extract_candidate_identity(observation)
        if identity is None:
            continue
        key = repr(sorted(identity.items()))
        if key in seen:
            continue
        seen.add(key)
        unique.append(identity)
    return unique


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
        "hold_baseline_delta": baseline_delta_normal,
        "no_trade_baseline_delta": net_normal,
        "incumbent_delta": None,
        "no_trade_opportunity_cost": no_trade_opportunity_cost,
        "confidence_interval": None,
        "lower_confidence_bound": lower_confidence_bound,
        "walk_forward_pass_rate": walk_forward_pass_rate,
        "pair_concentration": pair_concentration,
        "calendar_concentration": calendar_concentration,
        "data_quality_pass": data_quality_pass,
    }


def _runtime_candidate_decision(
    *, runtime: RuntimeRegimeSnapshot, candidate: dict[str, Any]
) -> dict[str, Any]:
    scorecard = candidate.get("scorecard", {})
    identity_validation = validate_candidate_identity(candidate)
    scorecard_rows = [
        row
        for row in scorecard.get("scorecard_by_regime", [])
        if row.get("market_regime") == runtime.current_regime
    ]
    row = scorecard_rows[0] if scorecard_rows else {}
    required_features = set(str(item) for item in candidate.get("required_features", []))
    available_features = set(str(item) for item in runtime.available_features)
    missing_features = sorted(required_features - available_features)
    feature_quality = feature_quality_passes_thresholds(
        runtime.feature_quality_report,
        candidate.get("feature_quality_thresholds"),
    ) if candidate.get("feature_quality_thresholds") else {"ok": True, "reason_codes": [], "checks": []}
    checks = [
        _check(
            "candidate_identity_valid",
            identity_validation["ok"],
            {"identity_checks": identity_validation["checks"]},
        ),
        _check(
            "runtime_process_control_disabled",
            runtime.process_control_allowed is False
            and runtime.paper_or_dry_run_process_running is False,
            {
                "process_control_allowed": runtime.process_control_allowed,
                "paper_or_dry_run_process_running": runtime.paper_or_dry_run_process_running,
            },
        ),
        _check(
            "runtime_data_quality_passed",
            runtime.data_quality_pass,
            {"data_quality_pass": runtime.data_quality_pass},
        ),
        _check(
            "runtime_regime_predeclared",
            runtime.current_regime in MARKET_REGIMES,
            {"current_regime": runtime.current_regime},
        ),
        _check(
            "runtime_regime_eligible",
            runtime.current_regime in set(candidate.get("eligible_regimes", [])),
            {
                "current_regime": runtime.current_regime,
                "eligible_regimes": candidate.get("eligible_regimes", []),
            },
        ),
        _check(
            "runtime_regime_not_blocked",
            runtime.current_regime not in set(candidate.get("blocked_regimes", [])),
            {
                "current_regime": runtime.current_regime,
                "blocked_regimes": candidate.get("blocked_regimes", []),
            },
        ),
        _check(
            "runtime_pair_allowed",
            runtime.pair in set(candidate.get("allowed_pairs", [])),
            {"pair": runtime.pair, "allowed_pairs": candidate.get("allowed_pairs", [])},
        ),
        _check(
            "runtime_timeframe_allowed",
            runtime.timeframe in set(candidate.get("allowed_timeframes", [])),
            {
                "timeframe": runtime.timeframe,
                "allowed_timeframes": candidate.get("allowed_timeframes", []),
            },
        ),
        _check(
            "runtime_pair_allowed_by_identity",
            identity_validation["ok"]
            and runtime.pair in set(identity_validation["candidate_identity"].get("allowed_pairs", [])),
            {
                "pair": runtime.pair,
                "identity_allowed_pairs": (
                    identity_validation.get("candidate_identity") or {}
                ).get("allowed_pairs", []),
            },
        ),
        _check(
            "runtime_timeframe_allowed_by_identity",
            identity_validation["ok"]
            and runtime.timeframe
            in set(identity_validation["candidate_identity"].get("allowed_timeframes", [])),
            {
                "timeframe": runtime.timeframe,
                "identity_allowed_timeframes": (
                    identity_validation.get("candidate_identity") or {}
                ).get("allowed_timeframes", []),
            },
        ),
        _check(
            "runtime_regime_classifier_version_matches",
            runtime.regime_classifier_version == candidate.get("regime_classifier_version"),
            {
                "runtime_regime_classifier_version": runtime.regime_classifier_version,
                "candidate_regime_classifier_version": candidate.get("regime_classifier_version"),
            },
        ),
        _check("runtime_required_features_available", not missing_features, {"missing_features": missing_features}),
        _check(
            "runtime_feature_quality_passed",
            feature_quality["ok"],
            {"reason_codes": feature_quality.get("reason_codes", [])},
        ),
        _check(
            "scorecard_decision_selector_eligible",
            candidate.get("scorecard_decision")
            in {"GLOBAL_SELECTOR_ELIGIBLE", "REGIME_SCOPED_SELECTOR_ELIGIBLE"},
            {"scorecard_decision": candidate.get("scorecard_decision")},
        ),
    ]
    selectable = all(check["passed"] for check in checks)
    selector_score = (
        float(row.get("net_pnl_stress_cost", 0.0) or 0.0) * 1.0
        + float(row.get("lower_confidence_bound", 0.0) or 0.0) * 0.5
        + float(row.get("net_pnl_normal_cost", 0.0) or 0.0) * 0.25
        - float(row.get("max_drawdown", 0.0) or 0.0) * 0.1
        + float(row.get("hold_baseline_delta", row.get("baseline_delta_normal_cost", 0.0)) or 0.0) * 0.25
    )
    return {
        "candidate_id": candidate.get("candidate_id"),
        "strategy_id": candidate.get("strategy_id"),
        "logic_id": candidate.get("logic_id"),
        "selectable": selectable,
        "checks": checks,
        "reason_codes": ["runtime_selection_passed"] if selectable else [
            check["name"] for check in checks if not check["passed"]
        ],
        "selector_score": round(selector_score, 6),
        "scorecard_summary": {
            "decision": candidate.get("scorecard_decision"),
            "current_regime": runtime.current_regime,
            "net_pnl_normal_cost": row.get("net_pnl_normal_cost", 0.0),
            "net_pnl_stress_cost": row.get("net_pnl_stress_cost", 0.0),
            "lower_confidence_bound": row.get("lower_confidence_bound", 0.0),
            "max_drawdown": row.get("max_drawdown", 0.0),
            "hold_baseline_delta": row.get("hold_baseline_delta", row.get("baseline_delta_normal_cost", 0.0)),
        },
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
