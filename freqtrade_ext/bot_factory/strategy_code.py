from __future__ import annotations

import hashlib
import json
import keyword
import re
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

from freqtrade_ext.bot_factory.freqai_backtest import candidate_freqai_identifier
from freqtrade_ext.bot_factory.safety import scan_paths
from freqtrade_ext.bot_factory.strategy_proposals import REQUIRED_PROPOSAL_SECTIONS


STRATEGY_CODE_GENERATOR_VERSION = "strategy_code_generator_v2"
PARAMETER_OPTIMIZATION_POLICY = "theory_fixed_parameters_no_freqtrade_hyperopt"
STRATEGY_CODE_NOTICE = (
    "Strategy code generation writes local strategy, metadata, and static-check "
    "artifacts only. It does not run backtests, start paper or dry-run trading, "
    "call exchange order endpoints, promote candidates, or manage any bot process."
)

DEFAULT_PARAMETER_DEFAULTS: dict[str, int | float] = {
    "buy_rsi_window": 14,
    "buy_pullback_lookback": 5,
    "buy_rsi_pullback": 32,
    "buy_rsi_recovery": 42,
    "buy_ema_fast": 12,
    "buy_ema_slow": 48,
    "buy_volume_window": 24,
    "buy_volume_factor": 1.0,
    "sell_rsi_exit": 65,
    "sell_timeout_candles": 96,
}
LOGIC_VARIANT_PARAMETER_DEFAULTS: dict[str, dict[str, int | float]] = {
    "amihud_illiquidity_premium": {
        "buy_rsi_window": 14,
        "buy_pullback_lookback": 36,
        "buy_rsi_pullback": 40,
        "buy_rsi_recovery": 48,
        "buy_ema_fast": 12,
        "buy_ema_slow": 48,
        "buy_volume_window": 72,
        "buy_volume_factor": 0.90,
        "sell_rsi_exit": 62,
        "sell_timeout_candles": 96,
    },
    "bipower_jump_decay": {
        "buy_rsi_window": 14,
        "buy_pullback_lookback": 24,
        "buy_rsi_pullback": 42,
        "buy_rsi_recovery": 50,
        "buy_ema_fast": 12,
        "buy_ema_slow": 48,
        "buy_volume_window": 72,
        "buy_volume_factor": 0.90,
        "sell_rsi_exit": 66,
        "sell_timeout_candles": 72,
    },
    "crowding_unwind_reaccumulation": {
        "buy_rsi_window": 14,
        "buy_pullback_lookback": 24,
        "buy_rsi_pullback": 38,
        "buy_rsi_recovery": 46,
        "buy_ema_fast": 12,
        "buy_ema_slow": 48,
        "buy_volume_window": 48,
        "buy_volume_factor": 0.90,
        "sell_rsi_exit": 66,
        "sell_timeout_candles": 72,
    },
    "calendar_turnover_seasonality": {
        "buy_rsi_window": 14,
        "buy_pullback_lookback": 24,
        "buy_rsi_pullback": 40,
        "buy_rsi_recovery": 48,
        "buy_ema_fast": 12,
        "buy_ema_slow": 48,
        "buy_volume_window": 48,
        "buy_volume_factor": 1.00,
        "sell_rsi_exit": 62,
        "sell_timeout_candles": 96,
    },
    "cross_asset_cointegration_spread": {
        "buy_rsi_window": 14,
        "buy_pullback_lookback": 36,
        "buy_rsi_pullback": 40,
        "buy_rsi_recovery": 48,
        "buy_ema_fast": 12,
        "buy_ema_slow": 48,
        "buy_volume_window": 72,
        "buy_volume_factor": 0.85,
        "sell_rsi_exit": 62,
        "sell_timeout_candles": 96,
    },
    "cross_asset_correlation_recovery": {
        "buy_rsi_window": 14,
        "buy_pullback_lookback": 36,
        "buy_rsi_pullback": 40,
        "buy_rsi_recovery": 48,
        "buy_ema_fast": 12,
        "buy_ema_slow": 48,
        "buy_volume_window": 72,
        "buy_volume_factor": 0.85,
        "sell_rsi_exit": 62,
        "sell_timeout_candles": 96,
    },
    "cross_asset_lead_lag": {
        "buy_rsi_window": 14,
        "buy_pullback_lookback": 24,
        "buy_rsi_pullback": 40,
        "buy_rsi_recovery": 48,
        "buy_ema_fast": 12,
        "buy_ema_slow": 48,
        "buy_volume_window": 48,
        "buy_volume_factor": 0.95,
        "sell_rsi_exit": 62,
        "sell_timeout_candles": 72,
    },
    "downside_liquidity_shock_reversal": {
        "buy_rsi_window": 14,
        "buy_pullback_lookback": 12,
        "buy_rsi_pullback": 34,
        "buy_rsi_recovery": 41,
        "buy_ema_fast": 12,
        "buy_ema_slow": 48,
        "buy_volume_window": 48,
        "buy_volume_factor": 0.95,
        "sell_rsi_exit": 58,
        "sell_timeout_candles": 72,
    },
    "directional_change_overshoot": {
        "buy_rsi_window": 14,
        "buy_pullback_lookback": 18,
        "buy_rsi_pullback": 40,
        "buy_rsi_recovery": 50,
        "buy_ema_fast": 12,
        "buy_ema_slow": 48,
        "buy_volume_window": 36,
        "buy_volume_factor": 0.95,
        "sell_rsi_exit": 66,
        "sell_timeout_candles": 72,
    },
    "entropy_regime_transition": {
        "buy_rsi_window": 14,
        "buy_pullback_lookback": 18,
        "buy_rsi_pullback": 38,
        "buy_rsi_recovery": 48,
        "buy_ema_fast": 12,
        "buy_ema_slow": 48,
        "buy_volume_window": 48,
        "buy_volume_factor": 0.95,
        "sell_rsi_exit": 63,
        "sell_timeout_candles": 96,
    },
    "fractal_long_memory_regime": {
        "buy_rsi_window": 14,
        "buy_pullback_lookback": 36,
        "buy_rsi_pullback": 40,
        "buy_rsi_recovery": 48,
        "buy_ema_fast": 12,
        "buy_ema_slow": 48,
        "buy_volume_window": 72,
        "buy_volume_factor": 0.90,
        "sell_rsi_exit": 62,
        "sell_timeout_candles": 96,
    },
    "funding_pressure_carry": {
        "buy_rsi_window": 14,
        "buy_pullback_lookback": 12,
        "buy_rsi_pullback": 40,
        "buy_rsi_recovery": 48,
        "buy_ema_fast": 12,
        "buy_ema_slow": 48,
        "buy_volume_window": 48,
        "buy_volume_factor": 0.90,
        "sell_rsi_exit": 62,
        "sell_timeout_candles": 96,
    },
    "intraday_session_liquidity_reclaim": {
        "buy_rsi_window": 14,
        "buy_pullback_lookback": 18,
        "buy_rsi_pullback": 42,
        "buy_rsi_recovery": 50,
        "buy_ema_fast": 12,
        "buy_ema_slow": 48,
        "buy_volume_window": 36,
        "buy_volume_factor": 1.05,
        "sell_rsi_exit": 62,
        "sell_timeout_candles": 48,
    },
    "liquidity_recovery_horizon": {
        "buy_rsi_window": 14,
        "buy_pullback_lookback": 18,
        "buy_rsi_pullback": 38,
        "buy_rsi_recovery": 46,
        "buy_ema_fast": 12,
        "buy_ema_slow": 48,
        "buy_volume_window": 72,
        "buy_volume_factor": 0.95,
        "sell_rsi_exit": 64,
        "sell_timeout_candles": 72,
    },
    "market_beta_drawdown_carry": {
        "buy_rsi_window": 14,
        "buy_pullback_lookback": 24,
        "buy_rsi_pullback": 40,
        "buy_rsi_recovery": 48,
        "buy_ema_fast": 12,
        "buy_ema_slow": 48,
        "buy_volume_window": 72,
        "buy_volume_factor": 0.90,
        "sell_rsi_exit": 78,
        "sell_timeout_candles": 288,
    },
    "mark_price_dislocation_reclaim": {
        "buy_rsi_window": 14,
        "buy_pullback_lookback": 12,
        "buy_rsi_pullback": 38,
        "buy_rsi_recovery": 46,
        "buy_ema_fast": 12,
        "buy_ema_slow": 48,
        "buy_volume_window": 72,
        "buy_volume_factor": 0.50,
        "sell_rsi_exit": 66,
        "sell_timeout_candles": 36,
    },
    "mark_discount_reclaim_continuation": {
        "buy_rsi_window": 14,
        "buy_pullback_lookback": 6,
        "buy_rsi_pullback": 38,
        "buy_rsi_recovery": 46,
        "buy_ema_fast": 12,
        "buy_ema_slow": 48,
        "buy_volume_window": 72,
        "buy_volume_factor": 0.0,
        "sell_rsi_exit": 66,
        "sell_timeout_candles": 36,
    },
    "mark_fair_value_momentum_lag": {
        "buy_rsi_window": 14,
        "buy_pullback_lookback": 12,
        "buy_rsi_pullback": 38,
        "buy_rsi_recovery": 46,
        "buy_ema_fast": 12,
        "buy_ema_slow": 48,
        "buy_volume_window": 48,
        "buy_volume_factor": 0.0,
        "sell_rsi_exit": 72,
        "sell_timeout_candles": 12,
    },
    "mean_reversion_pullback": dict(DEFAULT_PARAMETER_DEFAULTS),
    "microstructure_spread_reversion": {
        "buy_rsi_window": 14,
        "buy_pullback_lookback": 24,
        "buy_rsi_pullback": 40,
        "buy_rsi_recovery": 48,
        "buy_ema_fast": 12,
        "buy_ema_slow": 48,
        "buy_volume_window": 72,
        "buy_volume_factor": 0.50,
        "sell_rsi_exit": 64,
        "sell_timeout_candles": 48,
    },
    "range_quarticity_vol_of_vol_state": {
        "buy_rsi_window": 14,
        "buy_pullback_lookback": 24,
        "buy_rsi_pullback": 40,
        "buy_rsi_recovery": 48,
        "buy_ema_fast": 12,
        "buy_ema_slow": 48,
        "buy_volume_window": 48,
        "buy_volume_factor": 0.90,
        "sell_rsi_exit": 64,
        "sell_timeout_candles": 96,
    },
    "regime_state_reentry": {
        "buy_rsi_window": 14,
        "buy_pullback_lookback": 24,
        "buy_rsi_pullback": 40,
        "buy_rsi_recovery": 48,
        "buy_ema_fast": 12,
        "buy_ema_slow": 48,
        "buy_volume_window": 72,
        "buy_volume_factor": 0.75,
        "sell_rsi_exit": 82,
        "sell_timeout_candles": 48,
    },
    "realized_skewness_tail_shape": {
        "buy_rsi_window": 14,
        "buy_pullback_lookback": 24,
        "buy_rsi_pullback": 40,
        "buy_rsi_recovery": 48,
        "buy_ema_fast": 12,
        "buy_ema_slow": 48,
        "buy_volume_window": 48,
        "buy_volume_factor": 0.90,
        "sell_rsi_exit": 62,
        "sell_timeout_candles": 96,
    },
    "semivariance_asymmetry_regime": {
        "buy_rsi_window": 14,
        "buy_pullback_lookback": 24,
        "buy_rsi_pullback": 40,
        "buy_rsi_recovery": 48,
        "buy_ema_fast": 12,
        "buy_ema_slow": 48,
        "buy_volume_window": 48,
        "buy_volume_factor": 0.85,
        "sell_rsi_exit": 62,
        "sell_timeout_candles": 96,
    },
    "signed_volume_imbalance_accumulation": {
        "buy_rsi_window": 14,
        "buy_pullback_lookback": 24,
        "buy_rsi_pullback": 40,
        "buy_rsi_recovery": 48,
        "buy_ema_fast": 12,
        "buy_ema_slow": 48,
        "buy_volume_window": 24,
        "buy_volume_factor": 1.00,
        "sell_rsi_exit": 64,
        "sell_timeout_candles": 72,
    },
    "trend_continuation": {
        "buy_rsi_window": 14,
        "buy_pullback_lookback": 8,
        "buy_rsi_pullback": 45,
        "buy_rsi_recovery": 52,
        "buy_ema_fast": 16,
        "buy_ema_slow": 64,
        "buy_volume_window": 36,
        "buy_volume_factor": 1.10,
        "sell_rsi_exit": 58,
        "sell_timeout_candles": 144,
    },
    "variance_ratio_regime_switch": {
        "buy_rsi_window": 14,
        "buy_pullback_lookback": 24,
        "buy_rsi_pullback": 42,
        "buy_rsi_recovery": 50,
        "buy_ema_fast": 12,
        "buy_ema_slow": 48,
        "buy_volume_window": 72,
        "buy_volume_factor": 0.90,
        "sell_rsi_exit": 62,
        "sell_timeout_candles": 96,
    },
    "volatility_breakout": {
        "buy_rsi_window": 10,
        "buy_pullback_lookback": 10,
        "buy_rsi_pullback": 40,
        "buy_rsi_recovery": 50,
        "buy_ema_fast": 10,
        "buy_ema_slow": 40,
        "buy_volume_window": 48,
        "buy_volume_factor": 1.25,
        "sell_rsi_exit": 70,
        "sell_timeout_candles": 72,
    },
}
ALLOWED_LOGIC_VARIANTS = set(LOGIC_VARIANT_PARAMETER_DEFAULTS)
PARAMETER_OVERRIDE_ALIASES = {
    "local_falsification_hold_candles": "sell_timeout_candles",
}
PARAMETER_OVERRIDE_RANGES: dict[str, tuple[float, float]] = {
    "buy_rsi_window": (8.0, 30.0),
    "buy_pullback_lookback": (2.0, 24.0),
    "buy_rsi_pullback": (20.0, 55.0),
    "buy_rsi_recovery": (35.0, 65.0),
    "buy_ema_fast": (8.0, 30.0),
    "buy_ema_slow": (32.0, 120.0),
    "buy_volume_window": (12.0, 72.0),
    "buy_volume_factor": (0.0, 2.0),
    "sell_rsi_exit": (55.0, 80.0),
    "sell_timeout_candles": (2.0, 288.0),
}
STRUCTURAL_DATA_CODE_SUPPORTED_VARIANTS: set[str] = {
    "crowding_unwind_reaccumulation",
}

_STRUCTURAL_DATA_RE = re.compile(
    r"(?i)\b(open[-_ ]?interest|long[-_ /]?short[-_ ]?(?:account[-_ ]?)?ratio|"
    r"account[-_ ]?ratio|liquidations?|order[-_ ]?book|orderbook|"
    r"market[-_ ]?depth|book[-_ ]?imbalance|depth[-_ ]?imbalance)\b"
)

REQUIRED_PROPOSAL_METADATA_FIELDS = [
    "status",
    "proposal_status",
    "code_generation_eligible",
    "strategy_name",
    "target_exchange",
    "target_symbols",
    "timeframe",
    "spot_or_futures",
    "long_short",
    "proposal_path",
    "proposal_content_hash",
    "safety_scope",
]

_GENERATED_FORBIDDEN_PATTERNS: list[tuple[str, re.Pattern[str], str]] = [
    (
        "generated_code_no_short_entry_signal",
        re.compile(r"\benter_short\b"),
        "Generated strategy must not define short entry signals.",
    ),
    (
        "generated_code_no_short_exit_signal",
        re.compile(r"\bexit_short\b"),
        "Generated strategy must not define short exit signals.",
    ),
    (
        "generated_code_no_can_short_true",
        re.compile(r"\bcan_short\s*=\s*True\b"),
        "Generated strategy must not enable can_short.",
    ),
    (
        "generated_code_no_leverage_hook",
        re.compile(r"\bdef\s+leverage\s*\("),
        "Generated strategy must not define a leverage hook.",
    ),
    (
        "generated_code_no_shift_minus_one",
        re.compile(r"\.shift\s*\(\s*(?:periods\s*=\s*)?-\d+"),
        "Generated strategy must not use negative shifts.",
    ),
    (
        "generated_code_no_unsafe_iloc_minus_one",
        re.compile(r"\.iloc\s*\[\s*-1"),
        "Generated strategy must not use iloc[-1].",
    ),
    (
        "generated_code_no_direct_order_calls",
        re.compile(
            r"\b(create_order|private_post_order|fapiPrivatePostOrder|"
            r"request_order|requests\.post|httpx\.post)\b"
        ),
        "Generated strategy must not call order endpoints or direct HTTP POST.",
    ),
    (
        "generated_code_no_process_control",
        re.compile(r"\b(subprocess|Popen|os\.system|Start-Process|freqtrade\s+trade)\b"),
        "Generated strategy must not include process-control code.",
    ),
    (
        "generated_code_no_hardcoded_secret_assignment",
        re.compile(
            r"""(?ix)
            (api[_-]?key|secret|password|token|jwt_secret_key|ws_token)
            \s*[:=]\s*
            ["'][^"']{8,}["']
            """
        ),
        "Generated strategy must not include hardcoded credential-like values.",
    ),
    (
        "generated_code_no_single_timerange_fit",
        re.compile(r"\b20\d{6}\s*-\s*20\d{6}\b"),
        "Generated strategy must not include a hardcoded backtest timerange.",
    ),
]
ALLOWED_GENERATOR_MODES = {"rule_based", "freqai", "hybrid_ml"}


@dataclass(frozen=True)
class StrategyCodeInputs:
    root_dir: Path
    proposal_metadata_path: Path
    candidate_id: str | None = None
    output_root: Path = Path("registry/strategies/generated")
    created_by_agent: str = "codex"
    created_at: str | None = None
    command: Sequence[str] = field(default_factory=list)


@dataclass(frozen=True)
class StrategyCodeCheck:
    name: str
    status: str
    severity: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StrategyCodeArtifacts:
    strategy_code: str | None
    metadata: dict[str, Any]
    strategy_path: Path
    metadata_path: Path
    static_check_path: Path
    research_brief_path: Path


def build_strategy_code(inputs: StrategyCodeInputs) -> StrategyCodeArtifacts:
    created_at = inputs.created_at or datetime.now(UTC).replace(microsecond=0).isoformat()
    root_dir = inputs.root_dir.resolve()
    proposal_metadata_path = _resolve_workspace_path(inputs.proposal_metadata_path, root_dir)
    candidate_id = _candidate_id(inputs, created_at)
    checks: list[StrategyCodeCheck] = []

    metadata_path_within_workspace = _path_is_within_root(proposal_metadata_path, root_dir)
    checks.append(
        _check(
            "proposal_metadata_path_within_workspace",
            metadata_path_within_workspace,
            "blocker",
            "Proposal metadata path must resolve inside the repository workspace.",
            {"path": _safe_relative_path(proposal_metadata_path, root_dir)},
        )
    )
    metadata_file_present = metadata_path_within_workspace and proposal_metadata_path.is_file()
    checks.append(
        _check(
            "proposal_metadata_file_present",
            metadata_file_present,
            "blocker",
            "Proposal metadata path must exist as a local JSON file.",
            {"path": _safe_relative_path(proposal_metadata_path, root_dir)},
        )
    )

    proposal_metadata, metadata_load_check = _load_json_object_check(
        proposal_metadata_path, metadata_file_present
    )
    checks.append(metadata_load_check)

    strategy_name = str(proposal_metadata.get("strategy_name") or "unknown_strategy")
    class_name = strategy_name
    output_dir = _generated_output_dir(
        root_dir=root_dir,
        output_root=inputs.output_root,
        strategy_name=strategy_name,
        candidate_id=candidate_id,
    )
    strategy_path = output_dir / f"{_safe_filename(strategy_name)}.py"
    metadata_path = output_dir / "metadata.json"
    static_check_path = output_dir / "static_check.json"

    checks.extend(_metadata_schema_checks(proposal_metadata))
    checks.extend(_candidate_scope_checks(candidate_id, class_name, output_dir, root_dir))

    proposal_path = _proposal_path_from_metadata(proposal_metadata, root_dir)
    proposal_markdown = ""
    if proposal_path is not None:
        checks.extend(_proposal_file_checks(proposal_path, root_dir))
        if _path_is_within_root(proposal_path, root_dir) and proposal_path.is_file():
            proposal_markdown = proposal_path.read_text(encoding="utf-8")
            checks.extend(_proposal_hash_checks(proposal_markdown, proposal_metadata))
            checks.extend(_required_section_checks(proposal_markdown))
    else:
        checks.append(
            _check(
                "proposal_path_present",
                False,
                "blocker",
                "Proposal metadata must include a proposal_path.",
            )
        )

    checks.extend(_proposal_status_checks(proposal_metadata))
    checks.extend(_proposal_safety_scope_checks(proposal_metadata))
    checks.extend(_hypothesis_iteration_checks(proposal_metadata))

    strategy_code: str | None = None
    if not _has_blockers(checks):
        generator_mode = _generator_mode_from_proposal(proposal_metadata)
        strategy_code = _render_long_only_strategy_code(
            strategy_name=class_name,
            timeframe=str(proposal_metadata["timeframe"]),
            candidate_id=candidate_id,
            source_proposal_hash=str(proposal_metadata["proposal_content_hash"]),
            generator_mode=generator_mode,
            proposal_metadata=proposal_metadata,
        )
        checks.extend(_generated_code_safety_checks(strategy_code))
        if _has_blockers(checks):
            strategy_code = None

    metadata = _build_metadata(
        inputs=inputs,
        created_at=created_at,
        candidate_id=candidate_id,
        strategy_name=strategy_name,
        proposal_metadata=proposal_metadata,
        proposal_metadata_path=proposal_metadata_path,
        proposal_path=proposal_path,
        strategy_path=strategy_path,
        metadata_path=metadata_path,
        static_check_path=static_check_path,
        strategy_code=strategy_code,
        checks=checks,
        root_dir=root_dir,
    )
    return StrategyCodeArtifacts(
        strategy_code=strategy_code,
        metadata=metadata,
        strategy_path=strategy_path,
        metadata_path=metadata_path,
        static_check_path=static_check_path,
        research_brief_path=output_dir / "research_brief.json",
    )


def write_strategy_code_artifacts(artifacts: StrategyCodeArtifacts) -> None:
    artifacts.metadata_path.parent.mkdir(parents=True, exist_ok=True)
    artifacts.research_brief_path.write_text(
        json.dumps(artifacts.metadata.get("research_brief", {}), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if artifacts.strategy_code is not None:
        artifacts.strategy_path.write_text(artifacts.strategy_code, encoding="utf-8")
        static_report = scan_paths([artifacts.strategy_path])
        artifacts.static_check_path.write_text(static_report.to_json(), encoding="utf-8")
        _finalize_static_check_metadata(artifacts.metadata, static_report)
    else:
        artifacts.metadata["static_check"] = {
            "ran": False,
            "ok": False,
            "files_checked": 0,
            "findings": [],
        }

    artifacts.metadata_path.write_text(
        json.dumps(artifacts.metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def load_strategy_code_metadata(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Strategy code metadata must contain an object: {path}")
    return payload


def _build_metadata(
    *,
    inputs: StrategyCodeInputs,
    created_at: str,
    candidate_id: str,
    strategy_name: str,
    proposal_metadata: dict[str, Any],
    proposal_metadata_path: Path,
    proposal_path: Path | None,
    strategy_path: Path,
    metadata_path: Path,
    static_check_path: Path,
    strategy_code: str | None,
    checks: Sequence[StrategyCodeCheck],
    root_dir: Path,
) -> dict[str, Any]:
    blockers = [check for check in checks if check.status == "blocked"]
    source_proposal_hash = proposal_metadata.get("proposal_content_hash")
    generator_mode = _generator_mode_from_proposal(proposal_metadata)
    target_definition = proposal_metadata.get("target_definition")
    freqai_metadata = _freqai_execution_metadata(
        strategy_name=strategy_name,
        candidate_id=candidate_id,
        generator_mode=generator_mode,
        target_definition=target_definition,
    )
    structural_data_requirement = _proposal_structural_data_requirement(
        proposal_metadata
    )
    structural_data_quality_handoff = _proposal_structural_data_quality_handoff(
        proposal_metadata
    )
    structural_data_capability_handoff = (
        _proposal_structural_data_capability_handoff(proposal_metadata)
    )
    edge_discovery_handoff = _proposal_edge_discovery_handoff(proposal_metadata)
    local_falsification_handoff = _proposal_local_falsification_handoff(
        proposal_metadata
    )
    research_decision_novelty_handoff = (
        _proposal_research_decision_novelty_handoff(proposal_metadata)
    )
    research_decision_question_handoff = (
        _proposal_research_decision_question_handoff(proposal_metadata)
    )
    blocked_next_actions = _proposal_blocked_next_actions(proposal_metadata)
    research_handoff_summaries = _proposal_research_handoff_summaries(
        proposal_metadata
    )
    return {
        "generated_at": created_at,
        "phase": "strategy_generation",
        "factory": "strategy_code_generator",
        "generator_version": STRATEGY_CODE_GENERATOR_VERSION,
        "status": "blocked" if blockers else "pending_static_check",
        "strategy_code_generated": strategy_code is not None,
        "candidate_evaluation_eligible": False,
        "parameter_optimization_enabled": False,
        "parameter_optimization_policy": PARAMETER_OPTIMIZATION_POLICY,
        "strategy_name": strategy_name,
        "strategy_class_name": strategy_name,
        "candidate_id": candidate_id,
        "created_at": created_at,
        "created_by_agent": inputs.created_by_agent,
        "target_exchange": proposal_metadata.get("target_exchange"),
        "target_symbols": proposal_metadata.get("target_symbols", []),
        "timeframe": proposal_metadata.get("timeframe"),
        "spot_or_futures": proposal_metadata.get("spot_or_futures"),
        "long_short": proposal_metadata.get("long_short"),
        "source_proposal_metadata_path": _safe_relative_path(
            proposal_metadata_path, root_dir
        ),
        "source_proposal_metadata_hash": (
            _sha256_file(proposal_metadata_path)
            if proposal_metadata_path.is_file()
            and _path_is_within_root(proposal_metadata_path, root_dir)
            else None
        ),
        "source_proposal_path": (
            _safe_relative_path(proposal_path, root_dir) if proposal_path else None
        ),
        "source_proposal_content_hash": source_proposal_hash,
        "generated_strategy_path": _safe_relative_path(strategy_path, root_dir),
        "generated_strategy_content_hash": (
            _sha256_text(strategy_code) if strategy_code is not None else None
        ),
        "metadata_path": _safe_relative_path(metadata_path, root_dir),
        "static_check_report_path": _safe_relative_path(static_check_path, root_dir),
        "research_brief_path": _safe_relative_path(metadata_path.parent / "research_brief.json", root_dir),
        "generator_mode": generator_mode,
        "strategy_logic_variant": _metadata_logic_variant_from_proposal(proposal_metadata),
        "feature_list": list(proposal_metadata.get("feature_list", [])),
        "target_definition": target_definition,
        "label_horizon": proposal_metadata.get("label_horizon"),
        "prediction_threshold": proposal_metadata.get("prediction_threshold"),
        **freqai_metadata,
        "rule_filters": list(proposal_metadata.get("rule_filters", [])),
        "risk_policy": proposal_metadata.get("risk_policy"),
        "thesis_id": proposal_metadata.get("thesis_id"),
        "thesis_type": proposal_metadata.get("thesis_type"),
        "thesis_statement": proposal_metadata.get("thesis_statement"),
        "falsification_criteria": proposal_metadata.get("falsification_criteria"),
        "novelty_vs_previous": proposal_metadata.get("novelty_vs_previous"),
        "evidence_refs": list(proposal_metadata.get("evidence_refs", [])),
        "research_references": list(proposal_metadata.get("research_references", [])),
        "failure_taxonomy_codes": list(proposal_metadata.get("failure_taxonomy_codes", [])),
        "retry_budget_per_thesis": proposal_metadata.get("retry_budget_per_thesis"),
        "thesis_retry_count": proposal_metadata.get("thesis_retry_count"),
        "parameter_only_retry_count": proposal_metadata.get("parameter_only_retry_count"),
        "parameter_only_retry_limit": proposal_metadata.get("parameter_only_retry_limit"),
        "force_distinct_hypothesis_family": bool(proposal_metadata.get("force_distinct_hypothesis_family", False)),
        "structural_data_requirement": structural_data_requirement,
        "structural_data_quality_handoff": structural_data_quality_handoff,
        "structural_data_capability_handoff": structural_data_capability_handoff,
        "edge_discovery_handoff": edge_discovery_handoff,
        "local_falsification_handoff": local_falsification_handoff,
        "research_decision_novelty_handoff": research_decision_novelty_handoff,
        "research_decision_question_handoff": research_decision_question_handoff,
        "blocked_next_actions": blocked_next_actions,
        "research_handoff_summaries": research_handoff_summaries,
        "research_brief": {
            "thesis_id": proposal_metadata.get("thesis_id"),
            "thesis_statement": proposal_metadata.get("thesis_statement"),
            "candidate_id": candidate_id,
            "strategy_name": strategy_name,
            "evidence_refs": list(proposal_metadata.get("evidence_refs", [])),
            "research_references": list(proposal_metadata.get("research_references", [])),
            "failure_taxonomy_codes": list(proposal_metadata.get("failure_taxonomy_codes", [])),
            "strategy_logic_variant": _metadata_logic_variant_from_proposal(proposal_metadata),
            "novelty_vs_previous": proposal_metadata.get("novelty_vs_previous"),
            "structural_data_requirement": structural_data_requirement,
            "structural_data_quality_handoff": structural_data_quality_handoff,
            "structural_data_capability_handoff": structural_data_capability_handoff,
            "edge_discovery_handoff": edge_discovery_handoff,
            "local_falsification_handoff": local_falsification_handoff,
            "research_decision_novelty_handoff": (
                research_decision_novelty_handoff
            ),
            "research_decision_question_handoff": (
                research_decision_question_handoff
            ),
            "blocked_next_actions": blocked_next_actions,
            "research_handoff_summaries": research_handoff_summaries,
            "generated_at": created_at,
        },
        "parameter_overrides": _proposal_parameter_overrides(proposal_metadata),
        "parameter_defaults": _parameter_defaults_for_proposal(proposal_metadata),
        "checks": [check.to_dict() for check in checks],
        "blockers": [check.to_dict() for check in blockers],
        "rejection_reasons": [check.message for check in blockers],
        "static_check": {
            "ran": False,
            "ok": False,
            "files_checked": 0,
            "findings": [],
        },
        "safety_scope": {
            "command": "strategy code generation only",
            "long_only": True,
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
            "backtest_started": False,
            "freqtrade_hyperopt_parameter_optimization": False,
            "candidate_evaluation_started": False,
            "candidate_ranking_started": False,
            "paper_promotion_eligible": False,
            "promotion_authorized_by_this_command": False,
            "local_artifacts_source_of_truth": True,
        },
        "command": [_sanitize_command_token(token) for token in inputs.command],
        "notice": STRATEGY_CODE_NOTICE,
    }


def _freqai_execution_metadata(
    *,
    strategy_name: str,
    candidate_id: str,
    generator_mode: str,
    target_definition: Any,
) -> dict[str, Any]:
    if generator_mode not in {"freqai", "hybrid_ml"}:
        return {
            "freqai_identifier": None,
            "freqai_identifier_policy": "not_applicable",
            "freqai_expected_target_column": None,
            "freqai_cache_policy": {
                "candidate_specific_identifier_required": False,
                "reuse_existing_predictions_allowed": None,
            },
        }
    target_name = str(target_definition or "future_return").strip() or "future_return"
    return {
        "freqai_identifier": candidate_freqai_identifier(
            strategy_name, candidate_id, target_name
        ),
        "freqai_identifier_policy": "candidate_specific",
        "freqai_expected_target_column": f"&-{target_name}",
        "freqai_cache_policy": {
            "candidate_specific_identifier_required": True,
            "reuse_existing_predictions_allowed": False,
            "reason": (
                "Generated ML candidates must not reuse FreqAI model or prediction "
                "caches from a different candidate or target label."
            ),
        },
    }


def _finalize_static_check_metadata(metadata: dict[str, Any], static_report: Any) -> None:
    static_check = {
        "ran": True,
        "ok": static_report.ok,
        "files_checked": static_report.files_checked,
        "findings": [asdict(finding) for finding in static_report.findings],
    }
    static_check_result = _check(
        "generated_strategy_static_scan_ok",
        static_report.ok,
        "blocker",
        "Generated strategy static safety scan must pass before evaluation.",
        {
            "files_checked": static_report.files_checked,
            "finding_count": len(static_report.findings),
        },
    ).to_dict()
    metadata["static_check"] = static_check
    metadata["checks"].append(static_check_result)
    if static_report.ok:
        metadata["status"] = "generated"
        metadata["candidate_evaluation_eligible"] = True
        metadata["blockers"] = [
            check for check in metadata["checks"] if check["status"] == "blocked"
        ]
        metadata["rejection_reasons"] = [
            check["message"] for check in metadata["blockers"]
        ]
    else:
        metadata["status"] = "blocked"
        metadata["candidate_evaluation_eligible"] = False
        metadata["blockers"] = [
            check for check in metadata["checks"] if check["status"] == "blocked"
        ]
        metadata["rejection_reasons"] = [
            check["message"] for check in metadata["blockers"]
        ]



def _hypothesis_iteration_checks(proposal_metadata: dict[str, Any]) -> list[StrategyCodeCheck]:
    checks: list[StrategyCodeCheck] = []
    required_fields = [
        "thesis_id",
        "thesis_type",
        "thesis_statement",
        "falsification_criteria",
        "novelty_vs_previous",
        "evidence_refs",
    ]
    for field_name in required_fields:
        value = proposal_metadata.get(field_name)
        present = bool(value)
        if field_name == "evidence_refs":
            present = isinstance(value, list) and len(value) > 0
        checks.append(_check(f"hypothesis_{field_name}_present", present, "blocker", f"Proposal metadata must include {field_name} for hypothesis-driven iteration."))

    retry_budget = int(proposal_metadata.get("retry_budget_per_thesis") or 0)
    thesis_retry_count = int(proposal_metadata.get("thesis_retry_count") or 0)
    parameter_only_retry_count = int(proposal_metadata.get("parameter_only_retry_count") or 0)
    parameter_retry_limit = int(proposal_metadata.get("parameter_only_retry_limit") or 0)
    forced_family = bool(proposal_metadata.get("force_distinct_hypothesis_family"))

    checks.append(_check("thesis_retry_budget_configured", retry_budget > 0, "blocker", "Proposal metadata must configure retry_budget_per_thesis > 0."))
    checks.append(_check("thesis_retry_budget_not_exceeded", thesis_retry_count <= retry_budget if retry_budget > 0 else False, "blocker", "Thesis retry budget exceeded; generate a distinct hypothesis family candidate."))
    checks.append(_check("parameter_only_retry_limit_configured", parameter_retry_limit > 0, "blocker", "Proposal metadata must configure parameter_only_retry_limit > 0."))
    checks.append(_check("parameter_only_retry_guard", parameter_only_retry_count <= parameter_retry_limit if parameter_retry_limit > 0 else False, "blocker", "Parameter-only retries exceeded threshold; same-thesis tuning loop is blocked."))
    checks.append(_check("distinct_hypothesis_family_after_repeated_failure", forced_family or thesis_retry_count <= 1, "blocker", "Repeated failures require force_distinct_hypothesis_family=true."))

    failure_codes = proposal_metadata.get("failure_taxonomy_codes", [])
    allowed = {"FAIL_OVERFIT_WF_GAP", "FAIL_COST_SENSITIVE", "FAIL_REGIME_FRAGILE"}
    checks.append(_check("failure_taxonomy_codes_list", isinstance(failure_codes, list), "blocker", "failure_taxonomy_codes must be a list."))
    if isinstance(failure_codes, list):
        checks.append(_check("failure_taxonomy_codes_normalized", all(isinstance(code, str) and code in allowed for code in failure_codes), "blocker", "failure_taxonomy_codes must use normalized values.", {"allowed": sorted(allowed)}))
    research_refs = proposal_metadata.get("research_references", [])
    checks.append(
        _check(
            "research_references_present",
            isinstance(research_refs, list) and len(research_refs) > 0,
            "blocker",
            "Proposal metadata must include structured theory or literature references.",
        )
    )
    if isinstance(research_refs, list):
        thesis_id = str(proposal_metadata.get("thesis_id") or "").strip()
        checks.append(
            _check(
                "research_references_structured",
                all(
                    isinstance(ref, dict)
                    and ref.get("reference_id")
                    and ref.get("title")
                    and ref.get("source")
                    for ref in research_refs
                ),
                "blocker",
                "Research references must include reference_id, title, and source.",
            )
        )
        checks.append(
            _check(
                "research_references_have_relevance",
                all(
                    isinstance(ref, dict)
                    and bool(str(ref.get("relevance") or "").strip())
                    for ref in research_refs
                ),
                "blocker",
                "Research references must explain why each reference is relevant.",
            )
        )
        checks.append(
            _check(
                "research_references_record_publication_date",
                all(
                    isinstance(ref, dict)
                    and bool(str(ref.get("published_at") or "").strip())
                    for ref in research_refs
                ),
                "blocker",
                "Research references must record a publication date or version date.",
            )
        )
        checks.append(
            _check(
                "research_references_motivate_current_thesis",
                all(
                    isinstance(ref, dict)
                    and thesis_id
                    and thesis_id
                    in _motivated_thesis_ids(ref)
                    for ref in research_refs
                ),
                "blocker",
                "Research references must list the current thesis_id as motivated.",
                {"thesis_id": thesis_id},
            )
        )
    structural_data_requirement = _proposal_structural_data_requirement(
        proposal_metadata
    )
    structural_quality_handoff = _proposal_structural_data_quality_handoff(
        proposal_metadata
    )
    structural_capability_handoff = _proposal_structural_data_capability_handoff(
        proposal_metadata
    )
    edge_discovery_handoff = _proposal_edge_discovery_handoff(proposal_metadata)
    local_falsification_handoff = _proposal_local_falsification_handoff(
        proposal_metadata
    )
    research_decision_novelty_handoff = (
        _proposal_research_decision_novelty_handoff(proposal_metadata)
    )
    research_decision_question_handoff = (
        _proposal_research_decision_question_handoff(proposal_metadata)
    )
    checks.append(
        _check(
            "edge_discovery_handoff_passed",
            edge_discovery_handoff["passed"],
            "blocker",
            "Accepted proposal metadata must carry a passing Edge Discovery handoff before code generation.",
            {"handoff": edge_discovery_handoff},
        )
    )
    checks.append(
        _check(
            "local_falsification_handoff_passed",
            not local_falsification_handoff["required"]
            or local_falsification_handoff["passed"],
            "blocker",
            "High-risk cost-sensitive proposals must carry a passing proposal-stage local falsification handoff before code generation.",
            {"handoff": local_falsification_handoff},
        )
    )
    checks.append(
        _check(
            "research_decision_novelty_handoff_passed",
            not research_decision_novelty_handoff["required"]
            or research_decision_novelty_handoff["passed"],
            "blocker",
            "Accepted proposal metadata must not carry failed novelty or validated local-rejection matches from research decision constraints before code generation.",
            {"handoff": research_decision_novelty_handoff},
        )
    )
    checks.append(
        _check(
            "research_decision_question_handoff_passed",
            not research_decision_question_handoff["required"]
            or research_decision_question_handoff["passed"],
            "blocker",
            "Accepted proposal metadata must carry complete required research-question responses from research decision constraints before code generation.",
            {"handoff": research_decision_question_handoff},
        )
    )
    checks.append(
        _check(
            "structural_data_quality_handoff_passed",
            not structural_data_requirement["required"]
            or structural_quality_handoff["passed"],
            "blocker",
            "Structural-data proposals must carry a passing proposal-stage quality handoff before code generation.",
            {
                "structural_terms": structural_data_requirement["terms"],
                "handoff": structural_quality_handoff,
            },
        )
    )
    checks.append(
        _check(
            "structural_data_capability_handoff_passed",
            not structural_data_requirement["required"]
            or structural_capability_handoff["passed"],
            "blocker",
            "Structural-data proposals must carry a passing proposal-stage capability handoff before code generation.",
            {
                "structural_terms": structural_data_requirement["terms"],
                "handoff": structural_capability_handoff,
            },
        )
    )
    logic_variant = _logic_variant_from_proposal(proposal_metadata)
    checks.append(
        _check(
            "structural_data_code_generation_supported",
            not structural_data_requirement["required"]
            or logic_variant in STRUCTURAL_DATA_CODE_SUPPORTED_VARIANTS,
            "blocker",
            "Structural-data proposals require an explicitly supported code-generation variant before strategy code can be emitted.",
            {
                "logic_variant": logic_variant,
                "supported_structural_variants": sorted(
                    STRUCTURAL_DATA_CODE_SUPPORTED_VARIANTS
                ),
                "structural_terms": structural_data_requirement["terms"],
            },
        )
    )
    raw_logic_variant = proposal_metadata.get("strategy_logic_variant")
    checks.append(
        _check(
            "strategy_logic_variant_supported",
            raw_logic_variant is None or str(raw_logic_variant).strip().lower() in ALLOWED_LOGIC_VARIANTS,
            "blocker",
            "strategy_logic_variant must identify a supported hypothesis family.",
            {"allowed": sorted(ALLOWED_LOGIC_VARIANTS)},
        )
    )
    return checks


def _proposal_structural_data_requirement(
    proposal_metadata: dict[str, Any]
) -> dict[str, Any]:
    existing = proposal_metadata.get("structural_data_requirement")
    if isinstance(existing, dict):
        terms = _string_list(existing.get("terms", []))
        required = bool(existing.get("required")) or bool(terms)
        return {
            "required": required,
            "terms": terms,
        }
    text = " ".join(
        [
            str(proposal_metadata.get("strategy_name") or ""),
            str(proposal_metadata.get("strategy_type") or ""),
            str(proposal_metadata.get("strategy_logic_variant") or ""),
            str(proposal_metadata.get("thesis_type") or ""),
            str(proposal_metadata.get("thesis_statement") or ""),
            str(proposal_metadata.get("falsification_criteria") or ""),
            str(proposal_metadata.get("novelty_vs_previous") or ""),
            " ".join(_string_list(proposal_metadata.get("feature_list", []))),
            " ".join(_string_list(proposal_metadata.get("rule_filters", []))),
            " ".join(_string_list(proposal_metadata.get("evidence_refs", []))),
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
    }


def _proposal_structural_data_quality_handoff(
    proposal_metadata: dict[str, Any]
) -> dict[str, Any]:
    constraints = proposal_metadata.get("research_decision_constraints", [])
    if not isinstance(constraints, list):
        constraints = []
    candidates: list[dict[str, Any]] = []
    for item in constraints:
        if not isinstance(item, dict):
            continue
        paths = _string_list(item.get("structural_data_quality_report_paths", []))
        candidate = {
            "path": item.get("path"),
            "quality_report_paths": paths,
            "quality_report_paths_present": bool(paths),
            "quality_report_paths_exist": item.get(
                "structural_data_quality_reports_exist"
            )
            is True,
            "quality_reports_valid_check_passed": item.get(
                "structural_data_quality_reports_valid_check_passed"
            )
            is True,
            "structural_quality_check_passed": item.get(
                "structural_data_quality_check_passed"
            )
            is True,
            "proposal_gate_passed": item.get(
                "structural_data_quality_report_gate_passed"
            )
            is True,
        }
        candidate["passed"] = (
            candidate["quality_report_paths_present"]
            and candidate["quality_report_paths_exist"]
            and candidate["quality_reports_valid_check_passed"]
            and candidate["structural_quality_check_passed"]
            and candidate["proposal_gate_passed"]
        )
        candidates.append(candidate)
    return {
        "passed": any(item["passed"] for item in candidates),
        "candidate_count": len(candidates),
        "candidates": candidates,
    }


def _proposal_structural_data_capability_handoff(
    proposal_metadata: dict[str, Any]
) -> dict[str, Any]:
    constraints = proposal_metadata.get("research_decision_constraints", [])
    if not isinstance(constraints, list):
        constraints = []
    candidates: list[dict[str, Any]] = []
    for item in constraints:
        if not isinstance(item, dict):
            continue
        paths = _string_list(
            item.get("structural_data_capability_report_paths", [])
        )
        unsupported_required_classes = _string_list(
            item.get("structural_data_capability_unsupported_required_classes", [])
        )
        candidate = {
            "path": item.get("path"),
            "capability_report_paths": paths,
            "capability_report_paths_present": bool(paths),
            "capability_report_paths_exist": item.get(
                "structural_data_capability_reports_exist"
            )
            is True,
            "capability_reports_valid_check_passed": item.get(
                "structural_data_capability_reports_valid_check_passed"
            )
            is True,
            "structural_capability_check_passed": item.get(
                "structural_data_capability_check_passed"
            )
            is True,
            "structural_capability_support_check_passed": item.get(
                "structural_data_capability_support_check_passed"
            )
            is True,
            "proposal_gate_passed": item.get(
                "structural_data_capability_report_gate_passed"
            )
            is True,
            "required_classes_supported": item.get(
                "structural_data_capability_required_classes_supported"
            )
            is True,
            "usable_classes": _string_list(
                item.get("structural_data_capability_usable_classes", [])
            ),
            "unsupported_required_classes": unsupported_required_classes,
        }
        candidate["passed"] = (
            candidate["capability_report_paths_present"]
            and candidate["capability_report_paths_exist"]
            and candidate["capability_reports_valid_check_passed"]
            and candidate["structural_capability_check_passed"]
            and candidate["structural_capability_support_check_passed"]
            and candidate["proposal_gate_passed"]
            and candidate["required_classes_supported"]
            and not unsupported_required_classes
        )
        candidates.append(candidate)
    return {
        "passed": any(item["passed"] for item in candidates),
        "candidate_count": len(candidates),
        "candidates": candidates,
    }


def _proposal_edge_discovery_handoff(
    proposal_metadata: dict[str, Any]
) -> dict[str, Any]:
    handoff = proposal_metadata.get("edge_discovery_handoff")
    if not isinstance(handoff, dict):
        return {
            "required": True,
            "passed": False,
            "artifact_count": 0,
            "passing_edge_artifact_count": 0,
            "candidate_generation_allowed": False,
            "proposal_generation_allowed": False,
            "artifact_paths": [],
            "blocker_names": ["edge_discovery_handoff_missing"],
        }
    artifact_count = _non_negative_int(handoff.get("artifact_count"))
    passing_count = _non_negative_int(handoff.get("passing_edge_artifact_count"))
    passed = (
        handoff.get("passed") is True
        and artifact_count > 0
        and passing_count > 0
        and handoff.get("paths_valid") is True
        and handoff.get("factory_valid") is True
        and handoff.get("safety_scope_valid") is True
        and handoff.get("anti_parameter_search_valid") is True
        and handoff.get("direct_strategy_codegen_blocked") is True
        and handoff.get("candidate_generation_allowed") is True
        and handoff.get("proposal_generation_allowed") is True
    )
    return {
        "required": handoff.get("required") is not False,
        "passed": passed,
        "artifact_count": artifact_count,
        "parseable_artifact_count": _non_negative_int(
            handoff.get("parseable_artifact_count")
        ),
        "matching_thesis_artifact_count": _non_negative_int(
            handoff.get("matching_thesis_artifact_count")
        ),
        "passing_edge_artifact_count": passing_count,
        "paths_valid": handoff.get("paths_valid") is True,
        "factory_valid": handoff.get("factory_valid") is True,
        "safety_scope_valid": handoff.get("safety_scope_valid") is True,
        "anti_parameter_search_valid": (
            handoff.get("anti_parameter_search_valid") is True
        ),
        "direct_strategy_codegen_blocked": (
            handoff.get("direct_strategy_codegen_blocked") is True
        ),
        "candidate_generation_allowed": (
            handoff.get("candidate_generation_allowed") is True
        ),
        "proposal_generation_allowed": (
            handoff.get("proposal_generation_allowed") is True
        ),
        "artifact_paths": _string_list(handoff.get("artifact_paths", [])),
        "blocked_next_actions": _string_list(
            handoff.get("blocked_next_actions", [])
        ),
        "blocker_names": _string_list(handoff.get("blocker_names", [])),
    }


def _proposal_local_falsification_handoff(
    proposal_metadata: dict[str, Any]
) -> dict[str, Any]:
    constraints = proposal_metadata.get("research_decision_constraints", [])
    if not isinstance(constraints, list):
        constraints = []
    candidates: list[dict[str, Any]] = []
    for item in constraints:
        if not isinstance(item, dict):
            continue
        required = item.get("local_falsification_handoff_required")
        if required is not True and required is not False:
            required = (
                item.get("causal_risk_weights_present") is True
                and "cost_sensitive_mechanism"
                in _string_list(item.get("causal_required_categories_to_address", []))
            )
        candidate = {
            "path": item.get("path"),
            "required": required is True,
            "handoff_passed": item.get("local_falsification_handoff_passed") is True,
            "artifact_count": _non_negative_int(
                item.get("local_falsification_artifact_count")
            ),
            "parseable_artifact_count": _non_negative_int(
                item.get("local_falsification_parseable_artifact_count")
            ),
            "matching_thesis_artifact_count": _non_negative_int(
                item.get("local_falsification_matching_thesis_artifact_count")
            ),
            "passing_cost_edge_artifact_count": _non_negative_int(
                item.get("local_falsification_passing_cost_edge_artifact_count")
            ),
            "paths_valid": item.get("local_falsification_paths_valid") is True,
            "factory_valid": item.get("local_falsification_factory_valid") is True,
            "safety_scope_valid": (
                item.get("local_falsification_safety_scope_valid") is True
            ),
            "event_source_valid": (
                item.get("local_falsification_event_source_valid") is True
            ),
            "event_source_context_alignment_valid": (
                item.get(
                    "local_falsification_event_source_context_alignment_valid"
                )
                is True
            ),
            "event_source_failure_synthesis_guard_valid": (
                item.get(
                    "local_falsification_event_source_failure_synthesis_guard_valid"
                )
                is True
            ),
            "artifact_paths": _string_list(
                item.get("local_falsification_artifact_paths", [])
            ),
            "blocker_names": _string_list(
                item.get("local_falsification_blocker_names", [])
            ),
        }
        candidate["passed"] = (
            not candidate["required"]
            or (
                candidate["handoff_passed"]
                and candidate["artifact_count"] > 0
                and candidate["parseable_artifact_count"]
                == candidate["artifact_count"]
                and candidate["matching_thesis_artifact_count"] > 0
                and candidate["passing_cost_edge_artifact_count"] > 0
                and candidate["paths_valid"]
                and candidate["factory_valid"]
                and candidate["safety_scope_valid"]
                and candidate["event_source_valid"]
                and candidate["event_source_context_alignment_valid"]
                and candidate["event_source_failure_synthesis_guard_valid"]
                and not candidate["blocker_names"]
            )
        )
        candidates.append(candidate)
    required_candidates = [item for item in candidates if item["required"]]
    return {
        "required": bool(required_candidates),
        "passed": all(item["passed"] for item in required_candidates),
        "candidate_count": len(candidates),
        "required_candidate_count": len(required_candidates),
        "candidates": candidates,
    }


def _proposal_research_decision_novelty_handoff(
    proposal_metadata: dict[str, Any]
) -> dict[str, Any]:
    constraints = proposal_metadata.get("research_decision_constraints", [])
    if not isinstance(constraints, list):
        constraints = []
    candidates: list[dict[str, Any]] = []
    for item in constraints:
        if not isinstance(item, dict):
            continue
        repeated_failed_family_matches = list(
            dict.fromkeys(
                _string_list(item.get("repeated_failed_family_matches", []))
                + _string_list(item.get("repeated_family_matches", []))
            )
        )
        local_failed_mechanism_matches = _string_list(
            item.get("local_falsification_failed_mechanism_class_matches", [])
        )
        candidate = {
            "path": item.get("path"),
            "failed_thesis_id_match": item.get("failed_thesis_id_match") is True,
            "repeated_failed_family_matches": repeated_failed_family_matches,
            "local_falsification_failed_thesis_ids": _string_list(
                item.get("local_falsification_failed_thesis_ids", [])
            ),
            "local_falsification_failed_thesis_id_match": (
                item.get("local_falsification_failed_thesis_id_match") is True
            ),
            "local_falsification_failed_mechanism_tokens": _string_list(
                item.get("local_falsification_failed_mechanism_tokens", [])
            ),
            "local_falsification_failed_mechanism_class_matches": (
                local_failed_mechanism_matches
            ),
        }
        candidate["passed"] = (
            not candidate["failed_thesis_id_match"]
            and not candidate["repeated_failed_family_matches"]
            and not candidate["local_falsification_failed_thesis_id_match"]
            and not candidate["local_falsification_failed_mechanism_class_matches"]
        )
        candidates.append(candidate)
    failed_candidates = [item for item in candidates if not item["passed"]]
    return {
        "required": bool(candidates),
        "passed": not failed_candidates,
        "candidate_count": len(candidates),
        "failed_candidate_count": len(failed_candidates),
        "failed_paths": [
            str(item["path"])
            for item in failed_candidates
            if item.get("path") is not None
        ],
        "candidates": candidates,
    }


def _proposal_research_decision_question_handoff(
    proposal_metadata: dict[str, Any]
) -> dict[str, Any]:
    constraints = proposal_metadata.get("research_decision_constraints", [])
    if not isinstance(constraints, list):
        constraints = []
    candidates: list[dict[str, Any]] = []
    for item in constraints:
        if not isinstance(item, dict):
            continue
        required_questions = _string_list(item.get("required_research_questions", []))
        response_indexes = _positive_int_list(
            item.get("research_question_response_indexes", [])
        )
        reported_missing_indexes = _positive_int_list(
            item.get("missing_research_question_response_indexes", [])
        )
        upstream_computed_missing_indexes = _positive_int_list(
            item.get("computed_missing_research_question_response_indexes", [])
        )
        recomputed_missing_indexes = [
            index
            for index in range(1, len(required_questions) + 1)
            if index not in response_indexes
        ]
        missing_indexes = list(
            dict.fromkeys(
                [
                    *reported_missing_indexes,
                    *upstream_computed_missing_indexes,
                    *recomputed_missing_indexes,
                ]
            )
        )
        weak_indexes = _positive_int_list(
            item.get("weak_research_question_response_indexes", [])
        )
        required = item.get("requires_research_question_responses") is True
        candidate = {
            "path": item.get("path"),
            "required": required,
            "required_research_questions": required_questions,
            "research_question_response_indexes": response_indexes,
            "reported_missing_research_question_response_indexes": (
                reported_missing_indexes
            ),
            "upstream_computed_missing_research_question_response_indexes": (
                upstream_computed_missing_indexes
            ),
            "recomputed_missing_research_question_response_indexes": (
                recomputed_missing_indexes
            ),
            "missing_research_question_response_indexes": missing_indexes,
            "weak_research_question_response_indexes": weak_indexes,
        }
        candidate["passed"] = (
            not candidate["required"]
            or (
                bool(required_questions)
                and not missing_indexes
                and not weak_indexes
            )
        )
        candidates.append(candidate)
    required_candidates = [item for item in candidates if item["required"]]
    failed_candidates = [item for item in required_candidates if not item["passed"]]
    return {
        "required": bool(required_candidates),
        "passed": not failed_candidates,
        "candidate_count": len(candidates),
        "required_candidate_count": len(required_candidates),
        "failed_candidate_count": len(failed_candidates),
        "failed_paths": [
            str(item["path"])
            for item in failed_candidates
            if item.get("path") is not None
        ],
        "candidates": candidates,
    }


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


def _motivated_thesis_ids(reference: dict[str, Any]) -> set[str]:
    raw = reference.get("motivated_thesis_ids", [])
    if isinstance(raw, str):
        raw = [raw]
    return {str(item).strip() for item in raw if str(item).strip()}


def _render_long_only_strategy_code(
    *,
    strategy_name: str,
    timeframe: str,
    candidate_id: str,
    source_proposal_hash: str,
    generator_mode: str,
    proposal_metadata: dict[str, Any],
) -> str:
    freqai_block = ""
    freqai_start_source = ""
    logic_variant = _logic_variant_from_proposal(proposal_metadata)
    defaults = _parameter_defaults_for_proposal(proposal_metadata)
    sell_timeout_min = min(24, max(2, int(defaults["sell_timeout_candles"])))
    label_horizon = int(proposal_metadata.get("label_horizon") or 12)
    target_name = str(proposal_metadata.get("target_definition") or "future_return")
    threshold = float(proposal_metadata.get("prediction_threshold") or 0.0)
    entry_logic = _entry_logic_for_variant(logic_variant, generator_mode, target_name, threshold)
    exit_logic = _exit_logic_for_variant(logic_variant)
    extra_imports = ""
    structural_helper_block = ""
    startup_candle_count = 120
    informative_pairs_block = ""
    funding_indicator_source = ""
    if logic_variant == "crowding_unwind_reaccumulation":
        extra_imports = "\nfrom pathlib import Path\n\nimport pandas as pd"
        startup_candle_count = 900
        funding_indicator_source = '''
        dataframe = self._attach_local_crowding_features(dataframe, metadata)
'''
        structural_helper_block = '''
    open_interest_path = Path("data/market_structure/bybit/futures/BTC_USDT_USDT-1h-open_interest.parquet")
    long_short_ratio_path = Path("data/market_structure/bybit/futures/BTC_USDT_USDT-1h-long_short_ratio.parquet")
    _local_crowding_context_cache: dict[str, DataFrame] = {}

    def _attach_local_crowding_features(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if dataframe.empty:
            return self._neutral_crowding_features(dataframe)
        enriched = dataframe.sort_values("date").reset_index(drop=True).copy()
        enriched["date"] = pd.to_datetime(enriched["date"], utc=True, errors="coerce")
        base_step = self._infer_candle_step(enriched["date"])
        open_interest = self._load_structural_context(
            self.open_interest_path, ["open_interest"]
        )
        if not open_interest.empty:
            enriched = self._merge_structural_context(
                enriched,
                open_interest,
                ["open_interest"],
                base_step=base_step,
            )
        ratio = self._load_structural_context(
            self.long_short_ratio_path,
            ["long_account_ratio", "short_account_ratio", "long_short_ratio"],
        )
        if not ratio.empty:
            enriched = self._merge_structural_context(
                enriched,
                ratio,
                ["long_account_ratio", "short_account_ratio", "long_short_ratio"],
                base_step=base_step,
            )
        enriched = self._neutral_crowding_features(enriched)
        interest = enriched["open_interest"].astype(float)
        ratio_series = enriched["long_short_ratio"].astype(float)
        close = enriched["close"].astype(float)
        volume = enriched["volume"].astype(float)
        enriched["open_interest_delta_pct_288"] = (
            interest / interest.shift(288).replace(0, np.nan) - 1.0
        ) * 100.0
        ratio_mean = ratio_series.rolling(864, min_periods=864).mean()
        ratio_std = ratio_series.rolling(864, min_periods=864).std().replace(0, np.nan)
        enriched["long_short_ratio_zscore_864"] = (ratio_series - ratio_mean) / ratio_std
        sma_144 = close.rolling(144, min_periods=144).mean()
        enriched["sma_distance_bps_144"] = (close / sma_144.replace(0, np.nan) - 1.0) * 10000.0
        volume_mean = volume.rolling(288, min_periods=288).mean()
        volume_std = volume.rolling(288, min_periods=288).std().replace(0, np.nan)
        enriched["volume_zscore_288"] = (volume - volume_mean) / volume_std
        structural_columns = [
            "open_interest_delta_pct_288",
            "long_short_ratio_zscore_864",
            "sma_distance_bps_144",
            "volume_zscore_288",
        ]
        enriched[structural_columns] = (
            enriched[structural_columns]
            .replace([np.inf, -np.inf], 0.0)
            .fillna(0.0)
        )
        return enriched

    def _load_structural_context(self, path: Path, columns: list[str]) -> DataFrame:
        cache_key = str(path)
        if cache_key in self._local_crowding_context_cache:
            return self._local_crowding_context_cache[cache_key].copy()
        empty = DataFrame(columns=["date", *columns])
        if not path.is_file():
            return empty
        try:
            context = pd.read_parquet(path)
        except Exception:
            return empty
        if "date" not in context.columns:
            return empty
        context = context.copy()
        context["date"] = pd.to_datetime(context["date"], utc=True, errors="coerce")
        for column in columns:
            if column not in context.columns:
                context[column] = 0.0
            context[column] = pd.to_numeric(context[column], errors="coerce")
        context = (
            context[["date", *columns]]
            .dropna(subset=["date"])
            .drop_duplicates(subset=["date"], keep="last")
            .sort_values("date")
            .reset_index(drop=True)
        )
        if not context.empty:
            self._local_crowding_context_cache[cache_key] = context.copy()
        return context

    def _merge_structural_context(
        self,
        dataframe: DataFrame,
        context: DataFrame,
        columns: list[str],
        *,
        base_step: pd.Timedelta | None,
    ) -> DataFrame:
        if context.empty:
            return dataframe
        features = context.sort_values("date").reset_index(drop=True).copy()
        features["date_merge"] = self._closed_context_merge_dates(
            features["date"],
            context_step=self._infer_candle_step(features["date"]) or pd.Timedelta(hours=1),
            base_step=base_step,
        )
        merged = pd.merge_asof(
            dataframe.sort_values("date").reset_index(drop=True),
            features[["date_merge", *columns]],
            left_on="date",
            right_on="date_merge",
            direction="backward",
        )
        return merged.drop(columns=["date_merge"], errors="ignore")

    def _neutral_crowding_features(self, dataframe: DataFrame) -> DataFrame:
        neutral = dataframe.copy()
        defaults = {
            "open_interest": 0.0,
            "long_account_ratio": 0.0,
            "short_account_ratio": 0.0,
            "long_short_ratio": 0.0,
            "open_interest_delta_pct_288": 0.0,
            "long_short_ratio_zscore_864": 0.0,
            "sma_distance_bps_144": 0.0,
            "volume_zscore_288": 0.0,
        }
        for column, value in defaults.items():
            if column not in neutral.columns:
                neutral[column] = value
            neutral[column] = pd.to_numeric(neutral[column], errors="coerce").fillna(value)
        return neutral

    @staticmethod
    def _infer_candle_step(dates: pd.Series) -> pd.Timedelta | None:
        parsed = pd.to_datetime(dates, utc=True, errors="coerce").dropna().sort_values()
        diffs = parsed.diff().dropna()
        positive_diffs = diffs[diffs > pd.Timedelta(0)]
        if positive_diffs.empty:
            return None
        return positive_diffs.median()

    @staticmethod
    def _closed_context_merge_dates(
        dates: pd.Series,
        *,
        context_step: pd.Timedelta | None,
        base_step: pd.Timedelta | None,
    ) -> pd.Series:
        parsed = pd.to_datetime(dates, utc=True, errors="coerce")
        if context_step is None or base_step is None or context_step <= base_step:
            return parsed
        return parsed + (context_step - base_step)
'''
    if logic_variant == "funding_pressure_carry":
        informative_pairs_block = '''
    def informative_pairs(self):
        if not self.dp:
            return []
        return [
            (pair, "8h", "funding_rate")
            for pair in self.dp.current_whitelist()
        ]
'''
        funding_indicator_source = '''
        if self.dp:
            funding_dataframe = self.dp.get_pair_dataframe(
                pair=metadata["pair"],
                timeframe="8h",
                candle_type="funding_rate",
            )
            if not funding_dataframe.empty:
                funding_dataframe = funding_dataframe.copy()
                funding_dataframe["funding_rate_raw"] = funding_dataframe["open"].astype(float)
                funding_dataframe["funding_rate_mean_raw"] = funding_dataframe[
                    "funding_rate_raw"
                ].rolling(6, min_periods=1).mean()
                funding_dataframe["funding_rate_abs_mean_raw"] = funding_dataframe[
                    "funding_rate_raw"
                ].abs().rolling(6, min_periods=1).mean()
                dataframe = merge_informative_pair(
                    dataframe,
                    funding_dataframe[
                        [
                            "date",
                            "funding_rate_raw",
                            "funding_rate_mean_raw",
                            "funding_rate_abs_mean_raw",
                        ]
                    ],
                    self.timeframe,
                    "8h",
                    ffill=True,
                    append_timeframe=False,
                    suffix="funding",
                )
        if "funding_rate_raw_funding" in dataframe.columns:
            dataframe["funding_rate"] = dataframe["funding_rate_raw_funding"].fillna(0.0)
            dataframe["funding_rate_mean"] = dataframe[
                "funding_rate_mean_raw_funding"
            ].fillna(0.0)
            dataframe["funding_rate_abs_mean"] = dataframe[
                "funding_rate_abs_mean_raw_funding"
            ].fillna(0.0)
        else:
            dataframe["funding_rate"] = 0.0
            dataframe["funding_rate_mean"] = 0.0
            dataframe["funding_rate_abs_mean"] = 0.0
        dataframe["funding_pressure"] = dataframe["funding_rate"].rolling(
            12, min_periods=1
        ).mean()
        dataframe["funding_pressure_delta"] = dataframe["funding_pressure"].diff().fillna(0.0)
'''
    if logic_variant in {
        "mark_price_dislocation_reclaim",
        "mark_discount_reclaim_continuation",
        "mark_fair_value_momentum_lag",
    }:
        informative_pairs_block = '''
    def informative_pairs(self):
        if not self.dp:
            return []
        return [
            (pair, "4h", "mark")
            for pair in self.dp.current_whitelist()
        ]
'''
        funding_indicator_source = '''
        if self.dp:
            mark_dataframe = self.dp.get_pair_dataframe(
                pair=metadata["pair"],
                timeframe="4h",
                candle_type="mark",
            )
            if not mark_dataframe.empty:
                mark_dataframe = mark_dataframe.copy()
                mark_dataframe["mark_close_raw"] = mark_dataframe["close"].astype(float)
                mark_dataframe["mark_log_return_raw"] = np.log(
                    mark_dataframe["mark_close_raw"] / mark_dataframe["mark_close_raw"].shift(1)
                ).replace([np.inf, -np.inf], 0.0).fillna(0.0)
                mark_dataframe["mark_price_return_bps_raw"] = (
                    mark_dataframe["mark_close_raw"] / mark_dataframe["mark_close_raw"].shift(1)
                    - 1.0
                ).replace([np.inf, -np.inf], 0.0).fillna(0.0) * 10000.0
                dataframe = merge_informative_pair(
                    dataframe,
                    mark_dataframe[
                        ["date", "mark_close_raw", "mark_log_return_raw", "mark_price_return_bps_raw"]
                    ],
                    self.timeframe,
                    "4h",
                    ffill=True,
                    append_timeframe=False,
                    suffix="mark",
                )
        if "mark_close_raw_mark" in dataframe.columns:
            dataframe["mark_close"] = dataframe["mark_close_raw_mark"].fillna(dataframe["close"])
            dataframe["mark_log_return"] = dataframe["mark_log_return_raw_mark"].fillna(0.0)
            dataframe["mark_price_return_bps"] = dataframe[
                "mark_price_return_bps_raw_mark"
            ].fillna(0.0)
        else:
            dataframe["mark_close"] = dataframe["close"]
            dataframe["mark_log_return"] = 0.0
            dataframe["mark_price_return_bps"] = 0.0
        dataframe["mark_price_gap"] = (
            (dataframe["close"] - dataframe["mark_close"]) / dataframe["mark_close"].replace(0, np.nan)
        ).replace([np.inf, -np.inf], 0.0).fillna(0.0)
        dataframe["mark_price_gap_delta"] = dataframe["mark_price_gap"].diff().fillna(0.0)
        dataframe["mark_price_gap_delta_6"] = (
            dataframe["mark_price_gap"] - dataframe["mark_price_gap"].shift(6)
        ).fillna(0.0)
        dataframe["return_3"] = (
            dataframe["close"] / dataframe["close"].shift(3) - 1.0
        ).replace([np.inf, -np.inf], 0.0).fillna(0.0)
        dataframe["traded_lag_return_bps"] = (
            dataframe["close"] / dataframe["close"].shift(int(self.buy_pullback_lookback.value))
            - 1.0
        ).replace([np.inf, -np.inf], 0.0).fillna(0.0) * 10000.0
        volume_std = dataframe["volume"].rolling(
            int(self.buy_volume_window.value), min_periods=1
        ).std().replace(0, np.nan)
        dataframe["volume_zscore"] = (
            (dataframe["volume"] - dataframe["volume_mean"]) / volume_std
        ).replace([np.inf, -np.inf], 0.0).fillna(0.0)
        dataframe["mark_price_gap_mean"] = dataframe["mark_price_gap"].rolling(
            int(self.buy_volume_window.value), min_periods=1
        ).mean()
        dataframe["mark_price_gap_abs_mean"] = dataframe["mark_price_gap"].abs().rolling(
            int(self.buy_volume_window.value), min_periods=1
        ).mean().replace(0, 1e-9)
        dataframe["mark_price_trend"] = dataframe["mark_close"] / dataframe["mark_close"].shift(
            int(self.buy_pullback_lookback.value)
        ) - 1.0
        dataframe["mark_price_trend"] = dataframe["mark_price_trend"].replace(
            [np.inf, -np.inf], 0.0
        ).fillna(0.0)
'''
    if logic_variant in {
        "cross_asset_cointegration_spread",
        "cross_asset_correlation_recovery",
        "cross_asset_lead_lag",
    }:
        informative_pairs_block = '''
    def informative_pairs(self):
        return [("ETH/USDT:USDT", self.timeframe)]
'''
        funding_indicator_source = '''
        if self.dp:
            eth_dataframe = self.dp.get_pair_dataframe(
                pair="ETH/USDT:USDT",
                timeframe=self.timeframe,
            )
            if not eth_dataframe.empty:
                eth_dataframe = eth_dataframe.copy()
                eth_dataframe["eth_close_raw"] = eth_dataframe["close"].astype(float)
                eth_dataframe["eth_log_return_raw"] = np.log(
                    eth_dataframe["eth_close_raw"] / eth_dataframe["eth_close_raw"].shift(1)
                ).replace([np.inf, -np.inf], 0.0).fillna(0.0)
                eth_dataframe["eth_volume_raw"] = eth_dataframe["volume"].astype(float)
                dataframe = dataframe.merge(
                    eth_dataframe[["date", "eth_close_raw", "eth_log_return_raw", "eth_volume_raw"]],
                    on="date",
                    how="left",
                )
        if "eth_close_raw" in dataframe.columns:
            dataframe["eth_close"] = dataframe["eth_close_raw"].fillna(0.0)
        else:
            dataframe["eth_close"] = 0.0
        if "eth_log_return_raw" in dataframe.columns:
            dataframe["eth_log_return"] = dataframe["eth_log_return_raw"].fillna(0.0)
        else:
            dataframe["eth_log_return"] = 0.0
        if "eth_volume_raw" in dataframe.columns:
            dataframe["eth_volume"] = dataframe["eth_volume_raw"].fillna(0.0)
        else:
            dataframe["eth_volume"] = 0.0
'''
    entry_tag = {
        "amihud_illiquidity_premium": "amihud_illiquidity_premium",
        "bipower_jump_decay": "bipower_jump_decay",
        "calendar_turnover_seasonality": "calendar_turnover_seasonality",
        "crowding_unwind_reaccumulation": "crowding_unwind_reaccumulation",
        "cross_asset_cointegration_spread": "cross_asset_cointegration_spread",
        "cross_asset_correlation_recovery": "cross_asset_correlation_recovery",
        "cross_asset_lead_lag": "cross_asset_lead_lag",
        "downside_liquidity_shock_reversal": "downside_liquidity_shock_reversal",
        "directional_change_overshoot": "directional_change_overshoot",
        "entropy_regime_transition": "entropy_regime_transition",
        "fractal_long_memory_regime": "fractal_long_memory_regime",
        "funding_pressure_carry": "funding_pressure_carry",
        "intraday_session_liquidity_reclaim": "intraday_session_liquidity_reclaim",
        "liquidity_recovery_horizon": "liquidity_recovery_horizon",
        "market_beta_drawdown_carry": "market_beta_drawdown_carry",
        "mark_discount_reclaim_continuation": "mark_discount_reclaim_continuation",
        "mark_fair_value_momentum_lag": "mark_fair_value_momentum_lag",
        "mark_price_dislocation_reclaim": "mark_price_dislocation_reclaim",
        "microstructure_spread_reversion": "microstructure_spread_reversion",
        "range_quarticity_vol_of_vol_state": "range_quarticity_vol_of_vol_state",
        "realized_skewness_tail_shape": "realized_skewness_tail_shape",
        "regime_state_reentry": "regime_state_reentry",
        "semivariance_asymmetry_regime": "semivariance_asymmetry_regime",
        "signed_volume_imbalance_accumulation": "signed_volume_imbalance_accumulation",
        "trend_continuation": "trend_continuation",
        "variance_ratio_regime_switch": "variance_ratio_regime_switch",
        "volatility_breakout": "volatility_breakout",
    }.get(logic_variant, "rsi_pullback_recovery")
    exit_tag = {
        "amihud_illiquidity_premium": "illiquidity_or_resilience_exit",
        "bipower_jump_decay": "jump_decay_or_drift_exit",
        "calendar_turnover_seasonality": "calendar_turnover_or_midline_exit",
        "crowding_unwind_reaccumulation": "crowding_or_resilience_exit",
        "cross_asset_cointegration_spread": "cointegration_spread_or_resilience_exit",
        "cross_asset_correlation_recovery": "correlation_recovery_or_resilience_exit",
        "cross_asset_lead_lag": "cross_asset_spread_or_resilience_exit",
        "downside_liquidity_shock_reversal": "shock_reversal_exit",
        "directional_change_overshoot": "overshoot_failed_or_reversal_exit",
        "entropy_regime_transition": "entropy_regime_exit",
        "fractal_long_memory_regime": "fractal_memory_exit",
        "funding_pressure_carry": "funding_pressure_or_resilience_exit",
        "intraday_session_liquidity_reclaim": "session_vwap_or_time_exit",
        "liquidity_recovery_horizon": "liquidity_recovery_or_stress_exit",
        "market_beta_drawdown_carry": "beta_carry_risk_budget_exit",
        "mark_discount_reclaim_continuation": "mark_discount_reclaim_exit",
        "mark_fair_value_momentum_lag": "mark_fair_value_lag_exit",
        "mark_price_dislocation_reclaim": "mark_gap_or_resilience_exit",
        "microstructure_spread_reversion": "spread_normalization_or_reexpansion_exit",
        "range_quarticity_vol_of_vol_state": "range_quarticity_stress_exit",
        "realized_skewness_tail_shape": "tail_shape_or_skew_reversion_exit",
        "regime_state_reentry": "regime_state_or_volatility_exit",
        "semivariance_asymmetry_regime": "semivariance_risk_exit",
        "signed_volume_imbalance_accumulation": "imbalance_or_location_exit",
        "trend_continuation": "trend_exhaustion_or_timeout",
        "variance_ratio_regime_switch": "variance_ratio_or_autocorr_exit",
        "volatility_breakout": "breakout_failure_or_mean_reversion",
    }.get(logic_variant, "mean_reversion_or_momentum_failure")
    if generator_mode in {"freqai", "hybrid_ml"}:
        freqai_start_source = "        dataframe = self.freqai.start(dataframe, metadata, self)\n"
        freqai_block = f'''
    def feature_engineering_expand_all(self, dataframe: DataFrame, period: int, metadata: dict) -> DataFrame:
        dataframe[f"%-rsi-{{period}}"] = ta.RSI(dataframe, timeperiod=period)
        dataframe[f"%-ema-{{period}}"] = ta.EMA(dataframe, timeperiod=period)
        return dataframe

    def feature_engineering_expand_basic(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["%-pct-change"] = dataframe["close"].pct_change().fillna(0.0)
        dataframe["%-volume-z"] = (
            (dataframe["volume"] - dataframe["volume"].rolling(24, min_periods=1).mean())
            / dataframe["volume"].rolling(24, min_periods=1).std().replace(0, 1)
        ).fillna(0.0)
        return dataframe

    def feature_engineering_standard(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["%-atr"] = ta.ATR(dataframe, timeperiod=14).fillna(0.0)
        return dataframe

    def set_freqai_targets(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["&-{target_name}"] = (
            dataframe["close"].shift(-{label_horizon}) / dataframe["close"] - 1.0
        )
        return dataframe
'''
    return f'''from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any
{extra_imports}

import talib.abstract as ta
import numpy as np
from pandas import DataFrame
from technical import qtpylib

from freqtrade.strategy import DecimalParameter, IStrategy, IntParameter, merge_informative_pair, timeframe_to_minutes


class {strategy_name}(IStrategy):
    """
    Generated Bot Factory long-only strategy.

    Candidate ID: {candidate_id}
    Source proposal hash: {source_proposal_hash}
    Generator mode: {generator_mode}
    Strategy logic variant: {logic_variant}
    """

    INTERFACE_VERSION = 3

    can_short = False
    timeframe = {json.dumps(timeframe)}
    process_only_new_candles = True
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False
    startup_candle_count: int = {startup_candle_count}

    minimal_roi = {{"0": 0.03, "120": 0.01, "360": 0.0}}
    stoploss = -0.05
    trailing_stop = False

    buy_rsi_window = IntParameter(8, 30, default={int(defaults["buy_rsi_window"])}, space="buy", optimize=False, load=True)
    buy_pullback_lookback = IntParameter(2, 24, default={int(defaults["buy_pullback_lookback"])}, space="buy", optimize=False, load=True)
    buy_rsi_pullback = IntParameter(20, 55, default={int(defaults["buy_rsi_pullback"])}, space="buy", optimize=False, load=True)
    buy_rsi_recovery = IntParameter(35, 65, default={int(defaults["buy_rsi_recovery"])}, space="buy", optimize=False, load=True)
    buy_ema_fast = IntParameter(8, 30, default={int(defaults["buy_ema_fast"])}, space="buy", optimize=False, load=True)
    buy_ema_slow = IntParameter(32, 120, default={int(defaults["buy_ema_slow"])}, space="buy", optimize=False, load=True)
    buy_volume_window = IntParameter(12, 72, default={int(defaults["buy_volume_window"])}, space="buy", optimize=False, load=True)
    buy_volume_factor = DecimalParameter(
        0.80, 2.00, decimals=2, default={float(defaults["buy_volume_factor"]):.2f}, space="buy", optimize=False, load=True
    )
    sell_rsi_exit = IntParameter(55, 80, default={int(defaults["sell_rsi_exit"])}, space="sell", optimize=False, load=True)
    sell_timeout_candles = IntParameter(
        {sell_timeout_min}, 288, default={int(defaults["sell_timeout_candles"])}, space="sell", optimize=False, load=True
    )
{informative_pairs_block.rstrip()}
{structural_helper_block.rstrip()}

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=int(self.buy_rsi_window.value))
        dataframe["ema_fast"] = ta.EMA(dataframe, timeperiod=int(self.buy_ema_fast.value))
        dataframe["ema_slow"] = ta.EMA(dataframe, timeperiod=int(self.buy_ema_slow.value))
        dataframe["volume_mean"] = dataframe["volume"].rolling(
            int(self.buy_volume_window.value), min_periods=1
        ).mean()
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=14)
        dataframe["atr_mean"] = dataframe["atr"].rolling(24, min_periods=1).mean()
        dataframe["rolling_high"] = dataframe["close"].rolling(
            int(self.buy_pullback_lookback.value), min_periods=1
        ).max()
        dataframe["rolling_low"] = dataframe["close"].rolling(
            int(self.buy_pullback_lookback.value), min_periods=1
        ).min()
        date_series = dataframe["date"] if "date" in dataframe.columns else dataframe.index.to_series()
        dataframe["hour_utc"] = date_series.dt.hour
        dataframe["weekday"] = date_series.dt.dayofweek
        session_key = date_series.dt.strftime("%Y-%m-%d")
        typical_price = (dataframe["high"] + dataframe["low"] + dataframe["close"]) / 3.0
        cumulative_pv = (typical_price * dataframe["volume"]).groupby(session_key).cumsum()
        cumulative_volume = dataframe["volume"].groupby(session_key).cumsum().replace(0, 1)
        dataframe["session_vwap"] = cumulative_pv / cumulative_volume
        candle_direction = (
            (dataframe["close"] > dataframe["open"]).astype(int)
            - (dataframe["close"] < dataframe["open"]).astype(int)
        )
        dataframe["signed_volume"] = dataframe["volume"] * candle_direction
        rolling_signed_volume = dataframe["signed_volume"].rolling(
            int(self.buy_pullback_lookback.value), min_periods=1
        ).sum()
        rolling_volume = dataframe["volume"].rolling(
            int(self.buy_pullback_lookback.value), min_periods=1
        ).sum().replace(0, 1)
        dataframe["signed_volume_imbalance"] = rolling_signed_volume / rolling_volume
        candle_range = (dataframe["high"] - dataframe["low"]).replace(0, 1e-9)
        dataframe["close_location_value"] = (
            ((dataframe["close"] - dataframe["low"]) - (dataframe["high"] - dataframe["close"]))
            / candle_range
        )
        dataframe["close_location_mean"] = dataframe["close_location_value"].rolling(
            int(self.buy_pullback_lookback.value), min_periods=1
        ).mean()
        dataframe["range_pct"] = candle_range / dataframe["close"].replace(0, 1)
        dataframe["range_pct_mean"] = dataframe["range_pct"].rolling(24, min_periods=1).mean()
        dataframe["rolling_mid"] = (dataframe["rolling_high"] + dataframe["rolling_low"]) / 2.0
{funding_indicator_source.rstrip()}
        entropy_lookback = int(self.buy_pullback_lookback.value)
        direction_up_probability = (
            (dataframe["close"].diff().fillna(0.0) > 0).astype(float)
            .rolling(entropy_lookback, min_periods=1)
            .mean()
            .clip(0.001, 0.999)
        )
        dataframe["direction_entropy"] = -(
            direction_up_probability * np.log(direction_up_probability)
            + (1.0 - direction_up_probability)
            * np.log(1.0 - direction_up_probability)
        )
        dataframe["direction_entropy_baseline"] = dataframe["direction_entropy"].rolling(
            int(self.buy_volume_window.value), min_periods=1
        ).mean()
        range_sum = candle_range.rolling(entropy_lookback, min_periods=1).sum().replace(0, 1e-9)
        dataframe["range_efficiency"] = (
            dataframe["close"] - dataframe["close"].shift(entropy_lookback)
        ).abs() / range_sum
        dataframe["range_efficiency_mean"] = dataframe["range_efficiency"].rolling(
            int(self.buy_volume_window.value), min_periods=1
        ).mean()
        dataframe["entropy_drift"] = (
            dataframe["close"] / dataframe["close"].shift(entropy_lookback) - 1.0
        )
        log_return = np.log(dataframe["close"] / dataframe["close"].shift(1)).replace(
            [np.inf, -np.inf], 0.0
        ).fillna(0.0)
        dataframe["log_return"] = log_return
        dc_lookback = int(self.buy_pullback_lookback.value)
        dc_threshold = (
            (dataframe["atr"] / dataframe["close"].replace(0, np.nan))
            .rolling(dc_lookback, min_periods=1)
            .mean()
            .clip(lower=0.0025, upper=0.0300)
        ).replace([np.inf, -np.inf], 0.0).fillna(0.0060)
        dc_high = dataframe["close"].rolling(dc_lookback, min_periods=1).max()
        dc_low = dataframe["close"].rolling(dc_lookback, min_periods=1).min()
        pullback_from_high = (
            dataframe["close"] / dc_high.replace(0, np.nan) - 1.0
        ).replace([np.inf, -np.inf], 0.0).fillna(0.0)
        rebound_from_low = (
            dataframe["close"] / dc_low.replace(0, np.nan) - 1.0
        ).replace([np.inf, -np.inf], 0.0).fillna(0.0)
        dataframe["directional_change_threshold"] = dc_threshold
        dataframe["directional_change_state"] = np.select(
            [rebound_from_low >= dc_threshold, pullback_from_high <= -dc_threshold],
            [1.0, -1.0],
            default=0.0,
        )
        dataframe["directional_change_state"] = (
            dataframe["directional_change_state"].replace(0.0, np.nan).ffill().fillna(0.0)
        )
        dataframe["directional_change_event"] = (
            (
                dataframe["directional_change_state"]
                != dataframe["directional_change_state"].shift(1)
            )
            & (dataframe["directional_change_state"] != 0.0)
        ).astype(int)
        dataframe["bar_index"] = np.arange(len(dataframe), dtype=float)
        dataframe["directional_change_event_index"] = np.where(
            dataframe["directional_change_event"] > 0,
            dataframe["bar_index"],
            np.nan,
        )
        dataframe["directional_change_event_index"] = (
            dataframe["directional_change_event_index"].ffill()
        )
        dataframe["directional_change_event_age"] = (
            dataframe["bar_index"] - dataframe["directional_change_event_index"]
        ).replace([np.inf, -np.inf], dc_lookback + 1).fillna(dc_lookback + 1)
        dataframe["directional_change_extreme"] = np.where(
            dataframe["directional_change_state"] > 0.0,
            dc_low,
            np.where(dataframe["directional_change_state"] < 0.0, dc_high, dataframe["close"]),
        )
        bullish_overshoot = (
            dataframe["close"] / dataframe["directional_change_extreme"].replace(0, np.nan)
            - 1.0
        )
        bearish_overshoot = (
            dataframe["directional_change_extreme"] / dataframe["close"].replace(0, np.nan)
            - 1.0
        )
        dataframe["overshoot_return"] = np.where(
            dataframe["directional_change_state"] >= 0.0,
            bullish_overshoot,
            bearish_overshoot,
        )
        dataframe["overshoot_return"] = (
            dataframe["overshoot_return"].replace([np.inf, -np.inf], 0.0).fillna(0.0)
        )
        dataframe["overshoot_ratio"] = (
            dataframe["overshoot_return"] / dc_threshold.replace(0, np.nan)
        ).replace([np.inf, -np.inf], 0.0).fillna(0.0)
        dataframe["overshoot_length"] = dataframe["directional_change_event_age"]
        event_time_window = max(3, dc_lookback // 3)
        dataframe["event_time_trend"] = (
            log_return.rolling(event_time_window, min_periods=1).sum()
            * dataframe["directional_change_state"]
        ).replace([np.inf, -np.inf], 0.0).fillna(0.0)
        bullish_reversal = (
            dataframe["close"] / dc_high.replace(0, np.nan) - 1.0
        )
        bearish_reversal = (
            dc_low / dataframe["close"].replace(0, np.nan) - 1.0
        )
        dataframe["adverse_reversal_distance"] = np.where(
            dataframe["directional_change_state"] >= 0.0,
            bullish_reversal,
            bearish_reversal,
        )
        dataframe["adverse_reversal_distance"] = (
            dataframe["adverse_reversal_distance"]
            .replace([np.inf, -np.inf], 0.0)
            .fillna(0.0)
        )
        dataframe["turnover_proxy"] = (
            dataframe["volume"] / dataframe["volume_mean"].replace(0, 1e-9)
        ).replace([np.inf, -np.inf], 0.0).fillna(0.0)
        range_state_window = int(self.buy_volume_window.value)
        range_state_lookback = int(self.buy_pullback_lookback.value)
        range_min_periods = min(range_state_lookback, 8)
        dataframe["ohlc_range"] = dataframe["range_pct"]
        safe_high = dataframe["high"].replace(0, np.nan)
        safe_low = dataframe["low"].replace(0, np.nan)
        dataframe["range_return"] = np.log(safe_high / safe_low).replace(
            [np.inf, -np.inf], 0.0
        ).fillna(0.0)
        dataframe["range_quarticity_proxy"] = dataframe["range_return"].pow(4).rolling(
            range_state_lookback, min_periods=range_min_periods
        ).mean().replace([np.inf, -np.inf], 0.0).fillna(0.0)
        dataframe["range_quarticity_mean"] = dataframe["range_quarticity_proxy"].rolling(
            range_state_window * 2, min_periods=range_state_window
        ).mean().replace([np.inf, -np.inf], 0.0).fillna(0.0).replace(0, 1e-12)
        dataframe["range_quarticity_ratio"] = (
            dataframe["range_quarticity_proxy"] / dataframe["range_quarticity_mean"]
        ).replace([np.inf, -np.inf], 0.0).fillna(0.0)
        dataframe["range_quarticity_delta"] = dataframe["range_quarticity_ratio"].diff().fillna(0.0)
        dataframe["range_volatility"] = dataframe["range_pct"].rolling(
            range_state_lookback, min_periods=range_min_periods
        ).std().replace([np.inf, -np.inf], 0.0).fillna(0.0)
        dataframe["range_volatility_mean"] = dataframe["range_volatility"].rolling(
            range_state_window * 2, min_periods=range_state_window
        ).mean().replace([np.inf, -np.inf], 0.0).fillna(0.0).replace(0, 1e-9)
        dataframe["range_vol_of_vol_state"] = (
            dataframe["range_volatility"] / dataframe["range_volatility_mean"]
        ).replace([np.inf, -np.inf], 0.0).fillna(0.0)
        range_decay_window = max(3, range_state_lookback // 3)
        dataframe["range_state_decay"] = (
            dataframe["range_quarticity_ratio"]
            / dataframe["range_quarticity_ratio"].shift(range_decay_window).replace(0, np.nan)
        ).replace([np.inf, -np.inf], 1.0).fillna(1.0)
        dataframe["range_stress_ratio"] = (
            dataframe["range_quarticity_ratio"] * dataframe["range_vol_of_vol_state"]
        ).replace([np.inf, -np.inf], 0.0).fillna(0.0)
        dataframe["range_stress_recent"] = dataframe["range_stress_ratio"].rolling(
            range_state_lookback, min_periods=1
        ).max().replace([np.inf, -np.inf], 0.0).fillna(0.0)
        dataframe["participation_recovery"] = (
            dataframe["volume"] / dataframe["volume_mean"].replace(0, 1e-9)
        ).replace([np.inf, -np.inf], 0.0).fillna(0.0)
        stabilization_window = max(3, range_state_lookback // 3)
        dataframe["stabilization_drift"] = (
            dataframe["close"] / dataframe["close"].shift(stabilization_window) - 1.0
        ).replace([np.inf, -np.inf], 0.0).fillna(0.0)
        jump_lookback = int(self.buy_pullback_lookback.value)
        jump_min_periods = min(jump_lookback, 12)
        abs_log_return = log_return.abs()
        dataframe["realized_variance_fast"] = (
            log_return.pow(2)
            .rolling(jump_lookback, min_periods=jump_min_periods)
            .sum()
        ).replace([np.inf, -np.inf], 0.0).fillna(0.0)
        dataframe["bipower_variation"] = (
            (np.pi / 2.0)
            * (abs_log_return * abs_log_return.shift(1))
            .rolling(jump_lookback, min_periods=jump_min_periods)
            .sum()
        ).replace([np.inf, -np.inf], 0.0).fillna(0.0)
        dataframe["jump_variation"] = (
            dataframe["realized_variance_fast"] - dataframe["bipower_variation"]
        ).clip(lower=0.0).replace([np.inf, -np.inf], 0.0).fillna(0.0)
        dataframe["jump_variation_mean"] = dataframe["jump_variation"].rolling(
            jump_lookback * 2, min_periods=jump_min_periods
        ).mean().replace([np.inf, -np.inf], 0.0).fillna(0.0).replace(0, 1e-9)
        dataframe["jump_variation_ratio"] = (
            dataframe["jump_variation"] / dataframe["jump_variation_mean"]
        ).replace([np.inf, -np.inf], 0.0).fillna(0.0)
        decay_shift = max(2, jump_lookback // 3)
        dataframe["continuous_variance_decay"] = (
            dataframe["bipower_variation"]
            / dataframe["bipower_variation"].shift(decay_shift).replace(0, 1e-9)
        ).replace([np.inf, -np.inf], 0.0).fillna(0.0)
        dataframe["positive_jump_event"] = (
            (log_return > 0.0)
            & (dataframe["jump_variation_ratio"] > 1.25)
            & (dataframe["jump_variation"] > 0.0)
        ).astype(int)
        post_jump_window = max(3, jump_lookback // 4)
        dataframe["post_jump_drift"] = (
            dataframe["close"] / dataframe["close"].shift(post_jump_window) - 1.0
        ).replace([np.inf, -np.inf], 0.0).fillna(0.0)
        follow_through_window = max(3, jump_lookback // 3)
        dataframe["jump_follow_through"] = (
            dataframe["close"]
            / dataframe["close"]
            .rolling(follow_through_window, min_periods=3)
            .max()
            .shift(1)
            - 1.0
        ).replace([np.inf, -np.inf], 0.0).fillna(0.0)
        dataframe["jump_overextension"] = (
            dataframe["close"] / dataframe["rolling_high"].shift(1).replace(0, np.nan) - 1.0
        ).replace([np.inf, -np.inf], 0.0).fillna(0.0)
        spread_lookback = int(self.buy_pullback_lookback.value)
        spread_min_periods = min(spread_lookback, 8)
        spread_baseline_window = int(self.buy_volume_window.value) * 2
        return_autocovariance = (log_return * log_return.shift(1)).rolling(
            spread_lookback, min_periods=spread_min_periods
        ).mean()
        dataframe["return_autocovariance"] = return_autocovariance.replace(
            [np.inf, -np.inf], 0.0
        ).fillna(0.0)
        dataframe["roll_spread_proxy"] = (
            2.0 * np.sqrt((-return_autocovariance).clip(lower=0.0))
        ).replace([np.inf, -np.inf], 0.0).fillna(0.0)
        dataframe["roll_spread_mean"] = dataframe["roll_spread_proxy"].rolling(
            spread_baseline_window, min_periods=int(self.buy_volume_window.value)
        ).mean().replace([np.inf, -np.inf], 0.0).fillna(0.0).replace(0, 1e-9)
        dataframe["roll_spread_delta"] = dataframe["roll_spread_proxy"].diff().fillna(0.0)
        dataframe["hl_spread_proxy"] = dataframe["range_pct"].replace(
            [np.inf, -np.inf], 0.0
        ).fillna(0.0)
        dataframe["hl_spread_mean"] = dataframe["hl_spread_proxy"].rolling(
            spread_baseline_window, min_periods=int(self.buy_volume_window.value)
        ).mean().replace([np.inf, -np.inf], 0.0).fillna(0.0).replace(0, 1e-9)
        short_noise = log_return.abs().rolling(spread_lookback, min_periods=1).mean()
        long_noise = log_return.abs().rolling(
            spread_baseline_window, min_periods=spread_lookback
        ).mean().replace(0, 1e-9)
        dataframe["microstructure_noise_ratio"] = (
            short_noise / long_noise
        ).replace([np.inf, -np.inf], 0.0).fillna(0.0)
        variance_lookback = int(self.buy_pullback_lookback.value)
        variance_min_periods = min(variance_lookback, 8)
        one_step_variance = log_return.rolling(
            variance_lookback, min_periods=variance_min_periods
        ).var().replace(0, 1e-12)
        multi_step_return = np.log(
            dataframe["close"] / dataframe["close"].shift(variance_lookback)
        ).replace([np.inf, -np.inf], 0.0).fillna(0.0)
        multi_step_variance = multi_step_return.rolling(
            variance_lookback, min_periods=variance_min_periods
        ).var()
        dataframe["variance_ratio"] = (
            multi_step_variance / (one_step_variance * variance_lookback)
        ).replace([np.inf, -np.inf], 1.0).fillna(1.0)
        dataframe["variance_ratio_mean"] = dataframe["variance_ratio"].rolling(
            int(self.buy_volume_window.value), min_periods=1
        ).mean()
        dataframe["variance_ratio_delta"] = dataframe["variance_ratio"].diff().fillna(0.0)
        dataframe["return_autocorr"] = log_return.rolling(
            variance_lookback, min_periods=variance_min_periods
        ).corr(log_return.shift(1)).replace([np.inf, -np.inf], 0.0).fillna(0.0)
        dataframe["autocorr_mean"] = dataframe["return_autocorr"].rolling(
            int(self.buy_volume_window.value), min_periods=1
        ).mean()
        dataframe["regime_drift"] = (
            dataframe["close"] / dataframe["close"].shift(variance_lookback) - 1.0
        )
        normalized_atr = (
            dataframe["atr"] / dataframe["close"].replace(0, 1)
        ).replace([np.inf, -np.inf], 0.0).fillna(0.0).rolling(
            variance_lookback, min_periods=1
        ).mean().replace(0, 1e-9)
        dataframe["normalized_regime_return"] = (
            dataframe["regime_drift"] / normalized_atr
        ).replace([np.inf, -np.inf], 0.0).fillna(0.0)
        beta_volatility_window = int(self.buy_volume_window.value)
        beta_risk_window = beta_volatility_window * 4
        beta_volatility_min_periods = min(beta_volatility_window, 8)
        dataframe["realized_volatility"] = log_return.rolling(
            beta_volatility_window, min_periods=beta_volatility_min_periods
        ).std().replace([np.inf, -np.inf], 0.0).fillna(0.0)
        dataframe["realized_volatility_mean"] = dataframe[
            "realized_volatility"
        ].rolling(
            beta_risk_window, min_periods=beta_volatility_window
        ).mean().replace([np.inf, -np.inf], 0.0).fillna(0.0).replace(0, 1e-9)
        dataframe["market_beta_high"] = dataframe["close"].rolling(
            beta_risk_window, min_periods=1
        ).max()
        dataframe["market_beta_drawdown"] = (
            dataframe["close"] / dataframe["market_beta_high"].replace(0, np.nan) - 1.0
        ).replace([np.inf, -np.inf], 0.0).fillna(0.0)
        dataframe["market_beta_drift"] = (
            dataframe["close"] / dataframe["close"].shift(beta_volatility_window) - 1.0
        ).replace([np.inf, -np.inf], 0.0).fillna(0.0)
        regime_fast_window = int(self.buy_pullback_lookback.value)
        regime_state_window = int(self.buy_volume_window.value)
        regime_slow_window = regime_state_window * 2
        regime_min_periods = min(regime_state_window, 8)
        dataframe["regime_return_fast"] = (
            dataframe["close"] / dataframe["close"].shift(regime_fast_window) - 1.0
        ).replace([np.inf, -np.inf], 0.0).fillna(0.0)
        dataframe["regime_return_slow"] = (
            dataframe["close"] / dataframe["close"].shift(regime_slow_window) - 1.0
        ).replace([np.inf, -np.inf], 0.0).fillna(0.0)
        dataframe["regime_negative_frequency"] = (
            (log_return < 0.0)
            .astype(float)
            .rolling(regime_state_window, min_periods=regime_min_periods)
            .mean()
        ).replace([np.inf, -np.inf], 0.0).fillna(0.0)
        dataframe["regime_negative_frequency_mean"] = dataframe[
            "regime_negative_frequency"
        ].rolling(
            regime_slow_window, min_periods=regime_state_window
        ).mean().replace([np.inf, -np.inf], 0.0).fillna(0.5).replace(0, 1e-9)
        dataframe["regime_volatility"] = log_return.rolling(
            regime_state_window, min_periods=regime_min_periods
        ).std().replace([np.inf, -np.inf], 0.0).fillna(0.0)
        dataframe["regime_volatility_mean"] = dataframe["regime_volatility"].rolling(
            regime_slow_window, min_periods=regime_state_window
        ).mean().replace([np.inf, -np.inf], 0.0).fillna(0.0).replace(0, 1e-9)
        dataframe["regime_trendline"] = dataframe["close"].rolling(
            regime_slow_window, min_periods=regime_state_window
        ).mean()
        dataframe["regime_high"] = dataframe["close"].rolling(
            regime_slow_window, min_periods=regime_state_window
        ).max()
        dataframe["regime_drawdown"] = (
            dataframe["close"] / dataframe["regime_high"].replace(0, np.nan) - 1.0
        ).replace([np.inf, -np.inf], 0.0).fillna(0.0)
        fractal_lookback = int(self.buy_pullback_lookback.value)
        hurst_min_periods = min(fractal_lookback, 8)

        def _hurst_rs_proxy(values: np.ndarray) -> float:
            if len(values) < 8:
                return np.nan
            centered = values - np.nanmean(values)
            cumulative = np.nancumsum(centered)
            value_range = np.nanmax(cumulative) - np.nanmin(cumulative)
            value_std = np.nanstd(values)
            if value_range <= 0 or value_std <= 1e-12:
                return 0.5
            return float(np.clip(np.log(value_range / value_std) / np.log(len(values)), 0.0, 1.0))

        dataframe["hurst_proxy"] = log_return.rolling(
            fractal_lookback, min_periods=hurst_min_periods
        ).apply(_hurst_rs_proxy, raw=True)
        dataframe["hurst_baseline"] = dataframe["hurst_proxy"].rolling(
            int(self.buy_volume_window.value), min_periods=1
        ).mean()
        path_length = dataframe["close"].diff().abs().rolling(
            fractal_lookback, min_periods=1
        ).sum().replace(0, 1e-9)
        dataframe["fractal_efficiency"] = (
            dataframe["close"] - dataframe["close"].shift(fractal_lookback)
        ).abs() / path_length
        dataframe["fractal_efficiency_mean"] = dataframe["fractal_efficiency"].rolling(
            int(self.buy_volume_window.value), min_periods=1
        ).mean()
        dataframe["fractal_drift"] = (
            dataframe["close"] / dataframe["close"].shift(fractal_lookback) - 1.0
        )
        semivariance_lookback = int(self.buy_pullback_lookback.value)
        upside_squared_return = log_return.clip(lower=0.0).pow(2)
        downside_squared_return = log_return.clip(upper=0.0).pow(2)
        dataframe["upside_semivariance"] = upside_squared_return.rolling(
            semivariance_lookback, min_periods=1
        ).mean()
        dataframe["downside_semivariance"] = downside_squared_return.rolling(
            semivariance_lookback, min_periods=1
        ).mean()
        dataframe["downside_semivariance_mean"] = dataframe["downside_semivariance"].rolling(
            int(self.buy_volume_window.value), min_periods=1
        ).mean()
        semivariance_total = (
            dataframe["upside_semivariance"] + dataframe["downside_semivariance"]
        ).replace(0, 1e-12)
        dataframe["semivariance_balance"] = (
            dataframe["upside_semivariance"] - dataframe["downside_semivariance"]
        ) / semivariance_total
        dataframe["semivariance_balance_mean"] = dataframe["semivariance_balance"].rolling(
            int(self.buy_volume_window.value), min_periods=1
        ).mean()
        dataframe["semivariance_drift"] = (
            dataframe["close"] / dataframe["close"].shift(semivariance_lookback) - 1.0
        )
        higher_moment_lookback = int(self.buy_pullback_lookback.value)
        higher_moment_min_periods = min(higher_moment_lookback, 4)
        dataframe["realized_skewness"] = log_return.rolling(
            higher_moment_lookback, min_periods=higher_moment_min_periods
        ).skew().replace([np.inf, -np.inf], 0.0).fillna(0.0)
        dataframe["realized_kurtosis"] = log_return.rolling(
            higher_moment_lookback, min_periods=higher_moment_min_periods
        ).kurt().replace([np.inf, -np.inf], 0.0).fillna(0.0)
        dataframe["realized_skewness_mean"] = dataframe["realized_skewness"].rolling(
            int(self.buy_volume_window.value), min_periods=1
        ).mean()
        dataframe["realized_kurtosis_mean"] = dataframe["realized_kurtosis"].rolling(
            int(self.buy_volume_window.value), min_periods=1
        ).mean()
        dataframe["max_return"] = log_return.rolling(
            higher_moment_lookback, min_periods=1
        ).max()
        dataframe["max_return_mean"] = dataframe["max_return"].rolling(
            int(self.buy_volume_window.value), min_periods=1
        ).mean()
        dataframe["min_return"] = log_return.rolling(
            higher_moment_lookback, min_periods=1
        ).min()
        dataframe["tail_shape_drift"] = (
            dataframe["close"] / dataframe["close"].shift(higher_moment_lookback) - 1.0
        )
        dataframe["calendar_turnover_ratio"] = (
            dataframe["volume"] / dataframe["volume_mean"].replace(0, 1)
        ).replace([np.inf, -np.inf], 0.0).fillna(0.0)
        dataframe["calendar_turnover_ratio_mean"] = dataframe[
            "calendar_turnover_ratio"
        ].rolling(int(self.buy_volume_window.value), min_periods=1).mean()
        weekend_mask = dataframe["weekday"].isin([5, 6])
        weekday_mask = dataframe["weekday"] < 5
        dataframe["weekend_turnover_baseline"] = (
            dataframe["calendar_turnover_ratio"]
            .where(weekend_mask)
            .rolling(288, min_periods=1)
            .mean()
            .ffill()
            .fillna(1.0)
        )
        dataframe["weekday_turnover_baseline"] = (
            dataframe["calendar_turnover_ratio"]
            .where(weekday_mask)
            .rolling(288, min_periods=1)
            .mean()
            .ffill()
            .fillna(1.0)
        )
        dataframe["calendar_drift"] = (
            dataframe["close"] / dataframe["close"].shift(int(self.buy_pullback_lookback.value)) - 1.0
        )
        dataframe["dollar_volume"] = (
            dataframe["close"].abs() * dataframe["volume"].abs()
        ).replace(0, 1e-9)
        dataframe["amihud_illiquidity"] = (
            log_return.abs() / dataframe["dollar_volume"] * 1e9
        ).replace([np.inf, -np.inf], 0.0).fillna(0.0)
        dataframe["amihud_illiquidity_mean"] = dataframe[
            "amihud_illiquidity"
        ].rolling(int(self.buy_volume_window.value), min_periods=1).mean()
        dataframe["amihud_illiquidity_delta"] = (
            dataframe["amihud_illiquidity"].diff().fillna(0.0)
        )
        dataframe["illiquidity_drift"] = (
            dataframe["close"] / dataframe["close"].shift(int(self.buy_pullback_lookback.value)) - 1.0
        )
        recovery_lookback = int(self.buy_pullback_lookback.value)
        recovery_baseline = int(self.buy_volume_window.value)
        dataframe["amihud_illiquidity_ratio"] = (
            dataframe["amihud_illiquidity"]
            / dataframe["amihud_illiquidity_mean"].replace(0, 1e-9)
        ).replace([np.inf, -np.inf], 0.0).fillna(0.0)
        dataframe["range_recovery_ratio"] = (
            dataframe["range_pct"] / dataframe["range_pct_mean"].replace(0, 1e-9)
        ).replace([np.inf, -np.inf], 0.0).fillna(0.0)
        dataframe["volume_recovery_ratio"] = (
            dataframe["volume"] / dataframe["volume_mean"].replace(0, 1e-9)
        ).replace([np.inf, -np.inf], 0.0).fillna(0.0)
        liquidity_stress_flag = (
            (dataframe["amihud_illiquidity_ratio"] > 1.35)
            | (dataframe["range_recovery_ratio"] > 1.30)
        ).astype(float)
        dataframe["liquidity_stress_recent"] = liquidity_stress_flag.rolling(
            recovery_lookback, min_periods=1
        ).max()
        liquidity_normalized = (
            (dataframe["amihud_illiquidity_ratio"] <= 1.05)
            & (dataframe["range_recovery_ratio"] <= 1.15)
        ).astype(float)
        participation_recovered = (
            dataframe["volume_recovery_ratio"] >= self.buy_volume_factor.value
        ).astype(float)
        price_recovery_turn = (
            dataframe["close"].diff().fillna(0.0) > 0.0
        ).astype(float)
        dataframe["liquidity_recovery_score"] = (
            liquidity_normalized + participation_recovered + price_recovery_turn
        )
        dataframe["liquidity_recovery_anchor"] = (
            dataframe["rolling_mid"]
            + dataframe["close"].rolling(recovery_baseline, min_periods=1).mean()
        ) / 2.0
        dataframe["recovery_horizon_return"] = (
            dataframe["close"] / dataframe["close"].shift(recovery_lookback) - 1.0
        ).replace([np.inf, -np.inf], 0.0).fillna(0.0)
        if "eth_close" not in dataframe.columns:
            dataframe["eth_close"] = 0.0
        dataframe["eth_close"] = dataframe["eth_close"].fillna(0.0)
        if "eth_log_return" not in dataframe.columns:
            dataframe["eth_log_return"] = 0.0
        dataframe["eth_log_return"] = dataframe["eth_log_return"].fillna(0.0)
        dataframe["btc_log_return"] = log_return
        dataframe["eth_lead_return"] = dataframe["eth_log_return"].shift(1).fillna(0.0)
        dataframe["eth_lead_return_mean"] = dataframe["eth_lead_return"].rolling(
            int(self.buy_volume_window.value), min_periods=1
        ).mean()
        dataframe["eth_btc_return_spread"] = (
            dataframe["eth_lead_return"] - dataframe["btc_log_return"]
        )
        dataframe["eth_btc_spread_mean"] = dataframe["eth_btc_return_spread"].rolling(
            int(self.buy_volume_window.value), min_periods=1
        ).mean()
        dataframe["eth_btc_spread_abs_mean"] = dataframe["eth_btc_return_spread"].abs().rolling(
            int(self.buy_volume_window.value), min_periods=1
        ).mean()
        safe_eth_close = dataframe["eth_close"].replace(0, np.nan)
        safe_btc_close = dataframe["close"].replace(0, np.nan)
        dataframe["btc_eth_log_ratio"] = np.log(
            safe_btc_close / safe_eth_close
        ).replace([np.inf, -np.inf], 0.0).fillna(0.0)
        dataframe["btc_eth_ratio_mean"] = dataframe["btc_eth_log_ratio"].rolling(
            int(self.buy_volume_window.value), min_periods=1
        ).mean()
        btc_eth_ratio_std = dataframe["btc_eth_log_ratio"].rolling(
            int(self.buy_volume_window.value), min_periods=2
        ).std().replace(0, 1e-9)
        dataframe["btc_eth_ratio_zscore"] = (
            (dataframe["btc_eth_log_ratio"] - dataframe["btc_eth_ratio_mean"])
            / btc_eth_ratio_std
        ).replace([np.inf, -np.inf], 0.0).fillna(0.0)
        dataframe["btc_eth_ratio_zscore_delta"] = (
            dataframe["btc_eth_ratio_zscore"].diff().fillna(0.0)
        )
        dataframe["eth_regime_drift"] = (
            dataframe["eth_close"] / dataframe["eth_close"].shift(int(self.buy_pullback_lookback.value)) - 1.0
        ).replace([np.inf, -np.inf], 0.0).fillna(0.0)
        correlation_window = int(self.buy_volume_window.value)
        corr_min_periods = min(correlation_window, 8)
        dataframe["btc_eth_return_corr"] = dataframe["btc_log_return"].rolling(
            correlation_window, min_periods=corr_min_periods
        ).corr(dataframe["eth_log_return"]).replace([np.inf, -np.inf], 0.0).fillna(0.0)
        dataframe["btc_eth_corr_mean"] = dataframe["btc_eth_return_corr"].rolling(
            int(self.buy_volume_window.value), min_periods=1
        ).mean()
        dataframe["btc_eth_corr_delta"] = dataframe["btc_eth_return_corr"].diff().fillna(0.0)
        dataframe["btc_eth_relative_return"] = dataframe["btc_log_return"] - dataframe["eth_log_return"]
        dataframe["btc_eth_relative_return_mean"] = dataframe[
            "btc_eth_relative_return"
        ].rolling(int(self.buy_volume_window.value), min_periods=1).mean()
        dataframe["cross_asset_drift"] = (
            dataframe["close"] / dataframe["close"].shift(int(self.buy_pullback_lookback.value)) - 1.0
        )
{freqai_start_source.rstrip()}
        return dataframe
{freqai_block}

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
{entry_logic}
        dataframe.loc[entry_condition, ["enter_long", "enter_tag"]] = (
            1,
            "{entry_tag}",
        )
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
{exit_logic}
        dataframe.loc[exit_condition, ["exit_long", "exit_tag"]] = (
            1,
            "{exit_tag}",
        )
        return dataframe

    def custom_exit(
        self,
        pair: str,
        trade: Any,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs: Any,
    ) -> str | None:
        hold_minutes = int(self.sell_timeout_candles.value) * timeframe_to_minutes(
            self.timeframe
        )
        if current_time - trade.open_date_utc >= timedelta(minutes=hold_minutes):
            return "timeout_exit"
        return None
'''


def _generator_mode_from_proposal(proposal_metadata: dict[str, Any]) -> str:
    mode = str(proposal_metadata.get("generator_mode") or "rule_based").strip().lower()
    return mode if mode in ALLOWED_GENERATOR_MODES else "rule_based"


def _logic_variant_from_proposal(proposal_metadata: dict[str, Any]) -> str:
    variant = str(proposal_metadata.get("strategy_logic_variant") or "").strip().lower()
    if variant in ALLOWED_LOGIC_VARIANTS:
        return variant
    thesis_type = str(proposal_metadata.get("thesis_type") or "").strip().lower()
    failure_codes = set(proposal_metadata.get("failure_taxonomy_codes") or [])
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
        "account_ratio_reaccumulation",
        "crowding_unwind",
        "crowding_unwind_reaccumulation",
        "long_short_reaccumulation",
        "open_interest_unwind_reaccumulation",
        "positioning_unwind_reaccumulation",
    }:
        return "crowding_unwind_reaccumulation"
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
        "ohlc_quarticity_volatility_state_transition",
        "ohlc_range_quarticity",
        "quarticity_vol_of_vol_state",
        "range_quarticity_state_decay",
        "range_quarticity_vol_of_vol",
        "range_quarticity_vol_of_vol_state",
    }:
        return "range_quarticity_vol_of_vol_state"
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
        "downside_liquidity_shock",
        "downside_liquidity_shock_reversal",
        "capitulation_reversal",
    }:
        return "downside_liquidity_shock_reversal"
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
        "intraday_liquidity_timing",
        "intraday_session_liquidity",
        "session_liquidity_reclaim",
    }:
        return "intraday_session_liquidity_reclaim"
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


def _metadata_logic_variant_from_proposal(proposal_metadata: dict[str, Any]) -> str:
    explicit = str(proposal_metadata.get("strategy_logic_variant") or "").strip().lower()
    if explicit:
        return explicit
    return _logic_variant_from_proposal(proposal_metadata)


def _parameter_defaults_for_proposal(proposal_metadata: dict[str, Any]) -> dict[str, int | float]:
    explicit = str(proposal_metadata.get("strategy_logic_variant") or "").strip().lower()
    if explicit and explicit not in ALLOWED_LOGIC_VARIANTS:
        return {}
    defaults = dict(LOGIC_VARIANT_PARAMETER_DEFAULTS[_logic_variant_from_proposal(proposal_metadata)])
    defaults.update(_proposal_parameter_overrides(proposal_metadata))
    return defaults


def _proposal_parameter_overrides(proposal_metadata: dict[str, Any]) -> dict[str, int | float]:
    raw = proposal_metadata.get("parameter_overrides")
    if not isinstance(raw, dict):
        return {}
    overrides: dict[str, int | float] = {}
    for raw_name, raw_value in raw.items():
        name = str(raw_name).strip().lower()
        name = PARAMETER_OVERRIDE_ALIASES.get(name, name)
        if name not in PARAMETER_OVERRIDE_RANGES:
            continue
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            continue
        if value != value:
            continue
        low, high = PARAMETER_OVERRIDE_RANGES[name]
        if value < low or value > high:
            continue
        if isinstance(DEFAULT_PARAMETER_DEFAULTS.get(name), int):
            overrides[name] = int(round(value))
        else:
            overrides[name] = value
    return overrides


def _ml_filter_source(generator_mode: str, target_name: str, threshold: float) -> str:
    if generator_mode not in {"freqai", "hybrid_ml"}:
        return "        ml_filter = True"
    return (
        "        ml_filter = "
        f"dataframe.get({json.dumps('&-' + target_name)}, 0) > {threshold}"
    )


def _entry_logic_for_variant(
    logic_variant: str,
    generator_mode: str,
    target_name: str,
    threshold: float,
) -> str:
    ml_filter = _ml_filter_source(generator_mode, target_name, threshold)
    if logic_variant == "amihud_illiquidity_premium":
        return f'''        price_impact_premium = dataframe["amihud_illiquidity"] > dataframe["amihud_illiquidity_mean"]
        illiquidity_releasing = dataframe["amihud_illiquidity_delta"] < 0.0
        not_extreme_impact = dataframe["amihud_illiquidity"] <= (
            dataframe["amihud_illiquidity_mean"] * 3.0
        )
        price_resilience = dataframe["close"] > dataframe["rolling_mid"]
        positive_illiquidity_drift = dataframe["illiquidity_drift"] > 0.0
        controlled_range = dataframe["range_pct"] <= (dataframe["range_pct_mean"] * 1.5)
        volume_floor = dataframe["volume"] > (
            dataframe["volume_mean"] * self.buy_volume_factor.value
        )
{ml_filter}
        entry_condition = (
            price_impact_premium
            & illiquidity_releasing
            & not_extreme_impact
            & price_resilience
            & positive_illiquidity_drift
            & controlled_range
            & volume_floor
            & ml_filter
            & (dataframe["volume"] > 0)
        )'''
    if logic_variant == "bipower_jump_decay":
        return f'''        jump_window = max(2, int(self.buy_pullback_lookback.value) // 4)
        positive_jump_detected = (
            dataframe["positive_jump_event"].rolling(jump_window, min_periods=1).max() > 0
        )
        jump_dominates_continuous_variance = dataframe["jump_variation_ratio"] > 1.25
        continuous_variance_decaying = dataframe["continuous_variance_decay"] < 0.95
        post_jump_drift_positive = dataframe["post_jump_drift"] > 0.0
        not_overextended_after_jump = dataframe["jump_overextension"] < 0.018
        volume_positive = dataframe["volume"] > 0
{ml_filter}
        entry_condition = (
            positive_jump_detected
            & jump_dominates_continuous_variance
            & continuous_variance_decaying
            & post_jump_drift_positive
            & not_overextended_after_jump
            & volume_positive
            & ml_filter
        )'''
    if logic_variant == "crowding_unwind_reaccumulation":
        return f'''        open_interest_unwinding = dataframe["open_interest_delta_pct_288"] <= -0.75
        short_account_reaccumulation = dataframe["long_short_ratio_zscore_864"] <= -0.75
        price_above_sma = dataframe["sma_distance_bps_144"] >= 0.0
        volume_participation_floor = dataframe["volume_zscore_288"] >= -0.25
        price_resilience = dataframe["close"] > dataframe["rolling_mid"]
        controlled_range = dataframe["range_pct"] <= (dataframe["range_pct_mean"] * 1.6)
        not_overheated = dataframe["rsi"] < self.sell_rsi_exit.value
{ml_filter}
        entry_condition = (
            open_interest_unwinding
            & short_account_reaccumulation
            & price_above_sma
            & volume_participation_floor
            & price_resilience
            & controlled_range
            & not_overheated
            & ml_filter
            & (dataframe["volume"] > 0)
        )'''
    if logic_variant == "calendar_turnover_seasonality":
        return f'''        calendar_risk_window = dataframe["weekday"].isin([0, 3])
        weekend_discount_context = (
            dataframe["weekend_turnover_baseline"] <= dataframe["weekday_turnover_baseline"]
        )
        turnover_recovery = dataframe["calendar_turnover_ratio"] >= (
            dataframe["calendar_turnover_ratio_mean"] * self.buy_volume_factor.value
        )
        positive_calendar_drift = dataframe["calendar_drift"] > 0.0
        midline_hold = dataframe["close"] > dataframe["rolling_mid"]
        controlled_range = dataframe["range_pct"] <= (dataframe["range_pct_mean"] * 1.4)
        not_breakout_chase = dataframe["close"] <= dataframe["rolling_high"].shift(1)
{ml_filter}
        entry_condition = (
            calendar_risk_window
            & weekend_discount_context
            & turnover_recovery
            & positive_calendar_drift
            & midline_hold
            & controlled_range
            & not_breakout_chase
            & ml_filter
            & (dataframe["volume"] > 0)
        )'''
    if logic_variant == "cross_asset_lead_lag":
        return f'''        eth_positive_lead = dataframe["eth_lead_return"] > dataframe["eth_lead_return_mean"]
        btc_lag_discount = dataframe["eth_btc_return_spread"] > 0.0
        spread_not_extreme = dataframe["eth_btc_return_spread"] <= (
            dataframe["eth_btc_spread_mean"] + dataframe["eth_btc_spread_abs_mean"] * 2.0
        )
        btc_resilience = dataframe["close"] > dataframe["rolling_mid"]
        positive_cross_asset_drift = dataframe["cross_asset_drift"] > 0.0
        controlled_range = dataframe["range_pct"] <= (dataframe["range_pct_mean"] * 1.5)
        volume_filter = dataframe["volume"] > (
            dataframe["volume_mean"] * self.buy_volume_factor.value
        )
{ml_filter}
        entry_condition = (
            eth_positive_lead
            & btc_lag_discount
            & spread_not_extreme
            & btc_resilience
            & positive_cross_asset_drift
            & controlled_range
            & volume_filter
            & ml_filter
            & (dataframe["volume"] > 0)
        )'''
    if logic_variant == "cross_asset_cointegration_spread":
        return f'''        btc_discount_to_eth = dataframe["btc_eth_ratio_zscore"] < -0.50
        spread_reversion_turn = dataframe["btc_eth_ratio_zscore_delta"] > 0.0
        eth_market_support = dataframe["eth_regime_drift"] > 0.0
        btc_resilience = dataframe["close"] > dataframe["rolling_low"].shift(1)
        cointegration_spread_not_extreme = dataframe["btc_eth_ratio_zscore"].abs() <= 2.5
        controlled_range = dataframe["range_pct"] <= (dataframe["range_pct_mean"] * 1.5)
        volume_filter = dataframe["volume"] > (
            dataframe["volume_mean"] * self.buy_volume_factor.value
        )
{ml_filter}
        entry_condition = (
            btc_discount_to_eth
            & spread_reversion_turn
            & eth_market_support
            & btc_resilience
            & cointegration_spread_not_extreme
            & controlled_range
            & volume_filter
            & ml_filter
            & (dataframe["volume"] > 0)
        )'''
    if logic_variant == "cross_asset_correlation_recovery":
        return f'''        correlation_breakdown = dataframe["btc_eth_corr_mean"] < 0.35
        correlation_recovery = (
            (dataframe["btc_eth_return_corr"] > dataframe["btc_eth_corr_mean"])
            & (dataframe["btc_eth_corr_delta"] > 0.0)
        )
        btc_relative_recovery = (
            dataframe["btc_eth_relative_return"] > dataframe["btc_eth_relative_return_mean"]
        )
        eth_market_support = dataframe["eth_regime_drift"] > 0.0
        btc_resilience = dataframe["close"] > dataframe["rolling_mid"]
        controlled_range = dataframe["range_pct"] <= (dataframe["range_pct_mean"] * 1.5)
        volume_filter = dataframe["volume"] > (
            dataframe["volume_mean"] * self.buy_volume_factor.value
        )
{ml_filter}
        entry_condition = (
            correlation_breakdown
            & correlation_recovery
            & btc_relative_recovery
            & eth_market_support
            & btc_resilience
            & controlled_range
            & volume_filter
            & ml_filter
            & (dataframe["volume"] > 0)
        )'''
    if logic_variant == "downside_liquidity_shock_reversal":
        return f'''        shock_lookback = int(self.buy_pullback_lookback.value)
        lookback_return = dataframe["close"] / dataframe["close"].shift(shock_lookback) - 1.0
        normalized_atr = (dataframe["atr"] / dataframe["close"]).rolling(
            shock_lookback, min_periods=1
        ).mean()
        downside_shock = lookback_return <= -(normalized_atr * 1.5)
        rsi_washout = (
            dataframe["rsi"].rolling(shock_lookback, min_periods=1).min()
            <= self.buy_rsi_pullback.value
        )
        rsi_recovered = qtpylib.crossed_above(
            dataframe["rsi"], self.buy_rsi_recovery.value
        )
        quiet_volume = dataframe["volume"] < (
            dataframe["volume_mean"] * self.buy_volume_factor.value
        )
        local_low_reclaim = dataframe["close"] > dataframe["rolling_low"].shift(1)
{ml_filter}
        entry_condition = (
            downside_shock
            & rsi_washout
            & rsi_recovered
            & quiet_volume
            & local_low_reclaim
            & ml_filter
            & (dataframe["volume"] > 0)
        )'''
    if logic_variant == "entropy_regime_transition":
        return f'''        low_directional_entropy = dataframe["direction_entropy"] <= (
            dataframe["direction_entropy_baseline"] * 0.85
        )
        efficiency_expanding = dataframe["range_efficiency"] > dataframe["range_efficiency_mean"]
        positive_entropy_drift = dataframe["entropy_drift"] > 0.0
        midline_hold = dataframe["close"] > dataframe["rolling_mid"]
        range_not_extended = dataframe["close"] <= dataframe["rolling_high"].shift(1)
        volume_filter = dataframe["volume"] > (
            dataframe["volume_mean"] * self.buy_volume_factor.value
        )
{ml_filter}
        entry_condition = (
            low_directional_entropy
            & efficiency_expanding
            & positive_entropy_drift
            & midline_hold
            & range_not_extended
            & volume_filter
            & ml_filter
            & (dataframe["volume"] > 0)
        )'''
    if logic_variant == "fractal_long_memory_regime":
        return f'''        persistent_memory_regime = dataframe["hurst_proxy"] > 0.52
        efficient_path = dataframe["fractal_efficiency"] > (
            dataframe["fractal_efficiency_mean"] * 1.05
        )
        positive_fractal_drift = dataframe["fractal_drift"] > 0.0
        midline_hold = dataframe["close"] > dataframe["rolling_mid"]
        not_range_extension = dataframe["close"] <= dataframe["rolling_high"].shift(1)
        volume_filter = dataframe["volume"] > (
            dataframe["volume_mean"] * self.buy_volume_factor.value
        )
{ml_filter}
        entry_condition = (
            persistent_memory_regime
            & efficient_path
            & positive_fractal_drift
            & midline_hold
            & not_range_extension
            & volume_filter
            & ml_filter
            & (dataframe["volume"] > 0)
        )'''
    if logic_variant == "variance_ratio_regime_switch":
        return f'''        variance_ratio_expansion = dataframe["variance_ratio"] >= (
            dataframe["variance_ratio_mean"] * 0.98
        )
        positive_autocorr_regime = (
            (dataframe["return_autocorr"] > 0.0)
            & (dataframe["return_autocorr"] >= dataframe["autocorr_mean"])
        )
        positive_regime_drift = dataframe["regime_drift"] > 0.0
        controlled_regime_return = dataframe["normalized_regime_return"] <= 2.5
        midline_resilience = dataframe["close"] > dataframe["rolling_mid"]
        controlled_range = dataframe["range_pct"] <= (dataframe["range_pct_mean"] * 1.5)
        volume_filter = dataframe["volume"] > (
            dataframe["volume_mean"] * self.buy_volume_factor.value
        )
{ml_filter}
        entry_condition = (
            variance_ratio_expansion
            & positive_autocorr_regime
            & positive_regime_drift
            & controlled_regime_return
            & midline_resilience
            & controlled_range
            & volume_filter
            & ml_filter
            & (dataframe["volume"] > 0)
        )'''
    if logic_variant == "funding_pressure_carry":
        return f'''        negative_funding_pressure = dataframe["funding_pressure"] < 0.0
        funding_pressure_releasing = dataframe["funding_pressure_delta"] > 0.0
        price_resilience = dataframe["close"] > dataframe["rolling_mid"]
        not_positive_crowding = dataframe["funding_rate"] <= (
            dataframe["funding_rate_mean"] + dataframe["funding_rate_abs_mean"] * 0.25
        )
        controlled_range = dataframe["range_pct"] <= (dataframe["range_pct_mean"] * 1.4)
        volume_filter = dataframe["volume"] > (
            dataframe["volume_mean"] * self.buy_volume_factor.value
        )
{ml_filter}
        entry_condition = (
            negative_funding_pressure
            & funding_pressure_releasing
            & price_resilience
            & not_positive_crowding
            & controlled_range
            & volume_filter
            & ml_filter
            & (dataframe["volume"] > 0)
        )'''
    if logic_variant == "market_beta_drawdown_carry":
        return f'''        moderate_drawdown = (
            (dataframe["market_beta_drawdown"] <= -0.005)
            & (dataframe["market_beta_drawdown"] >= -0.055)
        )
        volatility_budget = dataframe["realized_volatility"] <= (
            dataframe["realized_volatility_mean"] * 1.45
        )
        positive_candle_reentry = dataframe["close"] > dataframe["open"]
        beta_resilience = dataframe["close"] > dataframe["rolling_mid"]
        participation_floor = dataframe["volume"] > (
            dataframe["volume_mean"] * self.buy_volume_factor.value
        )
        not_overheated = dataframe["rsi"] < self.sell_rsi_exit.value
{ml_filter}
        entry_condition = (
            moderate_drawdown
            & volatility_budget
            & positive_candle_reentry
            & beta_resilience
            & participation_floor
            & not_overheated
            & ml_filter
            & (dataframe["volume"] > 0)
        )'''
    if logic_variant == "mark_price_dislocation_reclaim":
        return f'''        mark_discount_pressure = dataframe["mark_price_gap"] <= -0.006
        mark_gap_reclaiming = dataframe["mark_price_gap_delta"] > 0.0
        mark_price_support = dataframe["mark_price_trend"] > -0.005
        discount_not_extreme = dataframe["mark_price_gap"] >= -0.035
        price_resilience = dataframe["close"] > dataframe["rolling_mid"]
        controlled_range = dataframe["range_pct"] <= (dataframe["range_pct_mean"] * 1.6)
        participation_floor = dataframe["volume"] > (
            dataframe["volume_mean"] * self.buy_volume_factor.value
        )
{ml_filter}
        entry_condition = (
            mark_discount_pressure
            & mark_gap_reclaiming
            & mark_price_support
            & discount_not_extreme
            & price_resilience
            & controlled_range
            & participation_floor
            & ml_filter
            & (dataframe["volume"] > 0)
        )'''
    if logic_variant == "mark_discount_reclaim_continuation":
        return f'''        mark_discount_pressure = dataframe["mark_price_gap"] <= -0.0005
        six_candle_discount_reclaim = dataframe["mark_price_gap_delta_6"] >= 0.0001
        short_return_nonnegative = dataframe["return_3"] >= 0.0
{ml_filter}
        entry_condition = (
            mark_discount_pressure
            & six_candle_discount_reclaim
            & short_return_nonnegative
            & ml_filter
            & (dataframe["volume"] > 0)
        )'''
    if logic_variant == "mark_fair_value_momentum_lag":
        return f'''        mark_fair_value_momentum = dataframe["mark_price_return_bps"] >= 25.0
        traded_price_lag = dataframe["traded_lag_return_bps"] <= 0.0
        range_budget = dataframe["range_pct"] <= 0.008
        participation_floor = dataframe["volume_zscore"] >= -1.0
{ml_filter}
        raw_entry_condition = (
            mark_fair_value_momentum
            & traded_price_lag
            & range_budget
            & participation_floor
            & ml_filter
            & (dataframe["volume"] > 0)
        )
        event_cooldown = np.zeros(len(dataframe), dtype=bool)
        next_allowed_index = 0
        cooldown_candles = max(1, int(self.buy_pullback_lookback.value))
        for row_index in np.flatnonzero(raw_entry_condition.fillna(False).to_numpy()):
            if int(row_index) < next_allowed_index:
                continue
            event_cooldown[int(row_index)] = True
            next_allowed_index = int(row_index) + cooldown_candles
        dataframe["mark_fair_value_event_cooldown"] = event_cooldown
        entry_condition = dataframe["mark_fair_value_event_cooldown"]'''
    if logic_variant == "microstructure_spread_reversion":
        return f'''        spread_pressure = dataframe["roll_spread_proxy"] > (
            dataframe["roll_spread_mean"] * 1.20
        )
        spread_compressing = dataframe["roll_spread_delta"] < 0.0
        hl_spread_normalizing = dataframe["hl_spread_proxy"] <= (
            dataframe["hl_spread_mean"] * 1.50
        )
        price_resilience = dataframe["close"] > dataframe["rolling_mid"]
        positive_recovery = dataframe["log_return"] > 0.0
        controlled_range = dataframe["range_pct"] <= (dataframe["range_pct_mean"] * 1.8)
        participation_floor = dataframe["volume"] > (
            dataframe["volume_mean"] * self.buy_volume_factor.value
        )
{ml_filter}
        entry_condition = (
            spread_pressure
            & spread_compressing
            & hl_spread_normalizing
            & price_resilience
            & positive_recovery
            & controlled_range
            & participation_floor
            & ml_filter
            & (dataframe["volume"] > 0)
        )'''
    if logic_variant == "regime_state_reentry":
        return f'''        positive_regime_drift = (
            (dataframe["regime_return_fast"] > 0.0)
            & (dataframe["regime_return_slow"] > 0.0)
        )
        state_stability = dataframe["regime_negative_frequency"] <= (
            dataframe["regime_negative_frequency_mean"] * 1.10
        )
        volatility_state_budget = dataframe["regime_volatility"] <= (
            dataframe["regime_volatility_mean"] * 1.60
        )
        trendline_support = dataframe["close"] > dataframe["regime_trendline"]
        closed_candle_reentry = dataframe["close"] > dataframe["open"]
        drawdown_state_intact = dataframe["regime_drawdown"] >= -0.030
        participation_floor = dataframe["volume"] > (
            dataframe["volume_mean"] * self.buy_volume_factor.value
        )
        not_overheated = dataframe["rsi"] < self.sell_rsi_exit.value
{ml_filter}
        entry_condition = (
            positive_regime_drift
            & state_stability
            & volatility_state_budget
            & trendline_support
            & closed_candle_reentry
            & drawdown_state_intact
            & participation_floor
            & not_overheated
            & ml_filter
            & (dataframe["volume"] > 0)
        )'''
    if logic_variant == "realized_skewness_tail_shape":
        return f'''        low_realized_skewness = dataframe["realized_skewness"] < dataframe["realized_skewness_mean"]
        kurtosis_risk_premium = dataframe["realized_kurtosis"] > dataframe["realized_kurtosis_mean"]
        lottery_tail_cooling = dataframe["max_return"] <= (dataframe["max_return_mean"] * 1.10)
        positive_tail_shape_drift = dataframe["tail_shape_drift"] > 0.0
        midline_hold = dataframe["close"] > dataframe["rolling_mid"]
        controlled_range = dataframe["range_pct"] <= (dataframe["range_pct_mean"] * 1.5)
        volume_filter = dataframe["volume"] > (
            dataframe["volume_mean"] * self.buy_volume_factor.value
        )
{ml_filter}
        entry_condition = (
            low_realized_skewness
            & kurtosis_risk_premium
            & lottery_tail_cooling
            & positive_tail_shape_drift
            & midline_hold
            & controlled_range
            & volume_filter
            & ml_filter
            & (dataframe["volume"] > 0)
        )'''
    if logic_variant == "semivariance_asymmetry_regime":
        return f'''        good_volatility_dominance = dataframe["semivariance_balance"] > 0.05
        bad_volatility_decay = dataframe["downside_semivariance"] < (
            dataframe["downside_semivariance_mean"] * 0.95
        )
        positive_semivariance_drift = dataframe["semivariance_drift"] > 0.0
        midline_hold = dataframe["close"] > dataframe["rolling_mid"]
        controlled_range = dataframe["range_pct"] <= (dataframe["range_pct_mean"] * 1.4)
        not_range_extension = dataframe["close"] <= dataframe["rolling_high"].shift(1)
        volume_filter = dataframe["volume"] > (
            dataframe["volume_mean"] * self.buy_volume_factor.value
        )
{ml_filter}
        entry_condition = (
            good_volatility_dominance
            & bad_volatility_decay
            & positive_semivariance_drift
            & midline_hold
            & controlled_range
            & not_range_extension
            & volume_filter
            & ml_filter
            & (dataframe["volume"] > 0)
        )'''
    if logic_variant == "intraday_session_liquidity_reclaim":
        return f'''        session_window = dataframe["hour_utc"].between(13, 20)
        weekday_liquidity = dataframe["weekday"] < 5
        vwap_reclaim = qtpylib.crossed_above(dataframe["close"], dataframe["session_vwap"])
        prior_vwap_discount = dataframe["close"].shift(1) < dataframe["session_vwap"].shift(1)
        volume_filter = dataframe["volume"] > (
            dataframe["volume_mean"] * self.buy_volume_factor.value
        )
        controlled_atr = dataframe["atr"] <= (dataframe["atr_mean"] * 1.5)
{ml_filter}
        entry_condition = (
            session_window
            & weekday_liquidity
            & prior_vwap_discount
            & vwap_reclaim
            & volume_filter
            & controlled_atr
            & ml_filter
            & (dataframe["volume"] > 0)
        )'''
    if logic_variant == "liquidity_recovery_horizon":
        return f'''        recent_liquidity_stress = dataframe["liquidity_stress_recent"] > 0.0
        liquidity_normalizing = dataframe["liquidity_recovery_score"] >= 2.0
        participation_recovered = (
            dataframe["volume_recovery_ratio"] >= self.buy_volume_factor.value
        )
        below_recovery_anchor = dataframe["close"] < dataframe["liquidity_recovery_anchor"]
        recovery_turn = dataframe["close"] > dataframe["close"].shift(1)
        controlled_cost_proxy = (
            dataframe["hl_spread_proxy"] <= dataframe["hl_spread_mean"] * 1.15
        )
        positive_recovery_horizon = dataframe["recovery_horizon_return"] > -0.015
{ml_filter}
        entry_condition = (
            recent_liquidity_stress
            & liquidity_normalizing
            & participation_recovered
            & below_recovery_anchor
            & recovery_turn
            & controlled_cost_proxy
            & positive_recovery_horizon
            & ml_filter
            & (dataframe["volume"] > 0)
        )'''
    if logic_variant == "signed_volume_imbalance_accumulation":
        return f'''        positive_signed_imbalance = dataframe["signed_volume_imbalance"] > 0.18
        close_location_accumulation = dataframe["close_location_mean"] > 0.20
        upper_close_location = dataframe["close_location_value"] > 0.0
        mid_reclaim = qtpylib.crossed_above(dataframe["close"], dataframe["rolling_mid"])
        not_breakout_chase = dataframe["close"] <= dataframe["rolling_high"].shift(1)
        controlled_range = dataframe["range_pct"] <= (dataframe["range_pct_mean"] * 1.6)
        volume_filter = dataframe["volume"] > (
            dataframe["volume_mean"] * self.buy_volume_factor.value
        )
{ml_filter}
        entry_condition = (
            positive_signed_imbalance
            & close_location_accumulation
            & upper_close_location
            & mid_reclaim
            & not_breakout_chase
            & controlled_range
            & volume_filter
            & ml_filter
            & (dataframe["volume"] > 0)
        )'''
    if logic_variant == "directional_change_overshoot":
        return f'''        directional_change_confirmed = (
            (dataframe["directional_change_state"] > 0.0)
            & (dataframe["directional_change_event_age"] >= 1.0)
            & (
                dataframe["directional_change_event_age"]
                <= int(self.sell_timeout_candles.value)
            )
        )
        overshoot_persisted = (
            (dataframe["overshoot_ratio"] >= 1.05)
            & (dataframe["overshoot_ratio"] <= 4.0)
            & (dataframe["overshoot_length"] >= 2.0)
        )
        event_time_trend_positive = dataframe["event_time_trend"] > 0.0
        adverse_reversal_absent = dataframe["adverse_reversal_distance"] >= (
            -dataframe["directional_change_threshold"] * 0.90
        )
        turnover_controlled = (
            (dataframe["turnover_proxy"] >= self.buy_volume_factor.value)
            & (dataframe["turnover_proxy"] <= 4.0)
        )
{ml_filter}
        entry_condition = (
            directional_change_confirmed
            & overshoot_persisted
            & event_time_trend_positive
            & adverse_reversal_absent
            & turnover_controlled
            & ml_filter
            & (dataframe["volume"] > 0)
        )'''
    if logic_variant == "range_quarticity_vol_of_vol_state":
        return f'''        range_quarticity_state_decay = (
            (dataframe["range_stress_recent"] >= 1.05)
            & (dataframe["range_state_decay"] <= 1.05)
            & (dataframe["range_quarticity_delta"] <= 0.25)
        )
        post_stress_stabilization = (
            (dataframe["range_vol_of_vol_state"] <= 1.35)
            & (dataframe["range_pct"] <= dataframe["range_pct_mean"] * 1.60)
        )
        participation_present = (
            dataframe["participation_recovery"] >= self.buy_volume_factor.value
        )
        range_not_reexpanding = (
            dataframe["range_stress_ratio"] <= dataframe["range_stress_recent"] * 0.98
        )
        positive_stabilization_drift = dataframe["stabilization_drift"] > -0.002
        turnover_controlled = (
            (dataframe["turnover_proxy"] >= self.buy_volume_factor.value)
            & (dataframe["turnover_proxy"] <= 4.0)
        )
{ml_filter}
        entry_condition = (
            range_quarticity_state_decay
            & post_stress_stabilization
            & participation_present
            & range_not_reexpanding
            & positive_stabilization_drift
            & turnover_controlled
            & ml_filter
            & (dataframe["volume"] > 0)
        )'''
    if logic_variant == "trend_continuation":
        return f'''        trend_filter = dataframe["ema_fast"] > dataframe["ema_slow"]
        momentum_confirmed = qtpylib.crossed_above(
            dataframe["rsi"], self.buy_rsi_recovery.value
        )
        atr_floor = dataframe["atr"] >= dataframe["atr_mean"]
        volume_filter = dataframe["volume"] > (
            dataframe["volume_mean"] * self.buy_volume_factor.value
        )
{ml_filter}
        entry_condition = (
            trend_filter
            & momentum_confirmed
            & atr_floor
            & volume_filter
            & ml_filter
            & (dataframe["volume"] > 0)
        )'''
    if logic_variant == "volatility_breakout":
        return f'''        prior_high = dataframe["rolling_high"].shift(1)
        breakout_filter = dataframe["close"] > prior_high
        atr_expansion = dataframe["atr"] > dataframe["atr_mean"]
        volume_filter = dataframe["volume"] > (
            dataframe["volume_mean"] * self.buy_volume_factor.value
        )
{ml_filter}
        entry_condition = (
            breakout_filter
            & atr_expansion
            & volume_filter
            & ml_filter
            & (dataframe["volume"] > 0)
        )'''
    return f'''        pullback_seen = (
            dataframe["rsi"].rolling(
                int(self.buy_pullback_lookback.value), min_periods=1
            ).min()
            <= self.buy_rsi_pullback.value
        )
        rsi_recovered = qtpylib.crossed_above(
            dataframe["rsi"], self.buy_rsi_recovery.value
        )
        trend_filter = dataframe["ema_fast"] >= dataframe["ema_slow"]
        volume_filter = dataframe["volume"] > (
            dataframe["volume_mean"] * self.buy_volume_factor.value
        )
{ml_filter}
        entry_condition = (
            pullback_seen
            & rsi_recovered
            & trend_filter
            & volume_filter
            & ml_filter
            & (dataframe["volume"] > 0)
        )'''


def _exit_logic_for_variant(logic_variant: str) -> str:
    if logic_variant == "amihud_illiquidity_premium":
        return '''        illiquidity_spike = dataframe["amihud_illiquidity"] > (
            dataframe["amihud_illiquidity_mean"] * 3.0
        )
        impact_premium_lost = dataframe["amihud_illiquidity"] < (
            dataframe["amihud_illiquidity_mean"] * 0.80
        )
        resilience_lost = dataframe["close"] < dataframe["rolling_mid"]
        rsi_target = dataframe["rsi"] >= self.sell_rsi_exit.value
        exit_condition = (
            (illiquidity_spike | impact_premium_lost | resilience_lost | rsi_target)
            & (dataframe["volume"] > 0)
        )'''
    if logic_variant == "bipower_jump_decay":
        return '''        jump_decay_failed = dataframe["continuous_variance_decay"] > 1.15
        jump_edge_faded = dataframe["jump_variation_ratio"] < 0.85
        drift_failed = dataframe["post_jump_drift"] < -0.002
        rsi_target = dataframe["rsi"] >= self.sell_rsi_exit.value
        exit_condition = (
            (jump_decay_failed | jump_edge_faded | drift_failed | rsi_target)
            & (dataframe["volume"] > 0)
        )'''
    if logic_variant == "crowding_unwind_reaccumulation":
        return '''        open_interest_reexpanded = dataframe["open_interest_delta_pct_288"] >= 0.75
        account_ratio_recrowded = dataframe["long_short_ratio_zscore_864"] >= 0.50
        sma_lost = dataframe["sma_distance_bps_144"] < -25.0
        volume_disappeared = dataframe["volume_zscore_288"] < -1.50
        rsi_target = dataframe["rsi"] >= self.sell_rsi_exit.value
        exit_condition = (
            (
                open_interest_reexpanded
                | account_ratio_recrowded
                | sma_lost
                | volume_disappeared
                | rsi_target
            )
            & (dataframe["volume"] > 0)
        )'''
    if logic_variant == "calendar_turnover_seasonality":
        return '''        calendar_window_expired = ~dataframe["weekday"].isin([0, 3])
        turnover_faded = dataframe["calendar_turnover_ratio"] < (
            dataframe["calendar_turnover_ratio_mean"] * 0.80
        )
        midline_lost = dataframe["close"] < dataframe["rolling_mid"]
        rsi_target = dataframe["rsi"] >= self.sell_rsi_exit.value
        exit_condition = (
            (calendar_window_expired | turnover_faded | midline_lost | rsi_target)
            & (dataframe["volume"] > 0)
        )'''
    if logic_variant == "cross_asset_lead_lag":
        return '''        eth_impulse_faded = dataframe["eth_lead_return"] < dataframe["eth_lead_return_mean"]
        spread_closed = dataframe["eth_btc_return_spread"] <= 0.0
        resilience_lost = dataframe["close"] < dataframe["rolling_mid"]
        rsi_target = dataframe["rsi"] >= self.sell_rsi_exit.value
        exit_condition = (
            (eth_impulse_faded | spread_closed | resilience_lost | rsi_target)
            & (dataframe["volume"] > 0)
        )'''
    if logic_variant == "cross_asset_cointegration_spread":
        return '''        spread_mean_reverted = dataframe["btc_eth_ratio_zscore"] >= -0.05
        spread_reversion_failed = dataframe["btc_eth_ratio_zscore_delta"] < -0.10
        eth_support_lost = dataframe["eth_regime_drift"] <= 0.0
        midline_lost = dataframe["close"] < dataframe["rolling_mid"]
        rsi_target = dataframe["rsi"] >= self.sell_rsi_exit.value
        exit_condition = (
            (
                spread_mean_reverted
                | spread_reversion_failed
                | eth_support_lost
                | midline_lost
                | rsi_target
            )
            & (dataframe["volume"] > 0)
        )'''
    if logic_variant == "cross_asset_correlation_recovery":
        return '''        correlation_recovery_faded = dataframe["btc_eth_corr_delta"] < 0.0
        relative_strength_lost = (
            dataframe["btc_eth_relative_return"] < dataframe["btc_eth_relative_return_mean"]
        )
        eth_support_lost = dataframe["eth_regime_drift"] <= 0.0
        midline_lost = dataframe["close"] < dataframe["rolling_mid"]
        rsi_target = dataframe["rsi"] >= self.sell_rsi_exit.value
        exit_condition = (
            (
                correlation_recovery_faded
                | relative_strength_lost
                | eth_support_lost
                | midline_lost
                | rsi_target
            )
            & (dataframe["volume"] > 0)
        )'''
    if logic_variant == "downside_liquidity_shock_reversal":
        return '''        mean_reversion_target = dataframe["rsi"] >= self.sell_rsi_exit.value
        shock_failure = dataframe["close"] < dataframe["rolling_low"].shift(1)
        exit_condition = (
            (mean_reversion_target | shock_failure)
            & (dataframe["volume"] > 0)
        )'''
    if logic_variant == "directional_change_overshoot":
        return '''        overshoot_failed = dataframe["overshoot_ratio"] < 0.75
        adverse_reversal = dataframe["adverse_reversal_distance"] < (
            -dataframe["directional_change_threshold"] * 1.25
        )
        event_time_trend_negative = dataframe["event_time_trend"] < 0.0
        directional_change_lost = dataframe["directional_change_state"] <= 0.0
        rsi_target = dataframe["rsi"] >= self.sell_rsi_exit.value
        exit_condition = (
            (
                overshoot_failed
                | adverse_reversal
                | event_time_trend_negative
                | directional_change_lost
                | rsi_target
            )
            & (dataframe["volume"] > 0)
        )'''
    if logic_variant == "range_quarticity_vol_of_vol_state":
        return '''        range_quarticity_reexpanding = (
            (dataframe["range_state_decay"] > 1.20)
            | (dataframe["range_quarticity_delta"] > 0.35)
        )
        range_destabilized = dataframe["range_pct"] > dataframe["range_pct_mean"] * 1.90
        stress_reappeared = (
            dataframe["range_stress_ratio"] > dataframe["range_stress_recent"].shift(1) * 1.05
        )
        stabilization_failed = dataframe["stabilization_drift"] < -0.006
        rsi_target = dataframe["rsi"] >= self.sell_rsi_exit.value
        exit_condition = (
            (
                range_quarticity_reexpanding
                | range_destabilized
                | stress_reappeared
                | stabilization_failed
                | rsi_target
            )
            & (dataframe["volume"] > 0)
        )'''
    if logic_variant == "entropy_regime_transition":
        return '''        entropy_reexpanded = dataframe["direction_entropy"] > dataframe["direction_entropy_baseline"]
        efficiency_lost = dataframe["range_efficiency"] < (
            dataframe["range_efficiency_mean"] * 0.75
        )
        midline_lost = dataframe["close"] < dataframe["rolling_mid"]
        rsi_target = dataframe["rsi"] >= self.sell_rsi_exit.value
        exit_condition = (
            (entropy_reexpanded | efficiency_lost | midline_lost | rsi_target)
            & (dataframe["volume"] > 0)
        )'''
    if logic_variant == "fractal_long_memory_regime":
        return '''        memory_decay = dataframe["hurst_proxy"] < 0.48
        efficiency_lost = dataframe["fractal_efficiency"] < (
            dataframe["fractal_efficiency_mean"] * 0.85
        )
        midline_lost = dataframe["close"] < dataframe["rolling_mid"]
        rsi_target = dataframe["rsi"] >= self.sell_rsi_exit.value
        exit_condition = (
            (memory_decay | efficiency_lost | midline_lost | rsi_target)
            & (dataframe["volume"] > 0)
        )'''
    if logic_variant == "variance_ratio_regime_switch":
        return '''        autocorr_faded = dataframe["return_autocorr"] < dataframe["autocorr_mean"]
        variance_ratio_faded = dataframe["variance_ratio"] < dataframe["variance_ratio_mean"]
        drift_lost = dataframe["regime_drift"] <= 0.0
        midline_lost = dataframe["close"] < dataframe["rolling_mid"]
        rsi_target = dataframe["rsi"] >= self.sell_rsi_exit.value
        exit_condition = (
            (autocorr_faded | variance_ratio_faded | drift_lost | midline_lost | rsi_target)
            & (dataframe["volume"] > 0)
        )'''
    if logic_variant == "funding_pressure_carry":
        return '''        funding_turns_expensive = dataframe["funding_rate"] > (
            dataframe["funding_rate_abs_mean"] * 0.25
        )
        funding_pressure_lost = dataframe["funding_pressure_delta"] < 0.0
        resilience_lost = dataframe["close"] < dataframe["rolling_mid"]
        rsi_target = dataframe["rsi"] >= self.sell_rsi_exit.value
        exit_condition = (
            (funding_turns_expensive | funding_pressure_lost | resilience_lost | rsi_target)
            & (dataframe["volume"] > 0)
        )'''
    if logic_variant == "market_beta_drawdown_carry":
        return '''        drawdown_budget_broken = dataframe["market_beta_drawdown"] < -0.060
        volatility_budget_broken = dataframe["realized_volatility"] > (
            dataframe["realized_volatility_mean"] * 1.80
        )
        rsi_target = dataframe["rsi"] >= self.sell_rsi_exit.value
        exit_condition = (
            (drawdown_budget_broken | volatility_budget_broken | rsi_target)
            & (dataframe["volume"] > 0)
        )'''
    if logic_variant == "mark_price_dislocation_reclaim":
        return '''        fair_value_reclaimed = dataframe["mark_price_gap"] >= -0.001
        mark_gap_deteriorated = dataframe["mark_price_gap_delta"] < -0.003
        resilience_lost = dataframe["close"] < dataframe["rolling_mid"]
        rsi_target = dataframe["rsi"] >= self.sell_rsi_exit.value
        exit_condition = (
            (fair_value_reclaimed | mark_gap_deteriorated | resilience_lost | rsi_target)
            & (dataframe["volume"] > 0)
        )'''
    if logic_variant == "mark_discount_reclaim_continuation":
        return '''        discount_reclaimed = dataframe["mark_price_gap"] >= -0.0001
        reclaim_failed = dataframe["mark_price_gap_delta_6"] < 0.0
        short_return_negative = dataframe["return_3"] < 0.0
        rsi_target = dataframe["rsi"] >= self.sell_rsi_exit.value
        exit_condition = (
            (discount_reclaimed | reclaim_failed | short_return_negative | rsi_target)
            & (dataframe["volume"] > 0)
        )'''
    if logic_variant == "mark_fair_value_momentum_lag":
        return '''        mark_momentum_faded = dataframe["mark_price_return_bps"] <= 0.0
        traded_lag_resolved = dataframe["traded_lag_return_bps"] >= 20.0
        range_budget_broken = dataframe["range_pct"] > 0.012
        rsi_target = dataframe["rsi"] >= self.sell_rsi_exit.value
        exit_condition = (
            (mark_momentum_faded | traded_lag_resolved | range_budget_broken | rsi_target)
            & (dataframe["volume"] > 0)
        )'''
    if logic_variant == "microstructure_spread_reversion":
        return '''        spread_normalized = dataframe["roll_spread_proxy"] <= dataframe["roll_spread_mean"]
        spread_reexpanding = dataframe["roll_spread_delta"] > 0.0
        noise_budget_broken = dataframe["microstructure_noise_ratio"] > 2.0
        resilience_lost = dataframe["close"] < dataframe["rolling_mid"]
        rsi_target = dataframe["rsi"] >= self.sell_rsi_exit.value
        exit_condition = (
            (
                spread_normalized
                | spread_reexpanding
                | noise_budget_broken
                | resilience_lost
                | rsi_target
            )
            & (dataframe["volume"] > 0)
        )'''
    if logic_variant == "regime_state_reentry":
        return '''        regime_fast_lost = dataframe["regime_return_fast"] <= 0.0
        trendline_lost = dataframe["close"] < dataframe["rolling_mid"]
        drawdown_state_broken = dataframe["regime_drawdown"] < -0.045
        volatility_state_broken = dataframe["regime_volatility"] > (
            dataframe["regime_volatility_mean"] * 2.00
        )
        rsi_target = dataframe["rsi"] >= self.sell_rsi_exit.value
        exit_condition = (
            (
                regime_fast_lost
                | trendline_lost
                | drawdown_state_broken
                | volatility_state_broken
                | rsi_target
            )
            & (dataframe["volume"] > 0)
        )'''
    if logic_variant == "realized_skewness_tail_shape":
        return '''        lottery_skew_reemerges = dataframe["realized_skewness"] > dataframe["realized_skewness_mean"]
        kurtosis_premium_lost = dataframe["realized_kurtosis"] < (
            dataframe["realized_kurtosis_mean"] * 0.80
        )
        midline_lost = dataframe["close"] < dataframe["rolling_mid"]
        rsi_target = dataframe["rsi"] >= self.sell_rsi_exit.value
        exit_condition = (
            (lottery_skew_reemerges | kurtosis_premium_lost | midline_lost | rsi_target)
            & (dataframe["volume"] > 0)
        )'''
    if logic_variant == "semivariance_asymmetry_regime":
        return '''        bad_volatility_spike = dataframe["downside_semivariance"] > (
            dataframe["downside_semivariance_mean"] * 1.25
        )
        balance_lost = dataframe["semivariance_balance"] < 0.0
        midline_lost = dataframe["close"] < dataframe["rolling_mid"]
        rsi_target = dataframe["rsi"] >= self.sell_rsi_exit.value
        exit_condition = (
            (bad_volatility_spike | balance_lost | midline_lost | rsi_target)
            & (dataframe["volume"] > 0)
        )'''
    if logic_variant == "intraday_session_liquidity_reclaim":
        return '''        vwap_loss = dataframe["close"] < dataframe["session_vwap"]
        session_finished = dataframe["hour_utc"] >= 22
        rsi_target = dataframe["rsi"] >= self.sell_rsi_exit.value
        exit_condition = (
            (vwap_loss | session_finished | rsi_target)
            & (dataframe["volume"] > 0)
        )'''
    if logic_variant == "liquidity_recovery_horizon":
        return '''        recovery_anchor_reached = dataframe["close"] >= dataframe["liquidity_recovery_anchor"]
        stress_reappeared = (
            (dataframe["amihud_illiquidity_ratio"] > 1.35)
            | (dataframe["range_recovery_ratio"] > 1.30)
        )
        participation_lost = dataframe["volume_recovery_ratio"] < 0.70
        rsi_target = dataframe["rsi"] >= self.sell_rsi_exit.value
        exit_condition = (
            (recovery_anchor_reached | stress_reappeared | participation_lost | rsi_target)
            & (dataframe["volume"] > 0)
        )'''
    if logic_variant == "signed_volume_imbalance_accumulation":
        return '''        imbalance_faded = dataframe["signed_volume_imbalance"] < 0.0
        close_location_failed = dataframe["close_location_value"] < -0.20
        mid_loss = dataframe["close"] < dataframe["rolling_mid"]
        rsi_target = dataframe["rsi"] >= self.sell_rsi_exit.value
        exit_condition = (
            (imbalance_faded | close_location_failed | mid_loss | rsi_target)
            & (dataframe["volume"] > 0)
        )'''
    if logic_variant == "trend_continuation":
        return '''        trend_exhaustion = dataframe["ema_fast"] < dataframe["ema_slow"]
        rsi_target = dataframe["rsi"] >= self.sell_rsi_exit.value
        exit_condition = (
            (trend_exhaustion | rsi_target)
            & (dataframe["volume"] > 0)
        )'''
    if logic_variant == "volatility_breakout":
        return '''        prior_low = dataframe["rolling_low"].shift(1)
        breakout_failure = dataframe["close"] < prior_low
        mean_reversion_target = dataframe["rsi"] >= self.sell_rsi_exit.value
        exit_condition = (
            (breakout_failure | mean_reversion_target)
            & (dataframe["volume"] > 0)
        )'''
    return '''        mean_reversion_target = dataframe["rsi"] >= self.sell_rsi_exit.value
        momentum_failure = dataframe["ema_fast"] < dataframe["ema_slow"]
        exit_condition = (
            (mean_reversion_target | momentum_failure)
            & (dataframe["volume"] > 0)
        )'''


def _metadata_schema_checks(proposal_metadata: dict[str, Any]) -> list[StrategyCodeCheck]:
    checks: list[StrategyCodeCheck] = []
    for field_name in REQUIRED_PROPOSAL_METADATA_FIELDS:
        checks.append(
            _check(
                f"proposal_metadata_{_safe_label(field_name)}_present",
                field_name in proposal_metadata,
                "blocker",
                f"Proposal metadata must include {field_name}.",
            )
        )
    checks.append(
        _check(
            "proposal_metadata_factory_valid",
            proposal_metadata.get("factory") == "strategy_proposal_generator",
            "blocker",
            "Proposal metadata must come from the strategy proposal generator.",
            {"factory": proposal_metadata.get("factory")},
        )
    )
    checks.append(
        _check(
            "proposal_metadata_phase_valid",
            proposal_metadata.get("phase") == "strategy_generation",
            "blocker",
            "Proposal metadata must belong to the strategy_generation phase.",
            {"phase": proposal_metadata.get("phase")},
        )
    )
    checks.append(
        _check(
            "proposal_metadata_has_no_blockers",
            not proposal_metadata.get("blockers"),
            "blocker",
            "Accepted proposal metadata must not include blockers.",
        )
    )
    return checks


def _candidate_scope_checks(
    candidate_id: str, class_name: str, output_dir: Path, root_dir: Path
) -> list[StrategyCodeCheck]:
    return [
        _check(
            "candidate_id_safe",
            bool(re.fullmatch(r"[A-Za-z0-9_.-]+", candidate_id)),
            "blocker",
            "Candidate ID must be a simple path-safe token.",
            {"candidate_id": candidate_id},
        ),
        _check(
            "strategy_class_name_valid",
            _is_valid_class_name(class_name),
            "blocker",
            "Strategy name from proposal metadata must be a valid Python class name.",
            {"strategy_name": class_name},
        ),
        _check(
            "generated_output_dir_within_workspace",
            _path_is_within_root(output_dir, root_dir),
            "blocker",
            "Generated strategy output directory must resolve inside the repository workspace.",
            {"path": _safe_relative_path(output_dir, root_dir)},
        ),
    ]


def _proposal_file_checks(proposal_path: Path, root_dir: Path) -> list[StrategyCodeCheck]:
    within_workspace = _path_is_within_root(proposal_path, root_dir)
    return [
        _check(
            "proposal_path_within_workspace",
            within_workspace,
            "blocker",
            "Source proposal path must resolve inside the repository workspace.",
            {"path": _safe_relative_path(proposal_path, root_dir)},
        ),
        _check(
            "proposal_file_present",
            within_workspace and proposal_path.is_file(),
            "blocker",
            "Source proposal Markdown file must exist.",
            {"path": _safe_relative_path(proposal_path, root_dir)},
        ),
    ]


def _proposal_hash_checks(
    proposal_markdown: str, proposal_metadata: dict[str, Any]
) -> list[StrategyCodeCheck]:
    actual_hash = _sha256_text(proposal_markdown)
    expected_hash = str(proposal_metadata.get("proposal_content_hash", ""))
    return [
        _check(
            "proposal_content_hash_matches",
            actual_hash == expected_hash,
            "blocker",
            "Source proposal Markdown content hash must match proposal metadata.",
            {"expected": expected_hash, "actual": actual_hash},
        )
    ]


def _required_section_checks(proposal_markdown: str) -> list[StrategyCodeCheck]:
    return [
        _check(
            f"proposal_markdown_section_{_safe_label(section)}_present",
            f"## {section}" in proposal_markdown,
            "blocker",
            f"Source proposal Markdown must include the {section} section.",
        )
        for section in REQUIRED_PROPOSAL_SECTIONS
    ]


def _proposal_status_checks(proposal_metadata: dict[str, Any]) -> list[StrategyCodeCheck]:
    return [
        _check(
            "proposal_status_accepted",
            proposal_metadata.get("status") == "accepted"
            and proposal_metadata.get("proposal_status") == "accepted",
            "blocker",
            "Proposal metadata status must be accepted.",
            {
                "status": proposal_metadata.get("status"),
                "proposal_status": proposal_metadata.get("proposal_status"),
            },
        ),
        _check(
            "proposal_code_generation_eligible",
            proposal_metadata.get("code_generation_eligible") is True,
            "blocker",
            "Proposal metadata must explicitly be code-generation eligible.",
            {"code_generation_eligible": proposal_metadata.get("code_generation_eligible")},
        ),
        _check(
            "proposal_long_short_long_only",
            _normalizes_to_long_only(str(proposal_metadata.get("long_short", ""))),
            "blocker",
            "Proposal long_short scope must be long-only.",
            {"long_short": proposal_metadata.get("long_short")},
        ),
    ]


def _proposal_safety_scope_checks(
    proposal_metadata: dict[str, Any]
) -> list[StrategyCodeCheck]:
    safety_scope = proposal_metadata.get("safety_scope", {})
    if not isinstance(safety_scope, dict):
        safety_scope = {}
    checks = [
        _safety_flag_check(safety_scope, "long_only", True),
        _safety_flag_check(safety_scope, "historical_evaluation_only", True),
        _safety_flag_check(safety_scope, "live_data", False),
        _safety_flag_check(safety_scope, "live_trading", False),
        _safety_flag_check(safety_scope, "paper_trading_started", False),
        _safety_flag_check(safety_scope, "dry_run_trading_started", False),
        _safety_flag_check(safety_scope, "exchange_order_placement", False),
        _safety_flag_check(safety_scope, "uses_api_keys_or_secrets", False),
        _safety_flag_check(safety_scope, "metadata_contains_secrets", False),
        _safety_flag_check(safety_scope, "leverage_above_one", False),
        _safety_flag_check(safety_scope, "shorting", False),
        _safety_flag_check(safety_scope, "process_control", False),
        _safety_flag_check(safety_scope, "backtest_started", False),
        _safety_flag_check(safety_scope, "local_artifacts_source_of_truth", True),
    ]
    checks.append(
        _check(
            "proposal_safety_leverage_capped_at_one",
            _as_float(safety_scope.get("leverage"), default=99.0) <= 1.0,
            "blocker",
            "Proposal safety scope leverage must be capped at 1.0.",
            {"leverage": safety_scope.get("leverage")},
        )
    )
    return checks


def _safety_flag_check(
    safety_scope: dict[str, Any], key: str, expected: bool
) -> StrategyCodeCheck:
    return _check(
        f"proposal_safety_{_safe_label(key)}_{str(expected).lower()}",
        safety_scope.get(key) is expected,
        "blocker",
        f"Proposal safety scope must record {key}={expected}.",
        {key: safety_scope.get(key)},
    )


def _generated_code_safety_checks(strategy_code: str) -> list[StrategyCodeCheck]:
    checks = [
        _check(
            "generated_code_can_short_false",
            re.search(r"\bcan_short\s*=\s*False\b", strategy_code) is not None,
            "blocker",
            "Generated strategy must explicitly set can_short = False.",
        ),
        _check(
            "generated_code_parameters_configurable",
            all(name in strategy_code for name in DEFAULT_PARAMETER_DEFAULTS),
            "blocker",
            "Generated strategy parameters must be explicit and configurable.",
        ),
        _check(
            "generated_code_freqtrade_hyperopt_disabled",
            "optimize=True" not in strategy_code and "optimize=False" in strategy_code,
            "blocker",
            "Generated strategy must keep Freqtrade hyperopt disabled for theory-fixed parameters.",
        ),
    ]
    for name, pattern, message in _GENERATED_FORBIDDEN_PATTERNS:
        if name == "generated_code_no_shift_minus_one":
            checks.append(
                _check(
                    name,
                    _negative_shifts_only_in_freqai_targets(strategy_code),
                    "blocker",
                    "Generated strategy may only use negative shifts inside set_freqai_targets.",
                )
            )
            continue
        checks.append(
            _check(
                name,
                pattern.search(strategy_code) is None,
                "blocker",
                message,
            )
        )
    return checks


def _negative_shifts_only_in_freqai_targets(strategy_code: str) -> bool:
    pattern = re.compile(r"\.shift\s*\(\s*(?:periods\s*=\s*)?-\d+")
    matches = list(pattern.finditer(strategy_code))
    if not matches:
        return True
    target_start = strategy_code.find("def set_freqai_targets(")
    if target_start == -1:
        return False
    next_method = strategy_code.find("\n    def ", target_start + 1)
    target_end = next_method if next_method != -1 else len(strategy_code)
    return all(target_start <= match.start() < target_end for match in matches)


def _load_json_object_check(
    path: Path, should_load: bool
) -> tuple[dict[str, Any], StrategyCodeCheck]:
    if not should_load:
        return {}, _check(
            "proposal_metadata_json_parseable",
            False,
            "blocker",
            "Proposal metadata JSON must be parseable.",
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {}, _check(
            "proposal_metadata_json_parseable",
            False,
            "blocker",
            "Proposal metadata JSON must be parseable.",
            {"error": str(exc)},
        )
    return (
        payload if isinstance(payload, dict) else {},
        _check(
            "proposal_metadata_json_parseable",
            isinstance(payload, dict),
            "blocker",
            "Proposal metadata JSON must contain an object.",
        ),
    )


def _proposal_path_from_metadata(
    proposal_metadata: dict[str, Any], root_dir: Path
) -> Path | None:
    value = proposal_metadata.get("proposal_path")
    if not isinstance(value, str) or not value.strip():
        return None
    return _resolve_workspace_path(Path(value), root_dir)


def _generated_output_dir(
    *,
    root_dir: Path,
    output_root: Path,
    strategy_name: str,
    candidate_id: str,
) -> Path:
    output_root_path = _resolve_workspace_path(output_root, root_dir)
    return output_root_path / _safe_filename(strategy_name) / candidate_id


def _candidate_id(inputs: StrategyCodeInputs, created_at: str) -> str:
    if inputs.candidate_id:
        return _safe_path_token(inputs.candidate_id)
    return _timestamp_slug(created_at)


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
    return token or "strategy"


def _safe_label(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_").lower()
    return token or "field"


def _string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _proposal_blocked_next_actions(proposal_metadata: dict[str, Any]) -> list[str]:
    actions: list[str] = []
    sources: list[Any] = [proposal_metadata.get("research_brief", {})]
    for key in ("failure_synthesis_constraints", "research_decision_constraints"):
        sources.extend(proposal_metadata.get(key, []) or [])
    for source in sources:
        if not isinstance(source, dict):
            continue
        for action in _string_list(source.get("blocked_next_actions", [])):
            if action not in actions:
                actions.append(action)
    return actions


def _proposal_research_handoff_summaries(
    proposal_metadata: dict[str, Any]
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    seen: set[str] = set()
    sources: list[Any] = [proposal_metadata.get("research_brief", {})]
    sources.extend(proposal_metadata.get("research_decision_constraints", []) or [])
    for source in sources:
        if not isinstance(source, dict):
            continue
        for raw in source.get("research_handoff_summaries", []) or []:
            if not isinstance(raw, dict):
                continue
            copied = _copy_jsonish(raw)
            key = json.dumps(copied, sort_keys=True, ensure_ascii=False)
            if key not in seen:
                seen.add(key)
                summaries.append(copied)
    return summaries


def _copy_jsonish(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _copy_jsonish(nested) for key, nested in value.items()}
    if isinstance(value, list):
        return [_copy_jsonish(item) for item in value]
    return value


def _safe_path_token(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._-")
    return token or "candidate"


def _is_valid_class_name(value: str) -> bool:
    return value.isidentifier() and not keyword.iskeyword(value)


def _normalizes_to_long_only(value: str) -> bool:
    normalized = re.sub(r"[^a-z]+", "-", value.strip().lower()).strip("-")
    return normalized in {"long", "long-only", "longonly"}


def _sanitize_command_token(token: str) -> str:
    text = str(token)
    text = re.sub(
        r"""(?ix)
        (?P<label>api[_-]?key|secret|password|token|jwt)
        (?P<sep>\s*[:=]\s*)
        (?P<quote>["'])?
        (?P<value>[A-Za-z0-9_./+=:-]{8,})
        (?P=quote)?
        """,
        lambda match: f"{match.group('label')}{match.group('sep')}[REDACTED]",
        text,
    )
    return re.sub(
        r"(?i)(\$\{[^}]*?(KEY|SECRET|TOKEN|PASSWORD|PASSWD|JWT)[^}]*?\}|"
        r"%[A-Z_]*(KEY|SECRET|TOKEN|PASSWORD|PASSWD|JWT)[A-Z0-9_]*%)",
        "[REDACTED_ENV]",
        text,
    )


def _check(
    name: str,
    passed: bool,
    severity: str,
    message: str,
    details: dict[str, Any] | None = None,
) -> StrategyCodeCheck:
    return StrategyCodeCheck(
        name=name,
        status="pass" if passed else "blocked",
        severity=severity,
        message=message,
        details=details or {},
    )


def _has_blockers(checks: Sequence[StrategyCodeCheck]) -> bool:
    return any(check.status == "blocked" for check in checks)


def _as_float(value: Any, *, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


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
