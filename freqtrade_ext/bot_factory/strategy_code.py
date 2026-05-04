from __future__ import annotations

import hashlib
import json
import keyword
import re
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

from freqtrade_ext.bot_factory.safety import scan_paths
from freqtrade_ext.bot_factory.strategy_proposals import REQUIRED_PROPOSAL_SECTIONS


STRATEGY_CODE_GENERATOR_VERSION = "strategy_code_generator_v2"
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
    )


def write_strategy_code_artifacts(artifacts: StrategyCodeArtifacts) -> None:
    artifacts.metadata_path.parent.mkdir(parents=True, exist_ok=True)
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
    return {
        "generated_at": created_at,
        "phase": "strategy_generation",
        "factory": "strategy_code_generator",
        "generator_version": STRATEGY_CODE_GENERATOR_VERSION,
        "status": "blocked" if blockers else "pending_static_check",
        "strategy_code_generated": strategy_code is not None,
        "candidate_evaluation_eligible": False,
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
        "generator_mode": _generator_mode_from_proposal(proposal_metadata),
        "feature_list": list(proposal_metadata.get("feature_list", [])),
        "target_definition": proposal_metadata.get("target_definition"),
        "label_horizon": proposal_metadata.get("label_horizon"),
        "prediction_threshold": proposal_metadata.get("prediction_threshold"),
        "rule_filters": list(proposal_metadata.get("rule_filters", [])),
        "risk_policy": proposal_metadata.get("risk_policy"),
        "parameter_defaults": dict(DEFAULT_PARAMETER_DEFAULTS),
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
            "candidate_evaluation_started": False,
            "candidate_ranking_started": False,
            "paper_promotion_eligible": False,
            "promotion_authorized_by_this_command": False,
            "local_artifacts_source_of_truth": True,
        },
        "command": [_sanitize_command_token(token) for token in inputs.command],
        "notice": STRATEGY_CODE_NOTICE,
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
    label_horizon = int(proposal_metadata.get("label_horizon") or 12)
    target_name = str(proposal_metadata.get("target_definition") or "future_return")
    threshold = float(proposal_metadata.get("prediction_threshold") or 0.0)
    if generator_mode in {"freqai", "hybrid_ml"}:
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

import talib.abstract as ta
from pandas import DataFrame
from technical import qtpylib

from freqtrade.strategy import DecimalParameter, IStrategy, IntParameter, timeframe_to_minutes


class {strategy_name}(IStrategy):
    """
    Generated Bot Factory long-only strategy.

    Candidate ID: {candidate_id}
    Source proposal hash: {source_proposal_hash}
    Generator mode: {generator_mode}
    """

    INTERFACE_VERSION = 3

    can_short = False
    timeframe = {json.dumps(timeframe)}
    process_only_new_candles = True
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False
    startup_candle_count: int = 120

    minimal_roi = {{"0": 0.03, "120": 0.01, "360": 0.0}}
    stoploss = -0.05
    trailing_stop = False

    buy_rsi_window = IntParameter(8, 30, default=14, space="buy", optimize=True, load=True)
    buy_pullback_lookback = IntParameter(2, 12, default=5, space="buy", optimize=True, load=True)
    buy_rsi_pullback = IntParameter(20, 45, default=32, space="buy", optimize=True, load=True)
    buy_rsi_recovery = IntParameter(35, 55, default=42, space="buy", optimize=True, load=True)
    buy_ema_fast = IntParameter(8, 24, default=12, space="buy", optimize=True, load=True)
    buy_ema_slow = IntParameter(32, 96, default=48, space="buy", optimize=True, load=True)
    buy_volume_window = IntParameter(12, 60, default=24, space="buy", optimize=True, load=True)
    buy_volume_factor = DecimalParameter(
        0.80, 2.00, decimals=2, default=1.00, space="buy", optimize=True, load=True
    )
    sell_rsi_exit = IntParameter(55, 80, default=65, space="sell", optimize=True, load=True)
    sell_timeout_candles = IntParameter(
        24, 288, default=96, space="sell", optimize=True, load=True
    )

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=int(self.buy_rsi_window.value))
        dataframe["ema_fast"] = ta.EMA(dataframe, timeperiod=int(self.buy_ema_fast.value))
        dataframe["ema_slow"] = ta.EMA(dataframe, timeperiod=int(self.buy_ema_slow.value))
        dataframe["volume_mean"] = dataframe["volume"].rolling(
            int(self.buy_volume_window.value), min_periods=1
        ).mean()
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=14)
        return dataframe
{freqai_block}

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        pullback_seen = (
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
        ml_filter = True
        if "{generator_mode}" in ("freqai", "hybrid_ml"):
            ml_filter = dataframe.get("&-{proposal_metadata.get("target_definition") or "future_return"}", 0) > {threshold}
        entry_condition = (
            pullback_seen
            & rsi_recovered
            & trend_filter
            & volume_filter
            & ml_filter
            & (dataframe["volume"] > 0)
        )
        dataframe.loc[entry_condition, ["enter_long", "enter_tag"]] = (
            1,
            "rsi_pullback_recovery",
        )
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        mean_reversion_target = dataframe["rsi"] >= self.sell_rsi_exit.value
        momentum_failure = dataframe["ema_fast"] < dataframe["ema_slow"]
        exit_condition = (
            (mean_reversion_target | momentum_failure)
            & (dataframe["volume"] > 0)
        )
        dataframe.loc[exit_condition, ["exit_long", "exit_tag"]] = (
            1,
            "mean_reversion_or_momentum_failure",
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
    ]
    for name, pattern, message in _GENERATED_FORBIDDEN_PATTERNS:
        checks.append(
            _check(
                name,
                pattern.search(strategy_code) is None,
                "blocker",
                message,
            )
        )
    return checks


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
