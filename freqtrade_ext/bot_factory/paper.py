from __future__ import annotations

import ast
import csv
import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

from freqtrade_ext.bot_factory.backtest_results import BacktestMetrics, evaluate_initial_gate
from freqtrade_ext.bot_factory.safety import SafetyReport
from freqtrade_ext.bot_factory.state_conditioning import (
    validate_state_conditioned_scorecard_for_selector,
)
from freqtrade_ext.bot_factory.strategy_suitability import (
    validate_strategy_suitability_matrix_for_selector,
)


READINESS_NOTICE = (
    "Paper readiness is a no-startup preflight. It does not start paper trading, "
    "dry-run trading, live trading, or any bot process."
)

_CREDENTIAL_KEY_RE = re.compile(
    r"(?i)(^key$|api[_-]?key|secret|password|passwd|token|uid|jwt|credential|chat_id)"
)
_PRIVATE_ENV_RE = re.compile(r"(?i)(\$\{?[A-Z_][A-Z0-9_]*\}?|env:|%[A-Z_][A-Z0-9_]*%)")

MAX_PAPER_MAX_OPEN_TRADES = 3
MAX_PAPER_STAKE_AMOUNT = 1_000.0
MAX_PAPER_DRY_RUN_WALLET = 10_000.0
SAFE_INITIAL_STATE = "stopped"


@dataclass(frozen=True)
class PaperReadinessInputs:
    root_dir: Path
    strategy: str
    run_id: str
    config_path: Path
    strategy_path: Path
    historical_dir: Path
    walk_forward_dir: Path
    training_dir: Path
    regime_scorecard_path: Path | None = None
    requires_regime_scorecard: bool = False
    market_state_scorecard_path: Path | None = None
    requires_market_state_scorecard: bool = False
    strategy_suitability_matrix_path: Path | None = None
    requires_strategy_suitability_matrix: bool = False
    output_root: Path = Path("data/paper_readiness")
    reviewer_notes: Sequence[str] = field(default_factory=list)
    command: Sequence[str] = field(default_factory=list)

    @property
    def output_dir(self) -> Path:
        return self.output_root / self.strategy / self.run_id


@dataclass(frozen=True)
class ReadinessCheck:
    name: str
    status: str
    severity: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StrategySafetyResult:
    ok: bool
    checks: list[ReadinessCheck]
    source_path: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "source_path": self.source_path,
            "checks": [check.to_dict() for check in self.checks],
        }


@dataclass(frozen=True)
class ConfigSafetyResult:
    ok: bool
    checks: list[ReadinessCheck]
    sanitized_summary: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "checks": [check.to_dict() for check in self.checks],
            "sanitized_summary": self.sanitized_summary,
            "metadata_contains_secrets": False,
        }


def load_json_file(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON file must contain an object: {path}")
    return payload


def build_candidate_artifacts(inputs: PaperReadinessInputs) -> dict[str, Any]:
    files = {
        "historical_metrics": inputs.historical_dir / "metrics.json",
        "historical_report": inputs.historical_dir / "report.md",
        "historical_metadata": inputs.historical_dir / "freqai_metadata.json",
        "historical_trades": inputs.historical_dir / "trades.csv",
        "walk_forward_metrics": inputs.walk_forward_dir / "walk_forward_metrics.json",
        "walk_forward_report": inputs.walk_forward_dir / "walk_forward_report.md",
        "training_manifest": inputs.training_dir / "training_manifest.json",
        "training_report": inputs.training_dir / "training_report.md",
    }
    if inputs.regime_scorecard_path is not None:
        files["regime_scorecard"] = inputs.regime_scorecard_path
    if inputs.market_state_scorecard_path is not None:
        files["market_state_scorecard"] = inputs.market_state_scorecard_path
    if inputs.strategy_suitability_matrix_path is not None:
        files["strategy_suitability_matrix"] = inputs.strategy_suitability_matrix_path
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "strategy": inputs.strategy,
        "run_id": inputs.run_id,
        "artifacts": {
            name: _artifact_info(path, inputs.root_dir) for name, path in files.items()
        },
    }


def evaluate_config_safety(config: dict[str, Any], *, strategy: str) -> ConfigSafetyResult:
    checks: list[ReadinessCheck] = []

    dry_run = config.get("dry_run")
    checks.append(
        _check(
            "dry_run_true",
            dry_run is True,
            "blocker",
            "Config must set dry_run=true for any future paper path.",
            {"present": "dry_run" in config},
        )
    )

    configured_strategy = config.get("strategy")
    checks.append(
        _check(
            "strategy_matches_candidate",
            configured_strategy == strategy,
            "blocker",
            "Config strategy must match the readiness candidate.",
            {"configured_strategy": _safe_scalar(configured_strategy), "candidate": strategy},
        )
    )

    timeframe = config.get("timeframe")
    checks.append(
        _check(
            "timeframe_explicit",
            bool(timeframe),
            "blocker",
            "Config must define an explicit timeframe.",
            {"present": bool(timeframe)},
        )
    )

    max_open_trades = config.get("max_open_trades")
    checks.append(
        _check(
            "max_open_trades_explicit",
            _positive_number(max_open_trades),
            "blocker",
            "Config must define a positive max_open_trades limit.",
            {"present": "max_open_trades" in config},
        )
    )
    checks.append(
        _check(
            "max_open_trades_conservative",
            _positive_integer(max_open_trades)
            and float(max_open_trades) <= MAX_PAPER_MAX_OPEN_TRADES,
            "blocker",
            "Config max_open_trades must stay within accepted simulation limits.",
            {
                "configured": _safe_scalar(max_open_trades),
                "maximum": MAX_PAPER_MAX_OPEN_TRADES,
            },
        )
    )

    stake_amount = config.get("stake_amount")
    checks.append(
        _check(
            "stake_amount_capped",
            _positive_number(stake_amount),
            "blocker",
            "Config must define a positive numeric stake_amount cap.",
            {"present": "stake_amount" in config, "numeric": _is_number(stake_amount)},
        )
    )
    checks.append(
        _check(
            "stake_amount_conservative",
            _positive_number(stake_amount)
            and float(stake_amount) <= MAX_PAPER_STAKE_AMOUNT,
            "blocker",
            "Config stake_amount must stay within accepted simulation limits.",
            {
                "configured": _safe_scalar(stake_amount),
                "maximum": MAX_PAPER_STAKE_AMOUNT,
            },
        )
    )

    dry_run_wallet = config.get("dry_run_wallet")
    checks.append(
        _check(
            "dry_run_wallet_conservative",
            _positive_number(dry_run_wallet)
            and float(dry_run_wallet) <= MAX_PAPER_DRY_RUN_WALLET,
            "blocker",
            "Config must define a conservative positive dry_run_wallet.",
            {
                "present": "dry_run_wallet" in config,
                "configured": _safe_scalar(dry_run_wallet),
                "maximum": MAX_PAPER_DRY_RUN_WALLET,
            },
        )
    )
    checks.append(
        _check(
            "stake_amount_within_dry_run_wallet",
            _positive_number(stake_amount)
            and _positive_number(dry_run_wallet)
            and float(stake_amount) <= float(dry_run_wallet),
            "blocker",
            "Config stake_amount must not exceed dry_run_wallet.",
            {
                "stake_amount": _safe_scalar(stake_amount),
                "dry_run_wallet": _safe_scalar(dry_run_wallet),
            },
        )
    )

    pairs = _pair_allowlist(config)
    checks.append(
        _check(
            "pair_allowlist_explicit",
            bool(pairs),
            "blocker",
            "Config must define an explicit non-empty exchange.pair_whitelist.",
            {"pair_count": len(pairs)},
        )
    )

    api_server = config.get("api_server", {})
    api_server_enabled = isinstance(api_server, dict) and api_server.get("enabled") is True
    checks.append(
        _check(
            "api_server_disabled",
            not api_server_enabled,
            "blocker",
            "Remote API server must not be enabled for this no-startup readiness layer.",
            {"api_server_present": isinstance(api_server, dict) and bool(api_server)},
        )
    )

    checks.append(
        _check(
            "force_entry_disabled",
            config.get("force_entry_enable") is not True,
            "blocker",
            "Config must not enable force_entry_enable for paper readiness.",
            {
                "present": "force_entry_enable" in config,
                "configured": bool(config.get("force_entry_enable")),
            },
        )
    )

    initial_state = config.get("initial_state")
    checks.append(
        _check(
            "initial_state_stopped",
            initial_state == SAFE_INITIAL_STATE,
            "blocker",
            "Config initial_state must be explicitly stopped.",
            {
                "present": "initial_state" in config,
                "configured": _safe_scalar(initial_state),
                "accepted": SAFE_INITIAL_STATE,
            },
        )
    )

    cancel_on_exit = config.get("cancel_open_orders_on_exit")
    checks.append(
        _check(
            "cancel_open_orders_on_exit_explicit",
            isinstance(cancel_on_exit, bool),
            "blocker",
            "Config must explicitly set cancel_open_orders_on_exit.",
            {
                "present": "cancel_open_orders_on_exit" in config,
                "configured": _safe_scalar(cancel_on_exit),
            },
        )
    )

    credential_paths = [
        finding for finding in _credential_findings(config) if finding["has_value"]
    ]
    checks.append(
        _check(
            "no_credential_values",
            not credential_paths,
            "blocker",
            "Config must not contain non-empty API keys, secrets, tokens, UIDs, or passwords.",
            {"credential_key_paths": [finding["path"] for finding in credential_paths]},
        )
    )

    private_env_paths = _private_env_findings(config)
    checks.append(
        _check(
            "no_private_env_values",
            not private_env_paths,
            "blocker",
            "Config must not contain private environment variable references.",
            {"env_reference_paths": private_env_paths},
        )
    )

    leverage_paths = _leverage_findings(config)
    checks.append(
        _check(
            "no_leverage_above_one",
            not leverage_paths,
            "blocker",
            "Config must not set leverage above 1.0.",
            {"leverage_key_paths": [finding["path"] for finding in leverage_paths]},
        )
    )

    endpoint_paths = _endpoint_findings(config)
    checks.append(
        _check(
            "no_order_endpoint_overrides",
            not endpoint_paths,
            "blocker",
            "Config must not include private or order endpoint overrides.",
            {"endpoint_key_paths": endpoint_paths},
        )
    )

    summary = {
        "dry_run": dry_run is True,
        "strategy": _safe_scalar(configured_strategy),
        "timeframe": _safe_scalar(timeframe),
        "exchange_name": _safe_scalar(_nested_get(config, ["exchange", "name"])),
        "pair_count": len(pairs),
        "max_open_trades": _safe_scalar(max_open_trades),
        "stake_currency": _safe_scalar(config.get("stake_currency")),
        "stake_amount": _safe_scalar(stake_amount),
        "dry_run_wallet": _safe_scalar(dry_run_wallet),
        "force_entry_enable": bool(config.get("force_entry_enable")),
        "initial_state": _safe_scalar(initial_state),
        "cancel_open_orders_on_exit": _safe_scalar(cancel_on_exit),
        "api_server_present": isinstance(api_server, dict) and bool(api_server),
        "api_server_enabled": api_server_enabled,
        "accepted_simulation_limits": {
            "max_open_trades_maximum": MAX_PAPER_MAX_OPEN_TRADES,
            "stake_amount_maximum": MAX_PAPER_STAKE_AMOUNT,
            "dry_run_wallet_maximum": MAX_PAPER_DRY_RUN_WALLET,
            "required_initial_state": SAFE_INITIAL_STATE,
        },
    }
    return ConfigSafetyResult(
        ok=all(check.status == "pass" for check in checks),
        checks=checks,
        sanitized_summary=summary,
    )


def evaluate_strategy_long_only(strategy_file: Path, strategy: str) -> StrategySafetyResult:
    text = strategy_file.read_text(encoding="utf-8")
    tree = ast.parse(text, filename=str(strategy_file))
    class_node = _find_class(tree, strategy)
    checks: list[ReadinessCheck] = []

    if class_node is None:
        checks.append(
            ReadinessCheck(
                name="strategy_class_found",
                status="blocked",
                severity="blocker",
                message=f"Strategy class was not found: {strategy}",
            )
        )
        return StrategySafetyResult(ok=False, checks=checks, source_path=str(strategy_file))

    can_short_value = _class_assignment_constant(class_node, "can_short")
    checks.append(
        _check(
            "can_short_false",
            can_short_value is False,
            "blocker",
            "Strategy must explicitly set can_short = False.",
            {"explicit_value": can_short_value if isinstance(can_short_value, bool) else None},
        )
    )

    short_signal_lines = _short_signal_lines(class_node)
    checks.append(
        _check(
            "no_short_signals",
            not short_signal_lines,
            "blocker",
            "Strategy must not reference enter_short or exit_short signals.",
            {"lines": short_signal_lines},
        )
    )

    leverage_checks = _leverage_hook_checks(class_node)
    checks.extend(leverage_checks)

    return StrategySafetyResult(
        ok=all(check.status == "pass" for check in checks),
        checks=checks,
        source_path=str(strategy_file),
    )


def evaluate_paper_readiness(
    inputs: PaperReadinessInputs,
    *,
    static_report: SafetyReport,
    config: dict[str, Any],
    strategy_file: Path,
) -> tuple[dict[str, Any], dict[str, Any], ConfigSafetyResult]:
    candidate_artifacts = build_candidate_artifacts(inputs)
    checks: list[ReadinessCheck] = []

    checks.extend(_candidate_artifact_checks(candidate_artifacts))

    config_safety = evaluate_config_safety(config, strategy=inputs.strategy)
    checks.extend(config_safety.checks)

    strategy_safety = evaluate_strategy_long_only(strategy_file, inputs.strategy)
    checks.extend(strategy_safety.checks)

    checks.append(
        _check(
            "static_strategy_check",
            static_report.ok,
            "blocker",
            "Static strategy safety check must pass without errors.",
            {
                "files_checked": static_report.files_checked,
                "finding_count": len(static_report.findings),
            },
        )
    )

    checks.extend(_phase2_evidence_checks(inputs))
    checks.append(
        _check(
            "reviewer_note_present",
            bool(inputs.reviewer_notes),
            "failure",
            "At least one explicit reviewer note is required before paper readiness can pass.",
            {"note_count": len(inputs.reviewer_notes)},
        )
    )

    status = _readiness_status(checks)
    readiness = {
        "generated_at": datetime.now(UTC).isoformat(),
        "phase": "3",
        "factory": "paper_readiness",
        "strategy": inputs.strategy,
        "run_id": inputs.run_id,
        "status": status,
        "readiness": status,
        "config_path": _safe_relative_path(inputs.config_path, inputs.root_dir),
        "strategy_path": _safe_relative_path(strategy_file, inputs.root_dir),
        "historical_dir": _safe_relative_path(inputs.historical_dir, inputs.root_dir),
        "walk_forward_dir": _safe_relative_path(inputs.walk_forward_dir, inputs.root_dir),
        "training_dir": _safe_relative_path(inputs.training_dir, inputs.root_dir),
        "regime_scorecard_path": _safe_relative_path(inputs.regime_scorecard_path, inputs.root_dir)
        if inputs.regime_scorecard_path is not None
        else None,
        "market_state_scorecard_path": _safe_relative_path(
            inputs.market_state_scorecard_path, inputs.root_dir
        )
        if inputs.market_state_scorecard_path is not None
        else None,
        "strategy_suitability_matrix_path": _safe_relative_path(
            inputs.strategy_suitability_matrix_path, inputs.root_dir
        )
        if inputs.strategy_suitability_matrix_path is not None
        else None,
        "checks": [check.to_dict() for check in checks],
        "blockers": [check.to_dict() for check in checks if check.status == "blocked"],
        "failures": [check.to_dict() for check in checks if check.status == "fail"],
        "reviewer_notes": list(inputs.reviewer_notes),
        "strategy_safety": strategy_safety.to_dict(),
        "static_check": static_report.to_dict(),
        "artifact_paths": {
            "paper_readiness": _safe_relative_path(
                inputs.output_dir / "paper_readiness.json", inputs.root_dir
            ),
            "paper_readiness_report": _safe_relative_path(
                inputs.output_dir / "paper_readiness_report.md", inputs.root_dir
            ),
            "candidate_artifacts": _safe_relative_path(
                inputs.output_dir / "candidate_artifacts.json", inputs.root_dir
            ),
            "config_safety": _safe_relative_path(
                inputs.output_dir / "config_safety.json", inputs.root_dir
            ),
            "command": _safe_relative_path(inputs.output_dir / "command.txt", inputs.root_dir),
        },
        "safety_scope": {
            "command": "paper readiness preflight only",
            "bot_startup": False,
            "freqtrade_trade": False,
            "paper_trading_started": False,
            "dry_run_trading_started": False,
            "live_trading": False,
            "exchange_order_placement": False,
            "uses_api_keys_or_secrets": False,
            "leverage_above_one": False,
            "shorting": False,
            "metadata_contains_secrets": False,
            "local_artifacts_source_of_truth": True,
        },
        "notice": READINESS_NOTICE,
    }
    return readiness, candidate_artifacts, config_safety


def write_paper_readiness_artifacts(
    *,
    inputs: PaperReadinessInputs,
    readiness: dict[str, Any],
    candidate_artifacts: dict[str, Any],
    config_safety: ConfigSafetyResult,
) -> None:
    output_dir = inputs.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "paper_readiness.json").write_text(
        json.dumps(readiness, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "candidate_artifacts.json").write_text(
        json.dumps(candidate_artifacts, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "config_safety.json").write_text(
        json.dumps(config_safety.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "command.txt").write_text(" ".join(inputs.command), encoding="utf-8")
    write_paper_readiness_report(readiness, output_dir / "paper_readiness_report.md")


def write_paper_readiness_report(readiness: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Paper Readiness Report",
        "",
        "## Summary",
        "",
        f"- Strategy: {readiness['strategy']}",
        f"- Run ID: {readiness['run_id']}",
        f"- Readiness: {readiness['readiness']}",
        f"- Historical artifacts: `{readiness['historical_dir']}`",
        f"- Walk-forward artifacts: `{readiness['walk_forward_dir']}`",
        f"- Training artifacts: `{readiness['training_dir']}`",
        f"- Regime scorecard: `{readiness.get('regime_scorecard_path') or 'not supplied'}`",
        f"- Market-state scorecard: `{readiness.get('market_state_scorecard_path') or 'not supplied'}`",
        f"- Strategy suitability matrix: `{readiness.get('strategy_suitability_matrix_path') or 'not supplied'}`",
        "",
        "## Checks",
        "",
    ]
    for check in readiness["checks"]:
        status = check["status"].upper()
        lines.append(f"- {status}: {check['name']} - {check['message']}")

    lines.extend(["", "## Reviewer Notes", ""])
    if readiness["reviewer_notes"]:
        lines.extend(f"- {note}" for note in readiness["reviewer_notes"])
    else:
        lines.append("- No reviewer notes were supplied.")

    lines.extend(
        [
            "",
            "## Notes",
            "",
            f"- {READINESS_NOTICE}",
            "- Failed Phase 2 gates block paper readiness.",
            "- A future human-approved infrastructure-only smoke test is a separate path.",
            "- Local JSON, CSV, and Markdown artifacts remain the source of truth.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _phase2_evidence_checks(inputs: PaperReadinessInputs) -> list[ReadinessCheck]:
    checks: list[ReadinessCheck] = []
    historical_metrics_path = inputs.historical_dir / "metrics.json"
    historical_trades_path = inputs.historical_dir / "trades.csv"
    walk_forward_metrics_path = inputs.walk_forward_dir / "walk_forward_metrics.json"
    training_manifest_path = inputs.training_dir / "training_manifest.json"
    checks.extend(_regime_scorecard_evidence_checks(inputs))
    checks.extend(_market_state_scorecard_evidence_checks(inputs))
    checks.extend(_strategy_suitability_matrix_evidence_checks(inputs))

    if historical_metrics_path.is_file():
        metrics_payload = load_json_file(historical_metrics_path)
        metrics = BacktestMetrics(**metrics_payload)
        gate = evaluate_initial_gate(metrics)
        checks.append(
            _check(
                "historical_backtest_gate",
                gate["recommendation"] == "pass",
                "failure",
                "Historical backtest gate must pass before paper readiness can pass.",
                {
                    "recommendation": gate["recommendation"],
                    "trade_count": metrics.trade_count,
                    "total_return_pct": metrics.total_return_pct,
                    "checks": gate["checks"],
                },
            )
        )
    else:
        checks.append(
            ReadinessCheck(
                name="historical_backtest_gate",
                status="blocked",
                severity="blocker",
                message="Historical metrics are missing, so the gate cannot be evaluated.",
                details={"path": _safe_relative_path(historical_metrics_path, inputs.root_dir)},
            )
        )

    if historical_trades_path.is_file():
        checks.extend(_historical_trade_safety_checks(historical_trades_path))
    else:
        checks.append(
            ReadinessCheck(
                name="historical_trades_long_only",
                status="blocked",
                severity="blocker",
                message="Historical trades are missing, so long-only trade evidence cannot be checked.",
                details={"path": _safe_relative_path(historical_trades_path, inputs.root_dir)},
            )
        )

    if walk_forward_metrics_path.is_file():
        walk_forward_metrics = load_json_file(walk_forward_metrics_path)
        recommendation = walk_forward_metrics.get("recommendation")
        checks.append(
            _check(
                "walk_forward_gate",
                recommendation == "pass",
                "failure",
                "Walk-forward recommendation must pass before paper readiness can pass.",
                {
                    "recommendation": recommendation,
                    "status": walk_forward_metrics.get("status"),
                    "summary": walk_forward_metrics.get("summary", {}),
                },
            )
        )
        checks.extend(_walk_forward_child_evidence_checks(walk_forward_metrics, inputs.root_dir))
    else:
        checks.append(
            ReadinessCheck(
                name="walk_forward_gate",
                status="blocked",
                severity="blocker",
                message="Walk-forward metrics are missing, so the gate cannot be evaluated.",
                details={"path": _safe_relative_path(walk_forward_metrics_path, inputs.root_dir)},
            )
        )

    if training_manifest_path.is_file():
        training_manifest = load_json_file(training_manifest_path)
        recommendation = training_manifest.get("recommendation")
        checks.append(
            _check(
                "training_factory_gate",
                recommendation == "pass",
                "failure",
                "Training factory recommendation must pass before paper readiness can pass.",
                {
                    "recommendation": recommendation,
                    "status": training_manifest.get("status"),
                    "summary": training_manifest.get("summary", {}),
                },
            )
        )
        checks.extend(_training_child_evidence_checks(training_manifest, inputs.root_dir))
    else:
        checks.append(
            ReadinessCheck(
                name="training_factory_gate",
                status="blocked",
                severity="blocker",
                message="Training manifest is missing, so the gate cannot be evaluated.",
                details={"path": _safe_relative_path(training_manifest_path, inputs.root_dir)},
            )
        )

    return checks


def _regime_scorecard_evidence_checks(inputs: PaperReadinessInputs) -> list[ReadinessCheck]:
    if not inputs.requires_regime_scorecard and inputs.regime_scorecard_path is None:
        return [
            _check(
                "regime_scorecard_not_required",
                True,
                "blocker",
                "No regime-scoped selector eligibility was claimed for this readiness request.",
                {},
            )
        ]
    path = inputs.regime_scorecard_path
    if path is None:
        return [
            ReadinessCheck(
                name="regime_scorecard_required",
                status="blocked",
                severity="blocker",
                message="Regime scorecard is required when regime-scoped selector eligibility is claimed.",
                details={"path": None},
            )
        ]
    resolved = path if path.is_absolute() else inputs.root_dir / path
    if not resolved.is_file():
        return [
            ReadinessCheck(
                name="regime_scorecard_required",
                status="blocked",
                severity="blocker",
                message="Regime scorecard artifact is missing.",
                details={"path": _safe_relative_path(resolved, inputs.root_dir)},
            )
        ]
    payload = load_json_file(resolved)
    return [
        _check(
            "regime_scorecard_required",
            True,
            "blocker",
            "Regime scorecard artifact is present.",
            {"path": _safe_relative_path(resolved, inputs.root_dir)},
        ),
        _check(
            "regime_scorecard_selector_eligible",
            payload.get("decision") in {"REGIME_SCOPED_SELECTOR_ELIGIBLE", "GLOBAL_SELECTOR_ELIGIBLE"},
            "failure",
            "Regime scorecard must be selector-eligible before paper readiness can pass.",
            {"decision": payload.get("decision")},
        ),
        _check(
            "regime_scorecard_does_not_authorize_promotion",
            payload.get("promotion_authorized_by_this_command") is False
            and payload.get("raw_aggregate_pnl_promotion_allowed") is False
            and payload.get("phase3_readiness_required_after_scorecard") is True,
            "blocker",
            "Regime scorecard must preserve no-promotion safety semantics.",
            {
                "promotion_authorized_by_this_command": payload.get("promotion_authorized_by_this_command"),
                "raw_aggregate_pnl_promotion_allowed": payload.get("raw_aggregate_pnl_promotion_allowed"),
                "phase3_readiness_required_after_scorecard": payload.get("phase3_readiness_required_after_scorecard"),
            },
        ),
    ]


def _market_state_scorecard_evidence_checks(inputs: PaperReadinessInputs) -> list[ReadinessCheck]:
    if not inputs.requires_market_state_scorecard and inputs.market_state_scorecard_path is None:
        return [
            _check(
                "market_state_scorecard_not_required",
                True,
                "blocker",
                "No market-state scorecard was supplied for this readiness request.",
                {},
            )
        ]
    path = inputs.market_state_scorecard_path
    if path is None:
        return [
            ReadinessCheck(
                name="market_state_scorecard_required",
                status="blocked",
                severity="blocker",
                message="Market-state scorecard is required when state-conditioned eligibility is claimed.",
                details={"path": None},
            )
        ]
    resolved = path if path.is_absolute() else inputs.root_dir / path
    if not resolved.is_file():
        return [
            ReadinessCheck(
                name="market_state_scorecard_required",
                status="blocked",
                severity="blocker",
                message="Market-state scorecard artifact is missing.",
                details={"path": _safe_relative_path(resolved, inputs.root_dir)},
            )
        ]
    payload = load_json_file(resolved)
    validation = validate_state_conditioned_scorecard_for_selector(payload)
    safety = payload.get("safety_scope") or {}
    return [
        _check(
            "market_state_scorecard_required",
            True,
            "blocker",
            "Market-state scorecard artifact is present.",
            {"path": _safe_relative_path(resolved, inputs.root_dir)},
        ),
        _check(
            "market_state_scorecard_full_schema",
            validation["ok"],
            "failure",
            "Market-state scorecard must pass the full state-conditioned schema, not only top-level flags.",
            {"reason_codes": validation["reason_codes"], "checks": validation["checks"]},
        ),
        _check(
            "market_state_scorecard_paper_readiness_allowed",
            payload.get("paper_readiness_input_allowed") is True,
            "failure",
            "Market-state scorecard must explicitly allow paper-readiness input after strict validation.",
            {
                "paper_readiness_input_allowed": payload.get(
                    "paper_readiness_input_allowed"
                )
            },
        ),
        _check(
            "market_state_scorecard_no_startup_scope",
            safety.get("freqtrade_trade_started") is False
            and safety.get("paper_trading_started") is False
            and safety.get("dry_run_trading_started") is False
            and safety.get("live_trading_started") is False
            and safety.get("exchange_order_placement") is False
            and safety.get("process_control") is False,
            "blocker",
            "Market-state scorecard must preserve no-startup safety scope.",
            {
                "freqtrade_trade_started": safety.get("freqtrade_trade_started"),
                "paper_trading_started": safety.get("paper_trading_started"),
                "dry_run_trading_started": safety.get("dry_run_trading_started"),
                "live_trading_started": safety.get("live_trading_started"),
                "exchange_order_placement": safety.get("exchange_order_placement"),
                "process_control": safety.get("process_control"),
            },
        ),
    ]


def _strategy_suitability_matrix_evidence_checks(
    inputs: PaperReadinessInputs,
) -> list[ReadinessCheck]:
    if (
        not inputs.requires_strategy_suitability_matrix
        and inputs.strategy_suitability_matrix_path is None
    ):
        return [
            _check(
                "strategy_suitability_matrix_not_required",
                True,
                "blocker",
                "No strategy suitability matrix was supplied for this readiness request.",
                {},
            )
        ]
    path = inputs.strategy_suitability_matrix_path
    if path is None:
        return [
            ReadinessCheck(
                name="strategy_suitability_matrix_required",
                status="blocked",
                severity="blocker",
                message="Strategy suitability matrix is required when state matching is claimed.",
                details={"path": None},
            )
        ]
    resolved = path if path.is_absolute() else inputs.root_dir / path
    if not resolved.is_file():
        return [
            ReadinessCheck(
                name="strategy_suitability_matrix_required",
                status="blocked",
                severity="blocker",
                message="Strategy suitability matrix artifact is missing.",
                details={"path": _safe_relative_path(resolved, inputs.root_dir)},
            )
        ]
    payload = load_json_file(resolved)
    validation = validate_strategy_suitability_matrix_for_selector(payload)
    safety = payload.get("safety_scope") or {}
    selector_rows = _selector_eligible_matrix_rows(payload)
    matching_rows = [
        row for row in selector_rows if _matrix_row_matches_strategy(row, inputs.strategy)
    ]
    return [
        _check(
            "strategy_suitability_matrix_required",
            True,
            "blocker",
            "Strategy suitability matrix artifact is present.",
            {"path": _safe_relative_path(resolved, inputs.root_dir)},
        ),
        _check(
            "strategy_suitability_matrix_full_schema",
            validation["ok"],
            "failure",
            "Strategy suitability matrix must pass full schema validation.",
            {"reason_codes": validation["reason_codes"], "checks": validation["checks"]},
        ),
        _check(
            "strategy_suitability_matrix_matches_strategy",
            bool(matching_rows),
            "failure",
            "Strategy suitability matrix must contain a selector-eligible row for the readiness strategy.",
            {
                "strategy": inputs.strategy,
                "selector_row_count": len(selector_rows),
                "matching_selector_row_count": len(matching_rows),
                "selector_strategies": [
                    _matrix_row_strategy_tokens(row) for row in selector_rows
                ],
            },
        ),
        _check(
            "strategy_suitability_matrix_no_startup_scope",
            safety.get("freqtrade_trade_started") is False
            and safety.get("paper_trading_started") is False
            and safety.get("dry_run_trading_started") is False
            and safety.get("live_trading_started") is False
            and safety.get("exchange_order_placement") is False
            and safety.get("process_control") is False,
            "blocker",
            "Strategy suitability matrix must preserve no-startup safety scope.",
            {
                "freqtrade_trade_started": safety.get("freqtrade_trade_started"),
                "paper_trading_started": safety.get("paper_trading_started"),
                "dry_run_trading_started": safety.get("dry_run_trading_started"),
                "live_trading_started": safety.get("live_trading_started"),
                "exchange_order_placement": safety.get("exchange_order_placement"),
                "process_control": safety.get("process_control"),
            },
        ),
    ]


def _selector_eligible_matrix_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = payload.get("rows", [])
    if not isinstance(rows, list):
        return []
    return [
        row
        for row in rows
        if isinstance(row, dict)
        and row.get("row_type") == "strategy"
        and row.get("decision") == "SELECTOR_ELIGIBLE"
    ]


def _matrix_row_matches_strategy(row: dict[str, Any], strategy: str) -> bool:
    return strategy in _matrix_row_strategy_tokens(row)


def _matrix_row_strategy_tokens(row: dict[str, Any]) -> list[str]:
    identity = row.get("strategy_identity_unit") or {}
    if not isinstance(identity, dict):
        identity = {}
    values = [
        row.get("strategy_id"),
        row.get("strategy_class_name"),
        identity.get("strategy_id"),
        identity.get("strategy_class_name"),
    ]
    return sorted({str(value) for value in values if value not in (None, "")})


def _walk_forward_child_evidence_checks(
    walk_forward_metrics: dict[str, Any], root_dir: Path
) -> list[ReadinessCheck]:
    checks: list[ReadinessCheck] = []
    windows = walk_forward_metrics.get("windows")
    windows_present = isinstance(windows, list) and bool(windows)
    checks.append(
        _check(
            "walk_forward_child_windows_present",
            windows_present,
            "blocker",
            "Walk-forward metrics must include child window evidence.",
            {"window_count": len(windows) if isinstance(windows, list) else 0},
        )
    )
    if not windows_present:
        return checks

    for index, window in enumerate(windows, start=1):
        if not isinstance(window, dict):
            checks.append(
                ReadinessCheck(
                    name=f"walk_forward_window_{index:02d}_shape",
                    status="blocked",
                    severity="blocker",
                    message="Walk-forward window evidence must be an object.",
                )
            )
            continue

        label = _safe_check_token(str(window.get("run_id") or f"window_{index:02d}"))
        run_dir = _resolve_optional_artifact_path(window.get("run_dir"), root_dir)
        artifacts = window.get("artifacts", {})
        if not isinstance(artifacts, dict):
            artifacts = {}

        required_paths = {
            "metrics": _artifact_path_from_payload(
                artifacts.get("metrics"),
                run_dir / "metrics.json" if run_dir is not None else None,
                root_dir,
            ),
            "trades": _artifact_path_from_payload(
                artifacts.get("trades"),
                run_dir / "trades.csv" if run_dir is not None else None,
                root_dir,
            ),
            "freqai_metadata": _artifact_path_from_payload(
                artifacts.get("freqai_metadata"),
                run_dir / "freqai_metadata.json" if run_dir is not None else None,
                root_dir,
            ),
        }
        checks.extend(
            _required_artifact_checks(
                f"walk_forward_{label}",
                required_paths,
                root_dir,
                "Walk-forward window child artifact must exist",
            )
        )

        trades_path = required_paths["trades"]
        if trades_path is not None and trades_path.is_file():
            checks.extend(
                _trade_safety_checks(
                    trades_path,
                    name_prefix=f"walk_forward_{label}_trades",
                    message_prefix="Walk-forward window exported trades",
                )
            )

    return checks


def _training_child_evidence_checks(
    training_manifest: dict[str, Any], root_dir: Path
) -> list[ReadinessCheck]:
    checks: list[ReadinessCheck] = []
    stages = training_manifest.get("stages")
    stages_present = isinstance(stages, list) and bool(stages)
    checks.append(
        _check(
            "training_child_stages_present",
            stages_present,
            "blocker",
            "Training manifest must include child stage evidence.",
            {"stage_count": len(stages) if isinstance(stages, list) else 0},
        )
    )
    if not stages_present:
        return checks

    backtest_stages = [
        stage
        for stage in stages
        if isinstance(stage, dict) and stage.get("name") == "freqai_backtest"
    ]
    checks.append(
        _check(
            "training_freqai_backtest_child_present",
            bool(backtest_stages),
            "blocker",
            "Training manifest must include a freqai_backtest child stage.",
            {"child_count": len(backtest_stages)},
        )
    )

    for index, stage in enumerate(backtest_stages, start=1):
        label = _safe_check_token(str(stage.get("run_id") or f"freqai_backtest_{index:02d}"))
        output_dir = _resolve_optional_artifact_path(stage.get("output_dir"), root_dir)
        artifacts = stage.get("artifacts", {})
        if not isinstance(artifacts, dict):
            artifacts = {}

        required_paths = {
            "metrics": _artifact_path_from_payload(
                artifacts.get("metrics"),
                output_dir / "metrics.json" if output_dir is not None else None,
                root_dir,
            ),
            "trades": _artifact_path_from_payload(
                artifacts.get("trades"),
                output_dir / "trades.csv" if output_dir is not None else None,
                root_dir,
            ),
            "freqai_metadata": _artifact_path_from_payload(
                artifacts.get("freqai_metadata"),
                output_dir / "freqai_metadata.json" if output_dir is not None else None,
                root_dir,
            ),
        }
        checks.extend(
            _required_artifact_checks(
                f"training_{label}",
                required_paths,
                root_dir,
                "Training child artifact must exist",
            )
        )

        trades_path = required_paths["trades"]
        if trades_path is not None and trades_path.is_file():
            checks.extend(
                _trade_safety_checks(
                    trades_path,
                    name_prefix=f"training_{label}_trades",
                    message_prefix="Training child exported trades",
                )
            )

    return checks


def _historical_trade_safety_checks(trades_path: Path) -> list[ReadinessCheck]:
    return _trade_safety_checks(
        trades_path,
        name_prefix="historical_trades",
        message_prefix="Historical exported trades",
    )


def _trade_safety_checks(
    trades_path: Path, *, name_prefix: str, message_prefix: str
) -> list[ReadinessCheck]:
    short_rows: list[int] = []
    high_leverage_rows: list[dict[str, Any]] = []

    with trades_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for index, row in enumerate(reader, start=2):
            if _truthy_string(row.get("is_short")):
                short_rows.append(index)
            leverage = row.get("leverage")
            if leverage not in (None, "") and _numeric_above_one(leverage):
                high_leverage_rows.append({"line": index})

    return [
        _check(
            f"{name_prefix}_no_shorts",
            not short_rows,
            "blocker",
            f"{message_prefix} must not contain short trades.",
            {"lines": short_rows},
        ),
        _check(
            f"{name_prefix}_no_leverage_above_one",
            not high_leverage_rows,
            "blocker",
            f"{message_prefix} must not contain leverage above 1.0.",
            {"lines": high_leverage_rows},
        ),
    ]


def _required_artifact_checks(
    name_prefix: str,
    paths: dict[str, Path | None],
    root_dir: Path,
    message_prefix: str,
) -> list[ReadinessCheck]:
    checks: list[ReadinessCheck] = []
    for artifact_name, artifact_path in paths.items():
        exists = artifact_path is not None and artifact_path.is_file()
        checks.append(
            _check(
                f"{name_prefix}_{artifact_name}_present",
                exists,
                "blocker",
                f"{message_prefix}: {artifact_name}.",
                {
                    "path": _safe_relative_path(artifact_path, root_dir)
                    if artifact_path is not None
                    else None
                },
            )
        )
    return checks


def _candidate_artifact_checks(candidate_artifacts: dict[str, Any]) -> list[ReadinessCheck]:
    checks: list[ReadinessCheck] = []
    for name, info in candidate_artifacts["artifacts"].items():
        checks.append(
            _check(
                f"artifact_present_{name}",
                bool(info["exists"]),
                "blocker",
                f"Required candidate artifact must exist: {name}.",
                {"path": info["path"]},
            )
        )
    return checks


def _readiness_status(checks: Sequence[ReadinessCheck]) -> str:
    if any(check.status == "blocked" for check in checks):
        return "blocked"
    if any(check.status == "fail" for check in checks):
        return "fail"
    return "pass"


def _check(
    name: str,
    passed: bool,
    severity: str,
    message: str,
    details: dict[str, Any] | None = None,
) -> ReadinessCheck:
    if passed:
        status = "pass"
    elif severity == "blocker":
        status = "blocked"
    else:
        status = "fail"
    return ReadinessCheck(
        name=name,
        status=status,
        severity=severity,
        message=message,
        details=details or {},
    )


def _find_class(tree: ast.AST, class_name: str) -> ast.ClassDef | None:
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return node
    return None


def _class_assignment_constant(class_node: ast.ClassDef, name: str) -> Any:
    for node in class_node.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return _constant_value(node.value)
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id == name:
                return _constant_value(node.value)
    return None


def _short_signal_lines(class_node: ast.ClassDef) -> list[int]:
    lines: list[int] = []
    for node in ast.walk(class_node):
        if isinstance(node, ast.Constant) and node.value in {"enter_short", "exit_short"}:
            lines.append(getattr(node, "lineno", 0))
        elif isinstance(node, ast.Attribute) and node.attr in {"enter_short", "exit_short"}:
            lines.append(getattr(node, "lineno", 0))
    return sorted(set(line for line in lines if line))


def _leverage_hook_checks(class_node: ast.ClassDef) -> list[ReadinessCheck]:
    leverage_hooks = [
        node for node in class_node.body if isinstance(node, ast.FunctionDef) and node.name == "leverage"
    ]
    if not leverage_hooks:
        return [
            ReadinessCheck(
                name="leverage_hook_absent_or_capped",
                status="pass",
                severity="blocker",
                message="No leverage hook was found.",
                details={"hook_present": False},
            )
        ]

    hook = leverage_hooks[0]
    high_constants = [
        {"line": getattr(node, "lineno", 0), "value": node.value}
        for node in ast.walk(hook)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, (int, float))
        and float(node.value) > 1.0
    ]
    nonconstant_returns = [
        getattr(node, "lineno", 0)
        for node in ast.walk(hook)
        if isinstance(node, ast.Return)
        and node.value is not None
        and not _constant_number_at_most_one(node.value)
    ]
    checks = [
        _check(
            "leverage_hook_no_constant_above_one",
            not high_constants,
            "blocker",
            "Leverage hook must not contain numeric constants above 1.0.",
            {"constants": high_constants},
        ),
        _check(
            "leverage_hook_returns_capped",
            not nonconstant_returns,
            "blocker",
            "Leverage hook returns must be statically capped at 1.0.",
            {"nonconstant_return_lines": nonconstant_returns},
        ),
    ]
    return checks


def _constant_number_at_most_one(node: ast.AST) -> bool:
    value = _constant_value(node)
    return isinstance(value, (int, float)) and float(value) <= 1.0


def _constant_value(node: ast.AST | None) -> Any:
    if node is None:
        return None
    if isinstance(node, ast.Constant):
        return node.value
    if (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, ast.USub)
        and isinstance(node.operand, ast.Constant)
        and isinstance(node.operand.value, (int, float))
    ):
        return -node.operand.value
    return None


def _credential_findings(payload: Any, prefix: str = "") -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if _CREDENTIAL_KEY_RE.search(str(key)):
                findings.append({"path": path, "has_value": _has_credential_value(value)})
            findings.extend(_credential_findings(value, path))
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            findings.extend(_credential_findings(value, f"{prefix}[{index}]"))
    return findings


def _private_env_findings(payload: Any, prefix: str = "") -> list[str]:
    findings: list[str] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            findings.extend(_private_env_findings(value, path))
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            findings.extend(_private_env_findings(value, f"{prefix}[{index}]"))
    elif isinstance(payload, str) and _PRIVATE_ENV_RE.search(payload):
        findings.append(prefix)
    return findings


def _leverage_findings(payload: Any, prefix: str = "") -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if "leverage" in str(key).lower() and _numeric_above_one(value):
                findings.append({"path": path})
            findings.extend(_leverage_findings(value, path))
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            findings.extend(_leverage_findings(value, f"{prefix}[{index}]"))
    return findings


def _endpoint_findings(payload: Any, prefix: str = "") -> list[str]:
    findings: list[str] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            lowered = path.lower()
            if isinstance(value, str) and value and value.startswith(("http://", "https://")):
                if any(token in lowered for token in ("private", "order", "endpoint", "url")):
                    findings.append(path)
            findings.extend(_endpoint_findings(value, path))
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            findings.extend(_endpoint_findings(value, f"{prefix}[{index}]"))
    return findings


def _has_credential_value(value: Any) -> bool:
    if value is None or value is False:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, list):
        return any(_has_credential_value(item) for item in value)
    if isinstance(value, dict):
        return any(_has_credential_value(item) for item in value.values())
    return True


def _numeric_above_one(value: Any) -> bool:
    try:
        return float(value) > 1.0
    except (TypeError, ValueError):
        return bool(value)


def _truthy_string(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _pair_allowlist(config: dict[str, Any]) -> list[str]:
    pairs = _nested_get(config, ["exchange", "pair_whitelist"])
    if not isinstance(pairs, list):
        return []
    return [str(pair) for pair in pairs if str(pair).strip()]


def _nested_get(payload: dict[str, Any], keys: Sequence[str]) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _positive_number(value: Any) -> bool:
    try:
        return float(value) > 0
    except (TypeError, ValueError):
        return False


def _positive_integer(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    return numeric > 0 and numeric.is_integer()


def _is_number(value: Any) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def _safe_scalar(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return type(value).__name__


def _artifact_info(path: Path, root_dir: Path) -> dict[str, Any]:
    exists = path.exists()
    info: dict[str, Any] = {
        "path": _safe_relative_path(path, root_dir),
        "exists": exists,
        "kind": "file" if path.is_file() else "directory" if path.is_dir() else "missing",
    }
    if exists and path.is_file():
        stat = path.stat()
        info.update(
            {
                "bytes": stat.st_size,
                "modified_at": datetime.fromtimestamp(stat.st_mtime, UTC).isoformat(),
                "sha256": _sha256_file(path),
            }
        )
    return info


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_path_from_payload(
    path_value: Any, fallback: Path | None, root_dir: Path
) -> Path | None:
    resolved = _resolve_optional_artifact_path(path_value, root_dir)
    return resolved or fallback


def _resolve_optional_artifact_path(path_value: Any, root_dir: Path) -> Path | None:
    if not isinstance(path_value, str) or not path_value.strip():
        return None
    path = Path(path_value)
    if path.is_absolute():
        return path
    return root_dir / path


def _safe_check_token(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_").lower()
    return token or "artifact"


def _safe_relative_path(path: Path, root_dir: Path) -> str:
    try:
        return str(path.resolve().relative_to(root_dir.resolve()))
    except ValueError:
        return path.name
