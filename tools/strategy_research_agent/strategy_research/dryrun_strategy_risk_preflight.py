#!/usr/bin/env python3
"""Dry-run strategy risk preflight.

This verifies the strategy as Freqtrade will actually load it, not only what
the strategy source appears to define.  It checks callable strategy hooks,
configuration overrides, and the final effective risk contract before a dry-run
bot can be started.
"""

from __future__ import annotations

import argparse
import copy
import inspect
import json
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from freqtrade.resolvers import StrategyResolver
from freqtrade.strategy.interface import IStrategy


def find_repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / ".venv").exists() and (parent / "user_data").exists():
            return parent
    return current.parents[2]


REPO_ROOT = find_repo_root()
DEFAULT_CONFIG = REPO_ROOT / "user_data/config_futures_dryrun.json"
DEFAULT_REGISTRY = REPO_ROOT / "user_data/strategy_research/strategy_registry.json"
EXPECTED_ROI = {0: 1.20, 180: 1.50, 360: 1.00}
EXPECTED_STOPLOSS = -0.60
EXPECTED_LEVERAGE = 50.0
ALLOWED_ENTRY_TIMEFRAMES = {"3m", "5m", "15m"}
REQUIRED_STOPLOSS_GUARD = {
    "method": "StoplossGuard",
    "lookback_period_candles": 96,
    "trade_limit": 3,
    "stop_duration_candles": 32,
    "only_per_pair": False,
}
OVERRIDE_ATTRIBUTES = [
    "minimal_roi",
    "timeframe",
    "stoploss",
    "trailing_stop",
    "trailing_stop_positive",
    "trailing_stop_positive_offset",
    "trailing_only_offset_is_reached",
    "use_custom_stoploss",
    "process_only_new_candles",
    "order_types",
    "order_time_in_force",
    "stake_currency",
    "stake_amount",
    "startup_candle_count",
    "unfilledtimeout",
    "use_exit_signal",
    "exit_profit_only",
    "ignore_roi_if_entry_signal",
    "exit_profit_offset",
    "disable_dataframe_checks",
    "ignore_buying_expired_candle_after",
    "position_adjustment_enable",
    "max_entry_position_adjustment",
    "max_open_trades",
]


@dataclass
class Check:
    name: str
    status: str
    detail: str


@dataclass
class StrategyAudit:
    strategy: str
    checks: list[Check] = field(default_factory=list)
    overrides: list[dict[str, Any]] = field(default_factory=list)

    def add(self, name: str, status: str, detail: str) -> None:
        self.checks.append(Check(name=name, status=status, detail=detail))

    @property
    def failed(self) -> bool:
        return any(check.status == "fail" for check in self.checks)


class FakeTrade:
    def __init__(self, open_date_utc: datetime) -> None:
        self.open_date_utc = open_date_utc


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def normalize_roi(value: Any) -> dict[int, float]:
    if not isinstance(value, dict):
        return {}
    return {int(key): float(val) for key, val in value.items()}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_config(config_path: Path, strategy: str) -> dict[str, Any]:
    config = load_json(config_path)
    config["strategy"] = strategy
    config["user_data_dir"] = REPO_ROOT / "user_data"
    config.setdefault("strategy_path", "user_data/strategies/research_generated")
    return config


def registry_strategies(path: Path) -> list[str]:
    if not path.exists():
        return []
    registry = load_json(path)
    names: list[str] = []
    for item in registry.get("strategies", []):
        name = item.get("name") or item.get("strategy") or item.get("strategy_name")
        if name and name not in names:
            names.append(name)
    return names


def selected_strategies(args: argparse.Namespace, config_path: Path) -> list[str]:
    names: list[str] = []
    if args.strategy:
        names.extend(args.strategy)
    else:
        config = load_json(config_path)
        if config.get("strategy"):
            names.append(config["strategy"])
    if args.all_registry:
        for name in registry_strategies(args.registry):
            if name not in names:
                names.append(name)
    return names


def owner_name(strategy: IStrategy, attr: str) -> str | None:
    for base in type(strategy).mro():
        if attr in base.__dict__:
            return base.__name__
    return None


def values_equal(left: Any, right: Any) -> bool:
    if isinstance(left, dict) and isinstance(right, dict):
        return left == right
    return left == right


def check_config_contract(audit: StrategyAudit, config: dict[str, Any]) -> None:
    if config.get("trading_mode") == "futures":
        audit.add("config:trading_mode", "ok", "futures")
    else:
        audit.add("config:trading_mode", "fail", f"expected futures, got {config.get('trading_mode')}")

    if config.get("margin_mode") == "isolated":
        audit.add("config:margin_mode", "ok", "isolated")
    else:
        audit.add("config:margin_mode", "fail", f"expected isolated, got {config.get('margin_mode')}")

    if config.get("max_open_trades") == 1:
        audit.add("config:max_open_trades", "ok", "1")
    else:
        audit.add("config:max_open_trades", "fail", f"expected 1, got {config.get('max_open_trades')}")

    if config.get("stake_amount") == 50:
        audit.add("config:stake_amount", "ok", "50 USDT margin")
    else:
        audit.add("config:stake_amount", "fail", f"expected 50, got {config.get('stake_amount')}")

    if config.get("tradable_balance_ratio") == 0.5:
        audit.add("config:tradable_balance_ratio", "ok", "0.5")
    else:
        audit.add(
            "config:tradable_balance_ratio",
            "fail",
            f"expected 0.5, got {config.get('tradable_balance_ratio')}",
        )


def check_overrides(audit: StrategyAudit, source: IStrategy, final: IStrategy, final_config: dict[str, Any]) -> None:
    for attr in OVERRIDE_ATTRIBUTES:
        if attr not in final_config:
            continue
        source_value = getattr(source, attr, "<missing>")
        final_value = getattr(final, attr, "<missing>")
        if not values_equal(source_value, final_value):
            audit.overrides.append(
                {
                    "attribute": attr,
                    "strategy_value": source_value,
                    "config_value": final_config[attr],
                    "effective_value": final_value,
                }
            )

    if audit.overrides:
        audit.add(
            "config_overrides",
            "ok",
            ", ".join(item["attribute"] for item in audit.overrides),
        )
    else:
        audit.add("config_overrides", "ok", "no config overrides changed strategy attributes")


def check_strategy_contract(audit: StrategyAudit, strategy: IStrategy) -> None:
    if strategy.timeframe in ALLOWED_ENTRY_TIMEFRAMES:
        audit.add("strategy:timeframe", "ok", strategy.timeframe)
    else:
        audit.add("strategy:timeframe", "fail", f"{strategy.timeframe} not in {sorted(ALLOWED_ENTRY_TIMEFRAMES)}")

    if getattr(strategy, "can_short", False):
        audit.add("strategy:can_short", "ok", "short-capable")
    else:
        audit.add("strategy:can_short", "fail", "strategy cannot short")

    roi = normalize_roi(getattr(strategy, "minimal_roi", {}))
    if roi == EXPECTED_ROI:
        audit.add("strategy:minimal_roi", "ok", str(roi))
    else:
        audit.add("strategy:minimal_roi", "fail", f"expected {EXPECTED_ROI}, got {roi}")

    stoploss = float(getattr(strategy, "stoploss", 0.0))
    if abs(stoploss - EXPECTED_STOPLOSS) < 1e-12:
        audit.add("strategy:stoploss", "ok", str(stoploss))
    else:
        audit.add("strategy:stoploss", "fail", f"expected {EXPECTED_STOPLOSS}, got {stoploss}")

    order_types = getattr(strategy, "order_types", {})
    if order_types.get("stoploss") == "market":
        audit.add("order_types:stoploss", "ok", "market")
    else:
        audit.add("order_types:stoploss", "fail", f"expected market, got {order_types.get('stoploss')}")

    if order_types.get("stoploss_on_exchange") is True:
        audit.add("order_types:stoploss_on_exchange", "ok", "true")
    else:
        audit.add(
            "order_types:stoploss_on_exchange",
            "fail",
            f"expected true, got {order_types.get('stoploss_on_exchange')}",
        )

    if order_types.get("stoploss_price_type") == "mark":
        audit.add("order_types:stoploss_price_type", "ok", "mark")
    else:
        audit.add("order_types:stoploss_price_type", "fail", f"expected mark, got {order_types.get('stoploss_price_type')}")


def check_callbacks(audit: StrategyAudit, strategy: IStrategy) -> None:
    required_overrides = ["populate_indicators", "populate_entry_trend", "populate_exit_trend", "leverage"]
    for attr in required_overrides:
        owner = owner_name(strategy, attr)
        if owner and owner != "IStrategy":
            audit.add(f"callback:{attr}", "ok", f"owner={owner}")
        else:
            audit.add(f"callback:{attr}", "fail", "not implemented by strategy")

    try:
        leverage = strategy.leverage(
            pair="BTC/USDT:USDT",
            current_time=datetime.now(UTC),
            current_rate=100.0,
            proposed_leverage=1.0,
            max_leverage=125.0,
            entry_tag=None,
            side="short",
        )
    except Exception as exc:
        audit.add("callback:leverage_value", "fail", f"{type(exc).__name__}: {exc}")
    else:
        if float(leverage) == EXPECTED_LEVERAGE:
            audit.add("callback:leverage_value", "ok", "50x")
        else:
            audit.add("callback:leverage_value", "fail", f"expected 50x, got {leverage}")

    custom_exit_owner = owner_name(strategy, "custom_exit")
    use_custom_exit = bool(getattr(strategy, "use_custom_exit", False))
    if use_custom_exit and custom_exit_owner and custom_exit_owner != "IStrategy":
        audit.add("callback:custom_exit", "ok", f"owner={custom_exit_owner}")
        signature = inspect.signature(strategy.custom_exit)
        required = {"pair", "trade", "current_time", "current_rate", "current_profit"}
        missing = sorted(required - set(signature.parameters))
        if missing:
            audit.add("callback:custom_exit_signature", "fail", "missing " + ", ".join(missing))
        else:
            audit.add("callback:custom_exit_signature", "ok", str(signature))
        try:
            result = strategy.custom_exit(
                pair="BTC/USDT:USDT",
                trade=FakeTrade(datetime.now(UTC) - timedelta(hours=9)),
                current_time=datetime.now(UTC),
                current_rate=100.0,
                current_profit=-0.01,
            )
        except Exception as exc:
            audit.add("callback:custom_exit_time_stop", "fail", f"{type(exc).__name__}: {exc}")
        else:
            if isinstance(result, str) and "time_stop" in result:
                audit.add("callback:custom_exit_time_stop", "ok", result)
            else:
                audit.add("callback:custom_exit_time_stop", "fail", f"expected time_stop reason, got {result!r}")
    else:
        audit.add("callback:custom_exit", "fail", f"use_custom_exit={use_custom_exit} owner={custom_exit_owner}")


def check_protections(audit: StrategyAudit, strategy: IStrategy) -> None:
    protections = getattr(strategy, "protections", [])
    stoploss_guards = [item for item in protections if item.get("method") == "StoplossGuard"]
    if not stoploss_guards:
        audit.add("protections:StoplossGuard", "fail", "missing")
        return
    guard = stoploss_guards[0]
    failures: list[str] = []
    for key, expected in REQUIRED_STOPLOSS_GUARD.items():
        if guard.get(key) != expected:
            failures.append(f"{key} expected {expected!r}, got {guard.get(key)!r}")
    required_profit = guard.get("required_profit")
    if required_profit is not None and float(required_profit) != 0.0:
        failures.append(f"required_profit expected 0.0 or absent, got {required_profit!r}")
    if failures:
        audit.add("protections:StoplossGuard", "fail", "; ".join(failures))
    else:
        audit.add("protections:StoplossGuard", "ok", json.dumps(guard, ensure_ascii=False, sort_keys=True))


def audit_strategy(strategy_name: str, config_path: Path) -> StrategyAudit:
    audit = StrategyAudit(strategy=strategy_name)
    final_config = build_config(config_path, strategy_name)
    source_config = {
        "strategy": strategy_name,
        "strategy_path": final_config.get("strategy_path"),
        "user_data_dir": final_config["user_data_dir"],
        "trading_mode": final_config.get("trading_mode", "futures"),
        "margin_mode": final_config.get("margin_mode", "isolated"),
        "dry_run": final_config.get("dry_run", True),
    }
    try:
        source_strategy = StrategyResolver._load_strategy(
            strategy_name,
            config=copy.deepcopy(source_config),
            extra_dir=source_config.get("strategy_path"),
        )
        final_strategy = StrategyResolver.load_strategy(copy.deepcopy(final_config))
    except Exception as exc:
        audit.add("strategy_load", "fail", f"{type(exc).__name__}: {exc}")
        return audit

    audit.add("strategy_load", "ok", f"{strategy_name} loaded")
    check_config_contract(audit, final_config)
    check_overrides(audit, source_strategy, final_strategy, final_strategy.config)
    check_strategy_contract(audit, final_strategy)
    check_callbacks(audit, final_strategy)
    check_protections(audit, final_strategy)
    return audit


def write_report(audits: list[StrategyAudit], report_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Dry-Run Strategy Risk Preflight",
        "",
        f"Generated: {datetime.now(UTC).isoformat()}",
        "",
    ]
    for audit in audits:
        lines.extend([f"## {audit.strategy}", ""])
        for check in audit.checks:
            marker = "PASS" if check.status == "ok" else "FAIL"
            lines.append(f"- {marker} `{check.name}`: {check.detail}")
        if audit.overrides:
            lines.extend(["", "### Config Overrides", ""])
            lines.append("| Attribute | Strategy value | Config value | Effective value |")
            lines.append("|---|---|---|---|")
            for item in audit.overrides:
                lines.append(
                    "| `{attribute}` | `{strategy_value}` | `{config_value}` | `{effective_value}` |".format(
                        attribute=item["attribute"],
                        strategy_value=json.dumps(item["strategy_value"], ensure_ascii=False, default=str),
                        config_value=json.dumps(item["config_value"], ensure_ascii=False, default=str),
                        effective_value=json.dumps(item["effective_value"], ensure_ascii=False, default=str),
                    )
                )
        lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")


def print_summary(audits: list[StrategyAudit], report_path: Path | None) -> None:
    for audit in audits:
        status = "FAIL" if audit.failed else "PASS"
        print(f"{status} {audit.strategy}")
        for check in audit.checks:
            marker = "ok" if check.status == "ok" else "FAIL"
            print(f"  [{marker}] {check.name}: {check.detail}")
        if audit.overrides:
            print("  overrides:")
            for item in audit.overrides:
                print(
                    f"    - {item['attribute']}: strategy={item['strategy_value']!r} "
                    f"config={item['config_value']!r} effective={item['effective_value']!r}"
                )
    if report_path:
        print(f"report: {rel(report_path)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--strategy", action="append", help="Strategy class to audit. Can be repeated.")
    parser.add_argument("--all-registry", action="store_true", help="Audit every strategy in strategy_registry.json too.")
    parser.add_argument("--json", action="store_true", help="Print JSON instead of a text summary.")
    parser.add_argument(
        "--report",
        type=Path,
        default=REPO_ROOT / "user_data/strategy_research/reports/latest_dryrun_strategy_risk_preflight.md",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.config.exists():
        print(f"missing config: {rel(args.config)}", file=sys.stderr)
        return 2
    names = selected_strategies(args, args.config)
    if not names:
        print("no strategies selected", file=sys.stderr)
        return 2

    audits = [audit_strategy(name, args.config) for name in names]
    write_report(audits, args.report)
    if args.json:
        print(
            json.dumps(
                {
                    "report": rel(args.report),
                    "audits": [
                        {
                            "strategy": audit.strategy,
                            "failed": audit.failed,
                            "checks": [check.__dict__ for check in audit.checks],
                            "overrides": audit.overrides,
                        }
                        for audit in audits
                    ],
                },
                ensure_ascii=False,
                indent=2,
                default=str,
            )
        )
    else:
        print_summary(audits, args.report)
    return 1 if any(audit.failed for audit in audits) else 0


if __name__ == "__main__":
    raise SystemExit(main())
