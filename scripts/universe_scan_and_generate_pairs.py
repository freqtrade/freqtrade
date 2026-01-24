import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import yaml

# Add project root to sys.path for direct script execution
sys.path.insert(0, os.getcwd())

from adapters.ccxt_shim.instrument import InstrumentSpec, InstrumentType, format_pair
from adapters.ccxt_shim.security_master import (
    SecurityMaster,
    find_latest_master_file,
    load_nfo_options_master,
)
from scripts.gen_option_whitelist import OptionSelection, _kolkata_today, select_option_pairs

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("universe_scan")


@dataclass(frozen=True)
class UniverseConfig:
    """Universe configuration derived from the strategy YAML."""

    indices: list[str]
    stocks: list[str]
    top_n_stocks: int | None
    total_pairs_cap: int | None


@dataclass(frozen=True)
class OptionPolicy:
    """Option selection policy configuration derived from the strategy YAML."""

    expiry_policy: str
    atm_breadth: int
    total_pairs_cap: int | None
    require_two_sided: bool
    include_cash_pair: bool | None


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle)
            if payload is None:
                return {}
            if not isinstance(payload, dict):
                raise ValueError("YAML root must be a mapping.")
            return payload
    except FileNotFoundError as exc:
        logger.error("Strategy config not found: %s", path)
        raise
    except (yaml.YAMLError, ValueError) as exc:
        logger.error("Failed to parse strategy config %s: %s", path, exc)
        raise


def _normalize_symbol_list(value: Any, label: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list of symbols.")
    return [str(item).upper() for item in value if str(item).strip()]


def _parse_universe_config(payload: dict[str, Any]) -> UniverseConfig:
    universe = payload.get("universe")
    if not isinstance(universe, dict):
        raise ValueError("Strategy config missing 'universe' mapping.")

    indices = _normalize_symbol_list(universe.get("indices"), "universe.indices")
    stocks = _normalize_symbol_list(universe.get("stocks"), "universe.stocks")
    top_n_stocks = universe.get("top_n_stocks")
    total_pairs_cap = universe.get("total_pairs_cap")

    return UniverseConfig(
        indices=indices,
        stocks=stocks,
        top_n_stocks=int(top_n_stocks) if top_n_stocks is not None else None,
        total_pairs_cap=int(total_pairs_cap) if total_pairs_cap is not None else None,
    )


def _parse_option_policy(payload: dict[str, Any]) -> OptionPolicy:
    option_policy = payload.get("option_policy", {})
    if option_policy is None:
        option_policy = {}
    if not isinstance(option_policy, dict):
        raise ValueError("Strategy config 'option_policy' must be a mapping.")

    expiry_policy = option_policy.get("expiry_policy", "nearest")
    if not isinstance(expiry_policy, str):
        raise ValueError("option_policy.expiry_policy must be a string.")

    atm_breadth = option_policy.get("atm_breadth", 2)
    if not isinstance(atm_breadth, int):
        raise ValueError("option_policy.atm_breadth must be an integer.")
    if atm_breadth < 0:
        raise ValueError("option_policy.atm_breadth must be non-negative.")

    total_pairs_cap = option_policy.get("total_pairs_cap")
    if total_pairs_cap is not None and not isinstance(total_pairs_cap, int):
        raise ValueError("option_policy.total_pairs_cap must be an integer.")
    total_pairs_cap_value = int(total_pairs_cap) if total_pairs_cap is not None else None

    require_two_sided = option_policy.get("require_two_sided", True)
    if not isinstance(require_two_sided, bool):
        raise ValueError("option_policy.require_two_sided must be a boolean.")

    include_cash_pair = option_policy.get("include_cash_pair")
    if include_cash_pair is not None and not isinstance(include_cash_pair, bool):
        raise ValueError("option_policy.include_cash_pair must be a boolean.")

    return OptionPolicy(
        expiry_policy=expiry_policy,
        atm_breadth=atm_breadth,
        total_pairs_cap=total_pairs_cap_value,
        require_two_sided=require_two_sided,
        include_cash_pair=include_cash_pair,
    )


def _load_contracts() -> SecurityMaster:
    master_file = find_latest_master_file("FONSEScripMaster.txt")
    if not master_file:
        logger.error("SecurityMaster file not found.")
        raise FileNotFoundError("FONSEScripMaster.txt not found")
    master = load_nfo_options_master(master_file)
    return SecurityMaster(master.get("by_contract", {}))


def _default_report_path(out_path: Path) -> Path:
    if out_path.name.endswith("_pairs.json"):
        return out_path.with_name(out_path.name.replace("_pairs.json", "_report.json"))
    return out_path.with_name(f"{out_path.stem}_report.json")


def _write_json(path: Path, payload: Any) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        logger.info("Wrote %s", path)
    except OSError as exc:
        logger.error("Failed to write %s: %s", path, exc)
        raise


def _scan_underlying(
    security_master: SecurityMaster,
    underlying: str,
    today: date,
    option_policy: OptionPolicy,
) -> OptionSelection:
    return select_option_pairs(
        security_master=security_master,
        underlying=underlying,
        expiry_policy=option_policy.expiry_policy,
        atm_breadth=option_policy.atm_breadth,
        n_expiries=1,
        today=today,
        spot_fetcher=None,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan universe and generate pairs.")
    parser.add_argument(
        "--strategy-config",
        required=True,
        help="Strategy YAML config path",
    )
    parser.add_argument("--out", required=True, help="Output pairs JSON path")
    parser.add_argument(
        "--report",
        default=None,
        help="Output report JSON path (default: derived from --out)",
    )
    parser.add_argument(
        "--mode",
        choices=["mock", "real"],
        default="mock",
        help="Reserved for compatibility (default: mock)",
    )
    return parser.parse_args()


def _include_cash_pair(option_policy: OptionPolicy, is_index: bool) -> bool:
    if option_policy.include_cash_pair is None:
        return is_index
    return option_policy.include_cash_pair


def _resolve_total_pairs_cap(universe: UniverseConfig, option_policy: OptionPolicy) -> int | None:
    if option_policy.total_pairs_cap is not None:
        return option_policy.total_pairs_cap
    return universe.total_pairs_cap


def _record_selection(
    underlying: str,
    selection: OptionSelection,
    chosen_expiry: dict[str, str | None],
    chosen_atm_strike: dict[str, float | None],
) -> None:
    expiries = selection.selected_expiries
    chosen_expiry[underlying] = expiries[0] if expiries else None
    chosen_atm_strike[underlying] = (
        selection.atm_strike_by_expiry.get(expiries[0]) if expiries else None
    )


def _append_cash_pair(pairs: list[str], underlying: str) -> None:
    cash_pair = format_pair(
        InstrumentSpec(type=InstrumentType.CASH, underlying=underlying, quote="INR")
    )
    pairs.append(cash_pair)


def _build_pairs_report(
    universe: UniverseConfig,
    option_policy: OptionPolicy,
    security_master: SecurityMaster,
    today: date,
) -> tuple[list[str], dict[str, Any]]:
    pairs: list[str] = []
    selected_indices: list[str] = []
    selected_stocks: list[str] = []
    skipped_underlyings: list[dict[str, str]] = []
    chosen_expiry: dict[str, str | None] = {}
    chosen_atm_strike: dict[str, float | None] = {}

    # 1. Process Stocks first (Priority)
    stock_entries: list[tuple[str, list[str]]] = []
    for underlying in universe.stocks:
        try:
            selection = _scan_underlying(security_master, underlying, today, option_policy)
            _record_selection(underlying, selection, chosen_expiry, chosen_atm_strike)

            if selection.option_count == 0:
                skipped_underlyings.append({"underlying": underlying, "reason": "no options"})
                continue
            if option_policy.require_two_sided and selection.ce_pe_pairs_count == 0:
                skipped_underlyings.append(
                    {"underlying": underlying, "reason": "no CE+PE available"}
                )
                continue

            stock_pairs = list(selection.option_pairs)
            if _include_cash_pair(option_policy, is_index=False):
                stock_pairs = [
                    format_pair(
                        InstrumentSpec(type=InstrumentType.CASH, underlying=underlying, quote="INR")
                    ),
                    *stock_pairs,
                ]
            stock_entries.append((underlying, stock_pairs))
        except Exception as exc:
            logger.error("Failed to scan stock %s: %s", underlying, exc)
            skipped_underlyings.append({"underlying": underlying, "reason": f"error: {exc}"})

    if universe.top_n_stocks:
        stock_entries = stock_entries[: universe.top_n_stocks]

    for underlying, s_pairs in stock_entries:
        selected_stocks.append(underlying)
        pairs.extend(s_pairs)

    # 2. Process Indices
    for underlying in universe.indices:
        try:
            selection = _scan_underlying(security_master, underlying, today, option_policy)
            _record_selection(underlying, selection, chosen_expiry, chosen_atm_strike)

            if selection.option_count == 0:
                skipped_underlyings.append({"underlying": underlying, "reason": "no options"})
                continue
            if option_policy.require_two_sided and selection.ce_pe_pairs_count == 0:
                skipped_underlyings.append(
                    {"underlying": underlying, "reason": "no CE+PE available"}
                )
                continue

            if _include_cash_pair(option_policy, is_index=True):
                _append_cash_pair(pairs, underlying)
            pairs.extend(selection.option_pairs)
            selected_indices.append(underlying)
        except Exception as exc:
            logger.error("Failed to scan index %s: %s", underlying, exc)
            skipped_underlyings.append({"underlying": underlying, "reason": f"error: {exc}"})

    # 3. Apply Cap and Deterministic Sort
    total_pairs_cap = _resolve_total_pairs_cap(universe, option_policy)
    if total_pairs_cap and len(pairs) > total_pairs_cap:
        logger.warning(
            "Total pairs cap hit: trimming %s pairs to %s",
            len(pairs),
            total_pairs_cap,
        )
        pairs = pairs[:total_pairs_cap]

    # Ensure deterministic output ordering
    pairs.sort()

    report = {
        "selected_indices": selected_indices,
        "selected_stocks": selected_stocks,
        "skipped_underlyings": skipped_underlyings,
        "chosen_expiry": chosen_expiry,
        "chosen_atm_strike": chosen_atm_strike,
        "status": "success" if pairs else "empty",
    }
    return pairs, report


def main() -> None:
    args = _parse_args()
    config_path = Path(args.strategy_config)
    out_path = Path(args.out)
    report_path = Path(args.report) if args.report else _default_report_path(out_path)

    payload = _load_yaml(config_path)
    universe = _parse_universe_config(payload)
    option_policy = _parse_option_policy(payload)

    security_master = _load_contracts()
    today = _kolkata_today()

    pairs, report = _build_pairs_report(universe, option_policy, security_master, today)

    _write_json(out_path, pairs)
    _write_json(report_path, report)


if __name__ == "__main__":
    main()
