import argparse
import json
import logging
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import yaml

from adapters.ccxt_shim.instrument import InstrumentSpec, InstrumentType, format_pair
from adapters.ccxt_shim.security_master import (
    SecurityMaster,
    find_latest_master_file,
    load_nfo_options_master,
)
from scripts.gen_option_whitelist import _kolkata_today, select_option_pairs

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("universe_scan")


@dataclass(frozen=True)
class UniverseConfig:
    """Universe configuration derived from the strategy YAML."""

    indices: list[str]
    stocks: list[str]
    top_n_stocks: int | None
    total_pairs_cap: int | None


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
) -> tuple[list[str], list[str], dict[str, float], int, int]:
    selection = select_option_pairs(
        security_master=security_master,
        underlying=underlying,
        expiry_policy="nearest",
        atm_breadth=2,
        n_expiries=1,
        today=today,
        spot_fetcher=None,
    )
    return (
        selection.option_pairs,
        selection.selected_expiries,
        selection.atm_strike_by_expiry,
        selection.option_count,
        selection.ce_pe_pairs_count,
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


def main() -> None:
    args = _parse_args()
    config_path = Path(args.strategy_config)
    out_path = Path(args.out)
    report_path = Path(args.report) if args.report else _default_report_path(out_path)

    payload = _load_yaml(config_path)
    universe = _parse_universe_config(payload)

    security_master = _load_contracts()
    today = _kolkata_today()

    pairs: list[str] = []
    selected_indices: list[str] = []
    selected_stocks: list[str] = []
    skipped_underlyings: list[dict[str, str]] = []
    chosen_expiry: dict[str, str | None] = {}
    chosen_atm_strike: dict[str, float | None] = {}

    for underlying in universe.indices:
        option_pairs, expiries, atm_by_expiry, _, _ = _scan_underlying(
            security_master, underlying, today
        )
        cash_pair = format_pair(
            InstrumentSpec(type=InstrumentType.CASH, underlying=underlying, quote="INR")
        )
        pairs.extend([cash_pair, *option_pairs])
        selected_indices.append(underlying)
        chosen_expiry[underlying] = expiries[0] if expiries else None
        chosen_atm_strike[underlying] = (
            atm_by_expiry.get(expiries[0]) if expiries else None
        )

    stock_entries: list[tuple[str, list[str]]] = []
    for underlying in universe.stocks:
        option_pairs, expiries, atm_by_expiry, option_count, ce_pe_pairs = _scan_underlying(
            security_master, underlying, today
        )
        chosen_expiry[underlying] = expiries[0] if expiries else None
        chosen_atm_strike[underlying] = (
            atm_by_expiry.get(expiries[0]) if expiries else None
        )

        if option_count == 0:
            skipped_underlyings.append({"underlying": underlying, "reason": "no options"})
            continue
        if ce_pe_pairs == 0:
            skipped_underlyings.append(
                {"underlying": underlying, "reason": "no CE+PE available"}
            )
            continue
        cash_pair = format_pair(
            InstrumentSpec(type=InstrumentType.CASH, underlying=underlying, quote="INR")
        )
        stock_entries.append((underlying, [cash_pair, *option_pairs]))

    if universe.top_n_stocks:
        stock_entries = stock_entries[: universe.top_n_stocks]

    for underlying, stock_pairs in stock_entries:
        selected_stocks.append(underlying)
        pairs.extend(stock_pairs)

    if universe.total_pairs_cap and len(pairs) > universe.total_pairs_cap:
        logger.warning(
            "Total pairs cap hit: trimming %s pairs to %s",
            len(pairs),
            universe.total_pairs_cap,
        )
        pairs = pairs[: universe.total_pairs_cap]

    report = {
        "selected_indices": selected_indices,
        "selected_stocks": selected_stocks,
        "skipped_underlyings": skipped_underlyings,
        "chosen_expiry": chosen_expiry,
        "chosen_atm_strike": chosen_atm_strike,
    }

    _write_json(out_path, pairs)
    _write_json(report_path, report)


if __name__ == "__main__":
    main()
