import argparse
import json
import logging
import os
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from statistics import median
from typing import Callable
from zoneinfo import ZoneInfo

sys.path.append(os.getcwd())

from adapters.ccxt_shim.breeze_ccxt import BreezeCCXT
from adapters.ccxt_shim.instrument import InstrumentSpec, InstrumentType, format_pair
from adapters.ccxt_shim.security_master import (
    SecurityMaster,
    find_latest_master_file,
    load_nfo_options_master,
)
from freqtrade.exceptions import OperationalException

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("gen_option_whitelist")


@dataclass(frozen=True)
class WhitelistInputs:
    """Inputs for option whitelist generation."""

    underlying: str
    expiry_policy: str
    atm_breadth: int
    n_expiries: int
    out_path: Path
    mode: str


@dataclass(frozen=True)
class OptionSelection:
    """Selected option contracts and related metadata for an underlying."""

    option_pairs: list[str]
    selected_expiries: list[str]
    atm_strike_by_expiry: dict[str, float]
    option_count: int
    ce_pe_pairs_count: int


def _kolkata_today() -> date:
    return datetime.now(tz=ZoneInfo("Asia/Kolkata")).date()


def _parse_expiry(expiry: str) -> date | None:
    try:
        return datetime.strptime(expiry, "%Y%m%d").date()
    except ValueError:
        logger.warning("Skipping invalid expiry: %s", expiry)
        return None


def _select_expiries(expiries: list[str], today: date, n_expiries: int) -> list[str]:
    sorted_expiries = sorted(expiries)
    future_expiries = [e for e in sorted_expiries if _parse_expiry(e) and _parse_expiry(e) >= today]
    if not future_expiries:
        logger.warning("No future expiries found; falling back to earliest available expiry.")
        future_expiries = sorted_expiries
    return future_expiries[: max(n_expiries, 1)]


def _most_common_step(differences: list[float]) -> float:
    counts = Counter(differences)
    if not counts:
        return 1.0
    max_count = max(counts.values())
    candidates = [diff for diff, count in counts.items() if count == max_count]
    return min(candidates)


def _compute_step(strikes: list[float]) -> float:
    unique_strikes = sorted(set(strikes))
    diffs = [b - a for a, b in zip(unique_strikes, unique_strikes[1:]) if b - a > 0]
    if diffs:
        return _most_common_step(diffs)
    return 1.0


def _resolve_spot(
    strikes: list[float],
    spot_fetcher: Callable[[], float | None] | None,
) -> float:
    spot = None
    if spot_fetcher is not None:
        try:
            spot = spot_fetcher()
        except OperationalException as exc:
            logger.warning("Ticker fetch failed, falling back to strike median: %s", exc)
        except Exception as exc:
            logger.exception("Unexpected error fetching ticker: %s", exc)
    if spot is None:
        spot = float(median(strikes))
        logger.info("Using median strike as spot: %s", spot)
    return spot


def _snap_to_strike(target: float, strikes: list[float]) -> float:
    return min(strikes, key=lambda strike: (abs(strike - target), strike))


def _select_strike_window(strikes: list[float], atm_strike: float, breadth: int) -> list[float]:
    sorted_strikes = sorted(strikes)
    if atm_strike not in sorted_strikes:
        sorted_strikes.append(atm_strike)
        sorted_strikes.sort()
    atm_index = sorted_strikes.index(atm_strike)
    start = max(atm_index - breadth, 0)
    end = min(atm_index + breadth + 1, len(sorted_strikes))
    return sorted_strikes[start:end]


def _build_spot_fetcher(underlying: str, mode: str) -> Callable[[], float | None]:
    api_key = os.environ.get("BREEZE_API_KEY", "mock_key")
    api_secret = os.environ.get("BREEZE_API_SECRET", "mock_secret")
    session_token = os.environ.get("BREEZE_SESSION_TOKEN", "mock_token")
    config = {"key": api_key, "secret": api_secret, "password": session_token}
    exchange = BreezeCCXT(config)

    def _fetch() -> float | None:
        ticker = exchange.fetch_ticker(f"{underlying}/INR")
        last = ticker.get("last")
        if last is None:
            logger.warning("Ticker did not include last price for %s", underlying)
            return None
        return float(last)

    if mode not in {"mock", "real"}:
        raise ValueError(f"Unsupported mode: {mode}")
    return _fetch


def generate_option_whitelist(
    security_master: SecurityMaster,
    inputs: WhitelistInputs,
    today: date,
    spot_fetcher: Callable[[], float | None] | None,
) -> list[str]:
    """Generate option whitelist pairs for the requested underlying."""
    selection = select_option_pairs(
        security_master=security_master,
        underlying=inputs.underlying,
        expiry_policy=inputs.expiry_policy,
        atm_breadth=inputs.atm_breadth,
        n_expiries=inputs.n_expiries,
        today=today,
        spot_fetcher=spot_fetcher,
    )
    if selection.ce_pe_pairs_count == 0:
        logger.error(
            "No CE/PE option pairs available for %s in selected expiries.", inputs.underlying
        )
        return []
    if selection.option_count == 0:
        logger.error("No option contracts found for %s", inputs.underlying)
        return []
    cash_pair = format_pair(
        InstrumentSpec(type=InstrumentType.CASH, underlying=inputs.underlying.upper(), quote="INR")
    )
    pairs = [cash_pair, *selection.option_pairs]
    return pairs


def select_option_pairs(
    security_master: SecurityMaster,
    underlying: str,
    expiry_policy: str,
    atm_breadth: int,
    n_expiries: int,
    today: date,
    spot_fetcher: Callable[[], float | None] | None,
) -> OptionSelection:
    """Select option pairs for an underlying using the P09 selection rules."""
    if expiry_policy != "nearest":
        raise ValueError(f"Unsupported expiry policy: {expiry_policy}")
    normalized_underlying = underlying.upper()
    relevant = [
        (
            normalized_underlying,
            key[1],
            float(key[2]),
            str(key[3]).upper(),
        )
        for key in security_master.by_contract
        if key[0] == normalized_underlying
    ]
    if not relevant:
        logger.warning("No option contracts found for %s", normalized_underlying)
        return OptionSelection([], [], {}, 0, 0)

    expiries = sorted({key[1] for key in relevant})
    selected_expiries = _select_expiries(expiries, today, n_expiries)
    if not selected_expiries:
        logger.warning("No expiries available for %s", normalized_underlying)
        return OptionSelection([], [], {}, 0, 0)

    option_pairs: list[str] = []
    atm_strike_by_expiry: dict[str, float] = {}
    ce_pe_pairs = 0

    for expiry in selected_expiries:
        strikes = sorted({float(key[2]) for key in relevant if key[1] == expiry})
        if not strikes:
            logger.warning("No strikes found for %s expiry %s", normalized_underlying, expiry)
            continue
        step = _compute_step(strikes)
        spot = _resolve_spot(strikes, spot_fetcher)
        atm_target = round(spot / step) * step
        atm_strike = _snap_to_strike(atm_target, strikes)
        atm_strike_by_expiry[expiry] = float(atm_strike)
        window = _select_strike_window(strikes, atm_strike, atm_breadth)
        for strike in window:
            current_strike = float(strike)
            for right in ("CE", "PE"):
                key = (normalized_underlying, expiry, current_strike, right)
                if key not in security_master.by_contract:
                    logger.debug("Contract not found in SecurityMaster: %s", key)
                    continue
                pair = format_pair(
                    InstrumentSpec(
                        type=InstrumentType.OPT,
                        underlying=normalized_underlying,
                        quote="INR",
                        expiry_yyyymmdd=expiry,
                        strike=current_strike,
                        right=right,
                    )
                )
                option_pairs.append(pair)
                if right == "CE":
                    has_ce = True
                if right == "PE":
                    has_pe = True

            # Check if we have both sides for stats, but pairs are added individually above
            if (
                normalized_underlying,
                expiry,
                current_strike,
                "CE",
            ) in security_master.by_contract and (
                normalized_underlying,
                expiry,
                current_strike,
                "PE",
            ) in security_master.by_contract:
                ce_pe_pairs += 1

    return OptionSelection(
        option_pairs=option_pairs,
        selected_expiries=selected_expiries,
        atm_strike_by_expiry=atm_strike_by_expiry,
        option_count=len(option_pairs),
        ce_pe_pairs_count=ce_pe_pairs,
    )


def _load_contracts() -> SecurityMaster | None:
    master_file = find_latest_master_file("FONSEScripMaster.txt")
    if not master_file:
        logger.error("SecurityMaster file not found.")
        return None
    master = load_nfo_options_master(master_file)
    return SecurityMaster(master.get("by_contract", {}))


def _write_pairs(path: Path, pairs: list[str]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(pairs, handle, indent=2)
        logger.info("Wrote %s pairs to %s", len(pairs), path)
    except OSError as exc:
        logger.error("Failed to write pairs to %s: %s", path, exc)
        raise


def parse_args() -> WhitelistInputs:
    parser = argparse.ArgumentParser(
        description="Generate option whitelist pairs from SecurityMaster."
    )
    parser.add_argument("--underlying", required=True, help="Underlying symbol, e.g. RELIANCE")
    parser.add_argument(
        "--expiry-policy",
        default="nearest",
        choices=["nearest"],
        help="Expiry selection policy (default: nearest)",
    )
    parser.add_argument("--atm-breadth", type=int, default=2, help="Strikes per side")
    parser.add_argument("--n-expiries", type=int, default=1, help="Number of expiries")
    parser.add_argument("--out", required=True, help="Output JSON path")
    parser.add_argument(
        "--mode",
        choices=["mock", "real"],
        default="mock",
        help="Ticker mode (default: mock)",
    )
    args = parser.parse_args()
    return WhitelistInputs(
        underlying=args.underlying,
        expiry_policy=args.expiry_policy,
        atm_breadth=max(args.atm_breadth, 0),
        n_expiries=max(args.n_expiries, 1),
        out_path=Path(args.out),
        mode=args.mode,
    )


def main() -> None:
    inputs = parse_args()
    if inputs.expiry_policy != "nearest":
        logger.error("Unsupported expiry policy: %s", inputs.expiry_policy)
        raise SystemExit(1)

    security_master = _load_contracts()
    if not security_master or not security_master.by_contract:
        raise SystemExit(1)

    today = _kolkata_today()
    spot_fetcher = _build_spot_fetcher(inputs.underlying.upper(), inputs.mode)
    pairs = generate_option_whitelist(security_master, inputs, today, spot_fetcher)
    if not pairs:
        logger.error("No pairs generated for %s", inputs.underlying)
        raise SystemExit(1)

    _write_pairs(inputs.out_path, pairs)


if __name__ == "__main__":
    main()
