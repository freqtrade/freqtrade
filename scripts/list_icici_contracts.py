import argparse
import logging
import os
import sys
from collections import defaultdict

sys.path.append(os.getcwd())

try:
    from freqtrade.adapters.ccxt_shim.instrument import InstrumentSpec, InstrumentType, format_pair
    from freqtrade.adapters.ccxt_shim.security_master import (
        find_latest_master_file,
        load_nfo_options_master,
    )
except ImportError:
    from adapters.ccxt_shim.instrument import InstrumentSpec, InstrumentType, format_pair
    from adapters.ccxt_shim.security_master import find_latest_master_file, load_nfo_options_master

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("list_icici_contracts")


def _collect_option_index(
    contracts: dict[tuple[str, str, float, str], dict[str, object]]
) -> dict[str, dict[str, dict[float, set[str]]]]:
    by_underlying: dict[str, dict[str, dict[float, set[str]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(set))
    )
    for (underlying, expiry, strike, right) in contracts:
        by_underlying[underlying][expiry][strike].add(right)
    return by_underlying


def _collect_future_index(
    futures: dict[tuple[str, str], dict[str, object]]
) -> dict[str, set[str]]:
    by_underlying: dict[str, set[str]] = defaultdict(set)
    for (underlying, expiry) in futures:
        by_underlying[underlying].add(expiry)
    return by_underlying


def list_contracts(
    expiries: int,
    strikes: int,
    underlyings: int,
    underlying_filter: list[str] | None,
    kind: str,
) -> int:
    """List canonical ICICI Breeze contracts.

    Args:
        expiries: Maximum expiries per underlying.
        strikes: Maximum strikes per expiry (options only).
        underlyings: Maximum underlying symbols when no filter is provided.
        underlying_filter: Optional list of underlyings to include, in order.
        kind: One of "opt", "fut", or "both" to control contract types.

    Returns:
        Exit code (0 for success, 1 for errors).
    """
    master_file = find_latest_master_file("FONSEScripMaster.txt")
    if not master_file:
        logger.error("SecurityMaster file not found.")
        return 1
    master = load_nfo_options_master(master_file)
    options = master.get("by_contract", {})
    futures = master.get("by_future", {})
    if not options and not futures:
        logger.error("No options or futures found in SecurityMaster.")
        return 1

    option_index = _collect_option_index(options)
    future_index = _collect_future_index(futures)
    available_underlyings = sorted(set(option_index.keys()) | set(future_index.keys()))
    if underlying_filter:
        requested = [item for item in underlying_filter if item in available_underlyings]
        missing = [item for item in underlying_filter if item not in available_underlyings]
        if missing:
            logger.warning("Requested underlyings not found: %s", ", ".join(missing))
        if not requested:
            logger.error("None of the requested underlyings were found.")
            return 1
        selected_underlyings = requested
    else:
        selected_underlyings = available_underlyings[:underlyings]

    print("ICICI Breeze Contracts (canonical pairs)")
    print("======================================")
    for underlying in selected_underlyings:
        print(f"\nUnderlying: {underlying}")
        option_expiries = sorted(option_index.get(underlying, {}).keys())[:expiries]
        future_expiries = sorted(future_index.get(underlying, set()))[:expiries]
        for expiry in sorted(set(option_expiries) | set(future_expiries)):
            if kind in {"fut", "both"} and expiry in future_index.get(underlying, set()):
                fut_spec = InstrumentSpec(
                    type=InstrumentType.FUT,
                    underlying=underlying,
                    expiry_yyyymmdd=expiry,
                )
                print(f"  FUT  {format_pair(fut_spec)}")
            if kind in {"opt", "both"}:
                strikes_for_expiry = sorted(
                    option_index.get(underlying, {}).get(expiry, {}).keys()
                )
                for strike in strikes_for_expiry[:strikes]:
                    rights = sorted(option_index[underlying][expiry][strike])
                    for right in rights:
                        opt_spec = InstrumentSpec(
                            type=InstrumentType.OPT,
                            underlying=underlying,
                            expiry_yyyymmdd=expiry,
                            strike=strike,
                            right=right,
                        )
                        print(f"  OPT  {format_pair(opt_spec)}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="List canonical ICICI Breeze futures/options pairs from SecurityMaster."
    )
    parser.add_argument("--expiries", type=int, default=3, help="Expiries per underlying.")
    parser.add_argument("--strikes", type=int, default=10, help="Strikes per expiry.")
    parser.add_argument("--underlyings", type=int, default=5, help="Underlyings to display.")
    parser.add_argument(
        "--underlying",
        type=str,
        default=None,
        help="Comma-separated list of underlyings to display.",
    )
    parser.add_argument(
        "--type",
        dest="kind",
        choices=("opt", "fut", "both"),
        default="both",
        help="Contract type to display.",
    )
    args = parser.parse_args()
    underlying_filter = None
    if args.underlying:
        underlying_filter = [item.strip() for item in args.underlying.split(",") if item.strip()]
        if not underlying_filter:
            logger.error("Underlying filter was provided but empty after parsing.")
            raise SystemExit(1)
    raise SystemExit(
        list_contracts(
            args.expiries,
            args.strikes,
            args.underlyings,
            underlying_filter,
            args.kind,
        )
    )


if __name__ == "__main__":
    main()
