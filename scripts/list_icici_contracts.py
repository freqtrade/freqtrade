import argparse
import logging
import os
import sys
from collections import defaultdict

sys.path.append(os.getcwd())

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


def list_contracts(expiries: int, strikes: int, underlyings: int) -> int:
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

    print("ICICI Breeze Contracts (canonical pairs)")
    print("======================================")
    for underlying in available_underlyings[:underlyings]:
        print(f"\nUnderlying: {underlying}")
        option_expiries = sorted(option_index.get(underlying, {}).keys())[:expiries]
        future_expiries = sorted(future_index.get(underlying, set()))[:expiries]
        for expiry in sorted(set(option_expiries) | set(future_expiries)):
            if expiry in future_index.get(underlying, set()):
                fut_spec = InstrumentSpec(
                    type=InstrumentType.FUT,
                    underlying=underlying,
                    expiry_yyyymmdd=expiry,
                )
                print(f"  FUT  {format_pair(fut_spec)}")
            strikes_for_expiry = sorted(option_index.get(underlying, {}).get(expiry, {}).keys())
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
    args = parser.parse_args()
    raise SystemExit(list_contracts(args.expiries, args.strikes, args.underlyings))


if __name__ == "__main__":
    main()
