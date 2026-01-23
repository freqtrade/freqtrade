import argparse
import logging
import os
import sys

sys.path.append(os.getcwd())

from adapters.ccxt_shim.instrument import InstrumentSpec, InstrumentType, format_pair, parse_pair

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("normalize_pair")


def _normalize_legacy(pair: str) -> str | None:
    import re
    from datetime import datetime

    opt_match = re.fullmatch(
        r"(?P<underlying>[A-Z0-9]+)-(?P<expiry>\d{4}-\d{2}-\d{2})-(?P<strike>\d+(?:\.\d+)?)-(?P<right>CE|PE)",
        pair,
        re.IGNORECASE,
    )
    if opt_match:
        expiry = datetime.strptime(opt_match.group("expiry"), "%Y-%m-%d").strftime("%Y%m%d")
        spec = InstrumentSpec(
            type=InstrumentType.OPT,
            underlying=opt_match.group("underlying").upper(),
            expiry_yyyymmdd=expiry,
            strike=float(opt_match.group("strike")),
            right=opt_match.group("right").upper(),
        )
        return format_pair(spec)

    fut_match = re.fullmatch(
        r"(?P<underlying>[A-Z0-9]+)-(?P<expiry>\d{4}-\d{2}-\d{2})-FUT",
        pair,
        re.IGNORECASE,
    )
    if fut_match:
        expiry = datetime.strptime(fut_match.group("expiry"), "%Y-%m-%d").strftime("%Y%m%d")
        spec = InstrumentSpec(
            type=InstrumentType.FUT,
            underlying=fut_match.group("underlying").upper(),
            expiry_yyyymmdd=expiry,
        )
        return format_pair(spec)

    return None


def normalize_pair(pair: str) -> str:
    try:
        return format_pair(parse_pair(pair))
    except ValueError:
        legacy = _normalize_legacy(pair)
        if legacy:
            return legacy
        raise


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Normalize legacy ICICI pair strings into canonical form."
    )
    parser.add_argument("pair", help="Pair string to normalize.")
    args = parser.parse_args()
    try:
        print(normalize_pair(args.pair))
    except ValueError as exc:
        logger.error("Invalid pair format: %s", args.pair)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
