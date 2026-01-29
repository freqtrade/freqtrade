#!/usr/bin/env python3
"""
P25 Build Security Master JSON
Parses raw TXT files (using adapters.ccxt_shim.security_master logic)
and emits a normalized, optimized JSON.
"""

import argparse
import datetime
import json
import logging
import os
import sys
from pathlib import Path

# Add project root to path to allow imports
sys.path.append(os.getcwd())

try:
    from adapters.ccxt_shim.security_master import load_nse_cash_master, load_nfo_options_master
except ImportError as e:
    print(f"Error importing security_master: {e}")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("P25Build")


def build(cash_path: str, fno_path: str, output_path: str):
    nse_path = Path(cash_path)
    fno_path = Path(fno_path)
    final_output = Path(output_path)

    if not nse_path.exists():
        # P25 Hardening requirement: exit 2 if input missing
        print(f"ERROR: Input file not found: {nse_path}")
        sys.exit(2)

    if not fno_path.exists():
        print(f"ERROR: Input file not found: {fno_path}")
        sys.exit(2)

    # Load Data
    logger.info(f"Parsing NSE Cash Master from {nse_path}...")
    nse_data = load_nse_cash_master(str(nse_path))

    logger.info(f"Parsing NFO Options Master from {fno_path}...")
    fno_data = load_nfo_options_master(str(fno_path))

    # Structuring Output
    output = {
        "meta": {
            "generated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "sources": {"cash": str(nse_path), "fno": str(fno_path)},
            "counts": {
                "cash": len(nse_data.get("by_symbol", {})),
                "options": len(fno_data.get("by_contract", {})),
                "futures": len(fno_data.get("by_future", {})),
            },
        },
        "cash": [],
        "options": [],
        "futures": [],
    }

    # Flatten Cash
    for info in nse_data.get("by_symbol", {}).values():
        output["cash"].append(info)
    output["cash"].sort(key=lambda x: x["symbol"])

    # Flatten Options
    for info in fno_data.get("by_contract", {}).values():
        output["options"].append(info)
    # Sort: Underlying, Expiry, Strike, Right
    output["options"].sort(
        key=lambda x: (x["underlying"], x["expiry_yyyymmdd"], x["strike"], x["right"])
    )

    # Flatten Futures
    for info in fno_data.get("by_future", {}).values():
        output["futures"].append(info)
    # Sort: Underlying, Expiry
    output["futures"].sort(key=lambda x: (x["underlying"], x["expiry_yyyymmdd"]))

    # Atomic Write
    tmp_path = final_output.with_suffix(".tmp")
    logger.info(f"Writing {tmp_path}...")
    with tmp_path.open("w") as f:
        json.dump(output, f, indent=None)

    logger.info(f"Renaming to {final_output}...")
    tmp_path.replace(final_output)

    logger.info("Build Complete.")
    logger.info(
        f"Stats: Cash={output['meta']['counts']['cash']}, "
        f"Options={output['meta']['counts']['options']}, "
        f"Futures={output['meta']['counts']['futures']}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cash", required=True, help="Path to NSEScripMaster.txt")
    parser.add_argument("--fno", required=True, help="Path to FONSEScripMaster.txt")
    parser.add_argument("--output", required=True, help="Path to output JSON")

    args = parser.parse_args()

    build(args.cash, args.fno, args.output)


if __name__ == "__main__":
    main()
