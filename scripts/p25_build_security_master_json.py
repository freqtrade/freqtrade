#!/usr/bin/env python3
"""
P25 Build Security Master JSON
Parses raw TXT files (using adapters.ccxt_shim.security_master logic)
and emits a normalized, optimized JSON.
"""

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

CACHE_DIR = Path("user_data/cache/security_master")
OUTPUT_FILE = CACHE_DIR / "latest.json"


def build():
    nse_path = CACHE_DIR / "NSEScripMaster.txt"
    fno_path = CACHE_DIR / "FONSEScripMaster.txt"

    if not nse_path.exists() or not fno_path.exists():
        logger.error(f"Missing input files in {CACHE_DIR}. Run fetch script first.")
        sys.exit(1)

    # Load Data
    logger.info("Parsing NSE Cash Master...")
    nse_data = load_nse_cash_master(str(nse_path))

    logger.info("Parsing NFO Options Master...")
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
    tmp_path = OUTPUT_FILE.with_suffix(".tmp")
    logger.info(f"Writing {tmp_path}...")
    with tmp_path.open("w") as f:
        json.dump(output, f, indent=None)

    logger.info(f"Renaming to {OUTPUT_FILE}...")
    tmp_path.replace(OUTPUT_FILE)

    logger.info("Build Complete.")
    logger.info(
        f"Stats: Cash={output['meta']['counts']['cash']}, "
        f"Options={output['meta']['counts']['options']}, "
        f"Futures={output['meta']['counts']['futures']}"
    )


if __name__ == "__main__":
    build()
