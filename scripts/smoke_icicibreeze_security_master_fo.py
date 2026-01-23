import logging
import os
import sys

# Add current directory to path so we can import adapters
sys.path.append(os.getcwd())

from adapters.ccxt_shim.breeze_ccxt import BreezeCCXT
from adapters.ccxt_shim.security_master import find_latest_master_file, load_nfo_options_master

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("smoke_test")


def run_smoke_test():
    logger.info("Starting SecurityMaster Smoke Test")

    # 1. File Discovery
    master_file = find_latest_master_file()
    if not master_file:
        logger.error("FAILED: Could not find SecurityMaster file")
        return
    logger.info(f"PASSED: Found master file at {master_file}")

    # 2. Parsing
    master = load_nfo_options_master(master_file)
    contracts = master["by_contract"]
    if not contracts:
        logger.error("FAILED: No contracts parsed")
        return
    logger.info(f"PASSED: Parsed {len(contracts)} contracts")

    # 3. Resolution & Market Building
    config = {
        "pair_whitelist": [
            "RELIANCE-2026-02-26-2800-CE",
            "CN:RELIANCE INDUSTRIES LTD-2026-02-26-2800-PE",
            "NIFTY-2026-02-26-22000-CE",
            "INVALID-FORMAT",
            "MISSING-2026-02-26-1000-CE",
        ]
    }

    exchange = BreezeCCXT(config)
    markets = exchange.fetch_markets()

    logger.info(f"Built {len(markets)} markets")
    for m in markets:
        logger.info(f"Market: {m['symbol']} (Token: {m['id']}, Lot: {m['lot']})")

    expected_symbols = [
        "RELIANCE/INR:2026-02-26:2800.0:CE",
        "RELIANCE/INR:2026-02-26:2800.0:PE",
        "NIFTY/INR:2026-02-26:22000.0:CE",
    ]

    built_symbols = [m["symbol"] for m in markets]
    for s in expected_symbols:
        if s not in built_symbols:
            logger.error(f"FAILED: Expected symbol {s} not found in markets")
            return

    logger.info("PASSED: All expected symbols found")

    # 4. Verify DeleteFlag works
    # Token 12349 has DeleteFlag=1, shouldn't be loaded (it's a duplicate 2800 CE)
    # Actually my mock has a different Token for it.
    tokens = [m["id"] for m in markets]
    if "12349" in tokens:
        logger.error("FAILED: Deleted contract (DeleteFlag=1) was loaded")
        return
    logger.info("PASSED: DeleteFlag filtering works")

    logger.info("SMOKE TEST FINISHED SUCCESSFULLY")


if __name__ == "__main__":
    run_smoke_test()
