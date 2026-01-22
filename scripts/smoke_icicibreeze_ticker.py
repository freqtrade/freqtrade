import sys
import asyncio
from pathlib import Path

sys.path.append(str(Path(".").resolve()))

from freqtrade.configuration import Configuration
from freqtrade.resolvers.exchange_resolver import ExchangeResolver
from freqtrade.exchange.icicibreeze import IcicibreezeShim


def test_shim_direct():
    print("--- Testing Direct Shim Instantiation ---")
    ex = IcicibreezeShim({"dry_run": True})
    # Force load_markets to populate has if needed (though describe should do it)
    ex.load_markets()
    print("Features property:", getattr(ex, "features", "MISSING"))
    print("Has fetchOrder:", ex.has.get("fetchOrder"))
    print("Has fetchTicker:", ex.has.get("fetchTicker"))


def test_via_resolver():
    print("\n--- Testing via ExchangeResolver ---")
    import freqtrade.exchange.icicibreeze

    print("Module file:", freqtrade.exchange.icicibreeze.__file__)
    config = Configuration.from_files(["user_data/config_icicibreeze.json"])
    # Force dry run
    config["dry_run"] = True

    exchange = ExchangeResolver.load_exchange(config, validate=False)
    print("Loaded exchange class:", exchange.__class__.__name__)
    print("Loaded api class:", exchange._api.__class__.__name__)

    print("Exchange has fetchOrder:", exchange._api.has.get("fetchOrder"))

    print("Reloading markets...")
    exchange.reload_markets(force=True)
    # Check markets
    print("Market Keys:", list(exchange.markets.keys()))

    print("Fetching ticker for BTC/USDT...")
    ticker = exchange.fetch_ticker("BTC/USDT")
    print("Ticker:", ticker)


if __name__ == "__main__":
    try:
        test_shim_direct()
        test_via_resolver()
    except Exception as e:
        print("Test failed:", e)
        import traceback

        traceback.print_exc()
