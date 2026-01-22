import ccxt
import sys
import pprint

# Ensure we can import freqtrade.exchange.icicibreeze
try:
    import freqtrade.exchange.icicibreeze
except ImportError as e:
    print(f"Could not import freqtrade.exchange.icicibreeze: {e}")
    sys.exit(1)

try:
    print("Initializing Exchange...")
    ex = ccxt.icicibreeze({"enableRateLimit": True})

    print("Loading markets...")
    mk = ex.load_markets()

    print(f"\nMarkets loaded: {len(mk)}")
    for sym in ["RELIANCE/INR", "TCS/INR", "BTC/USDT"]:
        print(f"Market {sym} present: {sym in mk}")
        if sym in mk:
            print(f"  Quote: {mk[sym]['quote']}")
            print(f"  Active: {mk[sym].get('active')}")

    print("\nCurrencies:")
    print(ex.currencies.keys())

    assert "INR" in ex.currencies, "INR missing from currencies"
    assert "RELIANCE/INR" in mk, "RELIANCE/INR missing from markets"

    print("\nSUCCESS: INR support verification passed.")

except Exception as e:
    print(f"\nERROR: {e}")
    import traceback

    traceback.print_exc()
