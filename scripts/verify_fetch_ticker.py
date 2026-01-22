import ccxt
import sys
import time
import pprint

# Ensure we can import freqtrade.exchange.icicibreeze
# This import is needed to register the shim
try:
    import freqtrade.exchange.icicibreeze
except ImportError as e:
    print(f"Could not import freqtrade.exchange.icicibreeze: {e}")
    sys.exit(1)

try:
    print("Initializing Exchange...")
    ex = ccxt.icicibreeze({"enableRateLimit": True})

    print("Loading markets...")
    ex.load_markets()

    print("Fetching Ticker for BTC/USDT...")
    t = ex.fetchTicker("BTC/USDT")

    print("\nTicker Result:")
    pprint.pprint(t)

    assert isinstance(t, dict), "Ticker must be a dictionary"
    assert t["symbol"] == "BTC/USDT", f"Symbol mismatch: {t.get('symbol')}"
    assert "last" in t, "Missing 'last' price"
    assert "timestamp" in t, "Missing 'timestamp'"
    assert "datetime" in t, "Missing 'datetime'"
    assert "bid" in t, "Missing 'bid'"
    assert "ask" in t, "Missing 'ask'"

    print("\nSUCCESS: fetchTicker verified.")

except Exception as e:
    print(f"\nERROR: {e}")
    import traceback

    traceback.print_exc()
