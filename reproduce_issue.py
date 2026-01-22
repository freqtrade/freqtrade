import ccxt
import sys
import os

# Ensure we import the shim
sys.path.insert(0, os.getcwd())
try:
    from freqtrade.exchange.icicibreeze import IcicibreezeShim

    print("Imported IcicibreezeShim")
except ImportError as e:
    print(f"Failed to import: {e}")
    sys.exit(1)

# Patch ccxt
ccxt.icicibreeze = IcicibreezeShim
if "icicibreeze" not in ccxt.exchanges:
    ccxt.exchanges.append("icicibreeze")

try:
    ex = ccxt.icicibreeze()
    print(f"IcicibreezeShim instantiated. Name: '{ex.name}' (Type: {type(ex.name)})")

    if ex.name is None:
        print("FAIL: Name is None")
        sys.exit(1)

    # Check describe
    d = ex.describe()
    print(f"Describe['name']: {d.get('name')}")

except Exception as e:
    print(f"Error instantiating: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)
