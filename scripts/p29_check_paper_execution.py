import sys
import os
import shutil
import sqlite3
import time
from pathlib import Path

# Ensure project root is in path
sys.path.append(os.getcwd())

from adapters.ccxt_shim.breeze_ccxt import BreezeCCXT


def verify_p29_paper_execution():
    # 1. Setup Config for P29
    ledger_path = Path("user_data/generated/temp_p29_ledger.sqlite")
    if ledger_path.exists():
        ledger_path.unlink()

    config = {
        "icicibreeze": {"paper_trading": {"enabled": True, "ledger_path": str(ledger_path)}},
        "options": {
            "key": os.environ.get("BREEZE_API_KEY", "test_key"),
            "secret": os.environ.get("BREEZE_API_SECRET", "test_secret"),
            "session_token": os.environ.get("BREEZE_SESSION_TOKEN", "test_token"),
        },
        "pair_whitelist": ["RELIANCE/INR"],
    }

    print(">>> Initializing BreezeCCXT in Real Mode + Paper Config...")
    exchange = BreezeCCXT(config)

    # Mock fetch_ticker to avoid real network call and authentication issues
    print(">>> Mocking fetch_ticker for Paper Logic verification...")
    exchange.fetch_ticker = lambda symbol, params=None: {
        "symbol": symbol,
        "last": 2500.0,
        "bid": 2499.0,
        "ask": 2501.0,
        "timestamp": int(time.time() * 1000),
    }

    # Assertions
    if exchange._is_mock_mode():
        print("ERROR: Exchange initialized in Mock Mode! P29 requires Real Mode context.")
        sys.exit(1)

    if not exchange.paper_mode:
        print("ERROR: Paper Mode not enabled!")
        sys.exit(1)

    if not exchange.paper_ledger:
        print("ERROR: Paper Ledger not initialized!")
        sys.exit(1)

    print(">>> Placing Buy Order (Should go to Ledger)...")
    try:
        # We need market data for create_order risk check (spread match)
        # In real mode this calls fetch_ticker.
        # If verification environment (CI) has no net/creds, fetch_ticker fails.
        # But we want to test ORDER ROUTING.
        # BreezeCCXT.create_order calls `fetch_ticker`.
        # We can mock fetch_ticker to return valid data to proceed to order creation,
        # OR we rely on "Shim" catching rate limit/network error and proceeding?
        # BreezeCCXT L608: catches Exception on ticker fetch and proceeds with empty price_surface.
        # So we should be fine even if network fails.

        order = exchange.create_order("RELIANCE/INR", "limit", "buy", 10, 2500.0)

        print(f">>> Order Created: {order['id']}")

        if "paper-" not in order["id"]:
            print(f"ERROR: Order ID {order['id']} does not look like a paper ID!")
            sys.exit(1)

        # Verify Ledger
        print(">>> Verifying Ledger...")
        if not ledger_path.exists():
            print("ERROR: Ledger DB file not found!")
            sys.exit(1)

        conn = sqlite3.connect(ledger_path)
        c = conn.cursor()
        rows = c.execute("SELECT id, symbol, amount FROM trades").fetchall()
        conn.close()

        if len(rows) == 0:
            print("ERROR: Ledger is empty!")
            sys.exit(1)

        row = rows[0]
        print(f"Ledger Row: {row}")

        if row[0] != order["id"]:
            print("ERROR: Ledger ID mismatch")
            sys.exit(1)

        if row[1] != "RELIANCE/INR":
            print("ERROR: Ledger Symbol mismatch")
            sys.exit(1)

        print(">>> P29 Verification Successful.")

    except Exception as e:
        print(f"ERROR: Unexpected exception during order placement: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    verify_p29_paper_execution()
