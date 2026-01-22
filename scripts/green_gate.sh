#!/bin/bash
set -euo pipefail

# Define paths to binaries in the virtual environment
FREQTRADE=".venv/bin/freqtrade"
PYTHON=".venv/bin/python"

echo "--- 1. Compile Check ---"
$PYTHON -m compileall -q freqtrade

echo "--- 2. Show Config ---"
$FREQTRADE show-config -c user_data/config_icicibreeze.json --userdir user_data >/tmp/show-config.json

echo "--- 3. List Markets ---"
$FREQTRADE list-markets -c user_data/config_icicibreeze.json --userdir user_data >/tmp/markets.txt

echo "--- 4. Ticker Smoke Test ---"
$PYTHON scripts/smoke_icicibreeze_ticker.py >/tmp/ticker.txt

echo "--- 5. Download Data Test ---"
$FREQTRADE download-data -c user_data/config_icicibreeze.json --userdir user_data --timeframes 5m --pairs BTC/USDT --days 2 -v >/tmp/dl_btc.txt

echo "--- 6. Dry Run Trade Test ---"
# Start trade in background, redirecting both stdout and stderr to capture logs
$FREQTRADE trade --dry-run -c user_data/config_icicibreeze.json --userdir user_data -s IcbcSmokeStrategy -vv >/tmp/trade.txt 2>&1 &
PID=$!
echo "Freqtrade started with PID $PID. Waiting 10s for startup..."
sleep 10
kill -INT $PID || true
wait $PID || true

echo "--- Verification ---"
# Check for key success markers in the logs
if grep -q "Using resolved exchange 'Icicibreeze'" /tmp/trade.txt; then
    echo "[OK] Exchange resolved"
else
    echo "[FAIL] Exchange resolution missing"
    cat /tmp/trade.txt
    exit 1
fi

if grep -q "Wallets synced" /tmp/trade.txt; then
    echo "[OK] Wallets synced"
else
    echo "[FAIL] Wallets verification missing"
    # Show the last part of the log for debugging
    tail -n 50 /tmp/trade.txt
    exit 1
fi

echo "GREEN_GATE=PASS"
