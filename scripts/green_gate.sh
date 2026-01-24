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

echo "--- 5. Download Data Test (BTC & INR) ---"
$FREQTRADE download-data -c user_data/config_icicibreeze.json --userdir user_data --timeframes 5m --pairs BTC/USDT --days 2 -v >/tmp/dl_btc.txt 2>&1
$FREQTRADE download-data -c user_data/config_icicibreeze.json --userdir user_data --timeframes 5m --pairs RELIANCE/INR --days 2 -v >/tmp/dl_inr.txt 2>&1

echo "--- 6. Dry Run Trade Test ---"
# Start trade in background, redirecting both stdout and stderr to capture logs
$FREQTRADE trade --dry-run -c user_data/config_icicibreeze.json --userdir user_data -s IcbcSmokeStrategy -vv >/tmp/trade.txt 2>&1 &
PID=$!
echo "Freqtrade started with PID $PID. Waiting 15s for startup/running state..."
sleep 15
kill -INT $PID || true
sleep 2
kill -9 $PID 2>/dev/null || true
wait $PID || true

echo "--- Verification ---"

# 1. Mode Detection
if grep -q "Stub mode" /tmp/trade.txt; then
    echo "[OK] Mode detected: stub"
elif grep -q "Real mode" /tmp/trade.txt; then
    echo "[OK] Mode detected: real"
else
    echo "[FAIL] Mode detection missing in logs"
    exit 1
fi

# 2. OHLCV Reliability (RELIANCE/INR)
# Expected log: "with length 751." (or meaningful number)
if grep -q "with length [1-9]" /tmp/dl_inr.txt; then
    LENGTH=$(grep -o "with length [0-9]*" /tmp/dl_inr.txt | head -n 1)
    echo "[OK] RELIANCE/INR data downloaded ($LENGTH)"
else
    echo "[FAIL] RELIANCE/INR data download failed or empty"
    cat /tmp/dl_inr.txt
    exit 1
fi

# 3. Trade State
if grep -q "Changing state to: RUNNING" /tmp/trade.txt; then
    echo "[OK] Reached RUNNING state"
else
    echo "[FAIL] Did not reach RUNNING state"
    # Show relevant logs
    grep "Changing state to" /tmp/trade.txt
    tail -n 20 /tmp/trade.txt
    exit 1
fi

# 4. Standard Checks (Exchange, Wallets)
if grep -q "Using resolved exchange 'Icicibreeze'" /tmp/trade.txt; then
    echo "[OK] Exchange resolved"
else
    echo "[FAIL] Exchange resolution missing"
    exit 1
fi

if grep -q "Wallets synced" /tmp/trade.txt; then
    echo "[OK] Wallets synced"
else
    echo "[FAIL] Wallets verification missing"
    exit 1
fi

echo "GREEN_GATE=PASS"
