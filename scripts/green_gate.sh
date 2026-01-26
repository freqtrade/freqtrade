#!/bin/bash
set -euo pipefail

# Accept OUT_DIR as env var or default to gate-specific path
OUT_DIR="${OUT_DIR:-user_data/generated/gates/p00_green_gate}"
mkdir -p "$OUT_DIR"

# Define paths to binaries in the virtual environment
FREQTRADE=".venv/bin/freqtrade"
PYTHON=".venv/bin/python"

# Standard Environment
TIMEFRAME=${TIMEFRAME:-5m}
DAYS=${DAYS:-2}
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

# --- Preflight Check ---
if [ ! -f "$PYTHON" ]; then
    echo "ERROR: $PYTHON not found. Activate a venv first."
    exit 1
fi
if [ ! -f "$FREQTRADE" ]; then
    echo "ERROR: $FREQTRADE not found. Ensure freqtrade is installed in the venv."
    exit 1
fi

echo "Using Python: $PYTHON"
$PYTHON -V

echo "--- 1. Compile Check ---"
$PYTHON -m compileall -q -x 'user_data/generated' freqtrade adapters scripts user_data tests

echo "--- 2. Show Config ---"
freqtrade show-config -c user_data/config_icicibreeze.json --userdir user_data >"$OUT_DIR/show-config.json"

echo "--- 3. List Markets ---"
freqtrade list-markets -c user_data/config_icicibreeze.json --userdir user_data >"$OUT_DIR/markets.txt"

echo "--- 4. Ticker Smoke Test ---"
$PYTHON scripts/smoke_icicibreeze_ticker.py >"$OUT_DIR/ticker.txt"

echo "--- 5. Download Data Test (BTC & INR) ---"
# Clear existing data to ensure non-zero download for verification
rm -f user_data/data/icicibreeze/RELIANCE_INR-*.feather
rm -f user_data/data/icicibreeze/RELIANCE_INR-*.json

if [ "${ENABLE_BTC_TEST:-0}" -eq 1 ]; then
    echo "Downloading BTC/USDT..."
    freqtrade download-data -c user_data/config_icicibreeze.json --userdir user_data --timeframes "$TIMEFRAME" --pairs BTC/USDT --days "$DAYS" -v >"$OUT_DIR/dl_btc.txt" 2>&1
else
    echo "Skipping BTC/USDT download (ENABLE_BTC_TEST=0)"
fi
freqtrade download-data -c user_data/config_icicibreeze.json --userdir user_data --timeframes "$TIMEFRAME" --pairs RELIANCE/INR --days "$DAYS" -v >"$OUT_DIR/dl_inr.txt" 2>&1

echo "--- 6. Dry Run Trade Test ---"
# Start trade in background, redirecting both stdout and stderr to capture logs
freqtrade trade --dry-run -c user_data/config_icicibreeze.json --userdir user_data -s IndiaEquitySmokeStrategy -vv >"$OUT_DIR/trade.txt" 2>&1 &
PID=$!
echo "Freqtrade started with PID $PID. Waiting 15s for startup/running state..."
sleep 15
kill -INT $PID || true
sleep 2
kill -9 $PID 2>/dev/null || true
wait $PID || true

echo "--- Verification ---"

# 1. Mode Detection
if grep -q "Stub mode" "$OUT_DIR/trade.txt"; then
    echo "[OK] Mode detected: stub"
elif grep -q "Real mode" "$OUT_DIR/trade.txt"; then
    echo "[OK] Mode detected: real"
else
    echo "[FAIL] Mode detection missing in logs"
    exit 1
fi

# 2. OHLCV Reliability (RELIANCE/INR)
if grep -q "with length [1-9]" "$OUT_DIR/dl_inr.txt"; then
    LENGTH=$(grep -o "with length [0-9]*" "$OUT_DIR/dl_inr.txt" | head -n 1)
    echo "[OK] RELIANCE/INR data downloaded ($LENGTH)"
else
    echo "[FAIL] RELIANCE/INR data download failed or empty"
    cat "$OUT_DIR/dl_inr.txt"
    exit 1
fi

# 3. Trade State
if grep -q "Changing state to: RUNNING" "$OUT_DIR/trade.txt"; then
    echo "[OK] Reached RUNNING state"
else
    echo "[FAIL] Did not reach RUNNING state"
    # Show relevant logs
    grep "Changing state to" "$OUT_DIR/trade.txt" || true
    tail -n 20 "$OUT_DIR/trade.txt"
    exit 1
fi

# 4. Standard Checks (Exchange, Wallets)
if grep -q "Using resolved exchange 'Icicibreeze'" "$OUT_DIR/trade.txt"; then
    echo "[OK] Exchange resolved"
else
    echo "[FAIL] Exchange resolution missing"
    exit 1
fi

if grep -q "Wallets synced" "$OUT_DIR/trade.txt"; then
    echo "[OK] Wallets synced"
else
    echo "[FAIL] Wallets verification missing"
    exit 1
fi

echo "GREEN_GATE=PASS"
echo "GATE_RESULT=PASS ARTIFACTS=$OUT_DIR"
