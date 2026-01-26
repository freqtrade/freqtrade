#!/bin/bash
# P12.C Mock 30d Backtesting Gate
set -euo pipefail

# Identify run context
source scripts/gates/common.sh "p12c"

export BREEZE_MOCK=1
export MOCK_OHLCV_SEED=42
export RISK_FORCE_SIGNAL=1
unset BREEZE_API_KEY BREEZE_API_SECRET BREEZE_SESSION_TOKEN

CFG="user_data/generated/config_p09x_v1.json"
STRAT="IndiaOptionsAutoStrategy"
PAIR="RELIANCE/INR"
TF="5m"
DAYS=30

echo "Step 0: Clean existing mock data"
rm -f "user_data/data/icicibreeze/mock_cache/RELIANCE_INR-5m.json"
rm -f "user_data/data/icicibreeze/RELIANCE_INR-5m.feather"
rm -f "user_data/data/icicibreeze/RELIANCE_INR-5m.json"

echo "Step 1: Download Mock Data (30 days)"
# This will call fetch_ohlcv which will synthesize and persist to JSON
freqtrade download-data -c "$CFG" --userdir user_data --pairs "$PAIR" --timeframes "$TF" --days "$DAYS" --erase

# JSON file produced by our mock persistence
# Note: Freqtrade replaces / with _ in filenames
JSON_FILE="user_data/data/icicibreeze/mock_cache/RELIANCE_INR-5m.json"

echo "Step 2: Assert Candle Count"
if [ ! -f "$JSON_FILE" ]; then
    echo "ERROR: Mock JSON data file not found: $JSON_FILE"
    ls -la user_data/data/icicibreeze/mock_cache/
    finish_gate 1
fi

COUNT=$(jq '. | length' "$JSON_FILE")
EXPECTED=7500 # Approx 30 days * 0.9 coverage
if [ "$COUNT" -lt "$EXPECTED" ]; then
    echo "ERROR: Insufficient candle count: $COUNT < $EXPECTED"
    finish_gate 1
fi
echo "Verified candle count: $COUNT"

echo "Step 3: Compute Timerange"
# We need to make sure the timerange is based on the data we just generated
TIMERANGE=$(bash scripts/p12_timerange.sh "$PAIR" "$TF")
echo "Timerange: $TIMERANGE"

echo "Step 4: Run Backtest"
LOG_FILE="$ARTIFACT_DIR/backtest.log"

# Use loose ROI to ensure trades close and we get metrics
BT_CFG="$ARTIFACT_DIR/config_bt.json"
jq '.minimal_roi = {"0": -1} | .stoploss = -0.99' "$CFG" > "$BT_CFG"

freqtrade backtesting -c "$BT_CFG" --userdir user_data -s "$STRAT" \
  --pairs "$PAIR" --timeframe "$TF" --timerange "$TIMERANGE" \
  --starting-balance 10000000 \
  --fee 0.0 \
  --export trades \
  --export-directory "$ARTIFACT_DIR" \
  --export-filename "backtest_results.json" > "$LOG_FILE" 2>&1 || finish_gate $?

echo "Step 5: Generate Metrics"
# Freqtrade might name the result file with a timestamp, so we find it
REAL_TRADES_FILE=$(find "$ARTIFACT_DIR" -name "*.json" ! -name "*.meta.json" ! -name "config*.json" ! -name "status.json" | head -n 1)

if [ -z "$REAL_TRADES_FILE" ]; then
    # Try search for zip
    ZIP_FILE=$(find "$ARTIFACT_DIR" -name "*.zip" | head -n 1)
    if [ -n "$ZIP_FILE" ]; then
        unzip -o "$ZIP_FILE" -d "$ARTIFACT_DIR"
        REAL_TRADES_FILE=$(find "$ARTIFACT_DIR" -name "*.json" ! -name "*.meta.json" ! -name "config*.json" ! -name "status.json" | head -n 1)
    fi
fi

if [ -z "$REAL_TRADES_FILE" ]; then
    echo "ERROR: Could not locate trades file in $ARTIFACT_DIR"
    finish_gate 1
fi

METRICS_FILE="$ARTIFACT_DIR/metrics_summary.json"
python3 scripts/p12_metrics_from_trades.py \
  --trades "$REAL_TRADES_FILE" \
  --out "$METRICS_FILE" \
  --pair "$PAIR" \
  --tf "$TF" \
  --timerange "$TIMERANGE" || finish_gate $?

echo "[OK] P12.C Mock 30d Backtesting passed"
finish_gate 0
