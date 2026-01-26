#!/bin/bash
# P12 Backtest Paper Validation and Metrics
set -euo pipefail

# Identify run context
source scripts/gates/common.sh "p12"

export BREEZE_MOCK=1
export RISK_FORCE_SIGNAL=1
unset BREEZE_API_KEY BREEZE_API_SECRET BREEZE_SESSION_TOKEN

CFG="user_data/generated/config_p09x_v1.json"
if [ ! -f "$CFG" ]; then
    echo "ERROR: Config missing: $CFG"
    finish_gate 1
fi

PAIR="RELIANCE/INR"
TF="5m"
DATADIR="user_data/data/icicibreeze"

echo "Step 1: Download Data (7 days)"
freqtrade download-data -c "$CFG" --userdir user_data --timeframes "$TF" --days 7 || finish_gate $?

echo "Step 2: Compute Timerange"
# Use underlying cash pair for timerange computation as it's most reliable
TIMERANGE=$(bash scripts/p12_timerange.sh "RELIANCE/INR" "$TF" "$DATADIR")
echo "Timerange from data: $TIMERANGE"

if [[ ! "$TIMERANGE" =~ ^[0-9]{8}-[0-9]{8}$ ]]; then
    echo "ERROR: Invalid timerange format: $TIMERANGE"
    finish_gate 1
fi

echo "Step 3: Run Backtesting"
LOG_FILE="$ARTIFACT_DIR/backtest.log"
TRADES_FILE="$ARTIFACT_DIR/backtest_trades.json"

# Create a backtest-specific config with loose ROI to ensure trades close
BT_CFG="$ARTIFACT_DIR/config_bt.json"
jq '.minimal_roi = {"0": -1} | .stoploss = -0.99' "$CFG" > "$BT_CFG"

freqtrade backtesting -c "$BT_CFG" \
  --userdir user_data \
  -s IndiaEquitySmokeStrategy \
  --pairs "$PAIR" \
  --timeframe "$TF" \
  --timerange "$TIMERANGE" \
  --starting-balance 10000000 \
  --fee 0.0 \
  --export trades \
  --export-directory "$ARTIFACT_DIR" \
  --export-filename "backtest_results.json" > "$LOG_FILE" 2>&1 || finish_gate $?

# Freqtrade might zip the results or name them based on strategy
# We search for any JSON in the artifact dir that contains trades (excluding meta and config)
REAL_TRADES_FILE=$(find "$ARTIFACT_DIR" -name "*.json" ! -name "*.meta.json" ! -name "config_bt.json" ! -name "status.json" | head -n 1)

# If no JSON but ZIP exists, unzip it
if [ -z "$REAL_TRADES_FILE" ]; then
    ZIP_FILE=$(find "$ARTIFACT_DIR" -name "*.zip" | head -n 1)
    if [ -n "$ZIP_FILE" ]; then
        echo "Step 3.1: Unzipping results"
        unzip -o "$ZIP_FILE" -d "$ARTIFACT_DIR"
        REAL_TRADES_FILE=$(find "$ARTIFACT_DIR" -name "*.json" ! -name "*.meta.json" ! -name "config_bt.json" ! -name "status.json" | head -n 1)
    fi
fi

if [ -z "$REAL_TRADES_FILE" ]; then
    echo "ERROR: Failed to locate backtest results JSON in $ARTIFACT_DIR"
    finish_gate 1
fi

echo "Step 4: Generate Metrics from $REAL_TRADES_FILE"
METRICS_FILE="$ARTIFACT_DIR/metrics_summary.json"
python3 scripts/p12_metrics_from_trades.py \
  --trades "$REAL_TRADES_FILE" \
  --out "$METRICS_FILE" \
  --pair "$PAIR" \
  --tf "$TF" \
  --timerange "$TIMERANGE" || finish_gate $?

echo "Step 5: Final Assertions"
if [ ! -f "$REAL_TRADES_FILE" ]; then
    echo "ERROR: Missing trades file: $REAL_TRADES_FILE"
    finish_gate 1
fi

if [ ! -f "$METRICS_FILE" ]; then
    echo "ERROR: Missing metrics file: $METRICS_FILE"
    finish_gate 1
fi

if grep -i "Traceback" "$LOG_FILE"; then
    echo "ERROR: Traceback found in backtest logs"
    tail -n 20 "$LOG_FILE"
    finish_gate 1
fi

echo "[OK] P12 Backtest Paper Validation and Metrics passed"
finish_gate 0
