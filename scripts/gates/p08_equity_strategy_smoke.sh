#!/bin/bash
# P08 Equity Strategy Smoke Gate
# Verifies strategy execution for equities

GATE_ID="p08"
source scripts/gates/common.sh "$GATE_ID"

echo "Step 1: Backtesting with IndiaEquitySmokeStrategy"
export BREEZE_MOCK=1
$PYTHON -m freqtrade backtesting --config user_data/config_icicibreeze.json --strategy IndiaEquitySmokeStrategy --timeframe 5m --timerange 20260119-20260124 || finish_gate $?

echo "Step 2: Dry-run Smoke Test"
LOG_FILE="$OUT_DIR/dry_run.log"
timeout 15s $PYTHON -m freqtrade trade --config user_data/config_icicibreeze.json --strategy IndiaEquitySmokeStrategy --dry-run > "$LOG_FILE" 2>&1 || true

if grep -q "Changing state to: RUNNING" "$LOG_FILE"; then
    echo "[OK] Bot reached RUNNING state with strategy"
else
    echo "[FAIL] Bot did NOT reach RUNNING state. Check $LOG_FILE"
    finish_gate 1
fi

echo "P08 Equity Strategy Smoke passed"
finish_gate 0
