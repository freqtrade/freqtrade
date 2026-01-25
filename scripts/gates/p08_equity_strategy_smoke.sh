#!/bin/bash
# P08 Equity Strategy Smoke Gate
# Verifies strategy execution for equities
set -euo pipefail

GATE_ID="p08"
source scripts/gates/common.sh "$GATE_ID"

require_timeout

TIMEFRAME=${TIMEFRAME:-5m}
DAYS=${DAYS:-2}
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

echo "Step 1: Backtesting with IndiaEquitySmokeStrategy"
export BREEZE_MOCK=1
freqtrade backtesting -c user_data/config_icicibreeze.json --userdir user_data --strategy IndiaEquitySmokeStrategy --timeframe "$TIMEFRAME" --days "$DAYS" || finish_gate $?

echo "Step 2: Dry-run Smoke Test"
LOG_FILE="$ARTIFACT_DIR/dry_run.log"
timeout 15s freqtrade trade -c user_data/config_icicibreeze.json --userdir user_data --strategy IndiaEquitySmokeStrategy --dry-run > "$LOG_FILE" 2>&1 || true

if grep -q "Changing state to: RUNNING" "$LOG_FILE"; then
    echo "[OK] Bot reached RUNNING state with strategy"
else
    echo "[FAIL] Bot did NOT reach RUNNING state. Check $LOG_FILE"
    finish_gate 1
fi

echo "P08 Equity Strategy Smoke passed"
finish_gate 0
