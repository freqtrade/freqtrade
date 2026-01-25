#!/bin/bash
# P05 Running State Gate
# Verifies that the bot can reach the RUNNING state in dry-run
set -euo pipefail

GATE_ID="p05"
source scripts/gates/common.sh "$GATE_ID"

require_timeout

echo "Step 1: Start Dry-run Smoke Test"
LOG_FILE="$ARTIFACT_DIR/dry_run.log"
export BREEZE_MOCK=1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

# Run for 15 seconds to ensure it has time to initialize
timeout 15s freqtrade trade -c user_data/config_icicibreeze.json --userdir user_data --strategy IndiaEquitySmokeStrategy --dry-run > "$LOG_FILE" 2>&1 || true

echo "Step 2: Assert 'Changing state to: RUNNING' exists in logs"
if grep -q "Changing state to: RUNNING" "$LOG_FILE"; then
    echo "[OK] Bot reached RUNNING state"
else
    echo "[FAIL] Bot did NOT reach RUNNING state. Check $LOG_FILE"
    finish_gate 1
fi

echo "P05 Running State passed"
finish_gate 0
