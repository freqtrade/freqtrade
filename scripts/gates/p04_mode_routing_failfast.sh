#!/bin/bash
# P04 Mode Routing Failfast Gate
# Verifies that real mode requires credentials and fails fast early

GATE_ID="p04"
source scripts/gates/common.sh "$GATE_ID"

echo "Step 1: Run in Real Mode without credentials"
LOG_FILE="$OUT_DIR/failfast.log"
export BREEZE_MOCK=0
# We use a command that initializes the exchange but doesn't do much else
$PYTHON -m freqtrade list-markets --config user_data/config_icicibreeze.json > "$LOG_FILE" 2>&1 || true

echo "Step 2: Assert expected log line exists"
if grep -q "Breeze API Key not found in Config or ENV" "$LOG_FILE"; then
    echo "[OK] Correct error message found in logs"
else
    echo "[FAIL] Expected 'Breeze API Key not found' message not found in logs"
    finish_gate 1
fi

echo "P04 Mode Routing Failfast passed"
finish_gate 0
