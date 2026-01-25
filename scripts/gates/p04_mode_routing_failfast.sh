#!/bin/bash
# P04 Mode Routing Failfast Gate
# Verifies that real mode requires credentials and fails fast early
set -euo pipefail

GATE_ID="p04"
source scripts/gates/common.sh "$GATE_ID"

echo "Step 1: Run in Real Mode without credentials"
LOG_FILE="$ARTIFACT_DIR/failfast.log"
export BREEZE_MOCK=0
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

# Explicitly unset credentials
unset BREEZE_API_KEY BREEZE_API_SECRET BREEZE_SESSION_TOKEN

# We use a command that initializes the exchange but doesn't do much else
freqtrade list-markets -c user_data/config_icicibreeze.json --userdir user_data > "$LOG_FILE" 2>&1 || true

echo "Step 2: Assert robust error detection"
# Accept any of these patterns
if grep -E "(API Key.*not found|BREEZE_API_KEY|credentials.*missing)" "$LOG_FILE"; then
    echo "[OK] Correct error message found in logs (regex matched)"
else
    echo "[FAIL] Expected credential error message not found in logs"
    finish_gate 1
fi

echo "Step 3: Ensure Mock mode was NOT enabled"
if grep -q "Mock mode enabled" "$LOG_FILE"; then
    echo "[FAIL] Mock mode was unexpectedly enabled in a real-mode test"
    finish_gate 1
fi

echo "P04 Mode Routing Failfast passed"
finish_gate 0
