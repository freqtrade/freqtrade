#!/bin/bash
# P04 Mode Routing Failfast Gate
# Verifies that real mode requires credentials and fails fast early
set -euo pipefail

GATE_ID="p04"
source scripts/gates/common.sh "$GATE_ID" "$@"

export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
LOG_FILE="$ARTIFACT_DIR/failfast.log"

if [ "$GATE_MODE" == "pos" ]; then
    echo "Step 1: Run in Real Mode without credentials (Positive - Expect Fail Fast)"
    export BREEZE_MOCK=0
    unset BREEZE_API_KEY BREEZE_API_SECRET BREEZE_SESSION_TOKEN
    
    freqtrade list-markets -c user_data/config_icicibreeze.json --userdir user_data > "$LOG_FILE" 2>&1 || true

    echo "Step 2: Assert robust error detection"
    if grep -E "(API Key.*not found|BREEZE_API_KEY|credentials.*missing)" "$LOG_FILE"; then
        echo "[OK] Correct error message found in logs (regex matched)"
    else
        echo "[FAIL] Expected credential error message not found in logs"
        finish_gate 1
    fi
    
    if grep -q "Mock mode enabled" "$LOG_FILE"; then
        echo "[FAIL] Mock mode was unexpectedly enabled in a real-mode test"
        finish_gate 1
    fi

elif [ "$GATE_MODE" == "neg" ]; then
    echo "Step 1: Run in Real Mode WITH credentials (Negative - Expect Success/No Fail Fast)"
    # We provide mock creds which should bypass the "Fail Fast" check but might fail later on connection
    # But the gate verifies "Fail Fast" logic. If we have creds, we pass the Fail Fast check.
    # Note: Use mock_key to trigger internal shim logic if possible, or just random key.
    export BREEZE_API_KEY="test_key"
    export BREEZE_API_SECRET="test_secret"
    export BREEZE_SESSION_TOKEN="test_token"
    export BREEZE_MOCK=0  # Real mode

    freqtrade list-markets -c user_data/config_icicibreeze.json --userdir user_data > "$LOG_FILE" 2>&1 || true

    if grep -E "(API Key.*not found|credentials.*missing)" "$LOG_FILE"; then
        echo "[FAIL] Fail Fast triggered despite credentials being present"
        finish_gate 1
    else
        echo "[OK] Did not Fail Fast on missing credentials"
    fi
    # It will likely fail on connection ("Failed to initialize Breeze"), that is fine.
    # We only care that it didn't block on "API Key not found".
fi

echo "P04 Mode Routing Failfast passed ($GATE_MODE)"
finish_gate 0
