#!/bin/bash
# P29: Real Mode Paper Trading
# Verifies interception of orders to local ledger.

set -euo pipefail

GATE_ID="p29"
source scripts/gates/common.sh "$GATE_ID" "$@"

if [ "$GATE_MODE" == "pos" ]; then
    echo ">>> Gate P29: Positive (Real Mode Paper Route)..."
    
    # Check for credentials
    if [ -z "${BREEZE_API_KEY:-}" ]; then
        echo ">>> WARNING: BREEZE_API_KEY not set."
        echo "P29_SKIP_MISSING_CREDS_POS"
        finish_gate 0
    fi
     
    # Only run if we didn't skip
    if python3 scripts/p29_check_paper_execution.py; then
        echo "P29_POS_PASS"
        finish_gate 0
    else
        echo "[FAIL] P29 Verification Script Failed."
        finish_gate 1
    fi

elif [ "$GATE_MODE" == "neg" ]; then
    echo ">>> Gate P29: Negative (Missing Creds Skip)..."
    
    # Force unset creds
    # Force unset creds
    unset BREEZE_API_KEY
    unset BREEZE_API_SECRET
    unset BREEZE_SESSION_TOKEN
    
    # We want to verify that the system detects missing creds and (conceptually) skips
    # or fails gracefully.
    # The requirement says "expected: SKIP with marker P29_SKIP_MISSING_CREDS".
    # We can write a tiny script to check init.
    
    cat <<EOF > "$ARTIFACT_DIR/neg_check.py"
import os
import sys
from adapters.ccxt_shim.breeze_ccxt import BreezeCCXT

def check_missing_creds():
    # Attempt init with empty config
    exchange = BreezeCCXT({})
    
    if exchange.breeze is None:
        print("Success: Breeze session is None (Graceful degradation)")
        sys.exit(0)
    else:
        print("Fail: Breeze session initialized despite missing creds!")
        sys.exit(1)

if __name__ == "__main__":
    check_missing_creds()
EOF

    if python3 "$ARTIFACT_DIR/neg_check.py"; then
        echo "P29_SKIP_MISSING_CREDS"
        finish_gate 0
    else
        echo "[FAIL] Neg Mode did not skip as expected."
        finish_gate 1
    fi

else
    echo "ERROR: Invalid mode"
    finish_gate 1
fi
