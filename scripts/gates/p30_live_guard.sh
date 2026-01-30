#!/bin/bash
# P30: Live Order Guard
# Verifies double-lock mechanism for live orders.

set -euo pipefail

GATE_ID="p30"
source scripts/gates/common.sh "$GATE_ID" "$@"

# Run the python validation suite which covers both Block (Neg) and Allow (Pos) paths
# using Mocks.
export BREEZE_MOCK=1

if [ "$GATE_MODE" == "pos" ]; then
    echo ">>> Gate P30: Positive (Checking Double Lock Logic)..."
    if python3 scripts/p30_check_live_guard.py; then
        echo "P30_POS_PASS_DEFAULT_BLOCK"
        finish_gate 0
    else
        echo "[FAIL] P30 Pos Logic Failed"
        finish_gate 1
    fi

elif [ "$GATE_MODE" == "neg" ]; then
    echo ">>> Gate P30: Negative (Checking Market Hours Layering)..."
    if python3 scripts/gates/p30_neg_check.py; then
        echo "P30_NEG_EXPECTED_BLOCK"
        finish_gate 0
    else
        echo "[FAIL] P30 Neg Logic Failed"
        finish_gate 1
    fi

else
    echo "ERROR: Invalid mode"
    finish_gate 1
fi
