#!/bin/bash
# P30: Live Order Guard
# Verifies double-lock mechanism for live orders.

set -euo pipefail

GATE_ID="p30"
source scripts/gates/common.sh "$GATE_ID" "$@"

# Run the python validation suite which covers both Block (Neg) and Allow (Pos) paths
# using Mocks.

echo ">>> Running P30 Live Guard Validation (Python)..."
if python3 scripts/p30_check_live_guard.py; then
    if [ "$GATE_MODE" == "pos" ]; then
        echo "P30_POS_PASS"
        finish_gate 0
    else
        # Neg mode implies we verified the BLOCKING property
        echo "P30_BLOCK_SUCCESS"
        finish_gate 0
    fi
else
    echo "[FAIL] P30 Validation Failed"
    finish_gate 1
fi
