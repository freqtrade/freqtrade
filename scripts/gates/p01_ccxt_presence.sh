#!/bin/bash
# P01 CCXT Presence Gate
# Verifies ccxt sync/async exchange exists
set -euo pipefail

GATE_ID="p01"
source scripts/gates/common.sh "$GATE_ID" "$@"

export BREEZE_MOCK=1

if [ "$GATE_MODE" == "pos" ]; then
    echo "Step 1: Verify ICICI Breeze in CCXT (Positive)"
    export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
    $PYTHON scripts/verify_ccxt_compliance.py || finish_gate $?

    echo "Step 2: Freqtrade list-markets (mock)"
    freqtrade list-markets -c user_data/config_icicibreeze.json --userdir user_data || finish_gate $?

elif [ "$GATE_MODE" == "neg" ]; then
    echo "Step 1: Verify ICICI Breeze in CCXT (Negative - Bad PYTHONPATH/No Site Packages)"
    # Reset PYTHONPATH to force import error
    export PYTHONPATH="/tmp"
    
    # Use -S to disable site-packages, ensuring we rely on PYTHONPATH/CWD which are broken/empty
    if $PYTHON -S scripts/verify_ccxt_compliance.py 2>/dev/null; then
        echo "[FAIL] verify_ccxt_compliance should have failed"
        finish_gate 1
    else
        echo "[OK] verify_ccxt_compliance failed as expected"
    fi
fi

echo "P01 CCXT Presence passed ($GATE_MODE)"
finish_gate 0
