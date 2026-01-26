#!/bin/bash
# P00 Governance Gate
# Verifies compilation and stable tests
set -euo pipefail

GATE_ID="p00"
source scripts/gates/common.sh "$GATE_ID" "$@"

export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

if [ "$GATE_MODE" == "pos" ]; then
    echo "Step 1: Python Compilation (Positive)"
    $PYTHON -m compileall -q -x 'user_data/generated' freqtrade adapters scripts user_data tests || finish_gate $?
    
    echo "Step 2: Subset of Stable Tests (Positive)"
    $PYTHON -m pytest -q tests/test_talib.py tests/test_instrument_parse_format.py || finish_gate $?

elif [ "$GATE_MODE" == "neg" ]; then
    echo "Step 1: Python Compilation (Negative - Expect Failure)"
    # Create invalid python file
    echo "def broken_syntax(:" > "$ARTIFACT_DIR/broken.py"
    
    if $PYTHON -m compileall -q "$ARTIFACT_DIR/broken.py" 2>/dev/null; then
        echo "[FAIL] compileall should have failed but succeeded"
        finish_gate 1
    else
        echo "[OK] compileall failed as expected"
    fi
fi

echo "P00 Governance passed ($GATE_MODE)"
finish_gate 0
