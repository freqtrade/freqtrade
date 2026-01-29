#!/bin/bash
# P27: Smart Money & Integrity
# Verifies FR-203 and FR-202 logic modules.

set -euo pipefail

GATE_ID="p27"
source scripts/gates/common.sh "$GATE_ID" "$@"

if [ "$GATE_MODE" == "pos" ]; then
    echo ">>> Gate P27: Positive (Unit Tests)..."
    
    # Run Valid Tests
    if pytest -v tests/strategy/test_p27_smart_money_fr203.py tests/strategy/test_p27_data_integrity_fr202.py; then
        echo "[OK] Unit tests passed."
        echo "P27_POS_PASS"
        echo ">>> Gate P27: SUCCESS"
        finish_gate 0
    else
        echo "[FAIL] Unit tests failed."
        finish_gate 1
    fi

elif [ "$GATE_MODE" == "neg" ]; then
    echo ">>> Gate P27: Negative (Logic Rejection)..."
    
    # Run the Negative Test which is DESIGNED TO FAIL
    # Logic returns False, Assert True => Test Fails
    
    set +e
    pytest tests/strategy/test_p27_negative_expected_fail.py > "$ARTIFACT_DIR/neg_test.log" 2>&1
    PYTEST_EXIT=$?
    set -e
    
    if [ $PYTEST_EXIT -ne 0 ]; then
        echo "[OK] Negative test failed as expected (Logic correctly rejected trade)."
        echo "P27_NEG_EXPECTED_FAIL"
        finish_gate 0
    else
        echo "[FAIL] Negative test PASSED! (Logic incorrectly allowed trade?)"
        cat "$ARTIFACT_DIR/neg_test.log"
        finish_gate 1
    fi

else
    echo "ERROR: Invalid mode"
    finish_gate 1
fi
