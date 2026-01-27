#!/bin/bash
set -euo pipefail

GATE_ID="p17_degraded_mode"
source scripts/gates/common.sh "$GATE_ID" "$@"

# P17 Degraded Mode Gate
echo "=========================================================="
echo "GATE: P17 Degraded Mode ($GATE_MODE)"
echo "=========================================================="

if [ "$GATE_MODE" == "pos" ]; then

    echo "1. Verify Forced Degraded Mode (Integration)"
    pytest -v tests/exchange/test_icicibreeze_integration_degraded.py || finish_gate $?
    echo "   [+] Integration Verified"

    echo "2. Verify Logic (Unit Tests)"
    pytest -v tests/test_degraded_mode_force_block.py tests/test_degraded_mode_auto_trigger.py || finish_gate $?
    echo "   [+] Logic Verified"

elif [ "$GATE_MODE" == "neg" ]; then
    echo "1. Verify Warn-Only Mode (Neg)"
    # Should allow entry despite degraded status
    pytest -v tests/test_degraded_mode_warn_only.py || finish_gate $?
    echo "   [+] Warn-Only Mode Verified"
fi

echo "----------------------------------------------------------"
echo "GATE P17-DegradedMode PASSED"
echo "----------------------------------------------------------"

finish_gate 0
