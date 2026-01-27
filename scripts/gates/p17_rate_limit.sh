#!/bin/bash
set -euo pipefail

GATE_ID="p17_rate_limit"
source scripts/gates/common.sh "$GATE_ID" "$@"

# P17 Rate Limit Gate
echo "=========================================================="
echo "GATE: P17 Rate Limiter ($GATE_MODE)"
echo "=========================================================="

if [ "$GATE_MODE" == "pos" ]; then
    echo "1. Verify Block Mode (Pytest Integration w/ Env)"
    # Uses tests/exchange/test_icicibreeze_rate_limit_applied.py
    # Tests handle their own mocking/logic.
    pytest -v tests/exchange/test_icicibreeze_rate_limit_applied.py || finish_gate $?
    echo "   [+] Block Mode Verified"

    echo "2. Verify Sleep Mode (Unit Test)"
    pytest -v tests/test_rate_limiter_sleep_mode.py || finish_gate $?
    echo "   [+] Sleep Mode Verified"

elif [ "$GATE_MODE" == "neg" ]; then
    echo "1. Verify Disabled Mode (Neg)"
    # Should run very fast and NOT block
    pytest -v tests/test_rate_limiter_disabled.py || finish_gate $?
    echo "   [+] Disabled Mode Verified"
fi

echo "----------------------------------------------------------"
echo "GATE P17-RateLimit PASSED"
echo "----------------------------------------------------------"

finish_gate 0
