#!/bin/bash
set -e
source scripts/gates/common.sh

# P17 Degraded Mode Gate
# Verifies degraded mode blocking.

echo "=========================================================="
echo "GATE: P17 Degraded Mode"
echo "=========================================================="

echo "1. Verify Forced Degraded Mode (Integration)"
# Uses tests/exchange/test_icicibreeze_integration_degraded.py
# Matches "degraded_block"
pytest -v tests/exchange/test_icicibreeze_integration_degraded.py
echo "   [+] Integration Verified"

echo "2. Verify Logic (Unit Tests)"
pytest -v tests/test_degraded_mode_force_block.py tests/test_degraded_mode_auto_trigger.py
echo "   [+] Logic Verified"

echo "----------------------------------------------------------"
echo "GATE P17-DegradedMode PASSED"
echo "----------------------------------------------------------"
