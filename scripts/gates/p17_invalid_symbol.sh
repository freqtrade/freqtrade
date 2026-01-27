#!/bin/bash
set -e
source scripts/gates/common.sh

# P17 Invalid Symbol Resilience Gate
# Verifies system handles bad data gracefully.

echo "=========================================================="
echo "GATE: P17 Invalid Symbol"
echo "=========================================================="

echo "1. Verify Invalid Symbol Handling (Pytest)"
pytest -v tests/exchange/test_icicibreeze_invalid_symbol.py
echo "   [+] Resilience Verified"

echo "----------------------------------------------------------"
echo "GATE P17-InvalidSymbol PASSED"
echo "----------------------------------------------------------"
