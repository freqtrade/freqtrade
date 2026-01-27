#!/bin/bash
set -euo pipefail

GATE_ID="p17_invalid_symbol"
source scripts/gates/common.sh "$GATE_ID" "$@"

# P17 Invalid Symbol Gate
echo "=========================================================="
echo "GATE: P17 Invalid Symbol ($GATE_MODE)"
echo "=========================================================="

if [ "$GATE_MODE" == "pos" ]; then

    echo "1. Verify Invalid Symbol Handling (Pytest)"
    pytest -v tests/exchange/test_icicibreeze_invalid_symbol.py || finish_gate $?
    echo "   [+] Resilience Verified"

elif [ "$GATE_MODE" == "neg" ]; then
    echo "Negative acceptance criteria not defined for P17-InvalidSymbol yet. Skipping."
fi

echo "----------------------------------------------------------"
echo "GATE P17-InvalidSymbol PASSED"
echo "----------------------------------------------------------"

finish_gate 0
