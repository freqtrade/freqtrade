#!/bin/bash
# P01 CCXT Presence Gate
# Verifies ccxt sync/async exchange exists
set -euo pipefail

GATE_ID="p01"
source scripts/gates/common.sh "$GATE_ID"

echo "Step 1: Verify ICICI Breeze in CCXT"
export BREEZE_MOCK=1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
$PYTHON scripts/verify_ccxt_compliance.py || finish_gate $?

echo "Step 2: Freqtrade list-markets (mock)"
freqtrade list-markets -c user_data/config_icicibreeze.json --userdir user_data || finish_gate $?

echo "P01 CCXT Presence passed"
finish_gate 0
