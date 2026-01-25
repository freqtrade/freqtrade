#!/bin/bash
# P03 INR Pairs Presence Gate
# Verifies that canonical INR pairs are present in market listing
set -euo pipefail

GATE_ID="p03"
source scripts/gates/common.sh "$GATE_ID"

echo "Step 1: Freqtrade list-markets and check for RELIANCE/INR"
export BREEZE_MOCK=1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
MARKETS_FILE="$ARTIFACT_DIR/markets.txt"
freqtrade list-markets -c user_data/config_icicibreeze.json --userdir user_data > "$MARKETS_FILE" || finish_gate $?

if grep -q "RELIANCE/INR" "$MARKETS_FILE"; then
    echo "[OK] RELIANCE/INR found in market list"
else
    echo "[FAIL] RELIANCE/INR NOT found in market list"
    finish_gate 1
fi

echo "P03 INR Pairs Presence passed"
finish_gate 0
