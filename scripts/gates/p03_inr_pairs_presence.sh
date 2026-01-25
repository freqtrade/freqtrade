#!/bin/bash
# P03 INR Pairs Presence Gate
# Verifies that canonical INR pairs are present in market listing

GATE_ID="p03"
source scripts/gates/common.sh "$GATE_ID"

echo "Step 1: Freqtrade list-markets and check for RELIANCE/INR"
export BREEZE_MOCK=1
MARKETS_FILE="$OUT_DIR/markets.txt"
$PYTHON -m freqtrade list-markets --config user_data/config_icicibreeze.json > "$MARKETS_FILE" || finish_gate $?

if grep -q "RELIANCE/INR" "$MARKETS_FILE"; then
    echo "[OK] RELIANCE/INR found in market list"
else
    echo "[FAIL] RELIANCE/INR NOT found in market list"
    finish_gate 1
fi

echo "P03 INR Pairs Presence passed"
finish_gate 0
