#!/bin/bash
# P01 CCXT Presence Gate
# Verifies ccxt sync/async exchange exists

GATE_ID="p01"
source scripts/gates/common.sh "$GATE_ID"

echo "Step 1: Verify ICICI Breeze in CCXT"
export BREEZE_MOCK=1
export PYTHONPATH=.
$PYTHON scripts/verify_ccxt_compliance.py || finish_gate $?

echo "Step 2: Freqtrade list-markets (mock)"
# Using a base config, we don't need a derived one yet
$PYTHON -m freqtrade list-markets --config user_data/config_icicibreeze.json || finish_gate $?

echo "P01 CCXT Presence passed"
finish_gate 0
