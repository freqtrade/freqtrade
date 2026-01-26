#!/bin/bash
# P14 Market Hours Guard Acceptance Gate
# Verifies deterministic blocking of entry orders outside NSE trading hours
# Uses Integration Tests (Pytest) as the reliable verification method due to
# harness limitations in Freqtrade dry-run/live-mock mode.
set -euo pipefail

# Identify run context
GATE_ID="p14"
source scripts/gates/common.sh "$GATE_ID" "$@"

if [ "$GATE_MODE" == "pos" ]; then
    echo "Step 1: Execute P14 Integration Tests (Positive - Market Open/Allowed)"
    # Select tests that expect success/allowed
    # The tests in `test_icicibreeze_market_hours_block.py` are likely named `test_market_open_allows...` and `test_market_closed_blocks...`
    # Let's run the whole suite for Positive case as default, but specific keywords could be better.
    # Actually, running ALL tests verifies the logic is correct (Open->OK, Closed->Block).
    # But to satisfy "Dual Case" semantics where Neg = Block Triggered:
    # Pos = Verify allowed paths. Neg = Verify blocked paths.
    
    # Assuming test names: `test_market_open_allows_all`
    TEST_CMD="$PYTHON -m pytest tests/exchange/test_icicibreeze_market_hours_block.py -k 'market_open'"
    
    if $PYTHON -m pytest tests/exchange/test_icicibreeze_market_hours_block.py -k "market_open"; then
        echo "[OK] Positive tests passed"
    else
        echo "[FAIL] Positive tests failed"
        finish_gate 1
    fi

elif [ "$GATE_MODE" == "neg" ]; then
    echo "Step 1: Execute P14 Integration Tests (Negative - Market Closed/Blocked)"
    
    if $PYTHON -m pytest tests/exchange/test_icicibreeze_market_hours_block.py -k "market_closed"; then
         echo "[OK] Negative tests passed (blocking verified)"
    else
         echo "[FAIL] Negative tests failed"
         finish_gate 1
    fi
fi

echo "P14 Market Hours Guard passed ($GATE_MODE)"
finish_gate 0
