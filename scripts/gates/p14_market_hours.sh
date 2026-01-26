#!/bin/bash
# P14 Market Hours Guard Acceptance Gate
# Verifies deterministic blocking of entry orders outside NSE trading hours
# Uses Integration Tests (Pytest) as the reliable verification method due to
# harness limitations in Freqtrade dry-run/live-mock mode.
set -euo pipefail

# Identify run context
source scripts/gates/common.sh "p14"

echo "Step 1: Execute P14 Integration Tests"
TEST_CMD="$PYTHON -m pytest tests/exchange/test_icicibreeze_market_hours_block.py"

echo "Running: $TEST_CMD"
if $TEST_CMD; then
    echo "[OK] P14 Integration Tests passed."
else
    echo "[FAIL] P14 Integration Tests failed."
    finish_gate 1
fi

echo "P14 Market Hours Guard passed"
finish_gate 0
