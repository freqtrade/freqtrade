#!/bin/bash
# P00 Governance Gate
# Verifies compilation and stable tests

GATE_ID="p00"
source scripts/gates/common.sh "$GATE_ID"

echo "Step 1: Python Compilation"
$PYTHON -m compileall -q freqtrade adapters scripts user_data tests || finish_gate $?

echo "Step 2: Subset of Stable Tests"
# Running a subset of tests that don't depend on external state/complex setup
$PYTHON -m pytest -q tests/test_talib.py tests/test_instrument_parse_format.py || finish_gate $?

echo "P00 Governance passed"
finish_gate 0
