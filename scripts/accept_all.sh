#!/bin/bash
# Master Acceptance Script
# Runs all phase gates in order with centralized logging and bundling

set -euo pipefail

RUN_ID=$(date +%Y%m%d_%H%M%S)
RUN_DIR="user_data/generated/accept_runs/$RUN_ID"
mkdir -p "$RUN_DIR/logs" "$RUN_DIR/status"

GATES=(
    "p00_governance"
    "p01_ccxt_presence"
    "p02_mock_download_ohlcv"
    "p03_inr_pairs_presence"
    "p04_mode_routing_failfast"
    "p05_running_state"
    "p06_green_gate"
    "p07_pair_naming_contract_listing"
    "p08_equity_strategy_smoke"
    "p09_options_strategy_accept"
    "p09x_universe_scanner_accept"
)

echo "=== STARTING FULL ACCEPTANCE SUITE (RUN_ID: $RUN_ID) ==="
echo "Run Directory: $RUN_DIR"

FAILED=0

for gate in "${GATES[@]}"; do
    GATE_SCRIPT="scripts/gates/${gate}.sh"
    if [ ! -f "$GATE_SCRIPT" ]; then
        echo "ERROR: Gate script missing: $GATE_SCRIPT"
        echo "FAIL" > "$RUN_DIR/status/${gate}.status"
        FAILED=1
        continue
    fi

    echo ""
    echo ">>> Running Gate: $gate"
    
    # Run gate and capture output to run dir log
    # We use a subshell to capture exit code while set -e is active
    if bash "$GATE_SCRIPT" 2>&1 | tee "$RUN_DIR/logs/${gate}.log"; then
        echo "PASS" > "$RUN_DIR/status/${gate}.status"
        echo ">>> Gate $gate PASSED"
    else
        echo "FAIL" > "$RUN_DIR/status/${gate}.status"
        echo ">>> Gate $gate FAILED (Check $RUN_DIR/logs/${gate}.log)"
        FAILED=1
    fi
done

echo ""
echo "=== ACCEPTANCE SUITE SUMMARY ==="
for gate in "${GATES[@]}"; do
    STATUS=$(cat "$RUN_DIR/status/${gate}.status" 2>/dev/null || echo "MISSING")
    printf "%-35s : %s\n" "$gate" "$STATUS"
done

# Create final bundle
BUNDLE="user_data/generated/accept_runs/${RUN_ID}.tar.gz"
tar -czf "$BUNDLE" -C "user_data/generated/accept_runs" "$RUN_ID"

echo ""
echo "Final Bundle: $BUNDLE"

if [ "$FAILED" -ne 0 ]; then
    echo "OVERALL STATUS: FAILED"
    exit 1
else
    echo "OVERALL STATUS: PASSED"
    exit 0
fi
