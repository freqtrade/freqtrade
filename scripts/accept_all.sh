#!/bin/bash
# Master Acceptance Script
# Runs all phase gates in order with run isolation and bundling
set -euo pipefail

RUN_ID=$(date +%Y%m%d_%H%M%S)
export RUN_ID

RUN_DIR="user_data/generated/accept_runs/$RUN_ID"
mkdir -p "$RUN_DIR"

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
echo "Run Folder: $RUN_DIR"

FAILED=0

for gate in "${GATES[@]}"; do
    GATE_SCRIPT="scripts/gates/${gate}.sh"
    if [ ! -f "$GATE_SCRIPT" ]; then
        echo "ERROR: Gate script missing: $GATE_SCRIPT"
        continue
    fi

    echo ""
    echo ">>> Executing Gate: $gate"
    
    # Run gate. Note: common.sh handles internal artifact routing via RUN_ID
    if bash "$GATE_SCRIPT"; then
        echo ">>> Gate $gate: PASS"
    else
        echo ">>> Gate $gate: FAIL"
        # Print log path to help debugging
        LOG_PATH="$RUN_DIR/gates/${gate//_*/}/gate.log"
        echo "Check log: $LOG_PATH"
        FAILED=1
        break # Fail fast
    fi
done

echo ""
echo "=== ACCEPTANCE SUITE SUMMARY ==="
if [ "$FAILED" -eq 0 ]; then
    echo "OVERALL STATUS: PASSED"
else
    echo "OVERALL STATUS: FAILED"
fi

# Create final bundle
BUNDLE="user_data/generated/accept_runs/${RUN_ID}.tar.gz"
tar -czf "$BUNDLE" -C "user_data/generated/accept_runs" "$RUN_ID"

echo ""
echo "Final Artifact: $BUNDLE"

exit "$FAILED"
