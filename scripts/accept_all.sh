#!/bin/bash
# Master Acceptance Script
# Runs all phase gates in order

set -e

GATES=(
    "p00_governance"
    "p01_ccxt_presence"
    "p02_mock_download_ohlcv"
    "p03_inr_pairs_presence"
    "p04_mode_routing_failfast"
    "p05_running_state"
    "p07_pair_naming_contract_listing"
    "p08_equity_strategy_smoke"
    "p09_options_strategy_accept"
    "p09x_universe_scanner_accept"
)

echo "=== STARTING FULL ACCEPTANCE SUITE ==="

FAILED=()

for gate in "${GATES[@]}"; do
    echo ""
    echo ">>> Running Gate: $gate"
    if bash "scripts/gates/${gate}.sh"; then
        echo ">>> Gate $gate PASSED"
    else
        echo ">>> Gate $gate FAILED"
        FAILED+=("$gate")
    fi
done

echo ""
echo "=== ACCEPTANCE SUITE SUMMARY ==="
if [ ${#FAILED[@]} -eq 0 ]; then
    echo "ALL GATES PASSED!"
    exit 0
else
    echo "GATES FAILED: ${FAILED[*]}"
    exit 1
fi
