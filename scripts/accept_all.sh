#!/bin/bash
# Master Acceptance Script
# Runs all phase gates in order with run isolation and bundling
set -euo pipefail

# Ensure we are running from the project root
cd "$(dirname "$0")/.."

RUN_ID=$(date +%Y%m%d_%H%M%S)
export RUN_ID

RUN_DIR="user_data/generated/accept_runs/$RUN_ID"
mkdir -p "$RUN_DIR"

# Full list of gates in sequence
ALL_GATES=(
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
    "p10_execution_surface"
    # "p11_risk_guardrails"
    "p12_backtest_paper_validation_and_metrics"
    "p12c_mock_30d_backtesting"
    "p13_ops_security_and_deployment"
    "p14_market_hours"
)

if [ "$#" -gt 0 ]; then
    GATES=()
    for TARGET in "$@"; do
        MATCH=""
        for g in "${ALL_GATES[@]}"; do
            if [[ "$g" == "$TARGET" ]]; then
                MATCH="$g"
                break
            fi
        done

        if [ -n "$MATCH" ]; then
            GATES+=("$MATCH")
        else
            echo "ERROR: Unknown gate: $TARGET"
            echo "Available gates: ${ALL_GATES[*]}"
            exit 1
        fi
    done
    echo "=== EXECUTING TARGET GATES: ${GATES[*]} (RUN_ID: $RUN_ID) ==="
else
    GATES=("${ALL_GATES[@]}")
    echo "=== STARTING FULL ACCEPTANCE SUITE (RUN_ID: $RUN_ID) ==="
fi
echo "Run Folder: $RUN_DIR"

FAILED=0

for gate in "${GATES[@]}"; do
    GATE_SCRIPT="scripts/gates/${gate}.sh"
    if [ ! -f "$GATE_SCRIPT" ]; then
        echo "ERROR: Gate script missing: $GATE_SCRIPT"
        FAILED=1
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
