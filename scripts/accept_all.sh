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
    "p12_backtest_paper_validation_and_metrics"
    "p12c_mock_30d_backtesting"
    "p13_ops_security_and_deployment"
    "p14_market_hours"
    "p15_risk_guardrails"
    "p16_order_router"
    "p17_rate_limit"
    "p17_degraded_mode"
    "p17_invalid_symbol"
    "p18_paper_forward_test"
    "p19_observability_audit"
    "p20_no_open_ports_pos"
    "p21_secrets_hygiene"
    "p22_real_mode_market_data"
    "p23_session_token_telegram"
    "p25_security_master_refresh"
    "p26_indicator_governance"
    "p27_smart_money"
    "p28_execution_microstructure"
    "p29_real_mode_paper_trade"
    "p30_live_guard"
)

# Parse flags
# Parse flags
TARGET_MODE="auto"
GATES_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --neg)
            TARGET_MODE="neg"
            shift
            ;;
        --pos)
            TARGET_MODE="pos"
            shift
            ;;
        *)
            GATES_ARGS+=("$1")
            shift
            ;;
    esac
done

# Gates that support Negative mode
HARDENED_GATES=(
    "p22_real_mode_market_data"
    "p25_security_master_refresh"
    "p26_indicator_governance"
    "p27_smart_money"
    "p28_execution_microstructure"
    "p29_real_mode_paper_trade"
    "p30_live_guard"
)

function is_hardened() {
    local gate=$1
    for h in "${HARDENED_GATES[@]}"; do
        if [[ "$h" == "$gate" ]]; then
            return 0
        fi
    done
    return 1
}

EXEC_PLAN=()

# 1. Resolve Execution Plan
if [ ${#GATES_ARGS[@]} -gt 0 ]; then
    # Custom Filter Mode
    for TARGET in "${GATES_ARGS[@]}"; do
        # Check for explicit suffixes
        if [[ "$TARGET" =~ ^(.*)_(pos|neg)$ ]]; then
            BASE_NAME="${BASH_REMATCH[1]}"
            SUFFIX="${BASH_REMATCH[2]}"
            
            # Validate Base Name
            MATCH=""
            for g in "${ALL_GATES[@]}"; do
                if [[ "$g" == "$BASE_NAME" ]]; then
                    MATCH="$g"
                    break
                fi
            done
            
            if [ -n "$MATCH" ]; then
                EXEC_PLAN+=("${MATCH}:${SUFFIX}")
            else
                echo "ERROR: Unknown gate base name: $BASE_NAME (from $TARGET)"
                echo "Available gates: ${ALL_GATES[*]}"
                exit 1
            fi
        else
            # No suffix - match base name exactly
            MATCH=""
            for g in "${ALL_GATES[@]}"; do
                if [[ "$g" == "$TARGET" ]]; then
                    MATCH="$g"
                    break
                fi
            done
            
            if [ -n "$MATCH" ]; then
                # Expand based on TARGET_MODE and Hardening
                if [[ "$TARGET_MODE" == "neg" ]]; then
                     EXEC_PLAN+=("${MATCH}:neg")
                elif [[ "$TARGET_MODE" == "pos" ]]; then
                     EXEC_PLAN+=("${MATCH}:pos")
                else
                     # Auto mode
                     EXEC_PLAN+=("${MATCH}:pos")
                     if is_hardened "$MATCH"; then
                         EXEC_PLAN+=("${MATCH}:neg")
                     fi
                fi
            else
                echo "ERROR: Unknown gate: $TARGET"
                echo "Available gates: ${ALL_GATES[*]}"
                exit 1
            fi
        fi
    done
    
    BANNER="CUSTOM_FILTER (count=${#EXEC_PLAN[@]})"

else
    # Full Suite
    # Iterate ALL_GATES and expand
    for g in "${ALL_GATES[@]}"; do
        if [[ "$TARGET_MODE" == "neg" ]]; then
             EXEC_PLAN+=("${g}:neg")
        elif [[ "$TARGET_MODE" == "pos" ]]; then
             EXEC_PLAN+=("${g}:pos")
        else
             # Auto - defaults to pos, plus neg if hardened
             EXEC_PLAN+=("${g}:pos")
             if is_hardened "$g"; then
                 EXEC_PLAN+=("${g}:neg")
             fi
        fi
    done
    
    if [[ "$TARGET_MODE" == "auto" ]]; then
        BANNER="suite(pos+neg)"
    else
        BANNER="suite(${TARGET_MODE}-only)"
    fi
fi

echo "=== STARTING ACCEPTANCE SUITE (RUN_ID: $RUN_ID, MODE: $BANNER) ==="
echo "Resolved Execution Plan:"
for item in "${EXEC_PLAN[@]}"; do
    echo " - $item"
done
echo "Run Folder: $RUN_DIR"

FAILED=0

for item in "${EXEC_PLAN[@]}"; do
    # Split item "gate:mode"
    # shellcheck disable=SC2001
    gate="${item%%:*}"
    CURRENT_MODE="${item##*:}"
    
    GATE_SCRIPT="scripts/gates/${gate}.sh"
    if [ ! -f "$GATE_SCRIPT" ]; then
        echo "ERROR: Gate script missing: $GATE_SCRIPT"
        FAILED=1
        continue
    fi

    echo ""
    echo ">>> Executing Gate: $gate (Mode: $CURRENT_MODE)"
    
    # Run gate with mode argument
    if bash "$GATE_SCRIPT" --mode="$CURRENT_MODE"; then
        echo ">>> Gate $gate ($CURRENT_MODE): PASS"
    else
        echo ">>> Gate $gate ($CURRENT_MODE): FAIL"
        # Print log path to help debugging (path now includes mode suffix)
        LOG_PATH="$RUN_DIR/gates/${gate}_${CURRENT_MODE}/gate.log"
        echo "Check log: $LOG_PATH"
        FAILED=1
        # Fail fast
        break 
    fi
done

if [ "$FAILED" -eq 1 ]; then
    # Break logic handled above, just ensure flow consistency
    :
fi

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
