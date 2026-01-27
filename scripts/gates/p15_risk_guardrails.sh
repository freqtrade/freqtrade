#!/bin/bash
# P15 Risk Guardrails Acceptance Gate
set -euo pipefail

# Identify run context
source scripts/gates/common.sh "p15" "$@"

export BREEZE_MOCK=1
export RISK_FORCE_SIGNAL=1
unset BREEZE_API_KEY BREEZE_API_SECRET BREEZE_SESSION_TOKEN

CFG="user_data/generated/config_p09x_v1.json"
if [ ! -f "$CFG" ]; then
    echo "ERROR: Config missing: $CFG"
    finish_gate 1
fi

# Prepare temporary config with increased max trades and position stacking enabled
GATE_CFG="$ARTIFACT_DIR/config_p15.json"
jq '.max_open_trades = 100 | .position_stacking = true | .dry_run_wallet = 10000000 | .stake_amount = 10' "$CFG" > "$GATE_CFG"

LOG_BLOCK="$ARTIFACT_DIR/dry_run_block.log"
LOG_ALLOW="$ARTIFACT_DIR/dry_run_allow.log"

# Clean up background process function
terminate_bot() {
    local pid=$1
    local name=$2
    echo "Terminating $name (SIGINT)..."
    kill -INT "$pid" || true
    for i in {1..20}; do
        if ! kill -0 "$pid" 2>/dev/null; then return 0; fi
        sleep 0.5
    done
    echo "$name still alive after 10s, sending SIGKILL..."
    kill -9 "$pid" || true
    wait "$pid" || true
}

if [ "$GATE_MODE" == "neg" ]; then
    # Case 1: Should block entries (Green Day Lock / High Risk)
    echo "Step 1: Case 1 - Should block entries (Green Day Lock) (Negative Mode)"
    export RISK_FORCE_DAILY_PROFIT_RATIO=0.015
    export RISK_FORCE_SIGNAL=1
    "$FREQTRADE" trade --dry-run \
      --db-url "sqlite:///$ARTIFACT_DIR/trades_case1.sqlite" \
      -c "$GATE_CFG" \
      --userdir user_data \
      -s IndiaEquitySmokeStrategy \
      -vv > "$LOG_BLOCK" 2>&1 &
    FT_PID=$!
    
    echo "Waiting for bot to evaluate risk (block)..."
    BLOCK_CONFIRMED=0
    for i in {1..60}; do
        if grep -q "RISK_BLOCK entry" "$LOG_BLOCK"; then
            BLOCK_CONFIRMED=1
            break
        fi
        sleep 0.5
    done
    
    terminate_bot "$FT_PID" "Case 1 Bot"
    
    if [ "$BLOCK_CONFIRMED" -eq 0 ]; then
        echo "ERROR: RISK_BLOCK not found in logs for Case 1 within 30s"
        echo "Last 20 lines of log:"
        tail -n 20 "$LOG_BLOCK"
        finish_gate 1
    fi
    echo "[OK] Risk block confirmed: $(grep "RISK_BLOCK entry" "$LOG_BLOCK" | head -n 1)"

elif [ "$GATE_MODE" == "pos" ]; then
    # Case 2: Should allow entries
    # Case 2: Should allow entries
    echo "Step 1: Case 2 - Should allow entries (Positive Mode)"
    export RISK_FORCE_DAILY_PROFIT_RATIO=0.0
    export RISK_FORCE_SIGNAL=1
    "$FREQTRADE" trade --dry-run \
      --db-url "sqlite:///$ARTIFACT_DIR/trades_case2.sqlite" \
      -c "$GATE_CFG" \
      --userdir user_data \
      -s IndiaEquitySmokeStrategy \
      -vv > "$LOG_ALLOW" 2>&1 &
    FT_PID=$!
    
    echo "Waiting for bot to reach TA Analysis (allow)..."
    ALLOW_CONFIRMED=0
    for i in {1..120}; do
        if grep -q "RISK_BLOCK entry" "$LOG_ALLOW"; then
             echo "ERROR: Unexpected RISK_BLOCK found in logs for Case 2"
             finish_gate 1
        fi
        if grep -q "RISK_OK entry" "$LOG_ALLOW"; then
            ALLOW_CONFIRMED=1
            break
        fi
        sleep 0.5
    done
    
    terminate_bot "$FT_PID" "Case 2 Bot"
    
    if [ "$ALLOW_CONFIRMED" -eq 0 ]; then
        echo "ERROR: Bot failed to reach TA analysis or log RISK_OK in Case 2"
        tail -n 20 "$LOG_ALLOW"
        finish_gate 1
    fi
    echo "[OK] Risk allow confirmed"
else
    echo "ERROR: Invalid --mode=$GATE_MODE (expected pos|neg)"
    finish_gate 1
fi

echo "P15 Risk Guardrails passed"
finish_gate 0
