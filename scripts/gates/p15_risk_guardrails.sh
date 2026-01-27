#!/bin/bash
# P15 Risk Guardrails Acceptance Gate
set -euo pipefail

# Identify run context
source scripts/gates/common.sh "p15" "$@"

export BREEZE_MOCK=1
export RISK_FORCE_SIGNAL=1
# Force Market Open to isolate Risk Guard logic (otherwise MarketHours blocks first)
export FT_FORCE_MARKET_OPEN=1

# Ensure we are in a clean state regarding time
unset FT_IST_NOW

CFG="user_data/generated/config_p09x_v1.json"
if [ ! -f "$CFG" ]; then
    echo "ERROR: Config missing: $CFG"
    finish_gate 1
fi

LOG_FILE="$ARTIFACT_DIR/gate.log"

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
    # Case 1: Should block entries (Max Trades = 0)
    echo "Step 1: Negative Mode - Verify Risk Block"
    
    GATE_CFG="$ARTIFACT_DIR/config_p15_neg.json"
    OVERLAY="user_data/examples/config_p15_neg_overlay.json"
    
    # Merge configs
    # We use jq to merge overlay into base config
    jq -s '.[0] * .[1]' "$CFG" "$OVERLAY" > "$GATE_CFG"
    
    # Force time to trading hours to avoid intraday cutoff (overlay sets cutoff 09:15)
    # Actually overlay sets max_trades_per_day=0, so even if time is good, it blocks.
    # But let's set time to 10:00 to be safe
    export FT_IST_NOW="2026-01-26T10:00:00+05:30"
    
    # Remove --dry-run to enforce Shim-level Risk Guard execution
    # BREEZE_MOCK=1 ensures we do not hit real API
    "$FREQTRADE" trade \
      --db-url "sqlite:///$ARTIFACT_DIR/trades_neg.sqlite" \
      -c "$GATE_CFG" \
      --userdir user_data \
      -s IndiaEquitySmokeStrategy \
      -vv > "$LOG_FILE" 2>&1 &
    FT_PID=$!
    
    echo "Waiting for risk block..."
    BLOCK_CONFIRMED=0
    for i in {1..60}; do
        if grep -q "risk_block:" "$LOG_FILE"; then
            BLOCK_CONFIRMED=1
            break
        fi
        sleep 0.5
    done
    
    terminate_bot "$FT_PID" "Neg Bot"
    
    if [ "$BLOCK_CONFIRMED" -eq 0 ]; then
        echo "ERROR: risk_block not found in logs for Negative Mode"
        tail -n 20 "$LOG_FILE"
        finish_gate 1
    fi
    echo "[OK] Risk block confirmed: $(grep "risk_block:" "$LOG_FILE" | head -n 1)"

elif [ "$GATE_MODE" == "pos" ]; then
    # Case 2: Should allow entries
    echo "Step 1: Positive Mode - Verify Trading Allowed"
    
    GATE_CFG="$ARTIFACT_DIR/config_p15_pos.json"
    OVERLAY="user_data/examples/config_p15_pos_overlay.json"
    
    # Merge configs
    jq -s '.[0] * .[1]' "$CFG" "$OVERLAY" > "$GATE_CFG"
    
    # Force time to 10:00 (trading hours)
    export FT_IST_NOW="2026-01-26T10:00:00+05:30"

    # Remove --dry-run
    "$FREQTRADE" trade \
      --db-url "sqlite:///$ARTIFACT_DIR/trades_pos.sqlite" \
      -c "$GATE_CFG" \
      --userdir user_data \
      -s IndiaEquitySmokeStrategy \
      -vv > "$LOG_FILE" 2>&1 &
    FT_PID=$!
    
    echo "Waiting for successful entry..."
    ALLOW_CONFIRMED=0
    for i in {1..120}; do
        if grep -q "risk_block:" "$LOG_FILE"; then
            echo "ERROR: Unexpected risk_block found in Positive Mode"
            finish_gate 1
        fi
        # Search for RISK_OK or just standard entry log if RISK_OK isn't logged by guard (guard logs warning on block)
        # But strategy logs "RISK_OK entry" if using the old strategy hook?
        # NO, we implemented this in SHIM. Shim only logs warning on block.
        # So we look for "Put Order" or standard freqtrade entry message.
        # "Buy RELIANCE/INR" or similar.
        if grep -q "Found open order" "$LOG_FILE"; then
             ALLOW_CONFIRMED=1
             break
        fi
        if grep -q "Put Order" "$LOG_FILE"; then
             ALLOW_CONFIRMED=1
             break
        fi
        sleep 0.5
    done
    
    terminate_bot "$FT_PID" "Pos Bot"
    
    if [ "$ALLOW_CONFIRMED" -eq 0 ]; then
        echo "ERROR: Entry not confirmed in Positive Mode"
        tail -n 20 "$LOG_FILE"
        finish_gate 1
    fi
    echo "[OK] Entry confirmed"
fi

echo "P15 Risk Guardrails passed"
finish_gate 0
