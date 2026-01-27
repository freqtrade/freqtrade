#!/bin/bash
# P16 Order Router Acceptance Gate
set -euo pipefail

# Identify run context
source scripts/gates/common.sh "p16" "$@"

# Determinism
export BREEZE_MOCK=1
export FT_FORCE_MARKET_OPEN=1
unset BREEZE_API_KEY BREEZE_API_SECRET BREEZE_SESSION_TOKEN

# Paths
GATE_CFG="$ARTIFACT_DIR/config_p16.json"
# We reuse P09x config as base
BASE_CFG="user_data/generated/config_p09x_v1.json"
if [ ! -f "$BASE_CFG" ]; then
    echo "ERROR: Config missing: $BASE_CFG"
    finish_gate 1
fi
# Relax RiskGuard for P16 tests (disable intraday cutoff/max trades)
jq '.risk_guard.enabled = false' "$BASE_CFG" > "$GATE_CFG"

PY_SCRIPT="$ARTIFACT_DIR/run_p16_test.py"

if [ "$GATE_MODE" == "pos" ]; then
    echo "Step 1: Positive Case - Entry should be allowed"
    
    # Create Python harness
    cat <<EOF > "$PY_SCRIPT"
import logging
import sys
import json
from adapters.ccxt_shim.breeze_ccxt import BreezeCCXT

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("p16_pos")

def run():
    try:
        # Load Gate Config
        with open("$GATE_CFG", "r") as f:
            config = json.load(f)
        
        # Ensure mock mode options preserved/merged if needed, 
        # but BreezeCCXT expects flat config or specific keys? 
        # BreezeCCXT checks config['risk_guard'] directly.
        
        # Force mock mode in options
        if "options" not in config:
            config["options"] = {}
        config["options"]["mode"] = "mock"

        ex = BreezeCCXT(config)
        # Pos Case: Buy Order
        logger.info("Placing BUY order...")
        order = ex.create_order("RELIANCE/INR", "limit", "buy", 10, 2500)
        logger.info(f"Order created: {order['id']}")
        
        # Cleanup
        ex.cancel_order(order['id'], "RELIANCE/INR")
        print("P16_POS_SUCCESS")
    except Exception as e:
        logger.error(f"FAILURE: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run()
EOF

    # Run Python Script
    $PYTHON "$PY_SCRIPT" > "$GATE_LOG" 2>&1
    
    if grep -q "P16_POS_SUCCESS" "$GATE_LOG"; then
        echo "[OK] Positive case success"
    else
        echo "ERROR: Positive case failed"
        tail -n 20 "$GATE_LOG"
        finish_gate 1
    fi

elif [ "$GATE_MODE" == "neg" ]; then
    echo "Step 1: Negative Case - Sell without position should be blocked"
    
    # Create Python harness
    cat <<EOF > "$PY_SCRIPT"
import logging
import sys
import json
from adapters.ccxt_shim.breeze_ccxt import BreezeCCXT
from freqtrade.exceptions import OperationalException

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("p16_neg")

def run():
    try:
        # Load Gate Config
        with open("$GATE_CFG", "r") as f:
            config = json.load(f)
            
        if "options" not in config:
            config["options"] = {}
        config["options"]["mode"] = "mock"

        ex = BreezeCCXT(config)
        # Neg Case: Sell Order without position
        logger.info("Placing SELL order...")
        ex.create_order("RELIANCE/INR", "limit", "sell", 10, 2500)
        logger.error("FAILURE: Sell order should have been blocked")
        sys.exit(1)
    except OperationalException as e:
        if "order_router_block:buyer_only" in str(e):
             print(f"P16_NEG_SUCCESS: Caught expected block: {e}")
        else:
             logger.error(f"FAILURE: Caught unexpected exception: {e}")
             sys.exit(1)
    except Exception as e:
        logger.error(f"FAILURE: Caught unexpected exception type: {type(e)} {e}")
        sys.exit(1)

if __name__ == "__main__":
    run()
EOF

    # Run Python Script
    $PYTHON "$PY_SCRIPT" > "$GATE_LOG" 2>&1
    
    if grep -q "P16_NEG_SUCCESS" "$GATE_LOG"; then
         echo "[OK] Negative case success (Block confirmed)"
    else
         echo "ERROR: Negative case failed (No block observed)"
         tail -n 20 "$GATE_LOG"
         finish_gate 1
    fi

else
    echo "ERROR: Invalid mode $GATE_MODE"
    finish_gate 1
fi

echo "P16 Order Router passed"
finish_gate 0
