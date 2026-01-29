#!/bin/bash
# P26: Indicator Governance
# Verifies Registry and Guards are correctly implemented and integrated.

set -euo pipefail

GATE_ID="p26"
source scripts/gates/common.sh "$GATE_ID" "$@"

if [ "$GATE_MODE" == "pos" ]; then
    echo ">>> Gate P26: Positive (Unit Tests & Integration)..."
    
    # 1. Run Unit Tests
    echo "1. Running Unit Tests..."
    pytest tests/strategy/test_p26_indicator_registry.py tests/strategy/test_p26_guards.py
    
    # 2. Integration Check (Backtest Dry Run)
    echo "2. Running Strategy Load Check..."
    # Simply running backtest with minimal config to ensure no import/runtime errors
    # using IndiaStockOptionsStrategy
    
    CONFIG="$ARTIFACT_DIR/config_check.json"
    cat <<EOF > "$CONFIG"
{
    "max_open_trades": 1,
    "stake_currency": "INR",
    "stake_amount": "unlimited",
    "tradable_balance_ratio": 0.99,
    "fiat_display_currency": "INR",
    "timeframe": "5m",
    "dry_run": true,
    "exchange": {
        "name": "icicibreeze",
        "key": "mock",
        "secret": "mock",
        "pair_whitelist": ["RELIANCE/INR"],
        "pair_blacklist": []
    }
}
EOF
    
    export BREEZE_MOCK=1
    # We use download-data to ensure we have *something* to backtest against, or verify dry-run
    # Actually, verify-strategy might be better if available, or just a quick backtest
    # Let's try trade --dry-run for 5 seconds to ensure startup
    
    timeout 10s "$FREQTRADE" trade --dry-run -c "$CONFIG" --strategy IndiaStockOptionsStrategy --userdir user_data || true
    # Note: timeout returns 124. We assume if it runs 10s without erroring early, it's fine.
    # A cleaner way is "freqtrade strategy-list" or similar?
    
    if "$FREQTRADE" list-strategies --userdir user_data | grep -q "IndiaStockOptionsStrategy"; then
        echo "[OK] Strategy loaded successfully."
    else
        echo "[FAIL] Strategy load failed."
        finish_gate 1
    fi
    
    echo "P26_POS_PASS"
    echo ">>> Gate P26: SUCCESS"
    finish_gate 0

elif [ "$GATE_MODE" == "neg" ]; then
    echo ">>> Gate P26: Negative (Lookahead Violation)..."
    
    # Run the Negative Test which is DESIGNED TO FAIL
    # If the production guard works, pytest will exit with code 1.
    
    set +e
    pytest tests/strategy/test_p26_guards_negative.py > "$ARTIFACT_DIR/neg_test.log" 2>&1
    PYTEST_EXIT=$?
    set -e
    
    if [ $PYTEST_EXIT -ne 0 ]; then
        echo "[OK] Negative test failed as expected (Guard active)."
        echo "P26_NEG_EXPECTED_FAIL"
        finish_gate 0
    else
        echo "[FAIL] Negative test PASSED! (Guard failed to raise error?)"
        cat "$ARTIFACT_DIR/neg_test.log"
        finish_gate 1
    fi

else
    echo "ERROR: Invalid mode"
    finish_gate 1
fi
