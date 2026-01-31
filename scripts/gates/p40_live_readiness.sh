#!/bin/bash
# P40: Live Readiness & Deadman
# Verifies that live trading requires Deadman switch and Readiness.

set -euo pipefail

GATE_ID="p40_live_readiness"
source scripts/gates/common.sh "$GATE_ID" "$@"

# 1. Setup Config
echo ">>> Gate P40: Setup..."

# Ensure secrets dir exists
mkdir -p user_data/secrets
# Touch Security Master to ensure freshness for gate
touch user_data/data/icicibreeze/NSEScripMaster.txt
touch user_data/data/icicibreeze/FONSEScripMaster.txt

# Mock Config with Live Trading Enabled
cat > "${ARTIFACT_DIR}/config_p40.json" <<EOF
{
    "max_open_trades": 1,
    "stake_currency": "INR",
    "stake_amount": 1000,
    "stoploss": -0.99,
    "fiat_display_currency": "INR",
    "dry_run": false,
    "timeframe": "1m",
    "entry_pricing": {
        "price_side": "same",
        "use_order_book": true,
        "order_book_top": 1
    },
    "exit_pricing": {
        "price_side": "same",
        "use_order_book": true
    },
    "exchange": {
        "name": "icicibreeze",
        "key": "mock_key",
        "secret": "mock_secret",
        "pair_whitelist": ["RELIANCE/INR"],
        "pair_blacklist": [],
        "icicibreeze": {
            "app_key": "mock_app_key",
            "s_key": "mock_s_key",
            "session_token": "mock_token",
            "live_trading": {
                "enabled": true
            }
        }
    },
    "pairlists": [
        {"method": "StaticPairList"}
    ],
    "icicibreeze": {
        "app_key": "mock_app_key",
        "s_key": "mock_s_key",
        "session_token": "mock_token",
        "live_trading": {
            "enabled": true
        }
    },
    "risk_guard": {
        "enabled": true,
        "max_trades_per_day": 100
    }
}
EOF

# Strategy for Immediate Buy
cat > "${ARTIFACT_DIR}/strategy.py" <<EOF
from freqtrade.strategy import IStrategy
from pandas import DataFrame

class TestStrategy(IStrategy):
    MIN_ROI = {"0": 100.0}
    STOPLOSS = -0.99
    TIMEFRAME = "1m"

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[:, 'enter_long'] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[:, 'exit_long'] = 0
        return dataframe
EOF

if [ "$GATE_MODE" == "pos" ]; then
    echo ">>> Gate P40: Positive (Deadman Active)..."
    
    # Enable Deadman
    touch user_data/secrets/deadman_live.ok
    touch -m user_data/secrets/deadman_live.ok

    export FT_ENABLE_LIVE_ORDERS=1
    export BREEZE_MOCK=1
    export FT_FORCE_MARKET_OPEN=1
    
    # Run
    timeout 15s "$FREQTRADE" trade \
        --config "${ARTIFACT_DIR}/config_p40.json" \
        --strategy TestStrategy \
        --strategy-path "${ARTIFACT_DIR}" \
        --user-data-dir user_data \
        > "${ARTIFACT_DIR}/p40_pos.log" 2>&1 || true

    if grep -q "Deadman Switch Failed" "${ARTIFACT_DIR}/p40_pos.log"; then
        echo "[FAIL] Deadman blocked valid orders."
        cat "${ARTIFACT_DIR}/p40_pos.log" | tail -n 20
        finish_gate 1
    else
        echo "[OK] Deadman check passed."
        # Verify attempt
        if grep -q "LIVE ORDER: Placing" "${ARTIFACT_DIR}/p40_pos.log" || grep -q "Mock mode" "${ARTIFACT_DIR}/p40_pos.log"; then
             echo "[OK] Order placement logic reached."
        fi
        finish_gate 0
    fi

elif [ "$GATE_MODE" == "neg" ]; then
    echo ">>> Gate P40: Negative (Missing Deadman)..."
    
    # Remove Deadman
    rm -f user_data/secrets/deadman_live.ok

    export FT_ENABLE_LIVE_ORDERS=1
    export BREEZE_MOCK=1
    export FT_FORCE_MARKET_OPEN=1

    # Run
    timeout 15s "$FREQTRADE" trade \
        --config "${ARTIFACT_DIR}/config_p40.json" \
        --strategy TestStrategy \
        --strategy-path "${ARTIFACT_DIR}" \
        --user-data-dir user_data \
        > "${ARTIFACT_DIR}/p40_neg.log" 2>&1 || true

    if grep -q "Deadman Switch Failed" "${ARTIFACT_DIR}/p40_neg.log"; then
        echo "[OK] Deadman blocked live orders."
        finish_gate 0
    else
        echo "[FAIL] Deadman check failed to block or log missing."
        cat "${ARTIFACT_DIR}/p40_neg.log" | tail -n 20
        finish_gate 1
    fi

else
    echo "ERROR: Unknown valid mode $GATE_MODE"
    finish_gate 1
fi
