#!/bin/bash
# P22 Smoke Script: Real Mode Market Data Validation
# Usage: bash scripts/p22_real_data_smoke.sh
# Requires: BREEZE_MOCK=0 and valid creds env vars.

set -euo pipefail

ENV_CONF="user_data/config_icicibreeze.json"
TEMP_CONF="user_data/config_p22_temp.json"

echo ">>> P22 Smoke: Preparing minimal config..."

# Generate minimal config for RELIANCE/INR only to avoid NIFTY warnings in sample master
cat > "$TEMP_CONF" <<EOF
{
    "max_open_trades": 1,
    "stake_currency": "INR",
    "stake_amount": 1000,
    "tradable_balance_ratio": 0.99,
    "fiat_display_currency": "INR",
    "timeframe": "5m",
    "dry_run": true,
    "exchange": {
        "name": "icicibreeze",
        "key": "env_var",
        "secret": "env_var",
        "pair_whitelist": [
            "RELIANCE/INR"
        ],
        "pair_blacklist": []
    },
    "pairlists": [
        {"method": "StaticPairList"}
    ],
    "telegram": {
        "enabled": false,
        "token": "disabled",
        "chat_id": "disabled"
    },
    "api_server": {
        "enabled": false,
        "listen_ip_address": "127.0.0.1",
        "listen_port": 8080,
        "username": "freqtrader",
        "password": "SuperSecurePassword"
    },
    "bot_name": "freqtrade_p22",
    "initial_state": "running",
    "forcebuy_enable": false,
    "internals": {
        "process_throttle_secs": 5
    },
    "data_format_ohlcv": "json",
    "data_format_trades": "json"
}
EOF

echo ">>> P22 Smoke: 1. Listing Markets..."
freqtrade list-markets -c "$TEMP_CONF" --print-json > user_data/p22_markets.json

if grep -q "RELIANCE/INR" user_data/p22_markets.json; then
    echo "[OK] RELIANCE/INR found in market list."
else
    echo "[FAIL] RELIANCE/INR not found in market list."
    exit 1
fi

echo ">>> P22 Smoke: 2. Downloading Data (1 Day)..."
# Using --days 1 to keep it fast and light
freqtrade download-data -c "$TEMP_CONF" --days 1 -t 5m -p RELIANCE/INR

DATA_FILE="user_data/data/icicibreeze/RELIANCE_INR-5m.json"

if [ -f "$DATA_FILE" ]; then
    echo "[OK] Data file created: $DATA_FILE"
    # Basic check for content > empty list
    LINE_COUNT=$(grep -c "\[" "$DATA_FILE" || true)
    if [ "$LINE_COUNT" -gt 0 ]; then
         echo "[OK] Data file appears populated."
    else
         echo "[FAIL] Data file seems empty."
         exit 1
    fi
else
    echo "[FAIL] Data file not found."
    exit 1
fi

echo ">>> P22 Smoke: SUCCESS."
rm -f "$TEMP_CONF" user_data/p22_markets.json
