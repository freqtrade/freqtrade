#!/bin/bash
# P20 API Smoke Test
# Verifies that the API server starts safely on 127.0.0.1 and responds to ping.

set -euo pipefail

# Setup
source .venv/bin/activate
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export BREEZE_MOCK=1

API_PORT=8080
API_URL="http://127.0.0.1:$API_PORT/api/v1/ping"
LOG_FILE="/tmp/p20_api_smoke.log"

echo ">>> P20: Starting Safe API Smoke Test..."

# Generate a temporary config with API enabled safely
cat user_data/config_icicibreeze.json | jq '.api_server = {
    "enabled": true,
    "listen_ip_address": "127.0.0.1",
    "listen_port": '$API_PORT',
    "username": "smoke_user",
    "password": "smoke_password",
    "jwt_secret_key": "smoke_secret_key",
    "verbosity": "info"
}' > /tmp/config_api_smoke.json

# Start Freqtrade in background (Dry-run)
echo "Starting Freqtrade API (Dry-Run)..."
freqtrade trade --dry-run \
    -c /tmp/config_api_smoke.json \
    --userdir user_data \
    -s IndiaEquitySmokeStrategy \
    --db-url sqlite:////tmp/freqtrade_smoke.sqlite \
    > "$LOG_FILE" 2>&1 &

PID=$!

cleanup() {
    echo "Stopping Freqtrade (PID: $PID)..."
    kill $PID || true
    wait $PID || true
    rm -f /tmp/config_api_smoke.json
}
trap cleanup EXIT

# Wait for API to come up
echo "Waiting for API to bind port $API_PORT..."
for i in {1..30}; do
    if grep -q "Uvicorn running on http://127.0.0.1:$API_PORT" "$LOG_FILE"; then
        echo "API confirmed listening on 127.0.0.1:$API_PORT"
        break
    fi
    if ! kill -0 $PID 2>/dev/null; then
        echo "Freqtrade process died unexpectedly!"
        cat "$LOG_FILE"
        exit 1
    fi
    sleep 1
done

# Verify Ping
echo "Verifying /api/v1/ping..."
RESPONSE=$(curl -s "$API_URL" || echo "FAIL")

if [[ "$RESPONSE" == *"status"* ]]; then
    echo "[OK] API Responded: $RESPONSE"
else
    echo "[FAIL] API Response invalid: $RESPONSE"
    cat "$LOG_FILE"
    exit 1
fi

echo ">>> P20: API Smoke Test SUCCESS"
