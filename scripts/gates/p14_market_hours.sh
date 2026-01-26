#!/bin/bash
# P14 Market Hours Guard Acceptance Gate
# Verifies deterministic blocking of entry orders outside NSE trading hours
set -euo pipefail

# Identify run context
source scripts/gates/common.sh "p14"

export BREEZE_MOCK=1
DB_URL="sqlite:///${ARTIFACT_DIR}/p14.sqlite"

echo "Step 1: Verify Market Closed Behavior (Blocking expected)"
LOG_CLOSED="$ARTIFACT_DIR/market_closed.log"

# Force market CLOSED
export FT_FORCE_MARKET_CLOSED=1
export FT_FORCE_MARKET_OPEN=0
# Force strategy to generate signals
export RISK_FORCE_SIGNAL=1

# Create P14 config with dry_run=false (to force exchange calls) but rely on BREEZE_MOCK for safety
CFG_BASE="user_data/generated/config_p09x_v1.json"
CFG_P14="$ARTIFACT_DIR/config_p14_live_mock.json"

if [ ! -f "$CFG_BASE" ]; then
    echo "ERROR: Config missing: $CFG_BASE"
    echo "Ensure P09X has run or config is available."
    finish_gate 1
fi

# Patch dry_run to false
jq '.dry_run = false | .trading_mode = "spot"' "$CFG_BASE" > "$CFG_P14"

echo "Starting Freqtrade with FT_FORCE_MARKET_CLOSED=1 (Live-Mock Mode)..."
"$FREQTRADE" trade \
  --db-url "$DB_URL" \
  -c "$CFG_P14" \
  --userdir user_data \
  -s IndiaEquitySmokeStrategy \
  -vv > "$LOG_CLOSED" 2>&1 &
FT_PID=$!

# Helper for robust termination
terminate_bot() {
    local PID=$1
    echo "Terminating bot (PID $PID)..."
    kill -INT "$PID" || true
    # Wait up to 10s
    for i in {1..20}; do
        if ! kill -0 "$PID" 2>/dev/null; then
            return 0
        fi
        sleep 0.5
    done
    echo "Bot still alive, forcing kill..."
    kill -9 "$PID" || true
    wait "$PID" || true
}

# Wait for potential block logs
FOUND_BLOCK=0
echo "Waiting for 'market_hours_block' event..."
for i in {1..40}; do
    if grep -q "market_hours_block" "$LOG_CLOSED"; then
        FOUND_BLOCK=1
        break
    fi
    # Also check generic message
    if grep -q "market_closed: blocking entry order" "$LOG_CLOSED"; then
        FOUND_BLOCK=1
        break
    fi
    sleep 1
done

terminate_bot "$FT_PID"

if [ "$FOUND_BLOCK" -eq 1 ]; then
    echo "[OK] Found expected block log."
else
    echo "[FAIL] Did NOT find 'market_hours_block' or blocking message in logs."
    echo "Tail of log:"
    tail -n 20 "$LOG_CLOSED"
    finish_gate 1
fi


echo "Step 2: Verify Market Open Behavior (Non-blocking)"
LOG_OPEN="$ARTIFACT_DIR/market_open.log"

# Force market OPEN
export FT_FORCE_MARKET_CLOSED=0
export FT_FORCE_MARKET_OPEN=1

# Use same DB or new? New to be clean.
DB_URL_OPEN="sqlite:///${ARTIFACT_DIR}/p14_open.sqlite"

echo "Starting Freqtrade with FT_FORCE_MARKET_OPEN=1 (Live-Mock Mode)..."
"$FREQTRADE" trade \
  --db-url "$DB_URL_OPEN" \
  -c "$CFG_P14" \
  --userdir user_data \
  -s IndiaEquitySmokeStrategy \
  -vv > "$LOG_OPEN" 2>&1 &
FT_PID_OPEN=$!

# Wait for indicators/signals
echo "Running for 15s..."
sleep 15

terminate_bot "$FT_PID_OPEN"

if grep -q "market_hours_block" "$LOG_OPEN"; then
    echo "[FAIL] Found unexpected BLOCK in forced-open market mode!"
    finish_gate 1
fi
if grep -q "market_closed: blocking entry order" "$LOG_OPEN"; then
    echo "[FAIL] Found unexpected BLOCK message in forced-open market mode!"
    finish_gate 1
fi

echo "[OK] No blocks found in forced-open mode."

echo "P14 Market Hours Guard passed"
finish_gate 0
