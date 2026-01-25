#!/bin/bash
# P10 Execution Surface Acceptance Gate
set -euo pipefail

# Identify run context
source scripts/gates/common.sh "p10"

export BREEZE_MOCK=1

echo "Step 1: Mock Order Lifecycle (Pytest)"
"$PYTHON" -m pytest tests/exchange/test_icicibreeze_orders_mock.py || finish_gate $?

echo "Step 2: Dry-run Smoke Test (Freqtrade)"
CFG="user_data/generated/config_p09x_v1.json"
if [ ! -f "$CFG" ]; then
    echo "ERROR: Config missing: $CFG"
    finish_gate 1
fi

LOG_FILE="$ARTIFACT_DIR/dry_run.log"
# Run freqtrade in background
"$FREQTRADE" trade --dry-run \
  -c "$CFG" \
  --userdir user_data \
  -s IndiaOptionsAutoStrategy -vv > "$LOG_FILE" 2>&1 &
FT_PID=$!

# Wait up to 20s for RUNNING and TA Analysis
echo "Waiting for bot to reach RUNNING and perform TA analysis..."
SUCCESS=0
for i in {1..40}; do
    if grep -qE "Changing state to: RUNNING|Starting worker" "$LOG_FILE"; then
        if grep -q "TA Analysis Launched" "$LOG_FILE"; then
            SUCCESS=1
            break
        fi
    fi
    sleep 0.5
done

# Terminate cleanly
echo "Terminating bot (SIGINT)..."
kill -INT "$FT_PID" || true

# Wait up to 10s for clean exit, then kill
for i in {1..20}; do
    if ! kill -0 "$FT_PID" 2>/dev/null; then
        break
    fi
    sleep 0.5
done

if kill -0 "$FT_PID" 2>/dev/null; then
    echo "Bot still alive after 10s, sending SIGKILL..."
    kill -9 "$FT_PID" || true
fi
wait "$FT_PID" || true

# Assertions
echo "Verifying logs..."
if [ "$SUCCESS" -eq 0 ]; then
    echo "ERROR: Bot failed to reach RUNNING state or perform TA analysis within 20s"
    # Show last few lines of log to help debug
    tail -n 20 "$LOG_FILE"
    finish_gate 1
fi

grep -q "Using config: $CFG" "$LOG_FILE" || { echo "ERROR: Wrong config logged"; finish_gate 1; }
grep -q "Using resolved strategy IndiaOptionsAutoStrategy" "$LOG_FILE" || { echo "ERROR: Wrong strategy logged"; finish_gate 1; }
grep -q "Wallets synced." "$LOG_FILE" || { echo "ERROR: Wallets not synced"; finish_gate 1; }

if grep -qiE "OperationalException|not implemented" "$LOG_FILE"; then
    echo "ERROR: Forbidden string (OperationalException or 'not implemented') found in logs"
    grep -iE "OperationalException|not implemented" "$LOG_FILE"
    finish_gate 1
fi

echo "P10 Execution Surface passed"
finish_gate 0
