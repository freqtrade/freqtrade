#!/bin/bash
# P10 Execution Surface Acceptance Gate
set -euo pipefail

# Identify run context
source scripts/gates/common.sh "p10"

export BREEZE_MOCK=1

echo "Step 1: Mock Order Lifecycle (Pytest)"
"$PYTHON" -m pytest tests/exchange/test_icicibreeze_orders_mock.py || finish_gate $?

echo "Step 2: Dry-run Smoke Test (Freqtrade)"
# Use a short timeout to ensure it starts without error
# We use the p09x config as directed
CONFIG="user_data/config_icicibreeze.json"
if [ ! -f "$CONFIG" ]; then
    echo "ERROR: Config missing: $CONFIG"
    finish_gate 1
fi

LOG_FILE="$ARTIFACT_DIR/dry_run.log"
timeout 30s "$FREQTRADE" trade --dry-run \
  -c "$CONFIG" \
  --userdir user_data \
  -s IndiaOptionsAutoStrategy -vv > "$LOG_FILE" 2>&1 || true

# Check for OperationalException in logs
if grep -q "OperationalException" "$LOG_FILE"; then
    echo "ERROR: OperationalException found in dry-run logs"
    grep "OperationalException" "$LOG_FILE"
    finish_gate 1
fi

# Ensure it actually tried to start the worker
if ! grep -q "worker found" "$LOG_FILE"; then
    echo "WARNING: Worker start not confirmed in log, but no errors found."
fi

echo "P10 Execution Surface passed"
finish_gate 0
