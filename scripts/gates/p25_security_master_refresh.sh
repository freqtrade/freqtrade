#!/bin/bash
# P25 Gate: Security Master Refresh
# Verifies atomic fetch and build of latest.json

set -euo pipefail

GATE_ID="p25"
source scripts/gates/common.sh "$GATE_ID" "$@"

echo ">>> Gate P25: Security Master... ($GATE_MODE)"

CACHE_DIR="user_data/cache/security_master"
JSON_FILE="$CACHE_DIR/latest.json"

# Clean start
rm -rf "$CACHE_DIR"

if [[ "$GATE_MODE" == "neg" ]]; then
    # Negative Case: Run build without fetch (expect fail)
    echo "1. Testing Negative Build (Missing Inputs)..."
    set +e
    python3 scripts/p25_build_security_master_json.py > /dev/null 2>&1
    RET=$?
    set -e
    
    if [ "$RET" -ne 0 ]; then
        echo "[OK] Build failed as expected without inputs."
        echo "P25_NEG_EXPECTED_FAIL"
        finish_gate 0
    else
        echo "[FAIL] Build succeeded despite missing inputs?"
        finish_gate 1
    fi
fi

# Positive Case
echo "1. Fetching Master (Mock Mode)..."
python3 scripts/p25_fetch_security_master.py --mock

if [[ ! -f "$CACHE_DIR/NSEScripMaster.txt" ]]; then
    echo "[FAIL] Fetch failed to produce NSEScripMaster.txt"
    finish_gate 1
fi

echo "2. Building JSON..."
python3 scripts/p25_build_security_master_json.py

if [[ ! -f "$JSON_FILE" ]]; then
    echo "[FAIL] JSON file not created."
    finish_gate 1
fi

# Verify JSON Validity and Content
echo "3. Verifying JSON Content..."
COUNTS=$(jq -r '.meta.counts | "\(.cash) \(.options)"' "$JSON_FILE")
CASH_COUNT=$(echo "$COUNTS" | awk '{print $1}')
OPT_COUNT=$(echo "$COUNTS" | awk '{print $2}')

if [[ "$CASH_COUNT" -gt 0 ]] && [[ "$OPT_COUNT" -ge 0 ]]; then
    echo "[OK] Valid counts: Cash=$CASH_COUNT, Options=$OPT_COUNT"
else
    echo "[FAIL] Invalid counts."
    finish_gate 1
fi

echo "P25_POS_PASS"
finish_gate 0
