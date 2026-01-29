#!/bin/bash
# P25: Security Master Refresh (Hardened)
# Verifies security master fetch and build process.
#
# Modes:
#   pos: 
#     - Build master.json from inputs
#     - Assert valid JSON, no tmp files, deterministic output
#     - Emit P25_POS_PASS
#   neg:
#     - Run builder with missing input file
#     - Assert exit code 2 and failure observation
#     - Emit P25_NEG_EXPECTED_FAIL

set -euo pipefail

GATE_ID="p25"
source scripts/gates/common.sh "$GATE_ID" "$@"

echo ">>> Gate P25: Security Master... ($GATE_MODE)"

CACHE_DIR="user_data/cache/security_master"
mkdir -p "$CACHE_DIR"

if [ "$GATE_MODE" == "pos" ]; then
    # Positive Case checks
    
    # 1. Fetch (Mock or Real)
    if [[ "${BREEZE_MOCK:-0}" == "1" ]]; then
        echo "1. Fetching Master (Mock Mode)..."
        # Manually copy fixtures to cache dir to simulate fetch
        # Assumes user_data/data/icicibreeze has fixtures (which it does in this env)
        cp "user_data/data/icicibreeze/NSEScripMaster.txt" "$CACHE_DIR/"
        cp "user_data/data/icicibreeze/FONSEScripMaster.txt" "$CACHE_DIR/"
    else
        echo "1. Fetching Master (Real Mode)..."
        $PYTHON scripts/p25_fetch_security_master.py --output "$CACHE_DIR"
    fi
    
    # 2. Build
    echo "2. Building JSON..."
    OUT_FILE="$CACHE_DIR/latest.json"
    rm -f "$OUT_FILE"
    
    $PYTHON scripts/p25_build_security_master_json.py \
        --cash "$CACHE_DIR/NSEScripMaster.txt" \
        --fno "$CACHE_DIR/FONSEScripMaster.txt" \
        --output "$OUT_FILE"
        
    # 3. Verify Constraints
    echo "3. Verifying JSON Content..."
    
    # Valid JSON
    if ! jq . "$OUT_FILE" > /dev/null; then
        echo "[FAIL] Invalid JSON generated"
        finish_gate 1
    fi
    
    # No tmp files
    if ls "$CACHE_DIR"/*.tmp 1> /dev/null 2>&1; then
        echo "[FAIL] Temporary files left behind"
        finish_gate 1
    fi
    
    # Deterministic Sort Check
    echo "4. Checking Determinism..."
    OUT_FILE_2="$CACHE_DIR/latest_2.json"
    $PYTHON scripts/p25_build_security_master_json.py \
        --cash "$CACHE_DIR/NSEScripMaster.txt" \
        --fno "$CACHE_DIR/FONSEScripMaster.txt" \
        --output "$OUT_FILE_2"
        
    if ! diff -q "$OUT_FILE" "$OUT_FILE_2"; then
        echo "[FAIL] Output is not deterministic!"
        finish_gate 1
    fi
    rm "$OUT_FILE_2"
    
    # Extract stats for logging
    CASH_COUNT=$(jq '.cash | length' "$OUT_FILE")
    OPT_COUNT=$(jq '.fno.options | length' "$OUT_FILE")
    echo "[OK] Valid counts: Cash=$CASH_COUNT, Options=$OPT_COUNT"
    
    echo "P25_POS_PASS"
    echo ">>> Gate P25: SUCCESS"
    finish_gate 0

elif [ "$GATE_MODE" == "neg" ]; then
    # Negative Case - Missing Input
    echo "1. Running Builder with Missing Input..."
    
    # Point to a non-existent file
    set +e # Disable exit on error for this command
    OUTPUT=$($PYTHON scripts/p25_build_security_master_json.py --cash "NON_EXISTENT_FILE.txt" --fno "ign" --output "ign" 2>&1)
    RES=$?
    set -e
    
    # Requirement: exit code 2 (as implemented in builder hardening)
    if [ $RES -eq 2 ]; then
        echo "[OK] Builder exited with code 2 as expected."
        if echo "$OUTPUT" | grep -q "ERROR: Input file not found"; then
             echo "[OK] Found expected error message."
             echo "P25_NEG_EXPECTED_FAIL"
             finish_gate 0
        else
             echo "[FAIL] Incorrect error message: $OUTPUT"
             finish_gate 1
        fi
    else
        echo "[FAIL] Unexpected exit code: $RES (expected 2)"
        echo "Output: $OUTPUT"
        finish_gate 1
    fi

else
    echo "ERROR: Invalid mode $GATE_MODE"
    finish_gate 1
fi
