#!/bin/bash
# P03 INR Pairs Presence Gate
# Verifies that canonical INR pairs are present in market listing
set -euo pipefail

GATE_ID="p03"
source scripts/gates/common.sh "$GATE_ID" "$@"

export BREEZE_MOCK=1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
MARKETS_FILE="$ARTIFACT_DIR/markets.txt"

if [ "$GATE_MODE" == "pos" ]; then
    echo "Step 1: Freqtrade list-markets and check for RELIANCE/INR (Positive)"
    # Capture all output
    freqtrade list-markets -c user_data/config_icicibreeze.json --userdir user_data > "$MARKETS_FILE" 2>&1 || finish_gate $?

    if grep -q "RELIANCE/INR" "$MARKETS_FILE"; then
        echo "[OK] RELIANCE/INR found in market list"
    else
        echo "[FAIL] RELIANCE/INR NOT found in market list"
        finish_gate 1
    fi

elif [ "$GATE_MODE" == "neg" ]; then
    echo "Step 1: Freqtrade list-markets with missing SecurityMaster (Negative)"
    
    # To properly simulate missing files, we must run from a directory where 
    # user_data/data/icicibreeze does not exist relative to CWD, 
    # because the Shim uses Path.cwd() to resolve files.
    
    ROOT_DIR="$(pwd)"
    EMPTY_DIR="$ARTIFACT_DIR/empty_data"
    mkdir -p "$EMPTY_DIR"
    
    CONFIG_ABS="$ROOT_DIR/$ARTIFACT_DIR/config_empty.json"
    cp user_data/config_icicibreeze.json "$CONFIG_ABS"
    
    MARKETS_FILE_ABS="$ROOT_DIR/$MARKETS_FILE"
    
    # Switch CWD to empty dir
    pushd "$EMPTY_DIR" > /dev/null
    
    # Run freqtrade, capturing all output
    if freqtrade list-markets -c "$CONFIG_ABS" --userdir . > "$MARKETS_FILE_ABS" 2>&1; then
        :
    fi
    
    # Check if RELIANCE/INR is present in a 'real' context (not warning/config log)
    # We inspect the file.
    # If the pair is active, it should be in the table or JSON. 
    # If it's missing, it might appear in "WARNING ... not found".
    # So we filter OUT the warning lines.
    
    if grep "RELIANCE/INR" "$MARKETS_FILE_ABS" | grep -v "not found" | grep -v "Using config" > /dev/null; then
        echo "[FAIL] RELIANCE/INR found in market list (non-warning context)"
        # Debug output
        popd > /dev/null
        finish_gate 1
    else
        echo "[OK] RELIANCE/INR correctly missing from market list (filtered warnings)"
    fi
    popd > /dev/null
fi

echo "P03 INR Pairs Presence passed ($GATE_MODE)"
finish_gate 0
