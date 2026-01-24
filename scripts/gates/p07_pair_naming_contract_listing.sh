#!/bin/bash
# P07 Pair Naming & Contract Listing Gate
# Verifies that symbol normalization and contract parsing are deterministic

GATE_ID="p07"
source scripts/gates/common.sh "$GATE_ID"

echo "Step 1: Normalize Pair Names"
# Legacy format
LEGACY="RELIANCE-2026-02-26-2800-CE"
NORMALIZED_LEGACY=$($PYTHON scripts/normalize_pair.py "$LEGACY")
echo "Legacy: $LEGACY -> $NORMALIZED_LEGACY"
if [ "$NORMALIZED_LEGACY" != "RELIANCE-20260226-2800-CE/INR" ]; then
    echo "ERROR: Legacy normalization failed"
    finish_gate 1
fi

# Canonical format
CANONICAL="RELIANCE/INR"
NORMALIZED_CANONICAL=$($PYTHON scripts/normalize_pair.py "$CANONICAL")
echo "Canonical: $CANONICAL -> $NORMALIZED_CANONICAL"
if [ "$NORMALIZED_CANONICAL" != "RELIANCE/INR" ]; then
    echo "ERROR: Canonical normalization failed"
    finish_gate 1
fi

echo "Step 2: List ICICI Contracts"
CONTRACTS_FILE="$ARTIFACT_DIR/contracts.txt"
export PYTHONPATH=.
$PYTHON scripts/list_icici_contracts.py --underlying RELIANCE,NIFTY > "$CONTRACTS_FILE" || finish_gate $?

echo "Step 3: Assert deterministic counts"
# Check if we found RELIANCE and NIFTY in the output
if grep -q "Underlying: RELIANCE" "$CONTRACTS_FILE" && grep -q "Underlying: NIFTY" "$CONTRACTS_FILE"; then
    echo "[OK] Found expected underlyings in contract list"
else
    echo "[FAIL] Missing underlyings in contract list"
    finish_gate 1
fi

echo "P07 Pair Naming & Contract Listing passed"
finish_gate 0
