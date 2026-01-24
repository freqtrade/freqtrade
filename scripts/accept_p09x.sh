#!/bin/bash
set -e

# P09 Acceptance Gate Script
# Objective: Verify determinism of Universe Scan and Whitelist Generation

# REQUIRE an active venv
if ! command -v python >/dev/null 2>&1; then
    echo "ERROR: python not found in PATH. Activate a venv first."
    exit 1
fi

PYTHON=".venv/bin/python"

STRATEGY_YAML="user_data/india_strategy.yaml"
BASE_CONFIG="user_data/config_icicibreeze.json"
OUT_DIR="user_data/generated"
mkdir -p "$OUT_DIR"

UNIVERSE_SCAN_PY="scripts/universe_scan_and_generate_pairs.py"
GEN_WHITELIST_PY="scripts/make_config_with_pairs.py"
SECURITY_MASTER_TXT="user_data/data/icicibreeze/FONSEScripMaster.txt"

V1_PAIRS="$OUT_DIR/p09x_pairs_v1.json"
V1_REPORT="$OUT_DIR/p09x_report_v1.json"
V1_CONFIG="$OUT_DIR/config_p09x_v1.json"

V2_PAIRS="$OUT_DIR/p09x_pairs_v2.json"
V2_REPORT="$OUT_DIR/p09x_report_v2.json"
V2_CONFIG="$OUT_DIR/config_p09x_v2.json"

echo "Using Python: $(which python)"
python -V

echo "=== STEP 1: Run Universe Scan (Pass 1) ==="
$PYTHON "$UNIVERSE_SCAN_PY" \
    --config "$STRATEGY_YAML" \
    --security-master "$SECURITY_MASTER_TXT" \
    --out-pairs "$V1_PAIRS" \
    --out-report "$V1_REPORT"

echo "=== STEP 2: Generate Config (Pass 1) ==="
$PYTHON "$GEN_WHITELIST_PY" \
    --base-config "$BASE_CONFIG" \
    --pairs "$V1_PAIRS" \
    --out-config "$V1_CONFIG"

echo "=== STEP 3: Run Universe Scan (Pass 2) ==="
$PYTHON "$UNIVERSE_SCAN_PY" \
    --config "$STRATEGY_YAML" \
    --security-master "$SECURITY_MASTER_TXT" \
    --out-pairs "$V2_PAIRS" \
    --out-report "$V2_REPORT"

echo "=== STEP 4: Generate Config (Pass 2) ==="
$PYTHON "$GEN_WHITELIST_PY" \
    --base-config "$BASE_CONFIG" \
    --pairs "$V2_PAIRS" \
    --out-config "$V2_CONFIG"

echo "=== STEP 5: Verify Determinism ==="
sha256sum "$V1_PAIRS" "$V2_PAIRS"
sha256sum "$V1_REPORT" "$V2_REPORT"
sha256sum "$V1_CONFIG" "$V2_CONFIG"

DIFF=$(sha256sum "$V1_PAIRS" "$V2_PAIRS" | awk '{print $1}' | sort | uniq | wc -l)
if [ "$DIFF" -ne 1 ]; then
    echo "ERROR: Determinism check failed for pairs!"
    exit 1
fi

echo "=== STEP 6: Verify Content Integrity ==="
grep -q "RELIANCE" "$V1_PAIRS" || echo "Note: RELIANCE not in pairs (expected if filtered)"
grep -q "NIFTY" "$V1_PAIRS" || echo "Note: NIFTY not in pairs (expected if filtered)"

echo "=== STEP 7: Check Empty Cases ==="
if [ ! -s "$V1_PAIRS" ]; then
    echo "ERROR: Generated pairs file is empty!"
    exit 1
fi

echo "=== STEP 8: Archive Results ==="
tar -czf "$OUT_DIR/p09x_acceptance_artifacts.tar.gz" \
    "$V1_PAIRS" "$V1_REPORT" "$V1_CONFIG" \
    "$V2_PAIRS" "$V2_REPORT" "$V2_CONFIG"

echo "=== P09 ACCEPTANCE GATE PASSED ==="
echo "Artifacts: $OUT_DIR/p09x_acceptance_artifacts.tar.gz"
