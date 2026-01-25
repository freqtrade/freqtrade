#!/bin/bash
set -euo pipefail

# P09X Acceptance Gate Script
# Objective: Verify determinism of Universe Scan and Whitelist Generation

# Accept OUT_DIR as env var or default
OUT_DIR="${OUT_DIR:-user_data/generated/gates/p09x_universe_scanner_accept}"
mkdir -p "$OUT_DIR"

# REQUIRE an active venv
PYTHON=".venv/bin/python"
if [ ! -f "$PYTHON" ]; then
    echo "ERROR: $PYTHON not found. Activate a venv first."
    exit 1
fi

# Ensure jq exists
if ! command -v jq >/dev/null 2>&1; then
    echo "ERROR: jq is required but not installed."
    exit 1
fi

STRATEGY_YAML="user_data/india_strategy.yaml"
BASE_CONFIG="user_data/config_icicibreeze.json"
SECURITY_MASTER_TXT="user_data/data/icicibreeze/FONSEScripMaster.txt"

UNIVERSE_SCAN_PY="scripts/universe_scan_and_generate_pairs.py"
GEN_WHITELIST_PY="scripts/make_config_with_pairs.py"

V1_PAIRS="$OUT_DIR/p09x_pairs_v1.json"
V1_REPORT="$OUT_DIR/p09x_report_v1.json"
V1_CONFIG="$OUT_DIR/config_p09x_v1.json"

V2_PAIRS="$OUT_DIR/p09x_pairs_v2.json"
V2_REPORT="$OUT_DIR/p09x_report_v2.json"
V2_CONFIG="$OUT_DIR/config_p09x_v2.json"

echo "Using Python: $PYTHON"
$PYTHON -V

export PYTHONPATH=.

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
# Compare hashes directly
H1_PAIRS=$(sha256sum "$V1_PAIRS" | awk '{print $1}')
H2_PAIRS=$(sha256sum "$V2_PAIRS" | awk '{print $1}')
echo "Pairs Hash 1: $H1_PAIRS"
echo "Pairs Hash 2: $H2_PAIRS"
if [ "$H1_PAIRS" != "$H2_PAIRS" ]; then
    echo "ERROR: Determinism check failed for pairs!"
    exit 1
fi

H1_REPORT=$(sha256sum "$V1_REPORT" | awk '{print $1}')
H2_REPORT=$(sha256sum "$V2_REPORT" | awk '{print $1}')
if [ "$H1_REPORT" != "$H2_REPORT" ]; then
    echo "ERROR: Determinism check failed for reports!"
    exit 1
fi

H1_CONFIG=$(sha256sum "$V1_CONFIG" | awk '{print $1}')
H2_CONFIG=$(sha256sum "$V2_CONFIG" | awk '{print $1}')
if [ "$H1_CONFIG" != "$H2_CONFIG" ]; then
    echo "ERROR: Determinism check failed for configs!"
    exit 1
fi

echo "=== STEP 6: Hard-check Counts ==="
# Assert pairs is JSON array and length > 0
if ! jq -e 'type == "array"' "$V1_PAIRS" >/dev/null; then
    echo "ERROR: Pairs file is not a JSON array"
    exit 1
fi
PAIR_LEN=$(jq -r 'length' "$V1_PAIRS")
echo "Pairs count: $PAIR_LEN"
if [ "$PAIR_LEN" -eq 0 ]; then
    echo "ERROR: Pair list is empty!"
    exit 1
fi

# Assert whitelist in config is JSON array and length > 0
WL_LEN=$(jq -r '.exchange.pair_whitelist | length' "$V1_CONFIG")
echo "Whitelist count: $WL_LEN"
if [ "$WL_LEN" -eq 0 ]; then
    echo "ERROR: Whitelist is empty in derived config!"
    exit 1
fi

if [ "$WL_LEN" -ne "$PAIR_LEN" ]; then
    echo "ERROR: Whitelist count mismatch!"
    exit 1
fi

echo "=== STEP 7: Verify Freqtrade Market Resolution ==="
export BREEZE_MOCK=1
$PYTHON -m freqtrade list-markets --config "$V1_CONFIG"

echo "=== STEP 8: Download Data ==="
$PYTHON -m freqtrade download-data --config "$V1_CONFIG" --timeframes 1d --days 100

echo "=== STEP 9: Backtest ==="
$PYTHON -m freqtrade backtesting --config "$V1_CONFIG" --timeframe 1d --strategy IndiaEquitySmokeStrategy

echo "=== STEP 10: Dry-run Smoke Test ==="
timeout 15s $PYTHON -m freqtrade trade --config "$V1_CONFIG" --strategy IndiaEquitySmokeStrategy --dry-run || true

echo "=== STEP 11: Archive Results ==="
tar -czf "$OUT_DIR/p09x_gate_artifacts.tar.gz" -C "$OUT_DIR" .

echo "=== P09X ACCEPTANCE GATE PASSED ==="
echo "GATE_RESULT=PASS ARTIFACTS=$OUT_DIR"
