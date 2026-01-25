#!/bin/bash
# P09X Universe Scanner Accept Gate
# Objective: Verify determinism of Universe Scan and Whitelist Generation (Options E2E)
set -euo pipefail

GATE_ID="p09x"
source scripts/gates/common.sh "$GATE_ID"

require_timeout

TIMEFRAME=${TIMEFRAME:-5m}
DAYS=${DAYS:-2}
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

STRATEGY_YAML="user_data/india_strategy.yaml"
BASE_CONFIG="user_data/config_icicibreeze.json"
SECURITY_MASTER_TXT="user_data/data/icicibreeze/FONSEScripMaster.txt"

UNIVERSE_SCAN_PY="scripts/universe_scan_and_generate_pairs.py"
GEN_WHITELIST_PY="scripts/make_config_with_pairs.py"

V1_PAIRS="$ARTIFACT_DIR/p09x_pairs_v1.json"
V1_REPORT="$ARTIFACT_DIR/p09x_report_v1.json"
V1_CONFIG="$ARTIFACT_DIR/config_p09x_v1.json"

V2_PAIRS="$ARTIFACT_DIR/p09x_pairs_v2.json"
V2_REPORT="$ARTIFACT_DIR/p09x_report_v2.json"
V2_CONFIG="$ARTIFACT_DIR/config_p09x_v2.json"

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
# Compare hashes directly for pairs, report, and config
H1_PAIRS=$(sha256sum "$V1_PAIRS" | awk '{print $1}')
H2_PAIRS=$(sha256sum "$V2_PAIRS" | awk '{print $1}')
if [ "$H1_PAIRS" != "$H2_PAIRS" ]; then echo "ERROR: Pairs mismatch"; finish_gate 1; fi

H1_REPORT=$(sha256sum "$V1_REPORT" | awk '{print $1}')
H2_REPORT=$(sha256sum "$V2_REPORT" | awk '{print $1}')
if [ "$H1_REPORT" != "$H2_REPORT" ]; then echo "ERROR: Report mismatch"; finish_gate 1; fi

H1_CONFIG=$(sha256sum "$V1_CONFIG" | awk '{print $1}')
H2_CONFIG=$(sha256sum "$V2_CONFIG" | awk '{print $1}')
if [ "$H1_CONFIG" != "$H2_CONFIG" ]; then echo "ERROR: Config mismatch"; finish_gate 1; fi

echo "=== STEP 6: Hard-check Counts ==="
PAIR_COUNT=$(jq '. | length' "$V1_PAIRS")
WL_COUNT=$(jq -r '.exchange.pair_whitelist | length' "$V1_CONFIG")
echo "Pairs count: $PAIR_COUNT, Whitelist count: $WL_COUNT"

if [ "$PAIR_COUNT" -le 0 ]; then echo "ERROR: Pair list is empty"; finish_gate 1; fi
if [ "$WL_COUNT" -ne "$PAIR_COUNT" ]; then echo "ERROR: count mismatch"; finish_gate 1; fi

echo "=== STEP 7: Verify Freqtrade Market Resolution ==="
export BREEZE_MOCK=1
MARKETS_FILE="$ARTIFACT_DIR/markets.txt"
freqtrade list-markets -c "$V1_CONFIG" --userdir user_data > "$MARKETS_FILE" || finish_gate $?

# Verify list-markets output contains at least one of the generated option pairs
FIRST_PAIR=$(jq -r '.[0]' "$V1_PAIRS")
if grep -q "$FIRST_PAIR" "$MARKETS_FILE"; then
    echo "[OK] Found $FIRST_PAIR in list-markets"
else
    echo "[FAIL] $FIRST_PAIR not found in list-markets"
    finish_gate 1
fi

echo "=== STEP 8: Download Data ($TIMEFRAME, $DAYS days) ==="
freqtrade download-data -c "$V1_CONFIG" --userdir user_data --timeframes "$TIMEFRAME" --days "$DAYS" || finish_gate $?

echo "=== STEP 9: Backtest with IndiaOptionsAutoStrategy ==="
freqtrade backtesting -c "$V1_CONFIG" --userdir user_data --strategy IndiaOptionsAutoStrategy --timeframe "$TIMEFRAME" || finish_gate $?

echo "=== STEP 10: Dry-run Smoke Test ==="
timeout 15s freqtrade trade -c "$V1_CONFIG" --userdir user_data --strategy IndiaOptionsAutoStrategy --dry-run || true

echo "P09X Universe Scanner Accept passed"
finish_gate 0
