#!/bin/bash
# P09X Universe Scanner Accept Gate
# Objective: Verify determinism of Universe Scan and Whitelist Generation

GATE_ID="p09x"
source scripts/gates/common.sh "$GATE_ID"

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

export PYTHONPATH=.

echo "=== STEP 1: Run Universe Scan (Pass 1) ==="
$PYTHON "$UNIVERSE_SCAN_PY" \
    --config "$STRATEGY_YAML" \
    --security-master "$SECURITY_MASTER_TXT" \
    --out-pairs "$V1_PAIRS" \
    --out-report "$V1_REPORT" || finish_gate $?

echo "=== STEP 2: Generate Config (Pass 1) ==="
$PYTHON "$GEN_WHITELIST_PY" \
    --base-config "$BASE_CONFIG" \
    --pairs "$V1_PAIRS" \
    --out-config "$V1_CONFIG" || finish_gate $?

echo "=== STEP 3: Run Universe Scan (Pass 2) ==="
$PYTHON "$UNIVERSE_SCAN_PY" \
    --config "$STRATEGY_YAML" \
    --security-master "$SECURITY_MASTER_TXT" \
    --out-pairs "$V2_PAIRS" \
    --out-report "$V2_REPORT" || finish_gate $?

echo "=== STEP 4: Generate Config (Pass 2) ==="
$PYTHON "$GEN_WHITELIST_PY" \
    --base-config "$BASE_CONFIG" \
    --pairs "$V2_PAIRS" \
    --out-config "$V2_CONFIG" || finish_gate $?

echo "=== STEP 5: Verify Determinism ==="
sha256sum "$V1_PAIRS" "$V2_PAIRS"
sha256sum "$V1_REPORT" "$V2_REPORT"
sha256sum "$V1_CONFIG" "$V2_CONFIG"

DIFF=$(sha256sum "$V1_PAIRS" "$V2_PAIRS" | awk '{print $1}' | sort | uniq | wc -l)
if [ "$DIFF" -ne 1 ]; then
    echo "ERROR: Determinism check failed for pairs!"
    finish_gate 1
fi

echo "=== STEP 6: Hard-check Counts ==="
PAIR_COUNT=$(jq '. | length' "$V1_PAIRS")
WL_COUNT=$(jq '.exchange.pair_whitelist | length' "$V1_CONFIG")
echo "Pairs count: $PAIR_COUNT"
echo "Whitelist count: $WL_COUNT"

if [ "$PAIR_COUNT" -le 0 ]; then
    echo "ERROR: Pair list is empty!"
    finish_gate 1
fi

if [ "$WL_COUNT" -ne "$PAIR_COUNT" ]; then
    echo "ERROR: Whitelist count mismatch!"
    finish_gate 1
fi

echo "=== STEP 7: Verify Freqtrade Market Resolution ==="
export BREEZE_MOCK=1
$PYTHON -m freqtrade list-markets --config "$V1_CONFIG" || finish_gate $?

echo "=== STEP 8: Download Data ==="
$PYTHON -m freqtrade download-data --config "$V1_CONFIG" --timeframes 1d --days 100 || finish_gate $?

echo "=== STEP 9: Backtest ==="
$PYTHON -m freqtrade backtesting --config "$V1_CONFIG" --timeframe 1d --strategy IndiaEquitySmokeStrategy || finish_gate $?

echo "=== STEP 10: Dry-run Smoke Test ==="
timeout 15s $PYTHON -m freqtrade trade --config "$V1_CONFIG" --strategy IndiaEquitySmokeStrategy --dry-run || true

echo "P09X Universe Scanner Accept passed"
finish_gate 0
