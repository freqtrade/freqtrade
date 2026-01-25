#!/bin/bash
# P09 Options Strategy Accept Gate
# Verifies options whitelist generation and strategy execution

GATE_ID="p09"
source scripts/gates/common.sh "$GATE_ID"

echo "Step 1: Generate Options Whitelist for RELIANCE"
PAIRS_FILE="$OUT_DIR/p09_pairs.json"
export BREEZE_MOCK=1
export PYTHONPATH=.
$PYTHON scripts/gen_option_whitelist.py --underlying RELIANCE --out "$PAIRS_FILE" || finish_gate $?

echo "Step 2: Verify pairs quantity"
PAIR_COUNT=$(jq '. | length' "$PAIRS_FILE")
echo "Generated $PAIR_COUNT pairs"
if [ "$PAIR_COUNT" -eq 0 ]; then
    echo "ERROR: No pairs generated"
    finish_gate 1
fi

echo "Step 3: Generate Config with Pairs"
CONFIG_FILE="$OUT_DIR/config_p09.json"
$PYTHON scripts/make_config_with_pairs.py --base-config user_data/config_icicibreeze.json --pairs "$PAIRS_FILE" --out-config "$CONFIG_FILE" || finish_gate $?

echo "Step 4: Verify derived config"
WL_COUNT=$(jq '.exchange.pair_whitelist | length' "$CONFIG_FILE")
echo "Whitelist has $WL_COUNT entries"
if [ "$WL_COUNT" -eq 0 ]; then
    echo "ERROR: Empty whitelist in derived config"
    finish_gate 1
fi

echo "Step 5: Download Data (5m, 2 days)"
# Including underlying RELIANCE/INR explicitly just in case download-data doesn't pick it up from informative_pairs
$PYTHON -m freqtrade download-data --config "$CONFIG_FILE" --timeframes 5m --days 2 || finish_gate $?

echo "Step 6: Backtesting with IndiaIndexOptionsStrategy"
$PYTHON -m freqtrade backtesting --config "$CONFIG_FILE" --strategy IndiaIndexOptionsStrategy --timeframe 5m --timerange 20260119-20260124 || finish_gate $?

echo "Step 7: Dry-run Smoke Test"
LOG_FILE="$OUT_DIR/dry_run.log"
timeout 15s $PYTHON -m freqtrade trade --config "$CONFIG_FILE" --strategy IndiaIndexOptionsStrategy --dry-run > "$LOG_FILE" 2>&1 || true

if grep -q "Changing state to: RUNNING" "$LOG_FILE"; then
    echo "[OK] Bot reached RUNNING state with options strategy"
else
    echo "[FAIL] Bot did NOT reach RUNNING state. Check $LOG_FILE"
    finish_gate 1
fi

echo "P09 Options Strategy Accept passed"
finish_gate 0
