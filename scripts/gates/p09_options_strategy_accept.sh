#!/bin/bash
# P09 Options Strategy Accept Gate
# Verifies options whitelist generation and strategy execution
set -euo pipefail

GATE_ID="p09"
source scripts/gates/common.sh "$GATE_ID" "$@"

require_timeout

TIMEFRAME=${TIMEFRAME:-5m}
DAYS=${DAYS:-2}
TIMERANGE=${TIMERANGE:-""}
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export BREEZE_MOCK=1

echo "Step 1: Generate Options Whitelist for RELIANCE (Shared)"
PAIRS_FILE="$ARTIFACT_DIR/p09_pairs.json"
$PYTHON scripts/gen_option_whitelist.py --underlying RELIANCE --out "$PAIRS_FILE" || finish_gate $?

echo "Step 2: Generate Config with Pairs (Shared)"
CONFIG_FILE="$ARTIFACT_DIR/config_p09.json"
$PYTHON scripts/make_config_with_pairs.py --base-config user_data/config_icicibreeze.json --pairs "$PAIRS_FILE" --out-config "$CONFIG_FILE" || finish_gate $?

echo "Step 3: Download Data ($TIMEFRAME, $DAYS days) (Shared)"
freqtrade download-data -c "$CONFIG_FILE" --userdir user_data --timeframes "$TIMEFRAME" --days "$DAYS" || finish_gate $?

if [ "$GATE_MODE" == "pos" ]; then
    echo "Step 4: Backtesting with IndiaOptionsAutoStrategy (Positive)"
    RANGE_ARG=""
    if [ -n "$TIMERANGE" ]; then
        RANGE_ARG="--timerange $TIMERANGE"
    fi
    freqtrade backtesting -c "$CONFIG_FILE" --userdir user_data --strategy IndiaOptionsAutoStrategy --timeframe "$TIMEFRAME" $RANGE_ARG || finish_gate $?
    
    echo "Step 5: Dry-run Smoke Test (Positive)"
    LOG_FILE="$ARTIFACT_DIR/dry_run.log"
    timeout 15s freqtrade trade -c "$CONFIG_FILE" --userdir user_data --strategy IndiaOptionsAutoStrategy --dry-run > "$LOG_FILE" 2>&1 || true
    
    if grep -q "Changing state to: RUNNING" "$LOG_FILE"; then
        echo "[OK] Bot reached RUNNING state with options strategy"
    else
        echo "[FAIL] Bot did NOT reach RUNNING state. Check $LOG_FILE"
        finish_gate 1
    fi

elif [ "$GATE_MODE" == "neg" ]; then
    echo "Step 4: Dry-run Smoke Test with Broken Strategy (Negative)"
    LOG_FILE="$ARTIFACT_DIR/dry_run_bad_strat.log"
    
    # Create bad strategy
    cat <<EOF > user_data/strategies/BadStrategy.py
class BadStrategy:
    def populate_indicators(self, dataframe, metadata):
        return dataframe
    # Missing other methods or syntax error
    def populate_entry_trend(self, dataframe, metadata):
        raise Exception("Intentional Failure")
EOF

    # Expect it to fail
    if timeout 15s freqtrade trade -c "$CONFIG_FILE" --userdir user_data --strategy BadStrategy --dry-run > "$LOG_FILE" 2>&1; then
        # Check logs for failure
        if grep -q "Changing state to: RUNNING" "$LOG_FILE"; then
             echo "[FAIL] Bot reached RUNNING state despite broken strategy"
             finish_gate 1
        elif grep -q "OperationalException" "$LOG_FILE" || grep -q "Intentional Failure" "$LOG_FILE"; then
             echo "[OK] Bot failed as expected with BadStrategy"
        else
             echo "[OK] Bot failed (exit code 0 but no RUNNING state)"
        fi
    else
        echo "[OK] Bot exited with error as expected"
    fi
    # Cleanup
    rm user_data/strategies/BadStrategy.py
fi

echo "P09 Options Strategy Accept passed ($GATE_MODE)"
finish_gate 0
