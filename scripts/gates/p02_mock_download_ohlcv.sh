#!/bin/bash
# P02 Mock Download OHLCV Gate
# Verifies data download functionality in mock mode
set -euo pipefail

GATE_ID="p02"
source scripts/gates/common.sh "$GATE_ID"

TIMEFRAME=${TIMEFRAME:-5m}
DAYS=${DAYS:-2}
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

echo "Step 1: Download Data (RELIANCE/INR, $TIMEFRAME, $DAYS days)"
export BREEZE_MOCK=1
freqtrade download-data -c user_data/config_icicibreeze.json --userdir user_data --pairs RELIANCE/INR --timeframes "$TIMEFRAME" --days "$DAYS" || finish_gate $?

echo "Step 2: Verify Data File"
DATA_FILE="user_data/data/icicibreeze/RELIANCE_INR-${TIMEFRAME}.feather"
# Fallback to json if feather is not used, although freqtrade 2025.12 defaults to feather
if [ ! -f "$DATA_FILE" ]; then
    DATA_FILE="user_data/data/icicibreeze/RELIANCE_INR-${TIMEFRAME}.json"
fi

if [ ! -f "$DATA_FILE" ]; then
    echo "ERROR: Data file not found at $DATA_FILE"
    finish_gate 1
fi

# Verify row count > 0 using python
$PYTHON -c "
import pandas as pd
import sys
try:
    if sys.argv[1].endswith('.feather'):
        df = pd.read_feather(sys.argv[1])
    else:
        df = pd.read_json(sys.argv[1], orient='values')
    print(f'Rows: {len(df)}')
    if len(df) == 0:
        sys.exit(1)
except Exception as e:
    print(f'Error reading file: {e}')
    sys.exit(1)
" "$DATA_FILE" || finish_gate $?

echo "P02 Mock Download OHLCV passed"
finish_gate 0
