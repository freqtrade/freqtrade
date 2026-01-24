#!/bin/bash
# P02 Mock Download OHLCV Gate
# Verifies data download functionality in mock mode

GATE_ID="p02"
source scripts/gates/common.sh "$GATE_ID"

echo "Step 1: Download Data (RELIANCE/INR, 5m, 2 days)"
export BREEZE_MOCK=1
$PYTHON -m freqtrade download-data --config user_data/config_icicibreeze.json --pairs RELIANCE/INR --timeframes 5m --days 2 || finish_gate $?

echo "Step 2: Verify Data File"
DATA_FILE="user_data/data/icicibreeze/RELIANCE_INR-5m.feather"
# Fallback to json if feather is not used, although freqtrade 2025.12 defaults to feather
if [ ! -f "$DATA_FILE" ]; then
    DATA_FILE="user_data/data/icicibreeze/RELIANCE_INR-5m.json"
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
