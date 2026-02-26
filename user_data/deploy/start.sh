#!/bin/bash

# Build db-url from Railway's DATABASE_URL
DB_ARG=""
if [ -n "$DATABASE_URL" ]; then
    DB_URL=$(echo "$DATABASE_URL" | sed 's|^postgres://|postgresql://|')
    DB_ARG="--db-url $DB_URL"
    echo "✅ PostgreSQL connected for trade tracking"
else
    echo "⚠️ No DATABASE_URL — using SQLite"
fi

# Start the monitor in the background
python3 -c "
import sys, os
sys.path.insert(0, '/freqtrade')
from monitor import start_monitor
start_monitor()
# Keep alive until freqtrade starts
import time
time.sleep(5)
" &

echo "✅ Monitor started"

# Start freqtrade
exec freqtrade trade \
    --config /freqtrade/config_mexc.json \
    --strategy TrendRiderStrategy \
    --userdir /freqtrade/user_data \
    $DB_ARG
