#!/bin/bash

echo "📥 Downloading historical data for backtesting..."

freqtrade download-data \
    --config /freqtrade/config_backtest.json \
    --timeframe 1h \
    --timerange 20250201- \
    --userdir /freqtrade/user_data \
    2>&1 | tail -5

echo "✅ Data downloaded — starting backtest server"

exec freqtrade webserver \
    --config /freqtrade/config_backtest.json \
    --userdir /freqtrade/user_data
