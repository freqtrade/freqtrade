#!/bin/sh
set -eu

mkdir -p /freqtrade/user_data/strategies

if [ ! -f /freqtrade/user_data/config.json ]; then
  echo "ERROR: Missing /freqtrade/user_data/config.json"
  exit 1
fi

exec freqtrade trade \
  --config /freqtrade/user_data/config.json \
  --strategy SampleStrategy \
  --dry-run
