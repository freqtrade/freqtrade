#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
source .venv/bin/activate
unset PYTHONPATH
export BREEZE_MOCK=1

UNDERLYING="${1:-RELIANCE}"

python scripts/gen_option_whitelist.py --underlying "$UNDERLYING" --atm-breadth 2 --expiry-policy nearest \
  --mode mock --out user_data/generated/p09_pairs.json

python scripts/make_config_with_pairs.py \
  --base user_data/config_icicibreeze.json \
  --pairs user_data/generated/p09_pairs.json \
  --out user_data/generated/config_p09.json

CFG="user_data/generated/config_p09.json"

freqtrade list-markets -c "$CFG" --userdir user_data | rg -n "$UNDERLYING|/INR" | head -n 120

freqtrade download-data -c "$CFG" --userdir user_data --timeframes 5m --days 5 -v | tail -n 140

freqtrade backtesting -c "$CFG" --userdir user_data -s IndiaOptionsAutoStrategy \
  > _p09_backtest.txt 2>&1 || true
rg -n "Traceback|Fatal exception|ERROR|Total trades|Backtesting report" _p09_backtest.txt | head -n 180

freqtrade trade --dry-run -c "$CFG" --userdir user_data -s IndiaOptionsAutoStrategy -vv | sed -n '1,260p'
