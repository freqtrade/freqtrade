#!/usr/bin/env bash
set -euo pipefail

# Periodically run Hyperopt -> Backtesting while OB collection continues.
# Usage: scripts/hpo_bt_loop.sh [WINDOW_DAYS] [THRESHOLD_MINUTES] [SLEEP_SECONDS]
# Defaults: WINDOW_DAYS=2, THRESHOLD_MINUTES=60, SLEEP_SECONDS=1800
#
# Optional env:
#   FREQTRADE_BIN   Path to freqtrade CLI. If not set, auto-detects `freqtrade`
#                   in PATH, otherwise falls back to `python3 -m freqtrade`.

WINDOW_DAYS="${1:-2}"
THRESHOLD_MINUTES="${2:-60}"
SLEEP_SECONDS="${3:-1800}"

USERDIR="user_data"
STRAT="TanakaAlpha5mV1"
STRAT_PATH="user_data/strategies"
CONFIG="user_data/config_bt_5mV1.json"
RESULT_DIR="user_data/backtest_results"
mkdir -p "$RESULT_DIR"

# --- Detect freqtrade CLI ---
if [[ -n "${FREQTRADE_BIN:-}" ]]; then
  # If provided, split into array to preserve words
  # shellcheck disable=SC2206
  FT_CMD=(${FREQTRADE_BIN})
else
  if command -v freqtrade >/dev/null 2>&1; then
    FT_CMD=(freqtrade)
  elif command -v python3 >/dev/null 2>&1 && python3 - <<'PY' >/dev/null 2>&1
import importlib
import sys
sys.exit(0 if importlib.util.find_spec('freqtrade') is not None else 1)
PY
  then
    FT_CMD=(python3 -m freqtrade)
  else
    echo "Error: freqtrade CLI not found. Install freqtrade in your current environment or set FREQTRADE_BIN=/path/to/freqtrade" >&2
    exit 1
  fi
fi

# Compute timerange start (UTC) as YYYYMMDD-
timerange_start() {
  env WINDOW_DAYS="$WINDOW_DAYS" python - << 'PY'
import os, datetime
days = int(os.environ.get('WINDOW_DAYS','2'))
start = (datetime.datetime.utcnow() - datetime.timedelta(days=days)).strftime('%Y%m%d')
print(f"{start}-")
PY
}

# Count distinct OB minute files for a symbol today (approx proxy for progress)
count_minutes() {
  python - << 'PY'
import os, re
from pathlib import Path
from datetime import datetime, timezone
symbol = os.environ['SYMBOL']
root = Path(f'user_data/featurestore/bybit/{symbol}/1s')
mins = 0
if root.exists():
    today = datetime.now(timezone.utc)
    daydir = root / str(today.year) / str(today.month) / str(today.day)
    paths = list(daydir.glob('ob_*.parquet')) if daydir.exists() else []
    keys = set()
    for p in paths:
        m = re.search(r'ob_(\d{12})_', p.name)
        if m:
            keys.add(m.group(1))
    mins = len(keys)
print(mins)
PY
}

run_once() {
  local TRANGE="$(timerange_start)"
  echo "[HPO] timerange=${TRANGE} spaces=buy,sell loss=ProfitDrawDownHyperOptLoss"
  "${FT_CMD[@]}" hyperopt -c "$CONFIG" --strategy "$STRAT" --strategy-path "$STRAT_PATH" \
    --timeframe 5m --timerange "$TRANGE" --spaces buy sell \
    --hyperopt-loss ProfitDrawDownHyperOptLoss -e 300 -j -1 \
    --print-json --logfile "$RESULT_DIR/hpo_autorun.log" || true

  echo "[BT ] timerange=${TRANGE}"
  "${FT_CMD[@]}" backtesting -c "$CONFIG" --strategy "$STRAT" --strategy-path "$STRAT_PATH" \
    --timeframe 5m --timerange "$TRANGE" --export none \
    --logfile "$RESULT_DIR/bt_autorun.log" || true
}

echo "Starting HPO/BT autorun loop: WINDOW_DAYS=${WINDOW_DAYS} THRESHOLD_MINUTES=${THRESHOLD_MINUTES} SLEEP_SECONDS=${SLEEP_SECONDS}"
echo "Using Freqtrade command: ${FT_CMD[*]}"

# Initial run (速報値)
run_once

prev_total=0
while true; do
  export SYMBOL="BTCUSDT"; btc=$(count_minutes)
  export SYMBOL="ETHUSDT"; eth=$(count_minutes)
  total=$((btc + eth))
  echo "[loop] OB minutes today: BTC=${btc} ETH=${eth} total=${total} (prev=${prev_total})"
  if [ "$total" -ge $((prev_total + THRESHOLD_MINUTES)) ]; then
    echo "[loop] Threshold reached (+${THRESHOLD_MINUTES}m). Running HPO/BT..."
    run_once
    prev_total=$total
  fi
  sleep "$SLEEP_SECONDS"
done
