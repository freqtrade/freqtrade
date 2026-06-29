#!/usr/bin/env bash
set -euo pipefail

ROOT="/Users/wangsen/Documents/我的Projects/freqtrade"
LOG_DIR="$ROOT/user_data/strategy_research/reports/automation"
LABELS=(
  "com.wangsen.freqtrade.strategy-research.daily"
  "com.wangsen.freqtrade.strategy-research.weekly-aux"
  "com.wangsen.freqtrade.strategy-research.weekly-knowledge"
)

for label in "${LABELS[@]}"; do
  echo "== $label =="
  if launchctl print "gui/$(id -u)/$label" >/tmp/freqtrade-strategy-research-launchd-status.txt 2>&1; then
    rg "state =|last exit code|program =|path =" /tmp/freqtrade-strategy-research-launchd-status.txt || true
  else
    echo "not installed"
  fi
  echo
done

echo "== recent logs =="
for log in "$LOG_DIR"/*.log; do
  [[ -e "$log" ]] || continue
  echo "-- $log --"
  tail -20 "$log"
done
