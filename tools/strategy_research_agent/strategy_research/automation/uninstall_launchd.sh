#!/usr/bin/env bash
set -euo pipefail

LAUNCH_AGENTS_DIR="$HOME/Library/LaunchAgents"
PLISTS=(
  "com.wangsen.freqtrade.strategy-research.daily.plist"
  "com.wangsen.freqtrade.strategy-research.weekly-aux.plist"
  "com.wangsen.freqtrade.strategy-research.weekly-knowledge.plist"
)

for plist in "${PLISTS[@]}"; do
  target_path="$LAUNCH_AGENTS_DIR/$plist"
  launchctl bootout "gui/$(id -u)" "$target_path" >/dev/null 2>&1 || true
  rm -f "$target_path"
  echo "Uninstalled ${plist%.plist}"
done

echo "Strategy research launchd jobs uninstalled."
