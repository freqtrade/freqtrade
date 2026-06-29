#!/usr/bin/env bash
set -euo pipefail

ROOT="/Users/wangsen/Documents/我的Projects/freqtrade"
AUTOMATION_DIR="$ROOT/user_data/strategy_research/automation"
LAUNCH_AGENTS_DIR="$HOME/Library/LaunchAgents"
LOG_DIR="$ROOT/user_data/strategy_research/reports/automation"
PLISTS=(
  "com.wangsen.freqtrade.strategy-research.daily.plist"
  "com.wangsen.freqtrade.strategy-research.weekly-aux.plist"
  "com.wangsen.freqtrade.strategy-research.weekly-knowledge.plist"
)

mkdir -p "$LAUNCH_AGENTS_DIR" "$LOG_DIR"

for plist in "${PLISTS[@]}"; do
  source_path="$AUTOMATION_DIR/$plist"
  target_path="$LAUNCH_AGENTS_DIR/$plist"
  cp "$source_path" "$target_path"
  launchctl bootout "gui/$(id -u)" "$target_path" >/dev/null 2>&1 || true
  launchctl bootstrap "gui/$(id -u)" "$target_path"
  launchctl enable "gui/$(id -u)/${plist%.plist}"
  echo "Installed ${plist%.plist}"
done

echo "Strategy research launchd jobs installed."
echo "Check status with: $AUTOMATION_DIR/status_launchd.sh"
