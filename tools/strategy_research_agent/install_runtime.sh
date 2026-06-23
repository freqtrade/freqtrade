#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SOURCE_ROOT="$ROOT/tools/strategy_research_agent"

mkdir -p "$ROOT/user_data/strategy_research"
mkdir -p "$ROOT/user_data/strategies/research_generated"

rsync -a --delete \
  --exclude '__pycache__' \
  --exclude 'reports/' \
  --exclude 'dashboard/' \
  --exclude 'cost_adjustments/' \
  --exclude 'cost_audits/' \
  --exclude 'matrix_summaries/' \
  --exclude 'walk_forward_summaries/' \
  --exclude 'promotion_candidates/' \
  --exclude 'promotion_blocks/' \
  --exclude 'promotion_reports/' \
  --exclude 'research_agendas/' \
  --exclude 'agenda_runs/' \
  --exclude 'trade_behavior/' \
  --exclude 'behavior_experiments/' \
  --exclude 'data_updates/' \
  --exclude 'strategy_assessments/' \
  --exclude 'candidates/*.json' \
  --exclude 'rejected/*.json' \
  --exclude 'watchlist/*.json' \
  --exclude 'experiments/autonomous_*' \
  --exclude 'experiments/iterative_*' \
  --exclude 'experiments/behavior_experiment_*' \
  --exclude 'experiments/walk_forward_validation_experiment.json' \
  --exclude 'sources/inbox/' \
  --exclude 'sources/reviews/' \
  --exclude 'sources/translation_drafts/' \
  "$SOURCE_ROOT/strategy_research/" \
  "$ROOT/user_data/strategy_research/"

rsync -a --delete \
  --exclude '__pycache__' \
  --exclude 'autonomous_research_strategies.py' \
  --exclude 'iterative_research_strategies.py' \
  --exclude 'behavior_experiment_strategies.py' \
  "$SOURCE_ROOT/strategies/research_generated/" \
  "$ROOT/user_data/strategies/research_generated/"

cp "$SOURCE_ROOT/download_binance_um_1m.py" "$ROOT/user_data/download_binance_um_1m.py"

chmod +x \
  "$ROOT/user_data/strategy_research/run_daily_research.sh" \
  "$ROOT/user_data/strategy_research/run_full_research_cycle.sh" \
  "$ROOT/user_data/strategy_research/automation/install_launchd.sh" \
  "$ROOT/user_data/strategy_research/automation/uninstall_launchd.sh" \
  "$ROOT/user_data/strategy_research/automation/status_launchd.sh"

echo "Installed strategy research agent runtime files into user_data/."
