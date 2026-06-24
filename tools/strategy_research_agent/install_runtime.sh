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
  --exclude 'context_sources/' \
  --exclude 'manual_playbook/' \
  --exclude 'family_diversity/' \
  --exclude 'sample_expansion/' \
  --exclude 'entry_quality/' \
  --exclude 'matrix_summaries/' \
  --exclude 'walk_forward_summaries/' \
  --exclude 'promotion_candidates/' \
  --exclude 'promotion_blocks/' \
  --exclude 'promotion_reports/' \
  --exclude 'research_agendas/' \
  --exclude 'agenda_runs/' \
  --exclude 'trade_behavior/' \
  --exclude 'behavior_experiments/' \
  --exclude 'failure_attribution/' \
  --exclude 'mature_researcher/' \
  --exclude 'agent_iterations/' \
  --exclude 'strategy_library/' \
  --exclude 'research_memory/' \
  --exclude 'source_discovery/' \
  --exclude 'data_updates/' \
  --exclude 'strategy_assessments/' \
  --exclude 'candidates/*.json' \
  --exclude 'rejected/*.json' \
  --exclude 'watchlist/*.json' \
  --exclude 'experiments/autonomous_*' \
  --exclude 'experiments/retired_seed_family_ledger.*' \
  --exclude 'experiments/iterative_*' \
  --exclude 'experiments/behavior_experiment_*' \
  --exclude 'experiments/memory_guided_*' \
  --exclude 'experiments/family_diversity_*' \
  --exclude 'experiments/sample_expansion_*' \
  --exclude 'experiments/entry_quality_*' \
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
  --exclude 'memory_guided_research_strategies.py' \
  --exclude 'sample_expansion_strategies.py' \
  --exclude 'entry_quality_strategies.py' \
  "$SOURCE_ROOT/strategies/research_generated/" \
  "$ROOT/user_data/strategies/research_generated/"

cp "$SOURCE_ROOT/download_binance_um_1m.py" "$ROOT/user_data/download_binance_um_1m.py"

chmod +x \
  "$ROOT/user_data/strategy_research/run_daily_research.sh" \
  "$ROOT/user_data/strategy_research/run_full_research_cycle.sh" \
  "$ROOT/user_data/strategy_research/run_strong_researcher_smoke.sh" \
  "$ROOT/user_data/strategy_research/automation/install_launchd.sh" \
  "$ROOT/user_data/strategy_research/automation/uninstall_launchd.sh" \
  "$ROOT/user_data/strategy_research/automation/status_launchd.sh"

echo "Installed strategy research agent runtime files into user_data/."
