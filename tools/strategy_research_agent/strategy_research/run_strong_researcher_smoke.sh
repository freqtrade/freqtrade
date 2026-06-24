#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

PYTHON="${PYTHON:-./.venv/bin/python}"
FREQTRADE="${FREQTRADE:-./.venv/bin/freqtrade}"
export PYTHONPATH="${PYTHONPATH:-user_data/offline_exchange}"

SMOKE_TIMERANGE="${SMOKE_TIMERANGE:-20260101-20260201}"

cat <<'EOF'
== Strong Strategy Researcher Smoke ==
Research-only flow:
  1. Preflight safety/data/output checks.
  2. Refresh external-source discovery queue.
  3. Rebuild strategy lineage and research memory.
  4. Plan and generate memory-guided strategy variants.
  5. Verify Freqtrade can discover generated variants.
  6. Smoke backtest generated variants on a short timerange.
  7. Build mature researcher diagnosis and next-experiment decisions.
  8. Rebuild lineage/memory and refresh dashboard/report.
EOF

"$PYTHON" user_data/strategy_research/preflight_research_agent.py
"$PYTHON" user_data/strategy_research/scout_external_sources.py
"$PYTHON" user_data/strategy_research/build_strategy_lineage.py
"$PYTHON" user_data/strategy_research/build_research_memory.py
"$PYTHON" user_data/strategy_research/plan_memory_guided_hypotheses.py
"$PYTHON" user_data/strategy_research/generate_memory_guided_strategies.py

echo "== Strong Strategy Researcher: list generated strategies =="
"$FREQTRADE" list-strategies \
  -c user_data/config_futures_dryrun.json \
  --strategy-path user_data/strategies/research_generated \
  | grep 'Memory.*Strategy'

echo "== Strong Strategy Researcher: memory-guided smoke backtest =="
"$PYTHON" user_data/strategy_research/run_research_agent.py \
  --experiment user_data/strategy_research/experiments/memory_guided_strategy_experiment.json \
  --timerange "$SMOKE_TIMERANGE"

"$PYTHON" user_data/strategy_research/analyze_strategy_research.py
"$PYTHON" user_data/strategy_research/attribute_strategy_failures.py
"$PYTHON" user_data/strategy_research/mature_researcher.py
"$PYTHON" user_data/strategy_research/build_strategy_lineage.py
"$PYTHON" user_data/strategy_research/build_research_memory.py
"$PYTHON" user_data/strategy_research/run_research_agent.py --skip-backtests

cat <<EOF
== Strong Strategy Researcher: outputs ==
Dashboard:  user_data/strategy_research/dashboard/index.html
Sources:    user_data/strategy_research/source_discovery/latest_source_discovery.md
Lineage:    user_data/strategy_research/strategy_library/latest_strategy_lineage.md
Memory:     user_data/strategy_research/research_memory/latest_research_memory.md
MemPlan:    user_data/strategy_research/experiments/memory_guided_hypothesis_ledger.md
MemStrat:   user_data/strategy_research/experiments/memory_guided_strategy_ledger.md
Assessment: user_data/strategy_research/strategy_assessments/latest_strategy_assessment.md
Failures:   user_data/strategy_research/failure_attribution/latest_failure_attribution.md
Researcher: user_data/strategy_research/mature_researcher/latest_researcher_decision.md
Reports:    user_data/strategy_research/reports/
EOF
