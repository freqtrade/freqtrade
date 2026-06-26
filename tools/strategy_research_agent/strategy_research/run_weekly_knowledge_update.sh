#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

PYTHON="${PYTHON:-./.venv/bin/python}"

echo "== Weekly External Knowledge Update: preflight =="
"$PYTHON" user_data/strategy_research/preflight_research_agent.py

echo "== Weekly External Knowledge Update: run =="
"$PYTHON" user_data/strategy_research/weekly_external_knowledge_update.py "$@"

echo "== Weekly External Knowledge Update: artifacts =="
cat <<'EOF'
Weekly:     user_data/strategy_research/knowledge_updates/latest_weekly_knowledge_update.md
Knowledge:  user_data/strategy_research/knowledge/latest_price_action_knowledge_layer_report.md
Graph:      user_data/strategy_research/knowledge/graph/knowledge_graph.md
GraphCtx:   user_data/strategy_research/knowledge/graph/strategy_agent_graph_context.json
Memory:     user_data/strategy_research/research_memory/latest_research_memory.md
Solidify:   user_data/strategy_research/consolidation/latest_research_consolidation.md
AgentRules: user_data/strategy_research/consolidation/agent_operating_rules.json
Dashboard:  user_data/strategy_research/dashboard/index.html
EOF
