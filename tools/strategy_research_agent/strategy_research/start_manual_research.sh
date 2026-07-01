#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

PYTHON="${PYTHON:-./.venv/bin/python}"

usage() {
  cat <<'EOF'
Usage: user_data/strategy_research/start_manual_research.sh [--quick|--source-scout|--price-action-knowledge|--bilibili-transcripts|--knowledge-graph|--knowledge-guided-hypotheses|--factor-research|--factor-to-strategy|--event-study|--agent-brain|--weekly-knowledge-update|--walk-forward|--promotion-gate|--family-risk-gate|--trade-behavior|--failure-attribution|--post-run-attribution|--mature-researcher|--mature-researcher-queue|--execute-mature-researcher|--strategy-lineage|--research-memory|--memory-guided-hypotheses|--memory-guided-strategies|--preflight-only] [--extra-agent-arg ARG ...]

Manual entrypoint for the research-only strategy agent.

Modes:
  --quick            Run preflight, then refresh report/dashboard without backtests.
  --source-scout     Build the external-source discovery and review queue.
  --price-action-knowledge
                     Build the local price-action knowledge base from public metadata/web snapshots and local knowledge cards.
  --bilibili-transcripts
                     Fetch Bilibili AI subtitles from the authenticated local browser cookie jar; does not download video.
  --knowledge-graph
                     Build the graph-structured price-action knowledge layer.
  --knowledge-guided-hypotheses
                     Build the curated price-action knowledge layer and plan hypotheses guarded by research memory.
  --factor-research  Mine 3m/5m/15m futures OHLCV factors before event study or strategy generation.
  --factor-to-strategy
                     Convert factor edge candidates into guarded event-study hypotheses; does not generate strategy classes directly.
  --event-study      Test measurable entry events before strategy generation.
  --agent-brain      Rebuild knowledge graph, research memory, knowledge/memory hypotheses, and consolidation policy.
  --weekly-knowledge-update
                     Refresh external/source knowledge weekly layer, rebuild Agent brain, and write a weekly knowledge update report.
  --walk-forward     Run fixed-window validation for current memory-guided strategies.
  --promotion-gate   Evaluate all-family promotion readiness with family-level risk controls and refresh report/dashboard.
  --family-risk-gate
                     Same gate as promotion-gate: strategy-family router + circuit-breaker dry-run readiness simulation.
  --trade-behavior  Analyze exported trades for behavior-level diagnostics.
  --failure-attribution
                     Build cross-evidence strategy failure attribution.
  --post-run-attribution
                     Run the mandatory after-backtest attribution gate and refresh memory.
  --mature-researcher
                     Build the senior researcher diagnosis and next-experiment decision plan.
  --mature-researcher-queue
                     Convert mature researcher decisions into a safe response queue.
  --execute-mature-researcher
                     Execute the highest priority safe mature researcher queue item.
  --strategy-lineage
                     Build strategy library lineage and refresh report/dashboard.
  --research-memory
                     Build durable research memory and refresh report/dashboard.
  --memory-guided-hypotheses
                     Plan next strategy hypotheses from research memory.
  --memory-guided-strategies
                     Generate isolated strategy variants from memory-guided hypotheses.
  --preflight-only  Only check environment, data, outputs, and safety flags.

Safety:
  - Does not start Freqtrade live trading.
  - Does not read private API keys.
  - Does not modify dry-run/live strategy config.
EOF
}

mode="quick"
extra_args=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --quick)
      mode="quick"
      shift
      ;;
    --source-scout)
      mode="source_scout"
      shift
      ;;
    --price-action-knowledge)
      mode="price_action_knowledge"
      shift
      ;;
    --bilibili-transcripts)
      mode="bilibili_transcripts"
      shift
      ;;
    --knowledge-graph)
      mode="knowledge_graph"
      shift
      ;;
    --knowledge-guided-hypotheses)
      mode="knowledge_guided_hypotheses"
      shift
      ;;
    --factor-research)
      mode="factor_research"
      shift
      ;;
    --factor-to-strategy)
      mode="factor_to_strategy"
      shift
      ;;
    --event-study)
      mode="event_study"
      shift
      ;;
    --agent-brain)
      mode="agent_brain"
      shift
      ;;
    --weekly-knowledge-update)
      mode="weekly_knowledge_update"
      shift
      ;;
    --walk-forward)
      mode="walk_forward"
      shift
      ;;
    --promotion-gate)
      mode="promotion_gate"
      shift
      ;;
    --family-risk-gate)
      mode="family_risk_gate"
      shift
      ;;
    --trade-behavior)
      mode="trade_behavior"
      shift
      ;;
    --failure-attribution)
      mode="failure_attribution"
      shift
      ;;
    --post-run-attribution)
      mode="post_run_attribution"
      shift
      ;;
    --mature-researcher)
      mode="mature_researcher"
      shift
      ;;
    --mature-researcher-queue)
      mode="mature_researcher_queue"
      shift
      ;;
    --execute-mature-researcher)
      mode="execute_mature_researcher"
      shift
      ;;
    --strategy-lineage)
      mode="strategy_lineage"
      shift
      ;;
    --research-memory)
      mode="research_memory"
      shift
      ;;
    --memory-guided-hypotheses)
      mode="memory_guided_hypotheses"
      shift
      ;;
    --memory-guided-strategies)
      mode="memory_guided_strategies"
      shift
      ;;
    --preflight-only)
      mode="preflight_only"
      shift
      ;;
    --extra-agent-arg)
      if [[ $# -lt 2 ]]; then
        echo "--extra-agent-arg requires a value." >&2
        exit 2
      fi
      extra_args+=("$2")
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

echo "== Strategy Research Agent: preflight =="
"$PYTHON" user_data/strategy_research/preflight_research_agent.py

echo "== Strategy Research Agent: fixed workflow gate =="
"$PYTHON" user_data/strategy_research/enforce_agent_workflow_gate.py

if [[ "$mode" == "preflight_only" ]]; then
  exit 0
fi

run_optional_script() {
  local script_path="$1"
  if [[ -f "$script_path" ]]; then
    "$PYTHON" "$script_path"
  else
    echo "WARN: optional script unavailable: $script_path"
  fi
}

run_agent_brain() {
  echo "== Strategy Research Agent: agent brain prerequisite =="
  "$PYTHON" user_data/strategy_research/build_price_action_knowledge_layer.py
  "$PYTHON" user_data/strategy_research/build_price_action_knowledge_graph.py
  run_optional_script user_data/strategy_research/build_strategy_lineage.py
  "$PYTHON" user_data/strategy_research/build_research_memory.py
  "$PYTHON" user_data/strategy_research/factor_research.py
  "$PYTHON" user_data/strategy_research/factor_to_strategy_plan.py
  "$PYTHON" user_data/strategy_research/plan_knowledge_guided_hypotheses.py
  "$PYTHON" user_data/strategy_research/plan_memory_guided_hypotheses.py
  "$PYTHON" user_data/strategy_research/build_research_consolidation.py
}

run_post_run_attribution() {
  echo "== Strategy Research Agent: post-run attribution gate =="
  run_optional_script user_data/strategy_research/build_strategy_lineage.py
  "$PYTHON" user_data/strategy_research/build_research_memory.py
  if ! "$PYTHON" user_data/strategy_research/analyze_trade_behavior.py; then
    echo "WARN: trade behavior diagnostics unavailable; continuing post-run attribution with failure evidence."
  fi
  if [[ -f user_data/strategy_research/entry_quality_review.py ]]; then
    if ! "$PYTHON" user_data/strategy_research/entry_quality_review.py; then
      echo "WARN: entry quality review unavailable; continuing post-run attribution with failure evidence."
    fi
  fi
  "$PYTHON" user_data/strategy_research/attribute_strategy_failures.py
  "$PYTHON" user_data/strategy_research/mature_researcher.py
  "$PYTHON" user_data/strategy_research/mature_researcher_queue.py
  "$PYTHON" user_data/strategy_research/build_research_memory.py
  "$PYTHON" user_data/strategy_research/build_research_consolidation.py
}

run_promotion_experience_update() {
  echo "== Strategy Research Agent: promotion experience update =="
  run_optional_script user_data/strategy_research/build_strategy_lineage.py
  "$PYTHON" user_data/strategy_research/build_research_memory.py
  "$PYTHON" user_data/strategy_research/build_research_consolidation.py
}

refresh_dashboard_if_available() {
  if [[ -f user_data/strategy_research/run_research_agent.py && -f user_data/strategy_research/run_strategy_research.py ]]; then
    "$PYTHON" user_data/strategy_research/run_research_agent.py --skip-backtests
  else
    echo "WARN: dashboard refresh skipped; run_research_agent dependencies are unavailable in runtime."
  fi
}

case "$mode" in
  quick)
    echo "== Strategy Research Agent: quick refresh =="
    "$PYTHON" user_data/strategy_research/run_research_agent.py --skip-backtests ${extra_args[@]+"${extra_args[@]}"}
    ;;
  source_scout)
    echo "== Strategy Research Agent: external source scout =="
    "$PYTHON" user_data/strategy_research/scout_external_sources.py
    "$PYTHON" user_data/strategy_research/run_research_agent.py --skip-backtests
    ;;
  price_action_knowledge)
    echo "== Strategy Research Agent: price action knowledge base =="
    "$PYTHON" user_data/strategy_research/build_price_action_knowledge_base.py
    "$PYTHON" user_data/strategy_research/build_price_action_knowledge_layer.py
    "$PYTHON" user_data/strategy_research/build_price_action_knowledge_graph.py
    "$PYTHON" user_data/strategy_research/query_price_action_knowledge.py breakout confirmation
    "$PYTHON" user_data/strategy_research/run_research_agent.py --skip-backtests
    ;;
  bilibili_transcripts)
    echo "== Strategy Research Agent: Bilibili transcript fetch =="
    "$PYTHON" user_data/strategy_research/fetch_bilibili_transcripts.py
    "$PYTHON" user_data/strategy_research/build_price_action_knowledge_base.py
    "$PYTHON" user_data/strategy_research/build_price_action_knowledge_layer.py
    "$PYTHON" user_data/strategy_research/build_price_action_knowledge_graph.py
    "$PYTHON" user_data/strategy_research/query_price_action_knowledge.py scalp crypto
    "$PYTHON" user_data/strategy_research/run_research_agent.py --skip-backtests
    ;;
  knowledge_graph)
    echo "== Strategy Research Agent: price action knowledge graph =="
    "$PYTHON" user_data/strategy_research/build_price_action_knowledge_layer.py
    "$PYTHON" user_data/strategy_research/build_price_action_knowledge_graph.py
    "$PYTHON" user_data/strategy_research/query_price_action_knowledge_graph.py pinbar --limit 5
    "$PYTHON" user_data/strategy_research/query_price_action_knowledge_graph.py scalp --limit 5
    ;;
  knowledge_guided_hypotheses)
    echo "== Strategy Research Agent: knowledge-guided hypotheses =="
    "$PYTHON" user_data/strategy_research/build_price_action_knowledge_layer.py
    "$PYTHON" user_data/strategy_research/build_price_action_knowledge_graph.py
    run_optional_script user_data/strategy_research/build_strategy_lineage.py
    "$PYTHON" user_data/strategy_research/build_research_memory.py
    "$PYTHON" user_data/strategy_research/plan_knowledge_guided_hypotheses.py
    "$PYTHON" user_data/strategy_research/build_research_consolidation.py
    "$PYTHON" user_data/strategy_research/run_research_agent.py --skip-backtests
    ;;
  factor_research)
    echo "== Strategy Research Agent: factor research =="
    "$PYTHON" user_data/strategy_research/build_price_action_knowledge_layer.py
    "$PYTHON" user_data/strategy_research/build_price_action_knowledge_graph.py
    "$PYTHON" user_data/strategy_research/build_research_memory.py
    "$PYTHON" user_data/strategy_research/factor_research.py
    "$PYTHON" user_data/strategy_research/build_research_consolidation.py
    refresh_dashboard_if_available
    ;;
  factor_to_strategy)
    echo "== Strategy Research Agent: factor-to-strategy guarded plan =="
    "$PYTHON" user_data/strategy_research/build_price_action_knowledge_layer.py
    "$PYTHON" user_data/strategy_research/build_price_action_knowledge_graph.py
    "$PYTHON" user_data/strategy_research/build_research_memory.py
    "$PYTHON" user_data/strategy_research/factor_research.py
    "$PYTHON" user_data/strategy_research/factor_to_strategy_plan.py
    "$PYTHON" user_data/strategy_research/build_research_consolidation.py
    refresh_dashboard_if_available
    ;;
  event_study)
    echo "== Strategy Research Agent: event study edge check =="
    "$PYTHON" user_data/strategy_research/run_event_study.py
    "$PYTHON" user_data/strategy_research/run_research_agent.py --skip-backtests
    ;;
  agent_brain)
    echo "== Strategy Research Agent: knowledge-memory-consolidation brain =="
    run_agent_brain
    "$PYTHON" user_data/strategy_research/run_research_agent.py --skip-backtests
    ;;
  weekly_knowledge_update)
    echo "== Strategy Research Agent: weekly external knowledge update =="
    user_data/strategy_research/run_weekly_knowledge_update.sh --with-bilibili
    ;;
  walk_forward)
    echo "== Strategy Research Agent: walk-forward validation =="
    "$PYTHON" user_data/strategy_research/walk_forward_validator.py build --source base --limit 6
    "$PYTHON" user_data/strategy_research/run_research_agent.py \
      --experiment user_data/strategy_research/experiments/walk_forward_validation_experiment.json \
      ${extra_args[@]+"${extra_args[@]}"}
    walk_forward_report=$("$PYTHON" - <<'PY'
import json
from pathlib import Path

index = json.loads(Path("user_data/strategy_research/reports/agent_report_index.json").read_text())
print(index["latest_report"]["path"])
PY
)
    "$PYTHON" user_data/strategy_research/walk_forward_validator.py summarize --report "$walk_forward_report"
    run_post_run_attribution
    "$PYTHON" user_data/strategy_research/run_research_agent.py --skip-backtests
    ;;
  promotion_gate)
    echo "== Strategy Research Agent: promotion gate =="
    "$PYTHON" user_data/strategy_research/family_risk_gate.py ${extra_args[@]+"${extra_args[@]}"}
    run_optional_script user_data/strategy_research/research_agenda.py
    run_promotion_experience_update
    "$PYTHON" user_data/strategy_research/run_research_agent.py --skip-backtests
    ;;
  family_risk_gate)
    echo "== Strategy Research Agent: family risk gate =="
    "$PYTHON" user_data/strategy_research/family_risk_gate.py ${extra_args[@]+"${extra_args[@]}"}
    run_optional_script user_data/strategy_research/research_agenda.py
    run_promotion_experience_update
    "$PYTHON" user_data/strategy_research/run_research_agent.py --skip-backtests
    ;;
  trade_behavior)
    echo "== Strategy Research Agent: trade behavior analysis =="
    "$PYTHON" user_data/strategy_research/analyze_trade_behavior.py
    "$PYTHON" user_data/strategy_research/run_research_agent.py --skip-backtests
    ;;
  failure_attribution)
    echo "== Strategy Research Agent: failure attribution =="
    "$PYTHON" user_data/strategy_research/attribute_strategy_failures.py
    "$PYTHON" user_data/strategy_research/run_research_agent.py --skip-backtests
    ;;
  post_run_attribution)
    run_post_run_attribution
    "$PYTHON" user_data/strategy_research/run_research_agent.py --skip-backtests
    ;;
  mature_researcher)
    echo "== Strategy Research Agent: mature researcher decision plan =="
    run_agent_brain
    "$PYTHON" user_data/strategy_research/analyze_strategy_research.py
    "$PYTHON" user_data/strategy_research/attribute_strategy_failures.py
    "$PYTHON" user_data/strategy_research/mature_researcher.py
    "$PYTHON" user_data/strategy_research/mature_researcher_queue.py
    "$PYTHON" user_data/strategy_research/run_research_agent.py --skip-backtests
    ;;
  mature_researcher_queue)
    echo "== Strategy Research Agent: mature researcher response queue =="
    run_agent_brain
    "$PYTHON" user_data/strategy_research/analyze_strategy_research.py
    "$PYTHON" user_data/strategy_research/attribute_strategy_failures.py
    "$PYTHON" user_data/strategy_research/mature_researcher.py
    "$PYTHON" user_data/strategy_research/mature_researcher_queue.py
    "$PYTHON" user_data/strategy_research/run_research_agent.py --skip-backtests
    ;;
  execute_mature_researcher)
    echo "== Strategy Research Agent: execute mature researcher response =="
    run_agent_brain
    "$PYTHON" user_data/strategy_research/analyze_strategy_research.py
    "$PYTHON" user_data/strategy_research/attribute_strategy_failures.py
    "$PYTHON" user_data/strategy_research/mature_researcher.py
    "$PYTHON" user_data/strategy_research/mature_researcher_queue.py --execute-next
    "$PYTHON" user_data/strategy_research/run_research_agent.py --skip-backtests
    ;;
  strategy_lineage)
    echo "== Strategy Research Agent: strategy lineage =="
    run_optional_script user_data/strategy_research/build_strategy_lineage.py
    "$PYTHON" user_data/strategy_research/run_research_agent.py --skip-backtests
    ;;
  research_memory)
    echo "== Strategy Research Agent: research memory =="
    run_optional_script user_data/strategy_research/build_strategy_lineage.py
    "$PYTHON" user_data/strategy_research/build_research_memory.py
    "$PYTHON" user_data/strategy_research/build_research_consolidation.py
    "$PYTHON" user_data/strategy_research/run_research_agent.py --skip-backtests
    ;;
  memory_guided_hypotheses)
    echo "== Strategy Research Agent: memory-guided hypotheses =="
    run_optional_script user_data/strategy_research/build_strategy_lineage.py
    "$PYTHON" user_data/strategy_research/build_research_memory.py
    "$PYTHON" user_data/strategy_research/plan_memory_guided_hypotheses.py
    "$PYTHON" user_data/strategy_research/build_research_consolidation.py
    "$PYTHON" user_data/strategy_research/run_research_agent.py --skip-backtests
    ;;
  memory_guided_strategies)
    echo "== Strategy Research Agent: memory-guided strategies =="
    run_optional_script user_data/strategy_research/build_strategy_lineage.py
    "$PYTHON" user_data/strategy_research/build_research_memory.py
    "$PYTHON" user_data/strategy_research/plan_memory_guided_hypotheses.py
    "$PYTHON" user_data/strategy_research/generate_memory_guided_strategies.py
    "$PYTHON" user_data/strategy_research/build_research_consolidation.py
    "$PYTHON" user_data/strategy_research/run_research_agent.py --skip-backtests
    ;;
esac

cat <<'EOF'
== Strategy Research Agent: outputs ==
Dashboard:  user_data/strategy_research/dashboard/index.html
Assessment: user_data/strategy_research/strategy_assessments/latest_strategy_assessment.md
MemPlan:    user_data/strategy_research/experiments/memory_guided_hypothesis_ledger.md
MemStrat:   user_data/strategy_research/experiments/memory_guided_strategy_ledger.md
Walk-Fwd:   user_data/strategy_research/walk_forward_summaries/latest_walk_forward_summary.md
Promotion:  user_data/strategy_research/promotion_reports/latest_promotion_report.md
FamilyGate: user_data/strategy_research/family_risk_gate/latest_family_risk_gate.md
Behavior:   user_data/strategy_research/trade_behavior/latest_trade_behavior.md
Failures:   user_data/strategy_research/failure_attribution/latest_failure_attribution.md
Researcher: user_data/strategy_research/mature_researcher/latest_researcher_decision.md
ResearchQ:  user_data/strategy_research/mature_researcher/latest_response_queue.md
Lineage:    user_data/strategy_research/strategy_library/latest_strategy_lineage.md
Memory:     user_data/strategy_research/research_memory/latest_research_memory.md
Factors:    user_data/strategy_research/factors/latest_factor_research.md
FactorPlan: user_data/strategy_research/factors/latest_factor_strategy_plan.md
EventStudy:user_data/strategy_research/event_studies/latest_event_study.md
MemPlan:    user_data/strategy_research/experiments/memory_guided_hypothesis_ledger.md
MemStrat:   user_data/strategy_research/experiments/memory_guided_strategy_ledger.md
Solidify:   user_data/strategy_research/consolidation/latest_research_consolidation.md
AgentRules: user_data/strategy_research/consolidation/agent_operating_rules.json
Sources:    user_data/strategy_research/source_discovery/latest_source_discovery.md
KnowWeekly: user_data/strategy_research/knowledge_updates/latest_weekly_knowledge_update.md
Knowledge:  user_data/strategy_research/knowledge/latest_price_action_knowledge_report.md
Graph:      user_data/strategy_research/knowledge/graph/knowledge_graph.md
GraphCtx:   user_data/strategy_research/knowledge/graph/strategy_agent_graph_context.json
KnowPlan:   user_data/strategy_research/experiments/knowledge_guided_hypothesis_ledger.md
BiliSubs:   user_data/strategy_research/knowledge/raw_sources/bilibili/bilibili_transcript_fetch_report.md
Reports:    user_data/strategy_research/reports/
EOF
