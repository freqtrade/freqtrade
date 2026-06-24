#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

PYTHON="${PYTHON:-./.venv/bin/python}"

usage() {
  cat <<'EOF'
Usage: user_data/strategy_research/start_manual_research.sh [--quick|--source-scout|--strong-researcher-smoke|--research-iteration|--autonomous-smoke|--iterate-smoke|--walk-forward|--promotion-gate|--agenda|--next-agenda|--execute-next-agenda|--trade-behavior|--behavior-experiments|--behavior-variants|--failure-attribution|--mature-researcher|--mature-researcher-queue|--execute-mature-researcher|--strategy-lineage|--research-memory|--memory-guided-hypotheses|--memory-guided-strategies|--full|--full-with-aux|--preflight-only] [--extra-agent-arg ARG ...]

Manual entrypoint for the research-only strategy agent.

Modes:
  --quick            Run preflight, then refresh report/dashboard without backtests.
  --source-scout     Build the external-source discovery and review queue.
  --strong-researcher-smoke
                     Run the integrated research-only scout/memory/generate/smoke loop.
  --research-iteration
                     Run the fixed experiment -> agent diagnosis -> improvement queue loop.
  --autonomous-smoke Generate autonomous hypotheses and run a short smoke backtest.
  --iterate-smoke    Generate V2 hypotheses from the latest autonomous failures and smoke test them.
  --walk-forward     Run fixed-window validation for current iterative strategies.
  --promotion-gate   Evaluate promotion readiness and refresh report/dashboard.
  --agenda           Build the next research agenda from promotion blockers.
  --next-agenda      Select the next safe agenda item and write a dry-run receipt.
  --execute-next-agenda
                     Execute the next safe non-long agenda item and write a receipt.
  --trade-behavior  Analyze exported trades for behavior-level diagnostics.
  --behavior-experiments
                     Plan follow-up experiments from behavior diagnostics.
  --behavior-variants
                     Generate strategy variants from behavior experiment plans.
  --failure-attribution
                     Build cross-evidence strategy failure attribution.
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
  --full            Run preflight, update 1m OHLCV, run matrix backtests, skip aux fetch.
  --full-with-aux   Same as --full, but also fetch funding/mark aux data.
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
    --strong-researcher-smoke)
      mode="strong_researcher_smoke"
      shift
      ;;
    --research-iteration)
      mode="research_iteration"
      shift
      ;;
    --full)
      mode="full"
      shift
      ;;
    --autonomous-smoke)
      mode="autonomous_smoke"
      shift
      ;;
    --iterate-smoke)
      mode="iterate_smoke"
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
    --agenda)
      mode="agenda"
      shift
      ;;
    --next-agenda)
      mode="next_agenda"
      shift
      ;;
    --execute-next-agenda)
      mode="execute_next_agenda"
      shift
      ;;
    --trade-behavior)
      mode="trade_behavior"
      shift
      ;;
    --behavior-experiments)
      mode="behavior_experiments"
      shift
      ;;
    --behavior-variants)
      mode="behavior_variants"
      shift
      ;;
    --failure-attribution)
      mode="failure_attribution"
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
    --full-with-aux)
      mode="full_with_aux"
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

if [[ "$mode" == "preflight_only" ]]; then
  exit 0
fi

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
  strong_researcher_smoke)
    echo "== Strategy Research Agent: strong researcher smoke =="
    user_data/strategy_research/run_strong_researcher_smoke.sh
    ;;
  research_iteration)
    echo "== Strategy Research Agent: research iteration loop =="
    "$PYTHON" user_data/strategy_research/plan_manual_trade_playbook.py
    "$PYTHON" user_data/strategy_research/generate_manual_direction_strategies.py
    "$PYTHON" user_data/strategy_research/autonomous_strategy_lab.py
    "$PYTHON" user_data/strategy_research/plan_context_sources.py
    "$PYTHON" user_data/strategy_research/build_strategy_lineage.py
    "$PYTHON" user_data/strategy_research/build_research_memory.py
    "$PYTHON" user_data/strategy_research/plan_memory_guided_hypotheses.py
    "$PYTHON" user_data/strategy_research/generate_memory_guided_strategies.py
    "$PYTHON" user_data/strategy_research/plan_family_diversity_experiment.py
    "$PYTHON" user_data/strategy_research/generate_sample_expansion_strategies.py
    "$PYTHON" user_data/strategy_research/generate_entry_quality_strategies.py
    "$PYTHON" user_data/strategy_research/run_research_agent.py \
      --experiment user_data/strategy_research/experiments/manual_direction_experiment.json \
      --timerange 20260101-20260201 \
      ${extra_args[@]+"${extra_args[@]}"}
    "$PYTHON" user_data/strategy_research/analyze_trade_behavior.py
    "$PYTHON" user_data/strategy_research/entry_quality_review.py
    "$PYTHON" user_data/strategy_research/attribute_strategy_failures.py
    "$PYTHON" user_data/strategy_research/mature_researcher.py
    "$PYTHON" user_data/strategy_research/mature_researcher_queue.py
    "$PYTHON" user_data/strategy_research/agent_iteration_review.py
    "$PYTHON" user_data/strategy_research/run_research_agent.py --skip-backtests
    ;;
  autonomous_smoke)
    echo "== Strategy Research Agent: autonomous strategy smoke =="
    "$PYTHON" user_data/strategy_research/autonomous_strategy_lab.py
    "$PYTHON" user_data/strategy_research/run_research_agent.py \
      --experiment user_data/strategy_research/experiments/autonomous_strategy_experiment.json \
      --timerange 20260101-20260201 \
      ${extra_args[@]+"${extra_args[@]}"}
    ;;
  iterate_smoke)
    echo "== Strategy Research Agent: failure-driven iteration smoke =="
    "$PYTHON" user_data/strategy_research/strategy_iteration_engine.py
    "$PYTHON" user_data/strategy_research/run_research_agent.py \
      --experiment user_data/strategy_research/experiments/iterative_strategy_experiment.json \
      --timerange 20260101-20260201 \
      ${extra_args[@]+"${extra_args[@]}"}
    ;;
  walk_forward)
    echo "== Strategy Research Agent: walk-forward validation =="
    "$PYTHON" user_data/strategy_research/strategy_iteration_engine.py
    "$PYTHON" user_data/strategy_research/walk_forward_validator.py build --source iterative --limit 6
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
    "$PYTHON" user_data/strategy_research/run_research_agent.py --skip-backtests
    ;;
  promotion_gate)
    echo "== Strategy Research Agent: promotion gate =="
    "$PYTHON" user_data/strategy_research/promotion_gate.py
    "$PYTHON" user_data/strategy_research/research_agenda.py
    "$PYTHON" user_data/strategy_research/run_research_agent.py --skip-backtests
    ;;
  agenda)
    echo "== Strategy Research Agent: research agenda =="
    "$PYTHON" user_data/strategy_research/research_agenda.py
    "$PYTHON" user_data/strategy_research/run_research_agent.py --skip-backtests
    ;;
  next_agenda)
    echo "== Strategy Research Agent: next agenda dry-run =="
    "$PYTHON" user_data/strategy_research/agenda_executor.py
    "$PYTHON" user_data/strategy_research/run_research_agent.py --skip-backtests
    ;;
  execute_next_agenda)
    echo "== Strategy Research Agent: execute next agenda =="
    "$PYTHON" user_data/strategy_research/agenda_executor.py --execute
    "$PYTHON" user_data/strategy_research/run_research_agent.py --skip-backtests
    ;;
  trade_behavior)
    echo "== Strategy Research Agent: trade behavior analysis =="
    "$PYTHON" user_data/strategy_research/analyze_trade_behavior.py
    "$PYTHON" user_data/strategy_research/run_research_agent.py --skip-backtests
    ;;
  behavior_experiments)
    echo "== Strategy Research Agent: behavior-driven experiments =="
    "$PYTHON" user_data/strategy_research/plan_behavior_experiments.py
    "$PYTHON" user_data/strategy_research/run_research_agent.py --skip-backtests
    ;;
  behavior_variants)
    echo "== Strategy Research Agent: behavior experiment strategy variants =="
    "$PYTHON" user_data/strategy_research/generate_behavior_experiment_strategies.py
    "$PYTHON" user_data/strategy_research/run_research_agent.py --skip-backtests
    ;;
  failure_attribution)
    echo "== Strategy Research Agent: failure attribution =="
    "$PYTHON" user_data/strategy_research/attribute_strategy_failures.py
    "$PYTHON" user_data/strategy_research/run_research_agent.py --skip-backtests
    ;;
  mature_researcher)
    echo "== Strategy Research Agent: mature researcher decision plan =="
    "$PYTHON" user_data/strategy_research/analyze_strategy_research.py
    "$PYTHON" user_data/strategy_research/attribute_strategy_failures.py
    "$PYTHON" user_data/strategy_research/mature_researcher.py
    "$PYTHON" user_data/strategy_research/mature_researcher_queue.py
    "$PYTHON" user_data/strategy_research/run_research_agent.py --skip-backtests
    ;;
  mature_researcher_queue)
    echo "== Strategy Research Agent: mature researcher response queue =="
    "$PYTHON" user_data/strategy_research/mature_researcher_queue.py
    "$PYTHON" user_data/strategy_research/run_research_agent.py --skip-backtests
    ;;
  execute_mature_researcher)
    echo "== Strategy Research Agent: execute mature researcher response =="
    "$PYTHON" user_data/strategy_research/mature_researcher_queue.py --execute-next
    "$PYTHON" user_data/strategy_research/run_research_agent.py --skip-backtests
    ;;
  strategy_lineage)
    echo "== Strategy Research Agent: strategy lineage =="
    "$PYTHON" user_data/strategy_research/build_strategy_lineage.py
    "$PYTHON" user_data/strategy_research/run_research_agent.py --skip-backtests
    ;;
  research_memory)
    echo "== Strategy Research Agent: research memory =="
    "$PYTHON" user_data/strategy_research/build_strategy_lineage.py
    "$PYTHON" user_data/strategy_research/build_research_memory.py
    "$PYTHON" user_data/strategy_research/run_research_agent.py --skip-backtests
    ;;
  memory_guided_hypotheses)
    echo "== Strategy Research Agent: memory-guided hypotheses =="
    "$PYTHON" user_data/strategy_research/build_strategy_lineage.py
    "$PYTHON" user_data/strategy_research/build_research_memory.py
    "$PYTHON" user_data/strategy_research/plan_memory_guided_hypotheses.py
    "$PYTHON" user_data/strategy_research/run_research_agent.py --skip-backtests
    ;;
  memory_guided_strategies)
    echo "== Strategy Research Agent: memory-guided strategies =="
    "$PYTHON" user_data/strategy_research/build_strategy_lineage.py
    "$PYTHON" user_data/strategy_research/build_research_memory.py
    "$PYTHON" user_data/strategy_research/plan_memory_guided_hypotheses.py
    "$PYTHON" user_data/strategy_research/generate_memory_guided_strategies.py
    "$PYTHON" user_data/strategy_research/run_research_agent.py --skip-backtests
    ;;
  full)
    echo "== Strategy Research Agent: full research cycle, aux fetch skipped =="
    user_data/strategy_research/run_full_research_cycle.sh --skip-aux-fetch
    ;;
  full_with_aux)
    echo "== Strategy Research Agent: full research cycle with aux fetch =="
    user_data/strategy_research/run_full_research_cycle.sh
    ;;
esac

cat <<'EOF'
== Strategy Research Agent: outputs ==
Dashboard:  user_data/strategy_research/dashboard/index.html
Assessment: user_data/strategy_research/strategy_assessments/latest_strategy_assessment.md
Matrix:     user_data/strategy_research/matrix_summaries/latest_matrix_summary.md
Hypotheses: user_data/strategy_research/experiments/autonomous_hypothesis_ledger.md
Iterations: user_data/strategy_research/experiments/iterative_hypothesis_ledger.md
Walk-Fwd:   user_data/strategy_research/walk_forward_summaries/latest_walk_forward_summary.md
Promotion:  user_data/strategy_research/promotion_reports/latest_promotion_report.md
Agenda:     user_data/strategy_research/research_agendas/latest_research_agenda.md
AgendaRun:  user_data/strategy_research/agenda_runs/latest_agenda_run.md
Behavior:   user_data/strategy_research/trade_behavior/latest_trade_behavior.md
BehaviorEx: user_data/strategy_research/behavior_experiments/latest_behavior_experiment_plan.md
BehaviorVar:user_data/strategy_research/experiments/behavior_experiment_hypothesis_ledger.md
Failures:   user_data/strategy_research/failure_attribution/latest_failure_attribution.md
Researcher: user_data/strategy_research/mature_researcher/latest_researcher_decision.md
ResearchQ:  user_data/strategy_research/mature_researcher/latest_response_queue.md
IterReview: user_data/strategy_research/agent_iterations/latest_iteration_review.md
ImproveQ:   user_data/strategy_research/agent_iterations/improvement_queue.json
Context:    user_data/strategy_research/context_sources/latest_context_source_plan.md
ManualPB:   user_data/strategy_research/manual_playbook/latest_manual_trade_playbook.md
ManualDir:  user_data/strategy_research/manual_playbook/latest_manual_direction_plan.md
FamilyDiv:  user_data/strategy_research/family_diversity/latest_family_diversity_plan.md
SampleEx:   user_data/strategy_research/sample_expansion/latest_sample_expansion_plan.md
EntryQual:  user_data/strategy_research/entry_quality/latest_entry_quality_review.md
EntryPlan:  user_data/strategy_research/entry_quality/latest_directed_experiment_plan.md
Lineage:    user_data/strategy_research/strategy_library/latest_strategy_lineage.md
Memory:     user_data/strategy_research/research_memory/latest_research_memory.md
MemPlan:    user_data/strategy_research/experiments/memory_guided_hypothesis_ledger.md
MemStrat:   user_data/strategy_research/experiments/memory_guided_strategy_ledger.md
Sources:    user_data/strategy_research/source_discovery/latest_source_discovery.md
Reports:    user_data/strategy_research/reports/
EOF
