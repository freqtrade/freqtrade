#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

PYTHON="${PYTHON:-./.venv/bin/python}"

usage() {
  cat <<'EOF'
Usage: user_data/strategy_research/start_manual_research.sh [--quick|--autonomous-smoke|--iterate-smoke|--walk-forward|--promotion-gate|--agenda|--next-agenda|--execute-next-agenda|--trade-behavior|--full|--full-with-aux|--preflight-only] [--extra-agent-arg ARG ...]

Manual entrypoint for the research-only strategy agent.

Modes:
  --quick            Run preflight, then refresh report/dashboard without backtests.
  --autonomous-smoke Generate autonomous hypotheses and run a short smoke backtest.
  --iterate-smoke    Generate V2 hypotheses from the latest autonomous failures and smoke test them.
  --walk-forward     Run fixed-window validation for current iterative strategies.
  --promotion-gate   Evaluate promotion readiness and refresh report/dashboard.
  --agenda           Build the next research agenda from promotion blockers.
  --next-agenda      Select the next safe agenda item and write a dry-run receipt.
  --execute-next-agenda
                     Execute the next safe non-long agenda item and write a receipt.
  --trade-behavior  Analyze exported trades for behavior-level diagnostics.
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
Reports:    user_data/strategy_research/reports/
EOF
