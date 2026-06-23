#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

PYTHON="${PYTHON:-./.venv/bin/python}"
export PYTHONPATH="${PYTHONPATH:-user_data/offline_exchange}"

usage() {
  cat <<'EOF'
Usage: user_data/strategy_research/run_full_research_cycle.sh [--skip-aux-fetch]

Runs the research-only strategy cycle:
  1. Incrementally update BTC/ETH Binance USDT-M 1m OHLCV.
  2. Optionally fetch Binance futures funding/mark aux data.
  3. Convert funding/mark aux data to Freqtrade futures candle format.
  4. Audit futures cost data.
  5. Generate autonomous strategy hypotheses and isolated code.
  6. Build BTC regime and base/stress cost experiments.
  7. Run autonomous smoke and failure-driven iteration smoke.
  8. Run walk-forward validation across fixed calendar windows.
  9. Run base and stress matrices with Freqtrade backtesting.
  10. Summarize matrix robustness.
  11. Build scorecards, failure diagnostics, and trade behavior diagnostics.
  12. Evaluate promotion readiness for manual dry-run review.
  13. Build the next research agenda from promotion blockers.
  14. Refresh the dashboard/report without live trading.

Safety:
  - Does not start live trading.
  - Does not read API keys.
  - Uses local dry-run/research configuration only.
EOF
}

skip_aux_fetch=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-aux-fetch)
      skip_aux_fetch=1
      shift
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

"$PYTHON" user_data/download_binance_um_1m.py --incremental

if [[ "$skip_aux_fetch" -eq 0 ]]; then
  "$PYTHON" user_data/strategy_research/fetch_futures_aux_data.py
fi

"$PYTHON" user_data/strategy_research/convert_aux_to_freqtrade_futures.py
"$PYTHON" user_data/strategy_research/audit_futures_cost_data.py
"$PYTHON" user_data/strategy_research/autonomous_strategy_lab.py
"$PYTHON" user_data/strategy_research/build_experiment_matrix.py

"$PYTHON" user_data/strategy_research/run_research_agent.py \
  --experiment user_data/strategy_research/experiments/autonomous_strategy_experiment.json \
  --timerange 20260101-20260201
autonomous_smoke_report=$("$PYTHON" - <<'PY'
import json
from pathlib import Path

index = json.loads(Path("user_data/strategy_research/reports/agent_report_index.json").read_text())
print(index["latest_report"]["path"])
PY
)

"$PYTHON" user_data/strategy_research/strategy_iteration_engine.py \
  --report "$autonomous_smoke_report"
"$PYTHON" user_data/strategy_research/run_research_agent.py \
  --experiment user_data/strategy_research/experiments/iterative_strategy_experiment.json \
  --timerange 20260101-20260201
iterative_smoke_report=$("$PYTHON" - <<'PY'
import json
from pathlib import Path

index = json.loads(Path("user_data/strategy_research/reports/agent_report_index.json").read_text())
print(index["latest_report"]["path"])
PY
)

"$PYTHON" user_data/strategy_research/walk_forward_validator.py build --source iterative --limit 6
"$PYTHON" user_data/strategy_research/run_research_agent.py \
  --experiment user_data/strategy_research/experiments/walk_forward_validation_experiment.json
walk_forward_report=$("$PYTHON" - <<'PY'
import json
from pathlib import Path

index = json.loads(Path("user_data/strategy_research/reports/agent_report_index.json").read_text())
print(index["latest_report"]["path"])
PY
)
"$PYTHON" user_data/strategy_research/walk_forward_validator.py summarize \
  --report "$walk_forward_report"

"$PYTHON" user_data/strategy_research/run_research_agent.py \
  --experiment user_data/strategy_research/experiments/candidate_regime_matrix_base_cost.json
base_report=$("$PYTHON" - <<'PY'
import json
from pathlib import Path

index = json.loads(Path("user_data/strategy_research/reports/agent_report_index.json").read_text())
print(index["latest_report"]["path"])
PY
)

"$PYTHON" user_data/strategy_research/run_research_agent.py \
  --experiment user_data/strategy_research/experiments/candidate_regime_matrix_stress_cost.json
stress_report=$("$PYTHON" - <<'PY'
import json
from pathlib import Path

index = json.loads(Path("user_data/strategy_research/reports/agent_report_index.json").read_text())
print(index["latest_report"]["path"])
PY
)

"$PYTHON" user_data/strategy_research/summarize_matrix.py \
  --report "$base_report" \
  --report "$stress_report"

"$PYTHON" user_data/strategy_research/analyze_strategy_research.py
"$PYTHON" user_data/strategy_research/analyze_trade_behavior.py
"$PYTHON" user_data/strategy_research/plan_behavior_experiments.py
"$PYTHON" user_data/strategy_research/generate_behavior_experiment_strategies.py
"$PYTHON" user_data/strategy_research/attribute_strategy_failures.py
"$PYTHON" user_data/strategy_research/promotion_gate.py
"$PYTHON" user_data/strategy_research/build_strategy_lineage.py
"$PYTHON" user_data/strategy_research/build_research_memory.py
"$PYTHON" user_data/strategy_research/plan_memory_guided_hypotheses.py
"$PYTHON" user_data/strategy_research/research_agenda.py
"$PYTHON" user_data/strategy_research/run_research_agent.py --skip-backtests

cat <<EOF
Research cycle complete.
Autonomous:   $autonomous_smoke_report
Iterative:    $iterative_smoke_report
Walk-forward: $walk_forward_report
Base report:   $base_report
Stress report: $stress_report
Hypotheses:    user_data/strategy_research/experiments/autonomous_hypothesis_ledger.md
Iterations:    user_data/strategy_research/experiments/iterative_hypothesis_ledger.md
Walk-Fwd:      user_data/strategy_research/walk_forward_summaries/latest_walk_forward_summary.md
Summary:       user_data/strategy_research/matrix_summaries/latest_matrix_summary.md
Assessment:    user_data/strategy_research/strategy_assessments/latest_strategy_assessment.md
Behavior:      user_data/strategy_research/trade_behavior/latest_trade_behavior.md
BehaviorEx:    user_data/strategy_research/behavior_experiments/latest_behavior_experiment_plan.md
BehaviorVar:   user_data/strategy_research/experiments/behavior_experiment_hypothesis_ledger.md
Failures:      user_data/strategy_research/failure_attribution/latest_failure_attribution.md
Promotion:     user_data/strategy_research/promotion_reports/latest_promotion_report.md
Lineage:       user_data/strategy_research/strategy_library/latest_strategy_lineage.md
Memory:        user_data/strategy_research/research_memory/latest_research_memory.md
MemPlan:       user_data/strategy_research/experiments/memory_guided_hypothesis_ledger.md
Agenda:        user_data/strategy_research/research_agendas/latest_research_agenda.md
Dashboard:     user_data/strategy_research/dashboard/index.html
EOF
