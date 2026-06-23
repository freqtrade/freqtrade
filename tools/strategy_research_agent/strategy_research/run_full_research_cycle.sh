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
  7. Run autonomous smoke, base, and stress matrices with Freqtrade backtesting.
  8. Summarize matrix robustness.
  9. Build scorecards and failure diagnostics.
  10. Refresh the dashboard/report without live trading.

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
"$PYTHON" user_data/strategy_research/run_research_agent.py --skip-backtests

cat <<EOF
Research cycle complete.
Autonomous:   $autonomous_smoke_report
Base report:   $base_report
Stress report: $stress_report
Hypotheses:    user_data/strategy_research/experiments/autonomous_hypothesis_ledger.md
Summary:       user_data/strategy_research/matrix_summaries/latest_matrix_summary.md
Assessment:    user_data/strategy_research/strategy_assessments/latest_strategy_assessment.md
Dashboard:     user_data/strategy_research/dashboard/index.html
EOF
