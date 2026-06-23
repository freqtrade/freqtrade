#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

PYTHON="${PYTHON:-./.venv/bin/python}"

usage() {
  cat <<'EOF'
Usage: user_data/strategy_research/start_manual_research.sh [--quick|--autonomous-smoke|--full|--full-with-aux|--preflight-only] [--extra-agent-arg ARG ...]

Manual entrypoint for the research-only strategy agent.

Modes:
  --quick            Run preflight, then refresh report/dashboard without backtests.
  --autonomous-smoke Generate autonomous hypotheses and run a short smoke backtest.
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
Reports:    user_data/strategy_research/reports/
EOF
