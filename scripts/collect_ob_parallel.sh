#!/usr/bin/env bash
set -euo pipefail

# Parallel launcher for orderbook collectors (BTCUSDT + ETHUSDT)
# Usage: scripts/collect_ob_parallel.sh [RUN_SECONDS] [EXCHANGE] [DEPTH]
# Defaults: RUN_SECONDS=7200, EXCHANGE=bybit, DEPTH=200

RUN_SECONDS="${1:-7200}"
EXCHANGE="${2:-bybit}"
DEPTH="${3:-200}"

ROOT_DIR="user_data/featurestore/${EXCHANGE}"
LOG_DIR="user_data/logs"
mkdir -p "${LOG_DIR}"

echo "Starting OB collectors in parallel (RUN_SECONDS=${RUN_SECONDS}, EXCHANGE=${EXCHANGE}, DEPTH=${DEPTH})"

start_collector() {
  local symbol="$1"
  local log_file="$2"
  echo "Launching ${symbol} → ${log_file}"
  nohup env \
    EXCHANGE="${EXCHANGE}" \
    SYMBOL="${symbol}" \
    DEPTH="${DEPTH}" \
    RUN_SECONDS="${RUN_SECONDS}" \
    ROOT_DIR="${ROOT_DIR}/${symbol}/1s" \
    python tools/ob_collector_ws.py >> "${log_file}" 2>&1 &
}

start_collector "BTCUSDT" "${LOG_DIR}/ob_BTCUSDT.log"
start_collector "ETHUSDT" "${LOG_DIR}/ob_ETHUSDT.log"

sleep 1
echo "Collectors started. PIDs:"
pgrep -f "tools/ob_collector_ws.py" || true
echo "Logs: ${LOG_DIR}/ob_BTCUSDT.log, ${LOG_DIR}/ob_ETHUSDT.log"
