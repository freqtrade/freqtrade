#!/usr/bin/env bash
set -euo pipefail

# Root of the repository
ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "${ROOT_DIR}"

# Allow caller to override compose command (e.g. COMPOSE_CMD="docker-compose")
if [[ -n "${COMPOSE_CMD:-}" ]]; then
    COMPOSE="${COMPOSE_CMD}"
elif docker compose version >/dev/null 2>&1; then
    COMPOSE="docker compose"
elif docker-compose version >/dev/null 2>&1; then
    COMPOSE="docker-compose"
else
    echo "[system_start] docker compose not found." >&2
    exit 1
fi

# Optionally pull latest freqtrade image
if [[ "${PULL_FT_IMAGE:-0}" == "1" ]]; then
    echo "[system_start] Pulling freqtrade image" >&2
    ${COMPOSE} pull freqtrade
fi

echo "[system_start] Starting freqtrade service" >&2
${COMPOSE} up -d freqtrade

if [[ "${START_COLLECTOR:-1}" == "1" ]]; then
    COLLECTOR_SCRIPT="${COLLECTOR_SCRIPT:-scripts/ob_collector_start.sh}"
    if [[ ! -x "${COLLECTOR_SCRIPT}" ]]; then
        echo "[system_start] Collector script ${COLLECTOR_SCRIPT} is not executable." >&2
        exit 1
    fi
    echo "[system_start] Launching orderbook collector via ${COLLECTOR_SCRIPT}" >&2
    bash "${COLLECTOR_SCRIPT}"
else
    echo "[system_start] Skipping orderbook collector startup (START_COLLECTOR=${START_COLLECTOR})" >&2
fi

echo "[system_start] System startup completed." >&2
