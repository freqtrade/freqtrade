#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "${ROOT_DIR}"

if [[ -n "${COMPOSE_CMD:-}" ]]; then
    COMPOSE="${COMPOSE_CMD}"
elif docker compose version >/dev/null 2>&1; then
    COMPOSE="docker compose"
elif docker-compose version >/dev/null 2>&1; then
    COMPOSE="docker-compose"
else
    echo "[system_redeploy] docker compose not found." >&2
    exit 1
fi

SERVICE="${FREQTRADE_SERVICE:-freqtrade}"

if [[ "${PULL_FT_IMAGE:-0}" == "1" ]]; then
    echo "[system_redeploy] Pulling image for ${SERVICE}" >&2
    ${COMPOSE} pull "${SERVICE}"
fi

echo "[system_redeploy] Updating service ${SERVICE}" >&2
if [[ "${FORCE_RECREATE:-1}" == "1" ]]; then
    ${COMPOSE} up -d --force-recreate "${SERVICE}"
else
    ${COMPOSE} up -d "${SERVICE}"
fi

if [[ "${RESTART_COLLECTOR:-1}" == "1" ]]; then
    COLLECTOR_SCRIPT="${COLLECTOR_SCRIPT:-scripts/ob_collector_redeploy.sh}"
    if [[ ! -x "${COLLECTOR_SCRIPT}" ]]; then
        echo "[system_redeploy] Collector script ${COLLECTOR_SCRIPT} is not executable." >&2
        exit 1
    fi
    echo "[system_redeploy] Rolling orderbook collector via ${COLLECTOR_SCRIPT}" >&2
    WAIT_SECONDS=${COLLECTOR_WAIT_SECONDS:-20} bash "${COLLECTOR_SCRIPT}"
else
    echo "[system_redeploy] Skipping collector redeploy (RESTART_COLLECTOR=${RESTART_COLLECTOR})" >&2
fi

echo "[system_redeploy] Redeploy finished." >&2
