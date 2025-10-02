#!/usr/bin/env bash
set -euo pipefail

# Configuration (override via environment variables)
# Avoid MSYS from rewriting docker volume paths on Windows
export MSYS_NO_PATHCONV=1
IMAGE_NAME="${IMAGE_NAME:-ob-collector:latest}"
DOCKERFILE_PATH="${DOCKERFILE_PATH:-Dockerfile.ob_collector}"
BASENAME="${BASENAME:-ob}"
CONTAINER_NAME="${CONTAINER_NAME:-${BASENAME}-green}"
HOST_FEATURESTORE="${HOST_FEATURESTORE:-$PWD/user_data/featurestore}"
EXCHANGE="${EXCHANGE:-bybit}"
SYMBOL="${SYMBOL:-BTCUSDT}"
CATEGORY="${CATEGORY:-linear}"  # bybit linear perpetual; use 'spot' for spot markets
DEPTH="${DEPTH:-200}"
RUN_SECONDS="${RUN_SECONDS:-0}"
HEARTBEAT_TIMEOUT="${HEARTBEAT_TIMEOUT:-15}"
BACKOFF_BASE="${BACKOFF_BASE:-3.0}"
PING_INTERVAL="${PING_INTERVAL:-10}"

resolve_path() {
    local target="$1"
    case "$target" in
        ~*) target="${target/#\~/$HOME}" ;;
    esac

    local resolved=""
    if command -v realpath >/dev/null 2>&1; then
        resolved="$(realpath "$target" 2>/dev/null || true)"
    fi
    if [[ -z "$resolved" ]] && command -v readlink >/dev/null 2>&1; then
        resolved="$(readlink -f "$target" 2>/dev/null || true)"
    fi
    if [[ -z "$resolved" ]]; then
        if [[ "$target" == /* ]]; then
            resolved="$target"
        else
            resolved="$(pwd -P)/$target"
        fi
    fi
    printf '%s\n' "$resolved"
}

HOST_PATH=$(resolve_path "$HOST_FEATURESTORE")
mkdir -p "$HOST_PATH"

echo "[ob_collector_start] Building image ${IMAGE_NAME} from ${DOCKERFILE_PATH}" >&2
docker build -f "${DOCKERFILE_PATH}" -t "${IMAGE_NAME}" .

echo "[ob_collector_start] Removing existing container ${CONTAINER_NAME} if present" >&2
docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true

echo "[ob_collector_start] Launching ${CONTAINER_NAME}" >&2
docker run -d \
  --name "${CONTAINER_NAME}" \
  --label "ob_collector.app=${BASENAME}" \
  -e EXCHANGE="${EXCHANGE}" \
  -e SYMBOL="${SYMBOL}" \
  -e CATEGORY="${CATEGORY}" \
  -e DEPTH="${DEPTH}" \
  -e ROOT_DIR="/data/featurestore/${EXCHANGE}/${SYMBOL}/1s" \
  -e RUN_SECONDS="${RUN_SECONDS}" \
  -e HEARTBEAT_TIMEOUT="${HEARTBEAT_TIMEOUT}" \
  -e BACKOFF_BASE="${BACKOFF_BASE}" \
  -e PING_INTERVAL="${PING_INTERVAL}" \
  -v "${HOST_PATH}:/data/featurestore" \
  "${IMAGE_NAME}"

echo "[ob_collector_start] Container ${CONTAINER_NAME} is running. Use 'docker logs -f ${CONTAINER_NAME}' to monitor." >&2
