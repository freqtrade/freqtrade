#!/usr/bin/env bash
set -euo pipefail

# Configuration (override via environment variables)
# Avoid MSYS from rewriting docker volume paths on Windows
export MSYS_NO_PATHCONV=1
IMAGE_NAME="${IMAGE_NAME:-ob-collector:latest}"
DOCKERFILE_PATH="${DOCKERFILE_PATH:-Dockerfile.ob_collector}"
BASENAME="${BASENAME:-ob}"
REBUILD_IMAGE="${REBUILD_IMAGE:-0}"
WAIT_SECONDS="${WAIT_SECONDS:-20}"
HOST_FEATURESTORE="${HOST_FEATURESTORE:-$PWD/user_data/featurestore}"
EXCHANGE="${EXCHANGE:-bybit}"
SYMBOL="${SYMBOL:-BTCUSDT}"
CATEGORY="${CATEGORY:-linear}"
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

CURRENT=$(docker ps --filter "label=ob_collector.app=${BASENAME}" --format '{{.Names}}' | head -n1)
if [[ -z "$CURRENT" ]]; then
    echo "[ob_collector_redeploy] No running container found for label ob_collector.app=${BASENAME}." >&2
    echo "[ob_collector_redeploy] Use scripts/ob_collector_start.sh to launch the first instance." >&2
    exit 1
fi

echo "[ob_collector_redeploy] Current active container: ${CURRENT}" >&2

SUFFIX=${CURRENT##*-}
PREFIX=${CURRENT%-${SUFFIX}}
case "$SUFFIX" in
    green) NEXT_SUFFIX=blue ;;
    blue)  NEXT_SUFFIX=green ;;
    *)     NEXT_SUFFIX=blue
           PREFIX=${BASENAME}
           ;;
esac
NEXT="${PREFIX}-${NEXT_SUFFIX}"

echo "[ob_collector_redeploy] Next container will be ${NEXT}" >&2

if [[ "$REBUILD_IMAGE" == "1" ]]; then
    echo "[ob_collector_redeploy] Rebuilding image ${IMAGE_NAME}" >&2
    docker build -f "${DOCKERFILE_PATH}" -t "${IMAGE_NAME}" .
fi

echo "[ob_collector_redeploy] Starting ${NEXT}" >&2
docker run -d \
  --name "${NEXT}" \
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

echo "[ob_collector_redeploy] Waiting ${WAIT_SECONDS}s before cutting traffic" >&2
sleep "$WAIT_SECONDS"

echo "[ob_collector_redeploy] Stopping old container ${CURRENT}" >&2
docker stop "$CURRENT" >/dev/null

echo "[ob_collector_redeploy] Removing old container ${CURRENT}" >&2
docker rm "$CURRENT" >/dev/null

echo "[ob_collector_redeploy] ${NEXT} is now the active collector." >&2
