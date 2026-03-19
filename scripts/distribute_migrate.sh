#!/usr/bin/env bash
# ============================================================================
# Distributed Strategy Migration — Exchange GA strategies between machines
# ============================================================================
#
# Bidirectional strategy exchange between two machines via SSH:
#   1. PUSH: Sends local outgoing_migrants/ → remote incoming_migrants/
#   2. PULL: Fetches remote outgoing_migrants/ → local incoming_migrants/
#
# Designed for asymmetric SSH setups (e.g. laptop can SSH to server, but
# not vice versa). Run this script on the machine that HAS SSH access.
#
# Works with GenericIslandModelEvolution's external_migration feature:
#   - The GA exports top individuals to outgoing_migrants/ each N generations
#   - This script exchanges them between machines
#   - Each GA's _load_external_migrants() picks up new arrivals
#
# Prerequisites:
#   - SSH key-based auth (ssh-copy-id) from THIS machine to REMOTE
#   - Same repo path on both machines (or set REMOTE_REPO)
#
# Usage:
#   ./scripts/distribute_migrate.sh                    # run once (push+pull)
#   ./scripts/distribute_migrate.sh --daemon           # run continuously
#   ./scripts/distribute_migrate.sh --daemon --interval 60
#   ./scripts/distribute_migrate.sh --status           # show pending files
#   ./scripts/distribute_migrate.sh --setup-keys       # set up SSH keys
#   ./scripts/distribute_migrate.sh --push-only        # only push to remote
#   ./scripts/distribute_migrate.sh --pull-only        # only pull from remote
#
# Configuration (override via environment or edit below):
#   REMOTE_HOST     - SSH host of the OTHER machine
#   REMOTE_USER     - SSH user on remote
#   REMOTE_REPO     - Repo path on remote (default: same as local)
#   POLL_INTERVAL   - Seconds between checks in daemon mode (default: 30)
# ============================================================================

set -uo pipefail

# ── Configuration ──
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
OUTGOING_DIR="${REPO_DIR}/genetic_algorithm/data/outgoing_migrants"
INCOMING_DIR="${REPO_DIR}/genetic_algorithm/data/incoming_migrants"
SENT_DIR="${REPO_DIR}/genetic_algorithm/data/outgoing_migrants/.sent"
PULLED_DIR="${REPO_DIR}/genetic_algorithm/data/incoming_migrants/.pulled_log"
LOG_DIR="${REPO_DIR}/genetic_algorithm/logs"
LOG_FILE="${LOG_DIR}/distribute_migrate.log"

# Remote configuration — edit these or set via environment
# Default: server IP (run this script from the LAPTOP to reach the SERVER)
if [ -z "${REMOTE_HOST}" ] || [ -z "${REMOTE_USER}" ]; then
    echo "ERROR: Set REMOTE_HOST and REMOTE_USER environment variables."
    echo "  Example: export REMOTE_HOST=192.168.1.100 REMOTE_USER=user"
    exit 1
fi
REMOTE_HOST="${REMOTE_HOST}"
REMOTE_USER="${REMOTE_USER}"
REMOTE_REPO="${REMOTE_REPO:-${REPO_DIR}}"
REMOTE_INCOMING="${REMOTE_REPO}/genetic_algorithm/data/incoming_migrants"
REMOTE_OUTGOING="${REMOTE_REPO}/genetic_algorithm/data/outgoing_migrants"

POLL_INTERVAL="${POLL_INTERVAL:-30}"
SSH_OPTS="-o ConnectTimeout=5 -o BatchMode=yes -o StrictHostKeyChecking=accept-new"

# ── Colours ──
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# ── Logging ──
mkdir -p "${LOG_DIR}" "${INCOMING_DIR}" "${OUTGOING_DIR}" "${SENT_DIR}" "${PULLED_DIR}"

log() {
    local level="$1"; shift
    local msg
    msg="[$(date '+%Y-%m-%d %H:%M:%S')] [${level}] $*"
    echo "${msg}" >> "${LOG_FILE}"
    if [[ "${level}" == "ERROR" ]]; then
        echo -e "${RED}${msg}${NC}" >&2
    elif [[ "${level}" == "WARN" ]]; then
        echo -e "${YELLOW}${msg}${NC}"
    else
        echo -e "${GREEN}${msg}${NC}"
    fi
}

# ── Functions ──

check_ssh_connectivity() {
    if ssh ${SSH_OPTS} "${REMOTE_USER}@${REMOTE_HOST}" "echo ok" &>/dev/null; then
        return 0
    else
        return 1
    fi
}

push_migrants() {
    # PUSH: local outgoing_migrants/ → remote incoming_migrants/
    local files=()
    while IFS= read -r -d '' f; do
        files+=("$f")
    done < <(find "${OUTGOING_DIR}" -maxdepth 1 -name '*.json' -print0 2>/dev/null)

    if [[ ${#files[@]} -eq 0 ]]; then
        return 0
    fi

    log "INFO" "[PUSH] Found ${#files[@]} outgoing migrant file(s) to send"

    ssh ${SSH_OPTS} "${REMOTE_USER}@${REMOTE_HOST}" \
        "mkdir -p '${REMOTE_INCOMING}'" 2>/dev/null

    local sent=0
    for fpath in "${files[@]}"; do
        local fname
        fname="$(basename "${fpath}")"

        if scp ${SSH_OPTS} -q "${fpath}" \
            "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_INCOMING}/${fname}" 2>/dev/null; then
            mv "${fpath}" "${SENT_DIR}/${fname}"
            sent=$((sent + 1))
            log "INFO" "[PUSH]   → Sent ${fname} to ${REMOTE_HOST}"
        else
            log "ERROR" "[PUSH]   ✗ Failed to send ${fname}"
        fi
    done

    if [[ ${sent} -gt 0 ]]; then
        log "INFO" "[PUSH] Sent ${sent}/${#files[@]} files to ${REMOTE_USER}@${REMOTE_HOST}"
    fi

    # Cleanup old .sent files (keep last 50)
    local sent_count
    sent_count="$(find "${SENT_DIR}" -name '*.json' 2>/dev/null | wc -l)"
    if [[ ${sent_count} -gt 50 ]]; then
        find "${SENT_DIR}" -name '*.json' -printf '%T@ %p\n' 2>/dev/null \
            | sort -n | head -n $((sent_count - 50)) | cut -d' ' -f2- \
            | xargs rm -f 2>/dev/null
    fi

    return 0
}

pull_migrants() {
    # PULL: remote outgoing_migrants/ → local incoming_migrants/
    # List remote outgoing files
    local remote_files
    remote_files="$(ssh ${SSH_OPTS} "${REMOTE_USER}@${REMOTE_HOST}" \
        "find '${REMOTE_OUTGOING}' -maxdepth 1 -name '*.json' -printf '%f\n' 2>/dev/null" 2>/dev/null)"

    if [[ -z "${remote_files}" ]]; then
        return 0
    fi

    local count
    count="$(echo "${remote_files}" | wc -l)"
    log "INFO" "[PULL] Found ${count} file(s) on remote to pull"

    local pulled=0
    while IFS= read -r fname; do
        [[ -z "${fname}" ]] && continue

        # Skip if we already pulled this file
        if [[ -f "${PULLED_DIR}/${fname}" ]]; then
            continue
        fi

        # SCP from remote outgoing → local incoming
        if scp ${SSH_OPTS} -q \
            "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_OUTGOING}/${fname}" \
            "${INCOMING_DIR}/${fname}" 2>/dev/null; then

            # Mark as pulled (touch a marker file)
            touch "${PULLED_DIR}/${fname}"

            # Remove from remote outgoing so it doesn't keep accumulating
            ssh ${SSH_OPTS} "${REMOTE_USER}@${REMOTE_HOST}" \
                "rm -f '${REMOTE_OUTGOING}/${fname}'" 2>/dev/null

            pulled=$((pulled + 1))
            log "INFO" "[PULL]   ← Pulled ${fname} from ${REMOTE_HOST}"
        else
            log "ERROR" "[PULL]   ✗ Failed to pull ${fname}"
        fi
    done <<< "${remote_files}"

    if [[ ${pulled} -gt 0 ]]; then
        log "INFO" "[PULL] Pulled ${pulled} files from ${REMOTE_USER}@${REMOTE_HOST}"
    fi

    # Cleanup old pulled markers (keep last 100)
    local marker_count
    marker_count="$(find "${PULLED_DIR}" -name '*.json' 2>/dev/null | wc -l)"
    if [[ ${marker_count} -gt 100 ]]; then
        find "${PULLED_DIR}" -name '*.json' -printf '%T@ %p\n' 2>/dev/null \
            | sort -n | head -n $((marker_count - 100)) | cut -d' ' -f2- \
            | xargs rm -f 2>/dev/null
    fi

    return 0
}

exchange_migrants() {
    push_migrants
    pull_migrants
}

show_status() {
    local incoming_count outgoing_count sent_count pulled_count
    incoming_count="$(find "${INCOMING_DIR}" -maxdepth 1 -name '*.json' 2>/dev/null | wc -l)"
    outgoing_count="$(find "${OUTGOING_DIR}" -maxdepth 1 -name '*.json' 2>/dev/null | wc -l)"
    sent_count="$(find "${SENT_DIR}" -name '*.json' 2>/dev/null | wc -l)"
    pulled_count="$(find "${PULLED_DIR}" -name '*.json' 2>/dev/null | wc -l)"

    echo -e "${BOLD}═══ Distributed Migration Status ═══${NC}"
    echo -e "  ${CYAN}This machine:${NC}  $(hostname) ($(hostname -I | awk '{print $1}'))"
    echo -e "  ${CYAN}Remote:${NC}        ${REMOTE_USER}@${REMOTE_HOST}"
    echo ""
    echo -e "  ${BOLD}Local:${NC}"
    echo -e "    ${BLUE}Incoming (waiting for GA pickup):${NC}  ${incoming_count} files"
    echo -e "    ${BLUE}Outgoing (waiting to send):${NC}        ${outgoing_count} files"
    echo -e "    ${BLUE}Sent (archived):${NC}                   ${sent_count} files"
    echo -e "    ${BLUE}Pulled (from remote, total):${NC}       ${pulled_count} files"
    echo ""

    if check_ssh_connectivity; then
        echo -e "  ${GREEN}SSH connectivity:${NC}  ✓ OK"

        local remote_in remote_out
        remote_in="$(ssh ${SSH_OPTS} "${REMOTE_USER}@${REMOTE_HOST}" \
            "find '${REMOTE_INCOMING}' -maxdepth 1 -name '*.json' 2>/dev/null | wc -l" 2>/dev/null)"
        remote_out="$(ssh ${SSH_OPTS} "${REMOTE_USER}@${REMOTE_HOST}" \
            "find '${REMOTE_OUTGOING}' -maxdepth 1 -name '*.json' 2>/dev/null | wc -l" 2>/dev/null)"
        echo -e "  ${BOLD}Remote:${NC}"
        echo -e "    ${BLUE}Incoming (waiting for GA pickup):${NC}  ${remote_in:-?} files"
        echo -e "    ${BLUE}Outgoing (waiting to pull):${NC}        ${remote_out:-?} files"
    else
        echo -e "  ${RED}SSH connectivity:${NC}  ✗ FAILED"
        echo -e "  ${YELLOW}Run: $0 --setup-keys${NC}"
    fi
    echo ""
}

setup_ssh_keys() {
    echo -e "${BOLD}═══ SSH Key Setup ═══${NC}"
    echo ""

    if [[ ! -f "${HOME}/.ssh/id_ed25519" ]] && [[ ! -f "${HOME}/.ssh/id_rsa" ]]; then
        echo -e "${YELLOW}No SSH key found. Generating one...${NC}"
        ssh-keygen -t ed25519 -f "${HOME}/.ssh/id_ed25519" -N "" -C "$(whoami)@$(hostname)"
        echo -e "${GREEN}Key generated.${NC}"
    else
        echo -e "${GREEN}SSH key already exists.${NC}"
    fi

    echo ""
    echo -e "Copying key to ${REMOTE_USER}@${REMOTE_HOST}..."
    echo -e "${YELLOW}You may be asked for the remote password ONE TIME:${NC}"
    ssh-copy-id -i "${HOME}/.ssh/id_ed25519.pub" "${REMOTE_USER}@${REMOTE_HOST}" 2>/dev/null \
        || ssh-copy-id "${REMOTE_USER}@${REMOTE_HOST}"

    echo ""
    if check_ssh_connectivity; then
        echo -e "${GREEN}✓ SSH key authentication works!${NC}"
    else
        echo -e "${RED}✗ SSH key authentication failed. Check manually.${NC}"
    fi
}

run_once() {
    if ! check_ssh_connectivity; then
        log "ERROR" "Cannot reach ${REMOTE_USER}@${REMOTE_HOST}. Run: $0 --setup-keys"
        return 1
    fi
    exchange_migrants
}

run_daemon() {
    log "INFO" "Starting migration daemon (poll every ${POLL_INTERVAL}s, push+pull)"
    log "INFO" "  Local:  $(hostname) ($(hostname -I | awk '{print $1}'))"
    log "INFO" "  Remote: ${REMOTE_USER}@${REMOTE_HOST}"

    trap 'log "INFO" "Daemon shutting down..."; exit 0' SIGINT SIGTERM

    while true; do
        if check_ssh_connectivity; then
            exchange_migrants
        else
            log "WARN" "Remote unreachable, will retry in ${POLL_INTERVAL}s"
        fi
        sleep "${POLL_INTERVAL}"
    done
}

# ── Main ──

case "${1:-}" in
    --status)
        show_status
        ;;
    --setup-keys)
        setup_ssh_keys
        ;;
    --push-only)
        check_ssh_connectivity && push_migrants
        ;;
    --pull-only)
        check_ssh_connectivity && pull_migrants
        ;;
    --daemon)
        if [[ "${2:-}" == "--interval" ]] && [[ -n "${3:-}" ]]; then
            POLL_INTERVAL="$3"
        fi
        run_daemon
        ;;
    --help|-h)
        head -40 "$0" | tail -35
        ;;
    *)
        run_once
        ;;
esac
