#!/usr/bin/env bash
# ============================================================================
# Distributed Strategy Migration — Exchange GA strategies between machines
# ============================================================================
#
# This script runs on each machine and:
#   1. Picks up outgoing migrant files from local outgoing_migrants/
#   2. SCPs them to the remote machine's incoming_migrants/
#   3. Cleans up sent files to avoid re-sending
#
# Designed to work with GenericIslandModelEvolution's external migration:
#   - The GA exports top individuals to outgoing_migrants/ each N generations
#   - This script ships them to the remote machine via SSH
#   - The remote GA's _load_external_migrants() picks them up
#
# Prerequisites:
#   - SSH key-based auth set up (ssh-copy-id) between machines
#   - Same repo path on both machines
#
# Usage:
#   ./scripts/distribute_migrate.sh                    # run once
#   ./scripts/distribute_migrate.sh --daemon           # run continuously
#   ./scripts/distribute_migrate.sh --daemon --interval 60
#   ./scripts/distribute_migrate.sh --status           # show pending files
#   ./scripts/distribute_migrate.sh --setup-keys       # set up SSH keys
#
# Configuration (override via environment or edit below):
#   REMOTE_HOST     - SSH host (default: from config)
#   REMOTE_USER     - SSH user (default: from config)
#   REMOTE_REPO     - Repo path on remote (default: same as local)
#   POLL_INTERVAL   - Seconds between checks in daemon mode (default: 30)
# ============================================================================

set -uo pipefail

# ── Configuration ──
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
OUTGOING_DIR="${REPO_DIR}/genetic_algorithm/data/outgoing_migrants"
INCOMING_DIR="${REPO_DIR}/genetic_algorithm/data/incoming_migrants"
SENT_DIR="${REPO_DIR}/genetic_algorithm/data/outgoing_migrants/.sent"
LOG_DIR="${REPO_DIR}/genetic_algorithm/logs"
LOG_FILE="${LOG_DIR}/distribute_migrate.log"

# Remote configuration — edit these or set via environment
REMOTE_HOST="${REMOTE_HOST:-192.168.178.30}"
REMOTE_USER="${REMOTE_USER:-periklis}"
REMOTE_REPO="${REMOTE_REPO:-${REPO_DIR}}"
REMOTE_INCOMING="${REMOTE_REPO}/genetic_algorithm/data/incoming_migrants"

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
mkdir -p "${LOG_DIR}" "${INCOMING_DIR}" "${OUTGOING_DIR}" "${SENT_DIR}"

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
    # Test SSH connection (key-based, no password prompt)
    if ssh ${SSH_OPTS} "${REMOTE_USER}@${REMOTE_HOST}" "echo ok" &>/dev/null; then
        return 0
    else
        return 1
    fi
}

send_migrants() {
    # Find all JSON files in outgoing directory (skip .sent/ and .tmp)
    local files=()
    while IFS= read -r -d '' f; do
        files+=("$f")
    done < <(find "${OUTGOING_DIR}" -maxdepth 1 -name '*.json' -print0 2>/dev/null)

    if [[ ${#files[@]} -eq 0 ]]; then
        return 0
    fi

    log "INFO" "Found ${#files[@]} outgoing migrant file(s) to send"

    # Ensure remote incoming directory exists
    ssh ${SSH_OPTS} "${REMOTE_USER}@${REMOTE_HOST}" \
        "mkdir -p '${REMOTE_INCOMING}'" 2>/dev/null

    local sent=0
    for fpath in "${files[@]}"; do
        local fname
        fname="$(basename "${fpath}")"

        # SCP the file to remote incoming_migrants
        if scp ${SSH_OPTS} -q "${fpath}" \
            "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_INCOMING}/${fname}" 2>/dev/null; then
            # Move to .sent/ to avoid re-sending
            mv "${fpath}" "${SENT_DIR}/${fname}"
            sent=$((sent + 1))
            log "INFO" "  → Sent ${fname} to ${REMOTE_HOST}"
        else
            log "ERROR" "  ✗ Failed to send ${fname} to ${REMOTE_HOST}"
        fi
    done

    if [[ ${sent} -gt 0 ]]; then
        log "INFO" "Sent ${sent}/${#files[@]} migrant files to ${REMOTE_USER}@${REMOTE_HOST}"
    fi

    # Cleanup old .sent files (keep last 50)
    local sent_count
    sent_count="$(find "${SENT_DIR}" -name '*.json' 2>/dev/null | wc -l)"
    if [[ ${sent_count} -gt 50 ]]; then
        find "${SENT_DIR}" -name '*.json' -printf '%T@ %p\n' 2>/dev/null \
            | sort -n | head -n $((sent_count - 50)) | cut -d' ' -f2- \
            | xargs rm -f 2>/dev/null
    fi

    return ${sent}
}

receive_status() {
    local incoming_count outgoing_count sent_count
    incoming_count="$(find "${INCOMING_DIR}" -maxdepth 1 -name '*.json' 2>/dev/null | wc -l)"
    outgoing_count="$(find "${OUTGOING_DIR}" -maxdepth 1 -name '*.json' 2>/dev/null | wc -l)"
    sent_count="$(find "${SENT_DIR}" -name '*.json' 2>/dev/null | wc -l)"

    echo -e "${BOLD}═══ Distributed Migration Status ═══${NC}"
    echo -e "  ${CYAN}Machine:${NC}   $(hostname)"
    echo -e "  ${CYAN}Remote:${NC}    ${REMOTE_USER}@${REMOTE_HOST}"
    echo ""
    echo -e "  ${BLUE}Incoming (waiting):${NC}  ${incoming_count} files"
    echo -e "  ${BLUE}Outgoing (pending):${NC}  ${outgoing_count} files"
    echo -e "  ${BLUE}Sent (archived):${NC}     ${sent_count} files"
    echo ""

    # Check SSH connectivity
    if check_ssh_connectivity; then
        echo -e "  ${GREEN}SSH connectivity:${NC}    ✓ OK"

        # Check remote side too
        local remote_in remote_out
        remote_in="$(ssh ${SSH_OPTS} "${REMOTE_USER}@${REMOTE_HOST}" \
            "find '${REMOTE_INCOMING}' -maxdepth 1 -name '*.json' 2>/dev/null | wc -l" 2>/dev/null)"
        remote_out="$(ssh ${SSH_OPTS} "${REMOTE_USER}@${REMOTE_HOST}" \
            "find '${REMOTE_REPO}/genetic_algorithm/data/outgoing_migrants' -maxdepth 1 -name '*.json' 2>/dev/null | wc -l" 2>/dev/null)"
        echo -e "  ${BLUE}Remote incoming:${NC}     ${remote_in:-?} files"
        echo -e "  ${BLUE}Remote outgoing:${NC}     ${remote_out:-?} files"
    else
        echo -e "  ${RED}SSH connectivity:${NC}    ✗ FAILED (is SSH key set up?)"
    fi
    echo ""
}

setup_ssh_keys() {
    echo -e "${BOLD}═══ SSH Key Setup ═══${NC}"
    echo ""

    # Check if we already have a key
    if [[ ! -f "${HOME}/.ssh/id_ed25519" ]] && [[ ! -f "${HOME}/.ssh/id_rsa" ]]; then
        echo -e "${YELLOW}No SSH key found. Generating one...${NC}"
        ssh-keygen -t ed25519 -f "${HOME}/.ssh/id_ed25519" -N "" -C "$(whoami)@$(hostname)"
        echo -e "${GREEN}Key generated.${NC}"
    else
        echo -e "${GREEN}SSH key already exists.${NC}"
    fi

    echo ""
    echo -e "Now copying key to ${REMOTE_USER}@${REMOTE_HOST}..."
    echo -e "${YELLOW}You will be asked for the remote password ONE TIME:${NC}"
    ssh-copy-id -i "${HOME}/.ssh/id_ed25519.pub" "${REMOTE_USER}@${REMOTE_HOST}" 2>/dev/null \
        || ssh-copy-id "${REMOTE_USER}@${REMOTE_HOST}"

    echo ""
    echo -e "Testing connection..."
    if check_ssh_connectivity; then
        echo -e "${GREEN}✓ SSH key authentication works!${NC}"
    else
        echo -e "${RED}✗ SSH key authentication failed. Please check manually.${NC}"
    fi
}

run_once() {
    if ! check_ssh_connectivity; then
        log "ERROR" "Cannot reach ${REMOTE_USER}@${REMOTE_HOST} via SSH. Run: $0 --setup-keys"
        return 1
    fi
    send_migrants
}

run_daemon() {
    log "INFO" "Starting distribute_migrate daemon (poll every ${POLL_INTERVAL}s)"
    log "INFO" "  Local:  $(hostname)"
    log "INFO" "  Remote: ${REMOTE_USER}@${REMOTE_HOST}"

    # Graceful shutdown
    trap 'log "INFO" "Daemon shutting down..."; exit 0' SIGINT SIGTERM

    while true; do
        if check_ssh_connectivity; then
            send_migrants
        else
            log "WARN" "Remote unreachable, will retry in ${POLL_INTERVAL}s"
        fi
        sleep "${POLL_INTERVAL}"
    done
}

# ── Main ──

case "${1:-}" in
    --status)
        receive_status
        ;;
    --setup-keys)
        setup_ssh_keys
        ;;
    --daemon)
        if [[ "${2:-}" == "--interval" ]] && [[ -n "${3:-}" ]]; then
            POLL_INTERVAL="$3"
        fi
        run_daemon
        ;;
    --help|-h)
        head -35 "$0" | tail -30
        ;;
    *)
        run_once
        ;;
esac
