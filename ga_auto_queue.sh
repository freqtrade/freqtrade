#!/usr/bin/env bash
# ============================================================================
# GA Auto-Queue Daemon — Keep the server busy with GA experiments 24/7
# ============================================================================
# Monitors running GA processes and launches queued experiments to maintain
# a target number of concurrent runs.
#
# Queue directory:  genetic_algorithm/config/queue/
#   - YAML configs named with priority prefix: 01_E19_name.yaml
#   - Lower number = higher priority (launched first)
#
# Done directory:   genetic_algorithm/config/done/
#   - Completed configs moved here with timestamp
#
# Usage:
#   ./ga_auto_queue.sh                   # run in foreground
#   ./ga_auto_queue.sh --status          # show running/queued counts
#   ./ga_auto_queue.sh --max 6           # override max concurrent (default: 5)
#   nohup ./ga_auto_queue.sh &           # run as background daemon
#   ./ga_auto_queue.sh --stop            # stop daemon gracefully
#
# Monitoring:
#   tail -f genetic_algorithm/logs/auto_queue.log
#   ./ga_monitor.sh                      # live experiment dashboard
# ============================================================================

set -uo pipefail
set +m  # Disable job control

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="${REPO_DIR}/.venv"
QUEUE_DIR="${REPO_DIR}/genetic_algorithm/config/queue"
DONE_DIR="${REPO_DIR}/genetic_algorithm/config/done"
LOG_DIR="${REPO_DIR}/genetic_algorithm/logs"
OUTPUT_BASE="${REPO_DIR}/genetic_algorithm/output/exploration/wave14"
LOG_FILE="${LOG_DIR}/auto_queue.log"

# Source .env file for API keys (GROQ_API_KEY needed for LLM experiments)
if [[ -f "${REPO_DIR}/.env" ]]; then
    set -a
    source "${REPO_DIR}/.env"
    set +a
fi
PID_FILE="${LOG_DIR}/auto_queue_daemon.pid"
TRACKED_PIDS_FILE="${LOG_DIR}/auto_queue_tracked.txt"

MAX_CONCURRENT=5
POLL_INTERVAL=30

# ── Colours ──
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# ── Parse arguments ──
MODE="run"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --status) MODE="status" ;;
        --stop) MODE="stop" ;;
        --max)
            shift
            MAX_CONCURRENT="${1:-5}"
            ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
    shift
done

# ── Logging ──
log() {
    local level="$1"
    shift
    local msg="$*"
    local ts
    ts=$(date '+%Y-%m-%d %H:%M:%S')
    echo "${ts} [${level}] ${msg}" >> "$LOG_FILE"
    if [[ "$level" == "ERROR" ]]; then
        echo -e "${RED}[${ts}] ${msg}${NC}" >&2
    else
        echo -e "${BLUE}[${ts}]${NC} ${msg}"
    fi
}

# ── Status command ──
if [[ "$MODE" == "status" ]]; then
    running=$(pgrep -cf "run_ga\.py" 2>/dev/null) || running=0
    queued=$(find "$QUEUE_DIR" -maxdepth 1 -name '*.yaml' 2>/dev/null | wc -l)
    done_count=$(find "$DONE_DIR" -maxdepth 1 -name '*.yaml' 2>/dev/null | wc -l)

    echo -e "${CYAN}╔════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║   GA Auto-Queue Status             ║${NC}"
    echo -e "${CYAN}╚════════════════════════════════════╝${NC}"
    echo ""
    echo -e "  ${BOLD}Running:${NC}  ${running} / ${MAX_CONCURRENT}"
    echo -e "  ${BOLD}Queued:${NC}   ${queued}"
    echo -e "  ${BOLD}Done:${NC}     ${done_count}"
    echo ""

    if [[ -f "$PID_FILE" ]]; then
        daemon_pid=$(cat "$PID_FILE")
        if kill -0 "$daemon_pid" 2>/dev/null; then
            echo -e "  ${GREEN}Daemon running (PID ${daemon_pid})${NC}"
        else
            echo -e "  ${YELLOW}Daemon not running (stale PID file)${NC}"
        fi
    else
        echo -e "  ${YELLOW}Daemon not running${NC}"
    fi

    if [[ $queued -gt 0 ]]; then
        echo ""
        echo -e "  ${BOLD}Queue:${NC}"
        find "$QUEUE_DIR" -maxdepth 1 -name '*.yaml' -printf '%f\n' 2>/dev/null | sort | while read -r f; do
            echo "    - ${f}"
        done
    fi

    if [[ -f "$TRACKED_PIDS_FILE" ]]; then
        echo ""
        echo -e "  ${BOLD}Tracked experiments:${NC}"
        while IFS=' ' read -r pid name; do
            [[ "$pid" == "#"* ]] && continue
            [[ -z "$pid" ]] && continue
            if kill -0 "$pid" 2>/dev/null; then
                echo -e "    ${GREEN}●${NC} ${name} (PID ${pid})"
            else
                echo -e "    ${RED}○${NC} ${name} (PID ${pid}, exited)"
            fi
        done < "$TRACKED_PIDS_FILE"
    fi
    exit 0
fi

# ── Stop command ──
if [[ "$MODE" == "stop" ]]; then
    if [[ -f "$PID_FILE" ]]; then
        daemon_pid=$(cat "$PID_FILE")
        if kill -0 "$daemon_pid" 2>/dev/null; then
            echo -e "${YELLOW}Sending SIGTERM to daemon (PID ${daemon_pid})...${NC}"
            kill -TERM "$daemon_pid"
            echo -e "${GREEN}Daemon stopped. Running experiments will continue to completion.${NC}"
        else
            echo -e "${YELLOW}Daemon not running (stale PID file). Cleaning up.${NC}"
            rm -f "$PID_FILE"
        fi
    else
        echo -e "${YELLOW}No daemon PID file found.${NC}"
    fi
    exit 0
fi

# ── Pre-flight checks ──
if [[ ! -f "${VENV_DIR}/bin/activate" ]]; then
    echo -e "${RED}ERROR: Virtual environment not found at ${VENV_DIR}${NC}"
    exit 1
fi

mkdir -p "$QUEUE_DIR" "$DONE_DIR" "$LOG_DIR" "$OUTPUT_BASE"

# Check for existing daemon
if [[ -f "$PID_FILE" ]]; then
    existing_pid=$(cat "$PID_FILE")
    if kill -0 "$existing_pid" 2>/dev/null; then
        echo -e "${RED}ERROR: Daemon already running (PID ${existing_pid}). Use --stop first.${NC}"
        exit 1
    else
        rm -f "$PID_FILE"
    fi
fi

# ── Activate venv ──
source "${VENV_DIR}/bin/activate"

# ── Track our PID ──
echo $$ > "$PID_FILE"

# ── Initialize tracked PIDs file ──
if [[ ! -f "$TRACKED_PIDS_FILE" ]]; then
    echo "# Auto-queue tracked PIDs" > "$TRACKED_PIDS_FILE"
fi

# ── State ──
declare -A TRACKED=()       # pid -> config_basename
declare -A TRACKED_LOG=()   # pid -> log_path

# Load existing tracked PIDs
while IFS=' ' read -r pid name; do
    [[ "$pid" == "#"* ]] && continue
    [[ -z "$pid" ]] && continue
    if kill -0 "$pid" 2>/dev/null; then
        TRACKED["$pid"]="$name"
    fi
done < "$TRACKED_PIDS_FILE" 2>/dev/null

# ── Functions ──

count_running_ga() {
    local count
    count=$(pgrep -cf "run_ga\.py" 2>/dev/null) || count=0
    echo "$count"
}

get_next_queued() {
    # Returns the lowest-priority-numbered YAML file from queue dir
    find "$QUEUE_DIR" -maxdepth 1 -name '*.yaml' -printf '%f\n' 2>/dev/null | sort | head -1
}

launch_experiment() {
    local config_basename="$1"
    local config_path="${QUEUE_DIR}/${config_basename}"
    local exp_name
    exp_name=$(basename "$config_basename" .yaml)
    local exp_log="${LOG_DIR}/queue_${exp_name}.log"
    local exp_output="${OUTPUT_BASE}/${exp_name}"

    # Check config still exists (may have been moved by concurrent check)
    if [[ ! -f "$config_path" ]]; then
        log "WARNING" "Config ${config_basename} no longer in queue — skipping"
        return 1
    fi

    mkdir -p "$exp_output"

    # Validate YAML before launching
    if ! python -c "import yaml; yaml.safe_load(open('${config_path}'))" 2>/dev/null; then
        log "ERROR" "YAML parse error in ${config_basename} — skipping!"
        mv "$config_path" "${DONE_DIR}/${exp_name}_ERROR_$(date '+%Y%m%d_%H%M%S').yaml"
        return 1
    fi

    # Move config to done directory BEFORE launching (prevents duplicate launches)
    local done_path="${DONE_DIR}/${config_basename}"
    mv "$config_path" "$done_path"

    # Launch with setsid for process isolation, using the done-dir copy
    export GA_OUTPUT_DIR="${exp_output}"

    setsid python genetic_algorithm/run_ga.py \
        --config "$done_path" \
        --no-monitor --yes \
        > "${exp_log}" 2>&1 &

    local pid=$!
    TRACKED["$pid"]="$config_basename"
    TRACKED_LOG["$pid"]="$exp_log"

    # Record in tracked file
    echo "${pid} ${exp_name}" >> "$TRACKED_PIDS_FILE"

    log "INFO" "LAUNCHED: ${exp_name} → PID ${pid} (log: ${exp_log})"
    return 0
}

check_completed() {
    local any_completed=false
    for pid in "${!TRACKED[@]}"; do
        if ! kill -0 "$pid" 2>/dev/null; then
            local config_basename="${TRACKED[$pid]}"
            local exp_name
            exp_name=$(basename "$config_basename" .yaml)
            local config_path="${QUEUE_DIR}/${config_basename}"

            # Get exit code
            wait "$pid" 2>/dev/null
            local exit_code=$?

            if [[ $exit_code -eq 0 ]]; then
                log "INFO" "COMPLETED: ${exp_name} (exit 0)"
            else
                log "WARNING" "COMPLETED: ${exp_name} (exit ${exit_code})"
            fi

            # Config already in done dir (moved at launch time)

            unset "TRACKED[$pid]"
            unset "TRACKED_LOG[$pid]"
            any_completed=true
        fi
    done

    if [[ "$any_completed" == true ]]; then
        # Rebuild tracked PIDs file with only running processes
        {
            echo "# Auto-queue tracked PIDs (updated $(date))"
            for pid in "${!TRACKED[@]}"; do
                local exp_name
                exp_name=$(basename "${TRACKED[$pid]}" .yaml)
                echo "${pid} ${exp_name}"
            done
        } > "$TRACKED_PIDS_FILE"
    fi
}

# ── Graceful shutdown ──
cleanup() {
    echo ""
    log "INFO" "Received shutdown signal — daemon stopping"
    log "INFO" "Running experiments will continue to completion"
    rm -f "$PID_FILE"
    exit 0
}
trap cleanup SIGINT SIGTERM

# ── Header ──
echo ""
echo -e "${CYAN}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║   GA Auto-Queue Daemon                                       ║${NC}"
echo -e "${CYAN}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "  ${BLUE}Max concurrent:${NC}  ${MAX_CONCURRENT}"
echo -e "  ${BLUE}Poll interval:${NC}   ${POLL_INTERVAL}s"
echo -e "  ${BLUE}Queue dir:${NC}       ${QUEUE_DIR}"
echo -e "  ${BLUE}Done dir:${NC}        ${DONE_DIR}"
echo -e "  ${BLUE}Log file:${NC}        ${LOG_FILE}"
echo -e "  ${BLUE}Daemon PID:${NC}      $$"
echo ""

log "INFO" "Auto-queue daemon started (max=${MAX_CONCURRENT}, poll=${POLL_INTERVAL}s)"

# Count existing GA processes from previous waves
existing_running=$(count_running_ga)
if [[ $existing_running -gt 0 ]]; then
    log "INFO" "Detected ${existing_running} existing GA process(es) from previous runs"
fi

# ── Main loop ──
while true; do
    # Check for completed experiments
    check_completed

    # Count all running GA processes (including from other waves)
    current_running=$(count_running_ga)
    queued_count=$(find "$QUEUE_DIR" -maxdepth 1 -name '*.yaml' 2>/dev/null | wc -l)

    # Launch new experiments if we have capacity
    slots_available=$((MAX_CONCURRENT - current_running))

    if [[ $slots_available -gt 0 && $queued_count -gt 0 ]]; then
        to_launch=$((slots_available < queued_count ? slots_available : queued_count))
        for ((i=0; i<to_launch; i++)); do
            next_config=$(get_next_queued)
            if [[ -n "$next_config" ]]; then
                launch_experiment "$next_config" || true
                sleep 2  # Brief pause between launches
            fi
        done
    fi

    # Check if we're done (nothing running, nothing queued)
    if [[ $queued_count -eq 0 && ${#TRACKED[@]} -eq 0 ]]; then
        # Double-check after a brief wait
        sleep 5
        queued_count=$(find "$QUEUE_DIR" -maxdepth 1 -name '*.yaml' 2>/dev/null | wc -l)
        if [[ $queued_count -eq 0 && ${#TRACKED[@]} -eq 0 ]]; then
            log "INFO" "All experiments complete — no more configs in queue. Daemon exiting."
            rm -f "$PID_FILE"
            echo -e "${GREEN}All experiments complete!${NC}"
            exit 0
        fi
    fi

    sleep "$POLL_INTERVAL"
done
