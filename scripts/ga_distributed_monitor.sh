#!/usr/bin/env bash
# ============================================================================
# Distributed GA Monitor — Server-side dashboard
# ============================================================================
# Shows: local processes, generation progress, fitness, migration status
#
# Usage:
#   ./scripts/ga_distributed_monitor.sh             # one-shot dashboard
#   ./scripts/ga_distributed_monitor.sh --watch      # auto-refresh every 15s
#   ./scripts/ga_distributed_monitor.sh --watch 30   # custom interval
#   ./scripts/ga_distributed_monitor.sh --compact    # less verbose
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="$REPO_DIR/genetic_algorithm/logs"
INCOMING_DIR="$REPO_DIR/genetic_algorithm/data/incoming_migrants"
OUTGOING_DIR="$REPO_DIR/genetic_algorithm/data/outgoing_migrants"

WATCH=false
COMPACT=false
INTERVAL=15

# Parse args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --watch|-w)
            WATCH=true
            if [[ ${2:-} =~ ^[0-9]+$ ]]; then
                INTERVAL=$2; shift
            fi
            shift ;;
        --compact|-c)
            COMPACT=true; shift ;;
        *) shift ;;
    esac
done

# ── Colors ──
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
WHITE='\033[1;37m'
GRAY='\033[0;90m'
NC='\033[0m'

header() {
    local line
    line=$(printf '=%.0s' {1..70})
    echo ""
    echo -e "${CYAN}${line}${NC}"
    echo -e "${WHITE}  $1${NC}"
    echo -e "${CYAN}${line}${NC}"
}

section() {
    echo ""
    echo -e "${YELLOW}--- $1 ---${NC}"
}

progress_bar() {
    local current=$1 total=$2
    local pct=$(( current * 100 / total ))
    local filled=$(( pct / 5 ))
    local empty=$(( 20 - filled ))
    local bar="["
    for ((i=0; i<filled; i++)); do bar+="#"; done
    for ((i=0; i<empty; i++)); do bar+="-"; done
    bar+="]"
    echo "$bar ${pct}%"
}

get_log_gen() {
    local logfile="$1"
    local max_gen=0
    if [[ -f "$logfile" ]]; then
        while IFS= read -r line; do
            if [[ "$line" =~ Creating\ generation\ ([0-9]+) ]]; then
                local g="${BASH_REMATCH[1]}"
                if (( g > max_gen )); then max_gen=$g; fi
            fi
        done < <(tail -500 "$logfile" 2>/dev/null | grep "Creating generation" || true)
    fi
    echo "$max_gen"
}

get_log_best_fitness() {
    local logfile="$1"
    if [[ -f "$logfile" ]]; then
        tail -500 "$logfile" 2>/dev/null | grep "NEW BEST" | tail -1 | grep -oP 'fitness=\K[\d.]+' || echo ""
    fi
}

show_dashboard() {
    clear 2>/dev/null || true
    local now
    now=$(date '+%Y-%m-%d %H:%M:%S')
    local hostname_str
    hostname_str=$(hostname)

    echo ""
    echo -e "  ${MAGENTA}DISTRIBUTED GA MONITOR${NC}  |  ${GRAY}${now}${NC}"
    echo -e "  ${GRAY}$(printf '=%.0s' {1..66})${NC}"

    # ── LOCAL SERVER ──
    header "SERVER ($hostname_str  —  $(hostname -I 2>/dev/null | awk '{print $1}'))"

    # System resources
    local load_avg mem_info
    load_avg=$(uptime | grep -oP 'load average: \K.*' || echo "?")
    mem_info=$(free -m 2>/dev/null | awk '/Mem:/{printf "%s/%sMB (%.0f%%)", $3, $2, $3/$2*100}' || echo "?")
    echo -e "  Load: ${load_avg}  |  Memory: ${mem_info}"

    # Git info
    local git_commit git_branch
    git_branch=$(cd "$REPO_DIR" && git branch --show-current 2>/dev/null || echo "?")
    git_commit=$(cd "$REPO_DIR" && git log --oneline -1 2>/dev/null || echo "?")
    echo -e "  Branch: ${git_branch}  |  Commit: ${git_commit}"

    # Processes
    section "GA Processes"
    local proc_found=false
    while IFS= read -r line; do
        if [[ -n "$line" ]]; then
            proc_found=true
            local pid cpu mem etime config_name
            pid=$(echo "$line" | awk '{print $1}')
            cpu=$(echo "$line" | awk '{print $2}')
            mem=$(echo "$line" | awk '{print $3}')
            etime=$(echo "$line" | awk '{print $4}')
            config_name=$(echo "$line" | grep -oP '(?<=--config[= ])\S+' | xargs -I{} basename {} .yaml 2>/dev/null || echo "")

            local cpu_color=$GREEN
            echo -e "  PID=${pid} ${cpu_color}CPU=${cpu}%${NC} MEM=${mem}% Uptime=${etime}  ${CYAN}${config_name}${NC}"
        fi
    done < <(ps -u "$(whoami)" -o pid,pcpu,pmem,etime,args --sort=-pcpu 2>/dev/null | grep "run_ga.py" | grep -v grep || true)

    if [[ "$proc_found" == false ]]; then
        echo -e "  ${RED}No GA processes running${NC}"
    fi

    # ── Distributed Server GA ──
    local dist_log="$LOG_DIR/distributed_server.log"
    section "Distributed Server GA"
    if [[ -f "$dist_log" ]]; then
        local stat_info size_kb mod_epoch mod_age
        size_kb=$(( $(stat --format='%s' "$dist_log" 2>/dev/null || echo 0) / 1024 ))
        mod_epoch=$(stat --format='%Y' "$dist_log" 2>/dev/null || echo 0)
        local now_epoch
        now_epoch=$(date +%s)
        mod_age=$(( (now_epoch - mod_epoch) / 60 ))
        echo "  Log: ${size_kb}KB, last updated ${mod_age}m ago"

        local gen
        gen=$(get_log_gen "$dist_log")
        if (( gen > 0 )); then
            local bar
            bar=$(progress_bar "$gen" 20)
            echo -e "  Generation: ${GREEN}${gen}/20  ${bar}${NC}"
        else
            echo -e "  Generation: ${YELLOW}0/20 (initial evaluation in progress)${NC}"
        fi

        local best_fit
        best_fit=$(get_log_best_fitness "$dist_log")
        if [[ -n "$best_fit" ]]; then
            echo -e "  Best fitness: ${GREEN}${best_fit}${NC}"
        fi

        # Island summaries
        if [[ "$COMPACT" == false ]]; then
            local island_data
            island_data=$(tail -500 "$dist_log" 2>/dev/null | grep -oP '\[(island_\S+)\s*\]\s*best=[\d.]+\s+avg=[\d.]+\s+diversity=[\d.]+' | sort -t'[' -k2 | while IFS= read -r il; do
                echo "  $il"
            done)
            if [[ -n "$island_data" ]]; then
                section "Server Islands"
                # Show latest per island
                declare -A islands
                while IFS= read -r il; do
                    local iname
                    iname=$(echo "$il" | grep -oP '\[(island_\S+)' | tr -d '[')
                    islands["$iname"]="$il"
                done < <(tail -500 "$dist_log" 2>/dev/null | grep 'best=.*avg=.*diversity=' || true)
                for key in $(echo "${!islands[@]}" | tr ' ' '\n' | sort); do
                    echo "  ${islands[$key]}"
                done
                unset islands
            fi
        fi

        # Migration events
        local mig_count
        mig_count=$(tail -500 "$dist_log" 2>/dev/null | grep -c "EXT-MIGRATION" || echo 0)
        if (( mig_count > 0 )); then
            echo "  External migrations: ${mig_count} events"
        fi
    else
        echo -e "  ${YELLOW}No log file found${NC}"
    fi

    # ── All GA Runs (generic log scanner) ──
    section "All GA Runs"
    local found_logs=0
    for log_file in "$LOG_DIR"/*.log logs/*.log; do
        [[ -f "$log_file" ]] || continue
        found_logs=1
        local log_name size_kb mod_epoch mod_age gen total_gen best_fit is_complete
        log_name=$(basename "$log_file" .log)
        size_kb=$(( $(stat --format='%s' "$log_file" 2>/dev/null || echo 0) / 1024 ))
        mod_epoch=$(stat --format='%Y' "$log_file" 2>/dev/null || echo 0)
        local now_epoch
        now_epoch=$(date +%s)
        mod_age=$(( (now_epoch - mod_epoch) / 60 ))
        gen=$(grep -oP 'GENERATION \K\d+' "$log_file" 2>/dev/null | tail -1)
        total_gen=$(grep -oP 'GENERATION \d+/\K\d+' "$log_file" 2>/dev/null | tail -1)
        best_fit=$(grep -oP 'fitness=\K[\d.]+' "$log_file" 2>/dev/null | sort -rn | head -1)
        is_complete=$(grep -c 'GA RUN COMPLETE' "$log_file" 2>/dev/null || echo 0)

        printf "  %-40s " "$log_name"
        if (( is_complete > 0 )); then
            echo -e "${GREEN}COMPLETE${NC}  best=${best_fit:-?}  (${size_kb}KB)"
        elif [[ -n "$total_gen" && -n "$gen" && "$gen" -gt 0 ]]; then
            local bar
            bar=$(progress_bar "$gen" "$total_gen")
            echo -e "${GREEN}Gen ${gen}/${total_gen}  ${bar}${NC}  best=${best_fit:-?}  (${mod_age}m ago)"
        else
            echo -e "${YELLOW}Starting...${NC}  (${size_kb}KB, ${mod_age}m ago)"
        fi
    done
    if (( found_logs == 0 )); then
        echo -e "  ${YELLOW}No GA log files found${NC}"
    fi

    # ── Migration Status ──
    section "Migration Status"
    local in_count out_count sent_count pulled_count
    in_count=$(find "$INCOMING_DIR" -maxdepth 1 -name '*.json' 2>/dev/null | wc -l)
    out_count=$(find "$OUTGOING_DIR" -maxdepth 1 -name '*.json' 2>/dev/null | wc -l)
    sent_count=$(find "$OUTGOING_DIR/.sent" -name '*.json' 2>/dev/null | wc -l)
    pulled_count=$(find "$INCOMING_DIR/.pulled_log" -name '*.json' 2>/dev/null | wc -l)
    echo "  Incoming: $in_count  Outgoing: $out_count  Sent: $sent_count  Pulled: $pulled_count"

    local total=$((sent_count + pulled_count))
    echo "  Total strategies exchanged: $total"

    echo ""
    if [[ "$WATCH" == true ]]; then
        echo -e "  ${GRAY}Auto-refreshing every ${INTERVAL}s. Press Ctrl+C to stop.${NC}"
    else
        echo -e "  ${GRAY}Run with --watch for auto-refresh. --compact for less detail.${NC}"
    fi
    echo ""
}

# ── Main ──
if [[ "$WATCH" == true ]]; then
    trap 'echo -e "\nMonitor stopped."; exit 0' INT
    while true; do
        show_dashboard
        sleep "$INTERVAL"
    done
else
    show_dashboard
fi
