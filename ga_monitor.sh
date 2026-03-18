#!/usr/bin/env bash
# ============================================================================
# GA Evolution Monitor v2 — Live Dashboard for All Running Experiments
# ============================================================================
# Auto-discovers experiments from log files and running processes.
# Shows: generation, fitness, diversity, SAFE/OVERFIT results, elapsed time.
#
# Usage:
#   ./ga_monitor.sh                     # auto-detect all running experiments
#   ./ga_monitor.sh wave3               # monitor specific wave only
#   ./ga_monitor.sh --all               # show all waves (running + completed)
#   ./ga_monitor.sh --once              # print once and exit
#   ./ga_monitor.sh --interval 10       # refresh every 10 seconds
# ============================================================================

set -uo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="${REPO_DIR}/genetic_algorithm/logs"
WAVE_FILTER=""
INTERVAL=5
ONCE=false
SHOW_ALL=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --once)     ONCE=true ;;
        --all)      SHOW_ALL=true ;;
        --interval) INTERVAL="${2:-5}"; shift ;;
        --help|-h)
            echo "Usage: $0 [wave_name] [--all] [--once] [--interval N]"
            echo ""
            echo "  wave_name       Filter to specific wave (e.g. wave3)"
            echo "  --all           Show completed experiments too"
            echo "  --once          Print status once and exit"
            echo "  --interval N    Refresh interval in seconds (default: 5)"
            exit 0
            ;;
        -*)         echo "Unknown option: $1"; exit 1 ;;
        *)          WAVE_FILTER="$1" ;;
    esac
    shift
done

# ── Pad a plain string to N chars ──
spad() { printf "%-${2}s" "$1"; }

# ── Pad a string that may contain ANSI codes to visual width N ──
apad() {
    local str="$1" width="$2"
    local plain
    plain=$(printf '%b' "$str" | sed $'s/\x1b\\[[0-9;]*m//g')
    local vlen=${#plain}
    local need=$(( width - vlen ))
    if [[ $need -gt 0 ]]; then
        printf '%b%*s' "$str" "$need" ""
    else
        printf '%b' "$str"
    fi
}

# ── Colours ──
RED='\e[31m'
GREEN='\e[32m'
YELLOW='\e[33m'
BLUE='\e[34m'
CYAN='\e[36m'
BOLD='\e[1m'
DIM='\e[2m'
NC='\e[0m'

# ── Discover running GA processes ──
declare -A RUNNING_PIDS=()

discover_running() {
    RUNNING_PIDS=()
    while IFS= read -r line; do
        [[ -z "$line" ]] && continue
        local pid cfg
        pid=$(echo "$line" | awk '{print $2}')
        cfg=$(echo "$line" | grep -oP '(?<=--config )\S+' || true)
        [[ -n "$cfg" && -n "$pid" ]] && RUNNING_PIDS["$cfg"]="$pid"
    done < <(ps aux 2>/dev/null | grep '[r]un_ga.py' || true)
}

# ── Discover log files ──
discover_logs() {
    local pat
    if [[ -n "$WAVE_FILTER" ]]; then
        pat="${LOG_DIR}/${WAVE_FILTER}_*.log"
    else
        pat="${LOG_DIR}/wave*_*.log"
    fi
    # shellcheck disable=SC2086
    ls -1 $pat 2>/dev/null | sort
}

# ── Parse wave + experiment name from log path ──
parse_log_name() {
    local bn
    bn=$(basename "$1" .log)
    local wave="${bn%%_*}"
    local exp="${bn#${wave}_}"
    echo "${wave}|${exp}"
}

# ── Find PID for an experiment ──
find_pid_for() {
    local exp="$1" wave="$2"
    for cfg in "${!RUNNING_PIDS[@]}"; do
        [[ "$cfg" == *"${exp}"* ]] && echo "${RUNNING_PIDS[$cfg]}" && return 0
    done
    local pf
    pf=$(ls -t "${LOG_DIR}/${wave}_pids_"*.txt 2>/dev/null | head -1)
    if [[ -n "$pf" && -f "$pf" ]]; then
        local p
        p=$(grep "$exp" "$pf" 2>/dev/null | awk '{print $1}')
        [[ -n "$p" ]] && echo "$p" && return 0
    fi
    return 1
}

# ── Extract all metrics from one log file ──
get_metrics() {
    local log="$1"
    [[ ! -f "$log" || ! -s "$log" ]] && echo "—|—|—|—|0|—|—|NO_LOG" && return

    local buf
    buf=$(tail -300 "$log" 2>/dev/null)

    # ── Generation ──
    local gen="init"
    local gl
    gl=$(echo "$buf" | grep -oP 'GENERATION \d+/\d+' | tail -1 || true)
    [[ -n "$gl" ]] && gen="${gl#GENERATION }"

    # ── Eval sub-progress ──
    local evp
    evp=$(echo "$buf" | grep -oP '\[EVAL\] Progress: \K\d+/\d+' | tail -1 || true)

    # ── Best fitness ──
    local best="—"
    local v
    v=$(echo "$buf" | grep -oP '\[STATS\] Best: \K[0-9.]+' | tail -1 || true)
    if [[ -n "$v" ]]; then
        best="$v"
    else
        v=$(echo "$buf" | grep -oP '\[SUMMARY\].*master=\K[0-9.]+' | tail -1 || true)
        if [[ -n "$v" ]]; then
            best="$v"
        else
            v=$(echo "$buf" | grep -oP '\[NEW BEST\].*fitness.?\K[0-9.]+' | tail -1 || true)
            [[ -n "$v" ]] && best="$v"
        fi
    fi

    # ── Average fitness ──
    local avg="—"
    v=$(echo "$buf" | grep -oP '\[STATS\].*Avg: \K[0-9.]+' | tail -1 || true)
    [[ -n "$v" ]] && avg="$v"

    # ── Diversity ──
    local div="—"
    v=$(echo "$buf" | grep -oP 'Diversity: \K[0-9.]+' | tail -1 || true)
    [[ -n "$v" ]] && div="$v"

    # ── Errors (only real ERROR/CRITICAL/Traceback) ──
    local errs
    errs=$(grep -cP '- (ERROR|CRITICAL) -|^Traceback' "$log" 2>/dev/null || echo 0)

    # ── Elapsed ──
    local elapsed="—"
    local t0
    t0=$(head -1 "$log" 2>/dev/null | grep -oP '^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}' || true)
    if [[ -n "$t0" ]]; then
        local e0 e1
        e0=$(date -d "$t0" +%s 2>/dev/null || echo 0)
        if [[ "$e0" -gt 0 ]]; then
            if grep -qE 'GA RUN COMPLETE|EVOLUTION COMPLETE' "$log" 2>/dev/null; then
                local tN
                tN=$(tac "$log" 2>/dev/null | grep -oP -m1 '^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}' || true)
                e1=$(date -d "${tN:-now}" +%s 2>/dev/null || date +%s)
            else
                e1=$(date +%s)
            fi
            local ds=$(( e1 - e0 ))
            [[ $ds -lt 0 ]] && ds=0
            local m=$(( ds / 60 )) s=$(( ds % 60 ))
            if [[ $m -ge 60 ]]; then
                elapsed="$(( m / 60 ))h$(( m % 60 ))m"
            else
                elapsed="${m}m${s}s"
            fi
        fi
    fi

    # ── Completion results ──
    local results="—"
    local _log_for_results="$log"
    if grep -qE 'GA RUN COMPLETE|EVOLUTION COMPLETE' "$log" 2>/dev/null; then
        # Try the experiment log first, fall back to queue log for post-evolution stats
        if ! grep -q 'SAFE:' "$log" 2>/dev/null; then
            # Post-evolution output may be in the corresponding queue_*.log
            local _bn _exp_name _qlog
            _bn=$(basename "$log" .log)
            _exp_name="${_bn#*_}"  # strip wave prefix
            _qlog=$(ls -t "${LOG_DIR}"/queue_*"${_exp_name}"*.log 2>/dev/null | head -1)
            [[ -n "$_qlog" && -f "$_qlog" ]] && _log_for_results="$_qlog"
        fi
        local sa wa ov sc
        sa=$(grep -oP 'SAFE: \K\d+' "$_log_for_results" 2>/dev/null | tail -1)
        wa=$(grep -oP 'WARNING: \K\d+' "$_log_for_results" 2>/dev/null | tail -1)
        ov=$(grep -oP 'OVERFIT: \K\d+' "$_log_for_results" 2>/dev/null | tail -1)
        sc=$(grep -oP 'Avg composite score: \K[0-9.]+' "$_log_for_results" 2>/dev/null | tail -1)
        if [[ -n "$sa" || -n "$wa" || -n "$ov" ]]; then
            results="${sa:-0}S/${wa:-0}W/${ov:-0}O sc=${sc:-?}"
        fi
    fi

    # ── Status ──
    local status="RUNNING"
    if grep -qE 'GA RUN COMPLETE|EVOLUTION COMPLETE' "$log" 2>/dev/null; then
        # Handle restarted experiments: if log has a new GA start AFTER the
        # last completion marker, the experiment was re-launched and is running.
        local last_complete last_start
        last_complete=$(grep -nE 'GA RUN COMPLETE|EVOLUTION COMPLETE' "$log" 2>/dev/null | tail -1 | cut -d: -f1)
        last_start=$(grep -n 'GENETIC ALGORITHM STARTING' "$log" 2>/dev/null | tail -1 | cut -d: -f1)
        if [[ -n "$last_start" && -n "$last_complete" && "$last_start" -gt "$last_complete" ]]; then
            status="RUNNING"
        else
            status="DONE"
        fi
    elif tail -30 "$log" 2>/dev/null | grep -qP 'Traceback|KeyboardInterrupt|FATAL'; then
        status="CRASHED"
    fi

    # Sub-progress in gen display
    if [[ "$status" == "RUNNING" && -n "$evp" && "$gen" != "init" ]]; then
        gen="${gen} [${evp}]"
    fi

    echo "${gen}|${best}|${avg}|${div}|${errs}|${elapsed}|${results}|${status}"
}

# ── Main dashboard ──
print_dashboard() {
    [[ "$ONCE" == false ]] && printf '\033[2J\033[H'

    discover_running

    local -a logs=()
    while IFS= read -r f; do
        [[ -n "$f" ]] && logs+=("$f")
    done < <(discover_logs)

    local n_run=0 n_done=0 n_crash=0 n_hidden=0

    # ── Title bar ──
    local title="GA EVOLUTION MONITOR"
    [[ -n "$WAVE_FILTER" ]] && title="GA MONITOR — ${WAVE_FILTER}"

    echo ""
    echo -e "  ${CYAN}╔════════════════════════════════════════════════════════════════════════════════════════════════════════╗${NC}"
    echo -ne "  ${CYAN}║${NC} ${BOLD}"; apad "$title" 80; echo -e "${NC}${DIM}$(date '+%H:%M:%S')${NC} ${CYAN}║${NC}"
    echo -e "  ${CYAN}╚════════════════════════════════════════════════════════════════════════════════════════════════════════╝${NC}"
    echo ""

    if [[ ${#logs[@]} -eq 0 ]]; then
        echo -e "  ${YELLOW}No experiment logs found.${NC}"
        [[ -n "$WAVE_FILTER" ]] && echo -e "  ${DIM}Searched: ${LOG_DIR}/${WAVE_FILTER}_*.log${NC}"
        echo -e "  ${DIM}Tip: $0 --all  |  $0 wave3${NC}"
        echo ""; return
    fi

    # ── Column header ──
    echo -ne "  "
    apad "${BOLD}WAVE${NC}"       6; echo -n " "
    apad "${BOLD}EXPERIMENT${NC}" 29; echo -n " "
    apad "${BOLD}GENERATION${NC}" 17; echo -n " "
    apad "${BOLD}BEST${NC}"       9; echo -n " "
    apad "${BOLD}AVG${NC}"        9; echo -n " "
    apad "${BOLD}DIV${NC}"        7; echo -n " "
    apad "${BOLD}ERR${NC}"        5; echo -n " "
    apad "${BOLD}TIME${NC}"       9; echo -n " "
    apad "${BOLD}RESULT${NC}"     20; echo -n " "
    echo -e "${BOLD}STATUS${NC}"
    echo -e "  ${DIM}───── ───────────────────────────── ───────────────── ───────── ───────── ─────── ───── ───────── ──────────────────── ──────────${NC}"

    local prev_wave=""

    for lf in "${logs[@]}"; do
        local parsed wave exp
        parsed=$(parse_log_name "$lf")
        wave="${parsed%%|*}"
        exp="${parsed#*|}"

        local raw
        raw=$(get_metrics "$lf")
        IFS='|' read -r gen best avg div errs elapsed results status <<< "$raw"

        # Verify RUNNING with actual PID
        if [[ "$status" == "RUNNING" ]]; then
            local pid
            pid=$(find_pid_for "$exp" "$wave" 2>/dev/null || true)
            if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
                : # confirmed
            else
                local age
                age=$(( $(date +%s) - $(stat -c%Y "$lf" 2>/dev/null || echo 0) ))
                [[ $age -gt 300 ]] && status="STALE"
            fi
        fi

        # Visibility filter
        if [[ "$SHOW_ALL" == false && "$status" == "DONE" ]]; then
            if [[ -n "$WAVE_FILTER" ]]; then
                : # show DONE for requested wave
            else
                ((n_done++)); ((n_hidden++)); continue   # hide by default
            fi
        fi

        case "$status" in
            DONE)    ((n_done++)) ;;
            RUNNING) ((n_run++)) ;;
            CRASHED) ((n_crash++)) ;;
        esac

        # Wave separator
        if [[ "$wave" != "$prev_wave" && -n "$prev_wave" ]]; then
            echo -e "  ${DIM}───── ───────────────────────────── ───────────────── ───────── ───────── ─────── ───── ───────── ──────────────────── ──────────${NC}"
        fi
        prev_wave="$wave"

        # ── Colorize ──
        local c_best c_err c_results c_status

        if [[ "$best" == "—" ]]; then
            c_best="${DIM}—${NC}"
        elif awk "BEGIN{exit(!($best >= 0.30))}" 2>/dev/null; then
            c_best="${GREEN}${best}${NC}"
        elif awk "BEGIN{exit(!($best >= 0.15))}" 2>/dev/null; then
            c_best="${YELLOW}${best}${NC}"
        else
            c_best="${RED}${best}${NC}"
        fi

        if [[ "${errs:-0}" -gt 0 ]] 2>/dev/null; then
            c_err="${RED}${errs}${NC}"
        else
            c_err="${DIM}0${NC}"
        fi

        case "$status" in
            RUNNING) c_status="${GREEN}● RUNNING${NC}" ;;
            DONE)    c_status="${GREEN}✓ DONE${NC}" ;;
            CRASHED) c_status="${RED}✗ CRASH${NC}" ;;
            STALE)   c_status="${YELLOW}? STALE${NC}" ;;
            NO_LOG)  c_status="${DIM}— NOLOG${NC}" ;;
            *)       c_status="${DIM}${status}${NC}" ;;
        esac

        if [[ "$results" == "—" ]]; then
            c_results="${DIM}—${NC}"
        else
            local sc_val
            sc_val=$(echo "$results" | grep -oP 'sc=\K[0-9.]+' || true)
            if [[ -n "$sc_val" ]]; then
                if awk "BEGIN{exit(!($sc_val < 0.15))}" 2>/dev/null; then
                    c_results="${GREEN}${results}${NC}"
                elif awk "BEGIN{exit(!($sc_val < 0.25))}" 2>/dev/null; then
                    c_results="${YELLOW}${results}${NC}"
                else
                    c_results="${RED}${results}${NC}"
                fi
            else
                c_results="$results"
            fi
        fi

        # ── Print row ──
        echo -ne "  "
        apad "$wave"      5; echo -n "  "
        apad "$exp"       28; echo -n "  "
        apad "$gen"       16; echo -n "  "
        apad "$c_best"     8; echo -n "  "
        apad "${avg}"      8; echo -n "  "
        apad "${div}"      6; echo -n "  "
        apad "$c_err"      4; echo -n "  "
        apad "${elapsed}"  8; echo -n "  "
        apad "$c_results" 19; echo -n "  "
        echo -e "$c_status"
    done

    echo ""

    # ── System bar ──
    local rss free load cores
    rss=$(ps aux 2>/dev/null | grep '[r]un_ga.py' | awk '{s+=$6} END{printf "%.0f",s/1024}' || echo 0)
    free=$(awk '/MemAvailable/{printf "%.0f",$2/1024}' /proc/meminfo 2>/dev/null || echo "?")
    load=$(cut -d' ' -f1 /proc/loadavg 2>/dev/null || echo "?")
    cores=$(nproc 2>/dev/null || echo "?")

    echo -ne "  ${BLUE}System${NC}: "
    [[ $n_run -gt 0 ]] && echo -ne "${GREEN}${n_run} running${NC} · "
    [[ $n_done -gt 0 ]] && echo -ne "${DIM}${n_done} done${NC} · "
    [[ $n_crash -gt 0 ]] && echo -ne "${RED}${n_crash} crashed${NC} · "
    echo -e "RSS ${rss}MB · Free ${free}MB · Load ${load}/${cores}"

    [[ $n_hidden -gt 0 ]] && echo -e "  ${DIM}${n_hidden} completed experiments hidden (use --all to show)${NC}"

    if [[ "$ONCE" == false ]]; then
        echo -e "  ${DIM}Refreshing ${INTERVAL}s · Ctrl+C to stop · S=SAFE W=WARNING O=OVERFIT · lower sc=better${NC}"
    fi
    echo ""
}

# ── Entrypoint ──
if [[ "$ONCE" == true ]]; then
    print_dashboard
else
    trap 'echo ""; echo -e "  ${DIM}Monitor stopped.${NC}"; exit 0' INT TERM
    while true; do
        print_dashboard
        sleep "$INTERVAL"
    done
fi
