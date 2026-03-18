#!/usr/bin/env bash
# ============================================================================
# Parallel Wave Runner — Launch 6 GA Experiments Simultaneously
# ============================================================================
# Launches all experiment configs in a wave directory as background processes.
# Each run is fully isolated (DB, checkpoints, logs, output, HoF).
#
# Usage:
#   ./run_parallel_wave.sh wave1                    # all 6 experiments
#   ./run_parallel_wave.sh wave1 E1 E3 E5           # only specific experiments
#   ./run_parallel_wave.sh wave1 --dry-run           # validate configs only
#
# Monitoring:
#   ./wave_monitor.sh wave1                          # live status dashboard
#
# Analysis:
#   python genetic_algorithm/scripts/wave_comparison.py wave1
#
# Hardware budget: 6 cores total
#   - 6 experiments × 1 worker each = 6 cores (parallel_evaluation: false)
#   - No --dashboard flags (saves ~1.2 GB RAM total)
#   - Expected total RAM: ~6-8 GB (well within 16 GB)
#
# Safety:
#   - Each process writes its own PID to a wave PID file
#   - Ctrl+C / SIGINT sends graceful shutdown to all GA processes
#   - Crashed experiments don't kill the others
#   - Exit codes collected and reported
# ============================================================================

set -uo pipefail

# Disable job control signals propagating to children
set +m

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="${REPO_DIR}/.venv"
WAVE_NAME="${1:-}"
DRY_RUN=false
CLEAN=false
SELECTED_EXPERIMENTS=()

# ── Parse arguments ──
if [[ -z "$WAVE_NAME" ]]; then
    echo "Usage: $0 <wave_name> [E1 E2 ...] [--dry-run]"
    echo "  wave_name: directory under genetic_algorithm/config/exploration/ (e.g., wave1)"
    echo "  E1 E2...: optional experiment filter (runs only matching configs)"
    echo "  --dry-run: validate configs without launching"
    exit 1
fi

shift
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=true ;;
        --clean) CLEAN=true ;;
        E*|e*) SELECTED_EXPERIMENTS+=("$1") ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
    shift
done

# ── Paths ──
CONFIG_DIR="${REPO_DIR}/genetic_algorithm/config/exploration/${WAVE_NAME}"
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
OUTPUT_BASE="${REPO_DIR}/genetic_algorithm/output/exploration/${WAVE_NAME}"
PID_FILE="${REPO_DIR}/genetic_algorithm/logs/${WAVE_NAME}_pids_${TIMESTAMP}.txt"
LOG_DIR="${REPO_DIR}/genetic_algorithm/logs"

# ── Colours ──
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# ── Validation ──
if [[ ! -d "$CONFIG_DIR" ]]; then
    echo -e "${RED}ERROR: Config directory not found: ${CONFIG_DIR}${NC}"
    echo "Available waves:"
    ls -1 "${REPO_DIR}/genetic_algorithm/config/exploration/" 2>/dev/null || echo "  (none)"
    exit 1
fi

if [[ ! -f "${VENV_DIR}/bin/activate" ]]; then
    echo -e "${RED}ERROR: Virtual environment not found at ${VENV_DIR}${NC}"
    exit 1
fi

source "${VENV_DIR}/bin/activate"

# ── Discover configs ──
declare -a CONFIG_FILES=()
declare -a EXPERIMENT_NAMES=()

for config_file in "${CONFIG_DIR}"/*.yaml; do
    [[ ! -f "$config_file" ]] && continue
    exp_name=$(basename "$config_file" .yaml)

    # Filter by selected experiments if specified
    if [[ ${#SELECTED_EXPERIMENTS[@]} -gt 0 ]]; then
        matched=false
        for sel in "${SELECTED_EXPERIMENTS[@]}"; do
            if [[ "$exp_name" == *"${sel}"* || "$exp_name" == *"$(echo "$sel" | tr '[:upper:]' '[:lower:]')"* ]]; then
                matched=true
                break
            fi
        done
        [[ "$matched" == false ]] && continue
    fi

    CONFIG_FILES+=("$config_file")
    EXPERIMENT_NAMES+=("$exp_name")
done

NUM_EXPERIMENTS=${#CONFIG_FILES[@]}

if [[ $NUM_EXPERIMENTS -eq 0 ]]; then
    echo -e "${RED}ERROR: No config files found in ${CONFIG_DIR}${NC}"
    exit 1
fi

# ── Header ──
echo ""
echo -e "${CYAN}╔══════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║   PARALLEL WAVE RUNNER — ${WAVE_NAME} (${NUM_EXPERIMENTS} experiments)                    ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "  ${BLUE}Timestamp:${NC}     ${TIMESTAMP}"
echo -e "  ${BLUE}Config dir:${NC}    ${CONFIG_DIR}"
echo -e "  ${BLUE}Output base:${NC}   ${OUTPUT_BASE}"
echo -e "  ${BLUE}Hardware:${NC}      $(nproc) cores, $(free -h | awk '/Mem:/ {print $2}') RAM"
echo -e "  ${BLUE}Dry run:${NC}       ${DRY_RUN}"
echo ""
echo -e "  ${BOLD}Experiments:${NC}"
for i in "${!EXPERIMENT_NAMES[@]}"; do
    echo -e "    $((i+1)). ${EXPERIMENT_NAMES[$i]}"
done
echo ""

# ── Pre-flight: validate all configs can be loaded ──
echo -e "${BLUE}── Pre-flight: validating configs ──${NC}"
all_valid=true
for i in "${!CONFIG_FILES[@]}"; do
    config="${CONFIG_FILES[$i]}"
    exp_name="${EXPERIMENT_NAMES[$i]}"

    # Quick YAML parse check
    if python -c "import yaml; yaml.safe_load(open('$config'))" 2>/dev/null; then
        echo -e "  ${GREEN}✓${NC} ${exp_name}"
    else
        echo -e "  ${RED}✗${NC} ${exp_name} — YAML parse error!"
        all_valid=false
    fi
done

if [[ "$all_valid" == false ]]; then
    echo -e "\n${RED}Config validation failed. Fix errors before launching.${NC}"
    exit 1
fi

echo -e "  ${GREEN}All configs valid.${NC}"
echo ""

if [[ "$DRY_RUN" == true ]]; then
    echo -e "${YELLOW}── DRY RUN — not launching processes ──${NC}"
    echo ""
    echo "Would launch:"
    for i in "${!EXPERIMENT_NAMES[@]}"; do
        echo "  python genetic_algorithm/run_ga.py --config ${CONFIG_FILES[$i]} --no-monitor --yes"
    done
    echo ""
    echo -e "${GREEN}All configs validated. Ready to launch.${NC}"
    exit 0
fi

# ── Setup directories ──
mkdir -p "${OUTPUT_BASE}" "${LOG_DIR}"

# ── Clean old logs if requested or if restarting ──
if [[ "$CLEAN" == true ]]; then
    echo -e "${YELLOW}── Cleaning old wave logs and outputs ──${NC}"
    for i in "${!EXPERIMENT_NAMES[@]}"; do
        exp_name="${EXPERIMENT_NAMES[$i]}"
        rm -f "${LOG_DIR}/${WAVE_NAME}_${exp_name}.log"
    done
    echo -e "  ${GREEN}Old logs cleaned.${NC}"
    echo ""
fi

# ── Launch all experiments ──
echo -e "${YELLOW}── Launching ${NUM_EXPERIMENTS} experiments in parallel ──${NC}"
echo ""

declare -a PIDS=()
declare -A PID_TO_NAME=()

for i in "${!CONFIG_FILES[@]}"; do
    config="${CONFIG_FILES[$i]}"
    exp_name="${EXPERIMENT_NAMES[$i]}"
    exp_log="${LOG_DIR}/${WAVE_NAME}_${exp_name}.log"
    exp_output="${OUTPUT_BASE}/${exp_name}"

    mkdir -p "${exp_output}"

    # Set output directory for this specific run
    export GA_OUTPUT_DIR="${exp_output}"

    # Launch in a new process group (setsid) so parent signals don't kill children
    setsid python genetic_algorithm/run_ga.py \
        --config "$config" \
        --no-monitor --yes \
        > "${exp_log}" 2>&1 &

    pid=$!
    PIDS+=("$pid")
    PID_TO_NAME["$pid"]="$exp_name"

    echo -e "  ${GREEN}✓${NC} ${exp_name} → PID ${pid} (log: ${exp_log})"
done

# ── Save PID file ──
{
    echo "# Wave: ${WAVE_NAME}"
    echo "# Started: $(date)"
    echo "# Timestamp: ${TIMESTAMP}"
    echo "#"
    for pid in "${PIDS[@]}"; do
        echo "${pid} ${PID_TO_NAME[$pid]}"
    done
} > "${PID_FILE}"

echo ""
echo -e "  ${BLUE}PIDs saved to:${NC} ${PID_FILE}"
echo ""

# ── Status report function (defined before cleanup trap) ──
generate_status_report() {
    local report_file="${OUTPUT_BASE}/wave_report_${TIMESTAMP}.txt"
    {
        echo "=============================================="
        echo "  WAVE REPORT: ${WAVE_NAME}"
        echo "  Timestamp: ${TIMESTAMP}"
        echo "  Completed: $(date)"
        echo "=============================================="
        echo ""

        for pid in "${PIDS[@]}"; do
            local exp_name="${PID_TO_NAME[$pid]}"
            local exit_code="${EXIT_CODES[$pid]:-unknown}"
            local exp_log="${LOG_DIR}/${WAVE_NAME}_${exp_name}.log"
            local status="?"

            if [[ "$exit_code" == "0" ]]; then
                status="OK"
            elif [[ "$exit_code" == "unknown" ]]; then
                status="UNKNOWN"
            else
                status="FAILED (exit ${exit_code})"
            fi

            echo "── ${exp_name}: ${status} ──"

            if [[ -f "$exp_log" ]]; then
                local best_fitness
                best_fitness=$(grep -oP '\[NEW BEST\].*fitness[= ]+\K[0-9.]+' "$exp_log" 2>/dev/null | tail -1)
                local last_gen
                last_gen=$(grep -oP 'GENERATION\s+\K\d+' "$exp_log" 2>/dev/null | tail -1)
                local safe_score
                safe_score=$(grep -oP '\d+/\d+ SAFE' "$exp_log" 2>/dev/null | tail -1)
                local errors
                errors=$(grep -ci 'error\|exception\|traceback' "$exp_log" 2>/dev/null || true)

                echo "  Best fitness:  ${best_fitness:-N/A}"
                echo "  Last gen:      ${last_gen:-N/A}"
                echo "  SAFE score:    ${safe_score:-N/A}"
                echo "  Log errors:    ${errors}"
            fi
            echo ""
        done

        echo "── Next Steps ──"
        echo "  Aggregate: python genetic_algorithm/scripts/wave_comparison.py ${OUTPUT_BASE}"
        echo "  Diff:      python genetic_algorithm/scripts/config_diff.py <config1> <config2>"
    } | tee "$report_file"
}

# ── Trap: graceful shutdown on Ctrl+C ──
cleanup() {
    echo ""
    echo -e "${YELLOW}── Received shutdown signal — stopping all experiments ──${NC}"
    for pid in "${PIDS[@]}"; do
        # Send to the process group (negative PID) via setsid
        if kill -0 "$pid" 2>/dev/null; then
            echo -e "  Sending SIGINT to ${PID_TO_NAME[$pid]} (PID ${pid})..."
            kill -INT -- "-$pid" 2>/dev/null || kill -INT "$pid" 2>/dev/null || true
        fi
    done

    # Wait up to 60s for graceful shutdown (checkpoint save)
    echo -e "  ${BLUE}Waiting up to 60s for checkpoint saves...${NC}"
    sleep 5
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            timeout 55 tail --pid="$pid" -f /dev/null 2>/dev/null || true
        fi
    done

    echo -e "${YELLOW}All experiments stopped.${NC}"
    generate_status_report
    exit 130
}
trap cleanup SIGINT SIGTERM

# ── Wait for all experiments ──
echo -e "${BLUE}── Waiting for all experiments to complete ──${NC}"
echo -e "  ${CYAN}Monitor progress:${NC} ./wave_monitor.sh ${WAVE_NAME}"
echo -e "  ${CYAN}Stop all:${NC}         Ctrl+C (graceful shutdown with checkpoint save)"
echo ""

declare -A EXIT_CODES=()
completed=0

while [[ $completed -lt $NUM_EXPERIMENTS ]]; do
    for pid in "${PIDS[@]}"; do
        # Skip already-tracked PIDs
        [[ -n "${EXIT_CODES[$pid]:-}" ]] && continue

        if ! kill -0 "$pid" 2>/dev/null; then
            wait "$pid" 2>/dev/null
            EXIT_CODES["$pid"]=$?
            exp_name="${PID_TO_NAME[$pid]}"
            exit_code="${EXIT_CODES[$pid]}"
            completed=$((completed + 1))

            if [[ $exit_code -eq 0 ]]; then
                echo -e "  ${GREEN}✓${NC} ${exp_name} completed (${completed}/${NUM_EXPERIMENTS})"
            else
                echo -e "  ${RED}✗${NC} ${exp_name} failed with exit code ${exit_code} (${completed}/${NUM_EXPERIMENTS})"
            fi
        fi
    done
    sleep 5
done

# ── Final report ──
generate_status_report

echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║   WAVE ${WAVE_NAME} COMPLETE                                                ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "  ${BLUE}Results:${NC}    ${OUTPUT_BASE}"
echo -e "  ${BLUE}Report:${NC}     ${OUTPUT_BASE}/wave_report_${TIMESTAMP}.txt"
echo -e "  ${BLUE}Aggregate:${NC}  python genetic_algorithm/scripts/wave_comparison.py ${OUTPUT_BASE}"
echo ""

# Return non-zero if any experiment failed
for pid in "${PIDS[@]}"; do
    if [[ "${EXIT_CODES[$pid]}" != "0" ]]; then
        exit 1
    fi
done
exit 0
