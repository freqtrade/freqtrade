#!/usr/bin/env bash
# ============================================================================
# Server Comparison Suite — 3 GA Modes Head-to-Head
# ============================================================================
# Runs sequentially:
#   A) Standard single-population (pop=50, gen=30, WF, full anti-overfit)
#   B) Island model with parallel islands (4 islands × pop=30, gen=20)
#   C) NSGA-II multi-objective with min_trades fix (pop=50, gen=25)
#
# Each run outputs to a timestamped directory for later aggregation.
#
# Usage:
#   cd /home/kali/trading/freqtradeForkGA
#   chmod +x run_server_comparison.sh
#   nohup ./run_server_comparison.sh &
#
# Or via tmux:
#   tmux new -s comparison './run_server_comparison.sh'
#
# Expected total runtime: ~14-22 hours
# ============================================================================

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="${REPO_DIR}/.venv"
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
OUTPUT_BASE="${REPO_DIR}/genetic_algorithm/output/server_comparison_${TIMESTAMP}"

# ── Colours ──
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# ── Run definitions ──
declare -a RUN_NAMES=("run_A_standard" "run_B_island" "run_C_nsga2")

declare -A RUN_CONFIGS=(
    ["run_A_standard"]="${REPO_DIR}/genetic_algorithm/config/ga_config_server_6core.yaml"
    ["run_B_island"]="${REPO_DIR}/genetic_algorithm/config/ga_config_server_island.yaml"
    ["run_C_nsga2"]="${REPO_DIR}/genetic_algorithm/config/ga_config_server_nsga2.yaml"
)

declare -A RUN_DESCRIPTIONS=(
    ["run_A_standard"]="Standard single-population (pop=50, gen=30, WF+anti-overfit)"
    ["run_B_island"]="Island model with parallel islands (4×pop=30, gen=20)"
    ["run_C_nsga2"]="NSGA-II multi-objective with min_trades fix (pop=50, gen=25)"
)

# ── Functions ──

print_header() {
    echo ""
    echo -e "${CYAN}╔══════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║       SERVER COMPARISON SUITE — ${TIMESTAMP}              ║${NC}"
    echo -e "${CYAN}╚══════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "  ${BLUE}Output directory:${NC}  ${OUTPUT_BASE}"
    echo -e "  ${BLUE}Number of runs:${NC}    ${#RUN_NAMES[@]}"
    echo -e "  ${BLUE}Server:${NC}            $(nproc) cores, $(free -h | awk '/Mem:/ {print $2}') RAM"
    echo ""
}

activate_venv() {
    if [[ -f "${VENV_DIR}/bin/activate" ]]; then
        source "${VENV_DIR}/bin/activate"
        echo -e "  ${GREEN}✓${NC} Virtual environment activated"
    else
        echo -e "  ${RED}✗${NC} Virtual environment not found at ${VENV_DIR}"
        exit 1
    fi
}

run_single() {
    local run_name="$1"
    local run_index="$2"
    local total_runs="$3"
    local config_file="${RUN_CONFIGS[${run_name}]}"
    local log_file="${OUTPUT_BASE}/${run_name}.log"
    local description="${RUN_DESCRIPTIONS[${run_name}]}"

    echo ""
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${YELLOW}  RUN ${run_index}/${total_runs}: ${description}${NC}"
    echo -e "${YELLOW}  Config: ${config_file}${NC}"
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""

    if [[ ! -f "$config_file" ]]; then
        echo -e "  ${RED}✗ Config file not found: ${config_file}${NC}"
        return 1
    fi

    mkdir -p "${OUTPUT_BASE}"

    local start_time
    start_time=$(date +%s)

    # Set output directory environment variable
    export GA_OUTPUT_DIR="${OUTPUT_BASE}/${run_name}"
    mkdir -p "$GA_OUTPUT_DIR"

    # Run the GA
    python genetic_algorithm/run_ga.py \
        --config "$config_file" \
        --no-monitor --yes \
        2>&1 | tee "${log_file}" || true

    local end_time
    end_time=$(date +%s)
    local elapsed=$(( end_time - start_time ))
    local elapsed_min=$(( elapsed / 60 ))

    echo ""
    echo -e "  ${GREEN}✓${NC} ${run_name} completed in ${elapsed_min} minutes"

    # Extract key metrics from log
    echo ""
    echo -e "  ${BLUE}── Quick Results ──${NC}"
    grep -E "(BEST|best_fitness|FITNESS|SAFE|profit)" "${log_file}" | tail -5 || true
    echo ""

    # Cooldown between runs
    if [[ "$run_index" -lt "$total_runs" ]]; then
        echo -e "  ${BLUE}Cooling down 10 seconds before next run...${NC}"
        sleep 10
    fi
}

generate_summary() {
    local summary_file="${OUTPUT_BASE}/comparison_summary.txt"
    echo ""
    echo -e "${CYAN}╔══════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║  COMPARISON SUMMARY                                            ║${NC}"
    echo -e "${CYAN}╚══════════════════════════════════════════════════════════════════╝${NC}"
    echo ""

    {
        echo "Server Comparison Suite — $(date)"
        echo "========================================"
        echo ""
        for run_name in "${RUN_NAMES[@]}"; do
            local log_file="${OUTPUT_BASE}/${run_name}.log"
            local description="${RUN_DESCRIPTIONS[${run_name}]}"
            echo "── ${description} ──"
            if [[ -f "$log_file" ]]; then
                echo "  Best fitness: $(grep -oP 'best_fitness.*?=\K[\d.]+' "$log_file" 2>/dev/null | tail -1 || echo 'N/A')"
                echo "  Best profit:  $(grep -oP 'profit.*?[\d.-]+%' "$log_file" 2>/dev/null | tail -1 || echo 'N/A')"
                echo "  SAFE score:   $(grep -oP '\d+/\d+ SAFE' "$log_file" 2>/dev/null | tail -1 || echo 'N/A')"
                echo "  Runtime:      $(grep -oP 'Total.*?[\d.]+ (min|sec|hour)' "$log_file" 2>/dev/null | tail -1 || echo 'N/A')"
            else
                echo "  (no log file found)"
            fi
            echo ""
        done
    } | tee "$summary_file"

    echo ""
    echo -e "  ${GREEN}Summary saved to: ${summary_file}${NC}"
}

# ── Main ──

print_header
activate_venv

echo ""
echo -e "${BLUE}── Pre-flight checks ──${NC}"
echo -e "  Python: $(python --version 2>&1)"
echo -e "  Working directory: ${REPO_DIR}"
echo ""

# Run all 3 comparison modes
total_runs=${#RUN_NAMES[@]}
for i in "${!RUN_NAMES[@]}"; do
    run_index=$(( i + 1 ))
    run_single "${RUN_NAMES[$i]}" "$run_index" "$total_runs"
done

# Generate summary
generate_summary

echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║  ALL COMPARISON RUNS COMPLETE                                  ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "  ${BLUE}Results:${NC} ${OUTPUT_BASE}"
echo -e "  ${BLUE}Aggregate:${NC} python genetic_algorithm/scripts/aggregate_comparison.py ${OUTPUT_BASE}"
echo ""
