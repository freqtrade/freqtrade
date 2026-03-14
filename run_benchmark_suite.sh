#!/usr/bin/env bash
# ============================================================================
# GA Benchmark Suite Runner
# ============================================================================
# Runs 8 benchmark experiments sequentially, capturing all output.
# Each run is time-limited via max_runtime_minutes in config.
#
# Usage:
#   cd /home/kali/trading/freqtradeForkGA
#   chmod +x run_benchmark_suite.sh
#   ./run_benchmark_suite.sh
#
# Expected total runtime: ~2.5–3 hours
# ============================================================================

set -euo pipefail

# ── Configuration ──
REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG_DIR="${REPO_DIR}/genetic_algorithm/config/benchmark"
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
OUTPUT_BASE="${REPO_DIR}/genetic_algorithm/output/benchmark_${TIMESTAMP}"
VENV_DIR="${REPO_DIR}/.venv"

# ── Colours ──
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'  # No Colour

# ── Run definitions: name → config file ──
declare -a RUN_NAMES=(
    "run1_baseline_raw"
    "run2_walkforward_only"
    "run3_full_antioverfit"
    "run4_island_regime"
    "run5_multi_pair"
    "run6_nsga2_multiobjective"
    "run7_short_selling"
    "run8_fee_noise_robust"
)

declare -A RUN_DESCRIPTIONS=(
    ["run1_baseline_raw"]="Baseline Raw GA (no anti-overfit)"
    ["run2_walkforward_only"]="Walk-Forward Only"
    ["run3_full_antioverfit"]="Full Anti-Overfit Stack"
    ["run4_island_regime"]="Island Model + Regime Specialists"
    ["run5_multi_pair"]="Multi-Pair Generalization (4 pairs)"
    ["run6_nsga2_multiobjective"]="NSGA-II Multi-Objective"
    ["run7_short_selling"]="Independent Short Selling"
    ["run8_fee_noise_robust"]="Fee Noise + Robustness"
)

# ── Functions ──

print_header() {
    echo ""
    echo -e "${CYAN}╔══════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║          GA BENCHMARK SUITE — ${TIMESTAMP}                ║${NC}"
    echo -e "${CYAN}╚══════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "  ${BLUE}Output directory:${NC} ${OUTPUT_BASE}"
    echo -e "  ${BLUE}Number of runs:${NC}   ${#RUN_NAMES[@]}"
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

extract_metric() {
    # Extract a metric from a log file using grep
    local logfile="$1"
    local pattern="$2"
    local default="${3:-N/A}"
    local result
    result=$(grep -oP "${pattern}" "${logfile}" 2>/dev/null | tail -1) || true
    echo "${result:-${default}}"
}

run_single_benchmark() {
    local run_name="$1"
    local run_index="$2"
    local total_runs="$3"
    local config_file="${CONFIG_DIR}/${run_name}.yaml"
    local run_output="${OUTPUT_BASE}/${run_name}"
    local log_file="${OUTPUT_BASE}/${run_name}.log"
    local description="${RUN_DESCRIPTIONS[${run_name}]}"

    echo ""
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${YELLOW}  RUN ${run_index}/${total_runs}: ${description}${NC}"
    echo -e "${YELLOW}  Config: ${config_file}${NC}"
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

    if [[ ! -f "${config_file}" ]]; then
        echo -e "  ${RED}✗ Config file not found: ${config_file}${NC}"
        echo "SKIPPED" > "${OUTPUT_BASE}/${run_name}.status"
        return 1
    fi

    # Create run output directory
    mkdir -p "${run_output}"

    # Record start time
    local start_time
    start_time=$(date +%s)
    echo "$(date '+%Y-%m-%d %H:%M:%S')" > "${OUTPUT_BASE}/${run_name}.start"

    # Run the GA
    echo -e "  ${BLUE}▸ Starting at $(date '+%H:%M:%S')...${NC}"

    set +e
    python "${REPO_DIR}/genetic_algorithm/run_ga.py" \
        --config "${config_file}" \
        --no-monitor \
        --yes \
        2>&1 | tee "${log_file}"
    local exit_code=$?
    set -e

    # Record end time
    local end_time
    end_time=$(date +%s)
    local duration=$(( end_time - start_time ))
    local duration_min=$(( duration / 60 ))
    local duration_sec=$(( duration % 60 ))
    echo "$(date '+%Y-%m-%d %H:%M:%S')" > "${OUTPUT_BASE}/${run_name}.end"

    # Record status
    if [[ ${exit_code} -eq 0 ]]; then
        echo "SUCCESS" > "${OUTPUT_BASE}/${run_name}.status"
        echo -e "  ${GREEN}✓ Completed in ${duration_min}m ${duration_sec}s (exit code: ${exit_code})${NC}"
    else
        echo "FAILED:${exit_code}" > "${OUTPUT_BASE}/${run_name}.status"
        echo -e "  ${RED}✗ Failed in ${duration_min}m ${duration_sec}s (exit code: ${exit_code})${NC}"
    fi

    # Record duration
    echo "${duration}" > "${OUTPUT_BASE}/${run_name}.duration"

    # Copy run output files to benchmark output
    local ga_output_dir="${REPO_DIR}/genetic_algorithm/output/benchmark/${run_name}"
    if [[ -d "${ga_output_dir}" ]]; then
        cp -r "${ga_output_dir}/." "${run_output}/" 2>/dev/null || true
    fi

    # Copy log files
    local ga_log="genetic_algorithm/logs/benchmark_${run_name#run*}.log"
    if [[ -f "${REPO_DIR}/${ga_log}" ]]; then
        cp "${REPO_DIR}/${ga_log}" "${run_output}/" 2>/dev/null || true
    fi

    # Copy generation_stats.csv if present
    find "${REPO_DIR}/genetic_algorithm/output/benchmark/${run_name}" \
        -name "generation_stats.csv" -exec cp {} "${run_output}/" \; 2>/dev/null || true

    # Brief cooldown between runs
    echo -e "  ${BLUE}▸ Cooldown 5s...${NC}"
    sleep 5

    return ${exit_code}
}

generate_summary() {
    local summary_file="${OUTPUT_BASE}/benchmark_summary.txt"

    echo ""
    echo -e "${CYAN}╔══════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║                    BENCHMARK SUITE SUMMARY                      ║${NC}"
    echo -e "${CYAN}╚══════════════════════════════════════════════════════════════════╝${NC}"

    {
        echo "=================================================================="
        echo "  GA BENCHMARK SUITE SUMMARY"
        echo "  Timestamp: ${TIMESTAMP}"
        echo "  Date: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "=================================================================="
        echo ""
        echo "ENVIRONMENT:"
        echo "  Python: $(python --version 2>&1)"
        echo "  Platform: $(uname -a)"
        echo "  CPU cores: $(nproc)"
        echo ""
        echo "=================================================================="
        echo "  INDIVIDUAL RUN RESULTS"
        echo "=================================================================="
        echo ""

        local total_passed=0
        local total_failed=0
        local total_skipped=0
        local total_duration=0

        for run_name in "${RUN_NAMES[@]}"; do
            local description="${RUN_DESCRIPTIONS[${run_name}]}"
            local status_file="${OUTPUT_BASE}/${run_name}.status"
            local duration_file="${OUTPUT_BASE}/${run_name}.duration"
            local log_file="${OUTPUT_BASE}/${run_name}.log"

            local status="UNKNOWN"
            local duration=0

            if [[ -f "${status_file}" ]]; then
                status=$(cat "${status_file}")
            fi
            if [[ -f "${duration_file}" ]]; then
                duration=$(cat "${duration_file}")
            fi
            total_duration=$(( total_duration + duration ))

            local duration_min=$(( duration / 60 ))
            local duration_sec=$(( duration % 60 ))

            echo "──────────────────────────────────────────────────────────────────"
            echo "  ${run_name}: ${description}"
            echo "  Status: ${status}"
            echo "  Duration: ${duration_min}m ${duration_sec}s"

            if [[ "${status}" == "SUCCESS" ]]; then
                total_passed=$((total_passed + 1))
            elif [[ "${status}" == "SKIPPED" ]]; then
                total_skipped=$((total_skipped + 1))
            else
                total_failed=$((total_failed + 1))
            fi

            # Extract key metrics from log
            if [[ -f "${log_file}" ]]; then
                # Best fitness
                local best_fitness
                best_fitness=$(grep -oP 'Best fitness.*?:\s*\K[\d.]+' "${log_file}" 2>/dev/null | tail -1) || true
                if [[ -n "${best_fitness}" ]]; then
                    echo "  Best fitness: ${best_fitness}"
                fi

                # Generations completed
                local gen_completed
                gen_completed=$(grep -oP 'GENERATION\s+\K\d+(?=/)' "${log_file}" 2>/dev/null | tail -1) || true
                local gen_total
                gen_total=$(grep -oP 'GENERATION\s+\d+/\K\d+' "${log_file}" 2>/dev/null | tail -1) || true
                if [[ -n "${gen_completed}" ]]; then
                    echo "  Generations: ${gen_completed}/${gen_total:-?}"
                fi

                # Convergence
                if grep -q "Converged:" "${log_file}" 2>/dev/null; then
                    echo "  Convergence: EARLY (converged)"
                elif grep -q "TIME LIMIT" "${log_file}" 2>/dev/null; then
                    echo "  Convergence: TIME LIMIT"
                else
                    echo "  Convergence: COMPLETED (all gens)"
                fi

                # Overfit assessment
                local safe_count
                safe_count=$(grep -coP 'SAFE' "${log_file}" 2>/dev/null) || safe_count=0
                local warning_count
                warning_count=$(grep -coP 'WARNING|CAUTION' "${log_file}" 2>/dev/null) || warning_count=0
                local overfit_count
                overfit_count=$(grep -coP 'OVERFIT|DANGER' "${log_file}" 2>/dev/null) || overfit_count=0
                echo "  Overfit classification: SAFE=${safe_count} / WARNING=${warning_count} / OVERFIT=${overfit_count}"
            fi
            echo ""
        done

        local total_min=$(( total_duration / 60 ))
        local total_sec=$(( total_duration % 60 ))

        echo "=================================================================="
        echo "  TOTALS"
        echo "=================================================================="
        echo "  Passed:  ${total_passed}/${#RUN_NAMES[@]}"
        echo "  Failed:  ${total_failed}/${#RUN_NAMES[@]}"
        echo "  Skipped: ${total_skipped}/${#RUN_NAMES[@]}"
        echo "  Total duration: ${total_min}m ${total_sec}s"
        echo ""
        echo "=================================================================="
        echo "  FILES"
        echo "=================================================================="
        echo "  Summary:  ${summary_file}"
        echo "  Logs:     ${OUTPUT_BASE}/*.log"
        echo "  Outputs:  ${OUTPUT_BASE}/<run_name>/"
        echo ""
        echo "  To generate comparison report:"
        echo "    python genetic_algorithm/scripts/benchmark_report.py ${OUTPUT_BASE}"
        echo ""
    } | tee "${summary_file}"
}

# ── Main ──

cd "${REPO_DIR}"

print_header
activate_venv

# Create output directory
mkdir -p "${OUTPUT_BASE}"

# Save run metadata
{
    echo "timestamp: ${TIMESTAMP}"
    echo "python_version: $(python --version 2>&1)"
    echo "platform: $(uname -a)"
    echo "cpu_cores: $(nproc)"
    echo "repo_dir: ${REPO_DIR}"
    echo "git_sha: $(git rev-parse --short HEAD 2>/dev/null || echo 'unknown')"
    echo "configs:"
    for run_name in "${RUN_NAMES[@]}"; do
        echo "  - ${run_name}"
    done
} > "${OUTPUT_BASE}/run_metadata.yaml"

# Run all benchmarks
suite_start=$(date +%s)
passed=0
failed=0

for i in "${!RUN_NAMES[@]}"; do
    run_name="${RUN_NAMES[$i]}"
    run_index=$((i + 1))

    if run_single_benchmark "${run_name}" "${run_index}" "${#RUN_NAMES[@]}"; then
        passed=$((passed + 1))
    else
        failed=$((failed + 1))
    fi
done

suite_end=$(date +%s)
suite_duration=$(( suite_end - suite_start ))
suite_min=$(( suite_duration / 60 ))
suite_sec=$(( suite_duration % 60 ))

echo ""
echo -e "${BLUE}Total suite duration: ${suite_min}m ${suite_sec}s${NC}"

# Generate summary
generate_summary

# Run comparison report
echo ""
echo -e "${BLUE}▸ Generating comparison report...${NC}"
if python "${REPO_DIR}/genetic_algorithm/scripts/benchmark_report.py" "${OUTPUT_BASE}" 2>&1; then
    echo -e "${GREEN}✓ Comparison report generated${NC}"
else
    echo -e "${YELLOW}⚠ Comparison report generation failed (non-critical)${NC}"
fi

echo ""
echo -e "${GREEN}════════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  BENCHMARK SUITE COMPLETE: ${passed} passed, ${failed} failed${NC}"
echo -e "${GREEN}  Results: ${OUTPUT_BASE}${NC}"
echo -e "${GREEN}════════════════════════════════════════════════════════════════${NC}"
