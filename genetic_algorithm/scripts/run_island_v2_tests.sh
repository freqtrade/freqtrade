#!/usr/bin/env bash
# ============================================================================
# Island Model v2 Test Runner — Sequential A/B/C/D
# ============================================================================
# Runs all 4 configs sequentially, capturing output and timing.
# Usage:  bash run_island_v2_tests.sh
#    or:  nohup bash run_island_v2_tests.sh > island_v2_tests.log 2>&1 &

set -euo pipefail
cd "$(dirname "$0")/../.."

START_FROM="${1:-A}"
if [[ ! "$START_FROM" =~ ^[ABCD]$ ]]; then
    echo "Usage: bash genetic_algorithm/scripts/run_island_v2_tests.sh [A|B|C|D]"
    echo "Example: bash genetic_algorithm/scripts/run_island_v2_tests.sh B"
    exit 1
fi

# Activate virtualenv if available
VENV_DIR="$(pwd)/.venv"
if [ -f "$VENV_DIR/bin/activate" ]; then
    source "$VENV_DIR/bin/activate"
fi

CONFIG_DIR="genetic_algorithm/config"
LOG_DIR="genetic_algorithm/logs"
mkdir -p "$LOG_DIR"

SUMMARY_FILE="$LOG_DIR/island_v2_summary.txt"
if [ "$START_FROM" = "A" ]; then
    echo "Island Model v2 Test Suite — $(date)" > "$SUMMARY_FILE"
    echo "==========================================" >> "$SUMMARY_FILE"
else
    echo "" >> "$SUMMARY_FILE"
    echo "Island Model v2 Test Suite RESUME from $START_FROM — $(date)" >> "$SUMMARY_FILE"
    echo "==========================================" >> "$SUMMARY_FILE"
fi

OVERALL_START=$(date +%s)

run_config() {
    local label="$1"
    local config_file="$2"
    local log_file="$3"

    echo ""
    echo "============================================"
    echo "  RUN $label — $(date '+%H:%M:%S')"
    echo "  Config: $config_file"
    echo "============================================"
    echo ""

    local start_ts=$(date +%s)

    python -m genetic_algorithm.run_ga \
        --config "$config_file" \
        --no-interactive \
        --yes \
        2>&1 | tee "$log_file"

    local exit_code=${PIPESTATUS[0]}
    local end_ts=$(date +%s)
    local elapsed=$(( end_ts - start_ts ))
    local minutes=$(( elapsed / 60 ))
    local seconds=$(( elapsed % 60 ))

    local status="PASS"
    if [ "$exit_code" -ne 0 ]; then
        status="FAIL (exit=$exit_code)"
    fi

    echo "" >> "$SUMMARY_FILE"
    echo "Run $label: $status  (${minutes}m ${seconds}s)" >> "$SUMMARY_FILE"
    echo "  Config: $config_file" >> "$SUMMARY_FILE"
    echo "  Log:    $log_file" >> "$SUMMARY_FILE"

    echo ""
    echo "──── Run $label: $status (${minutes}m ${seconds}s) ────"
    echo ""
}

# Run A: Baseline — single-TF advanced_ensemble
if [[ "$START_FROM" =~ ^A$ ]]; then
    run_config "A (Baseline)" \
        "$CONFIG_DIR/ga_config_island_v2_A_baseline.yaml" \
        "$LOG_DIR/run_v2_A_baseline.log"
fi

# Run B: MTF Hierarchical + advanced_ensemble
if [[ "$START_FROM" =~ ^(A|B)$ ]]; then
    run_config "B (MTF Hierarchical)" \
        "$CONFIG_DIR/ga_config_island_v2_B_hierarchical.yaml" \
        "$LOG_DIR/run_v2_B_hierarchical.log"
fi

# Run C: MTF Weighted Voting + advanced_ensemble
if [[ "$START_FROM" =~ ^(A|B|C)$ ]]; then
    run_config "C (MTF Weighted)" \
        "$CONFIG_DIR/ga_config_island_v2_C_weighted.yaml" \
        "$LOG_DIR/run_v2_C_weighted.log"
fi

# Run D: MTF Hierarchical + advanced_ensemble + RegimeGene
if [[ "$START_FROM" =~ ^(A|B|C|D)$ ]]; then
    run_config "D (Full Stack)" \
        "$CONFIG_DIR/ga_config_island_v2_D_full.yaml" \
        "$LOG_DIR/run_v2_D_full.log"
fi

OVERALL_END=$(date +%s)
TOTAL_ELAPSED=$(( OVERALL_END - OVERALL_START ))
TOTAL_MIN=$(( TOTAL_ELAPSED / 60 ))
TOTAL_SEC=$(( TOTAL_ELAPSED % 60 ))

echo "" >> "$SUMMARY_FILE"
echo "==========================================" >> "$SUMMARY_FILE"
echo "Total time: ${TOTAL_MIN}m ${TOTAL_SEC}s" >> "$SUMMARY_FILE"
echo "Completed:  $(date)" >> "$SUMMARY_FILE"

echo ""
echo "============================================"
echo "  ALL RUNS COMPLETE — ${TOTAL_MIN}m ${TOTAL_SEC}s total"
echo "  Summary: $SUMMARY_FILE"
echo "============================================"
cat "$SUMMARY_FILE"
