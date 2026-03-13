#!/usr/bin/env bash
# ============================================================================
# Sequential Runner: Island Model + MTF Regime Detection Comparison
# ============================================================================
# Runs 4 GA experiments back-to-back, each ~30-45 min:
#   A) Baseline   — Single-TF ensemble, no MTF
#   B) Hierarchical — MTF with hierarchical combination
#   C) Weighted   — MTF with weighted voting combination
#   D) Full Stack — MTF hierarchical + in-strategy RegimeGene
#
# Usage:  ./run_island_mtf_tests.sh
# Output: Logs + summary in genetic_algorithm/output/island_mtf_comparison/
# ============================================================================

set -euo pipefail

# ── Paths ──
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONFIG_DIR="genetic_algorithm/config"
OUTPUT_BASE="genetic_algorithm/output/island_mtf_comparison"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR="${OUTPUT_BASE}/run_${TIMESTAMP}"

mkdir -p "$RUN_DIR"

# ── Activate venv if available ──
if [[ -f ".venv/bin/activate" ]]; then
    source .venv/bin/activate
fi

# ── Define runs ──
declare -A CONFIGS
CONFIGS[A]="${CONFIG_DIR}/ga_config_island_mtf_A_baseline.yaml"
CONFIGS[B]="${CONFIG_DIR}/ga_config_island_mtf_B_hierarchical.yaml"
CONFIGS[C]="${CONFIG_DIR}/ga_config_island_mtf_C_weighted.yaml"
CONFIGS[D]="${CONFIG_DIR}/ga_config_island_mtf_D_full.yaml"

declare -A LABELS
LABELS[A]="Baseline (Single-TF, No MTF)"
LABELS[B]="MTF Hierarchical"
LABELS[C]="MTF Weighted Voting"
LABELS[D]="MTF Hierarchical + In-Strategy Regime"

RUN_ORDER=(A B C D)

# ── Summary file ──
SUMMARY="${RUN_DIR}/summary.txt"
cat > "$SUMMARY" << EOF
================================================================================
ISLAND MODEL + MTF COMPARISON — Run started: $(date)
================================================================================
Host: $(hostname)
Python: $(python --version 2>&1)
Working dir: $SCRIPT_DIR
Run directory: $RUN_DIR
================================================================================

EOF

echo "============================================================"
echo "  Island Model + MTF Comparison Runner"
echo "  Started: $(date)"
echo "  Output:  $RUN_DIR"
echo "============================================================"
echo ""

TOTAL_START=$(date +%s)
PASS_COUNT=0
FAIL_COUNT=0

for RUN_ID in "${RUN_ORDER[@]}"; do
    CONFIG="${CONFIGS[$RUN_ID]}"
    LABEL="${LABELS[$RUN_ID]}"
    LOG_FILE="${RUN_DIR}/run_${RUN_ID}.log"

    echo "────────────────────────────────────────────────────────────"
    echo "  RUN ${RUN_ID}: ${LABEL}"
    echo "  Config: ${CONFIG}"
    echo "  Log:    ${LOG_FILE}"
    echo "  Started: $(date)"
    echo "────────────────────────────────────────────────────────────"

    RUN_START=$(date +%s)

    # Run GA with --no-monitor --yes (non-interactive)
    set +e
    python genetic_algorithm/run_ga.py \
        --config "$CONFIG" \
        --no-monitor \
        --yes \
        2>&1 | tee "$LOG_FILE"
    EXIT_CODE=$?
    set -e

    RUN_END=$(date +%s)
    DURATION=$(( RUN_END - RUN_START ))
    MINUTES=$(( DURATION / 60 ))
    SECONDS=$(( DURATION % 60 ))

    if [[ $EXIT_CODE -eq 0 ]]; then
        STATUS="✓ PASSED"
        PASS_COUNT=$(( PASS_COUNT + 1 ))
    else
        STATUS="✗ FAILED (exit code: $EXIT_CODE)"
        FAIL_COUNT=$(( FAIL_COUNT + 1 ))
    fi

    # Extract key metrics from log
    BEST_FITNESS=$(grep -oP 'Best fitness:\s*\K[\d.]+' "$LOG_FILE" | tail -1 || echo "N/A")
    STRATEGIES=$(grep -oP 'Evaluated \K\d+ strategies' "$LOG_FILE" | tail -1 || echo "N/A")
    MTF_INFO=$(grep -oP 'MTF detection produced \K.*' "$LOG_FILE" | head -1 || echo "N/A (single-TF)")
    OVERFITTING=$(grep -cP 'SAFE|CAUTION|DANGER' "$LOG_FILE" 2>/dev/null || echo "0")

    # Append to summary
    cat >> "$SUMMARY" << EOF
────────────────────────────────────────────────────────────────────
RUN ${RUN_ID}: ${LABEL}
────────────────────────────────────────────────────────────────────
  Config:         ${CONFIG}
  Status:         ${STATUS}
  Duration:       ${MINUTES}m ${SECONDS}s
  Best Fitness:   ${BEST_FITNESS}
  Strategies:     ${STRATEGIES}
  MTF Segments:   ${MTF_INFO}
  Overfitting:    ${OVERFITTING} assessments
  Log:            ${LOG_FILE}

EOF

    echo ""
    echo "  → ${STATUS}  (${MINUTES}m ${SECONDS}s)"
    echo ""

    # Copy island results if they exist
    ISLAND_RESULTS="genetic_algorithm/output/island_results"
    if [[ -d "$ISLAND_RESULTS" ]]; then
        cp -r "$ISLAND_RESULTS" "${RUN_DIR}/island_results_${RUN_ID}"
        echo "  → Results saved to ${RUN_DIR}/island_results_${RUN_ID}"
    fi

    # Copy regime chart if it exists
    REGIME_CHART="genetic_algorithm/output/regime_chart.png"
    if [[ -f "$REGIME_CHART" ]]; then
        cp "$REGIME_CHART" "${RUN_DIR}/regime_chart_${RUN_ID}.png"
    fi

    # Small delay between runs to let system settle
    if [[ "$RUN_ID" != "D" ]]; then
        echo "  → Cooling down 10s before next run..."
        sleep 10
    fi
done

TOTAL_END=$(date +%s)
TOTAL_DURATION=$(( TOTAL_END - TOTAL_START ))
TOTAL_MIN=$(( TOTAL_DURATION / 60 ))
TOTAL_SEC=$(( TOTAL_DURATION % 60 ))

# ── Final summary ──
cat >> "$SUMMARY" << EOF
================================================================================
FINAL SUMMARY
================================================================================
  Total runs:     ${#RUN_ORDER[@]}
  Passed:         ${PASS_COUNT}
  Failed:         ${FAIL_COUNT}
  Total duration: ${TOTAL_MIN}m ${TOTAL_SEC}s
  Completed:      $(date)
================================================================================
EOF

echo ""
echo "============================================================"
echo "  ALL RUNS COMPLETE"
echo "  Passed: ${PASS_COUNT}/${#RUN_ORDER[@]}"
echo "  Failed: ${FAIL_COUNT}/${#RUN_ORDER[@]}"
echo "  Total:  ${TOTAL_MIN}m ${TOTAL_SEC}s"
echo "  Summary: ${SUMMARY}"
echo "============================================================"

cat "$SUMMARY"
