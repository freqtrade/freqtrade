#!/usr/bin/env bash
# ============================================================================
# Phase 1 Sequential Runner — WSL-safe (max 4 workers, one run at a time)
# ============================================================================
# Usage:  ./run_phase1_sequential.sh [a1|a2|a3]
#   No args  → runs all three sequentially (A1→A2→A3)
#   "a1"     → runs only A1 (15m)
#   "a2"     → runs only A2 (5m)
#   "a3"     → runs only A3 (1h)
# ============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

CONFIG_DIR="genetic_algorithm/config/production"
LOG_DIR="genetic_algorithm/logs"
mkdir -p "$LOG_DIR"

run_experiment() {
    local name="$1"
    local config="$2"
    local logfile="$LOG_DIR/${name}.log"

    echo ""
    echo "================================================================"
    echo "  STARTING: $name"
    echo "  Config:   $config"
    echo "  Log:      $logfile"
    echo "  Time:     $(date '+%Y-%m-%d %H:%M:%S')"
    echo "================================================================"
    echo ""

    python genetic_algorithm/run_ga.py \
        --config "$config" \
        --yes \
        --no-monitor \
        2>&1 | tee "$logfile"

    local exit_code=${PIPESTATUS[0]}

    echo ""
    echo "================================================================"
    echo "  FINISHED: $name  (exit code: $exit_code)"
    echo "  Time:     $(date '+%Y-%m-%d %H:%M:%S')"
    echo "================================================================"
    echo ""

    return $exit_code
}

# Which runs to execute
TARGET="${1:-all}"

case "$TARGET" in
    a1)
        run_experiment "phase1_a1_15m" "$CONFIG_DIR/phase1_a1_15m.yaml"
        ;;
    a2)
        run_experiment "phase1_a2_5m" "$CONFIG_DIR/phase1_a2_5m.yaml"
        ;;
    a3)
        run_experiment "phase1_a3_1h" "$CONFIG_DIR/phase1_a3_1h.yaml"
        ;;
    all)
        echo "Running all 3 Phase 1 experiments SEQUENTIALLY..."
        echo "Estimated total time: ~6-9 hours (3x ~2-3h each)"
        echo ""

        run_experiment "phase1_a1_15m" "$CONFIG_DIR/phase1_a1_15m.yaml"
        echo "--- A1 done, sleeping 10s before A2 ---"
        sleep 10

        run_experiment "phase1_a2_5m" "$CONFIG_DIR/phase1_a2_5m.yaml"
        echo "--- A2 done, sleeping 10s before A3 ---"
        sleep 10

        run_experiment "phase1_a3_1h" "$CONFIG_DIR/phase1_a3_1h.yaml"

        echo ""
        echo "============================================================"
        echo "  ALL PHASE 1 RUNS COMPLETE"
        echo "  Results in: genetic_algorithm/output/phase1/"
        echo "============================================================"
        ;;
    *)
        echo "Usage: $0 [a1|a2|a3|all]"
        exit 1
        ;;
esac
