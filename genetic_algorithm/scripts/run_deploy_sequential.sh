#!/usr/bin/env bash
# ============================================================================
# Sequential Deployment Runs Launcher
# ============================================================================
#
# PURPOSE:
#   Execute 3 deployment runs sequentially with comprehensive error handling
#   and progress tracking. Each run has different anti-overfitting configurations.
#
# RUNS:
#   Run 1: Basic exploration (no anti-overfitting)      ~2-3 hours
#   Run 2: Balanced validation (walk-forward + holdout) ~2-3 hours  
#   Run 3: Full robustness (regime-aware + all features) ~3-4 hours
#
#   Total estimated time: 8-10 hours
#
# USAGE:
#   # Start all runs from beginning
#   bash genetic_algorithm/scripts/run_deploy_sequential.sh
#
#   # Resume from a specific run (if interrupted)
#   bash genetic_algorithm/scripts/run_deploy_sequential.sh [1|2|3]
#
#   # Run in background with nohup
#   nohup bash genetic_algorithm/scripts/run_deploy_sequential.sh > deploy_runs.log 2>&1 &
#   echo $! > deploy_runs.pid
#
# MONITORING:
#   # Watch summary file
#   tail -f genetic_algorithm/logs/deploy_runs_summary.txt
#
#   # Check individual run logs
#   tail -f genetic_algorithm/logs/deploy_run1_basic.log
#   tail -f genetic_algorithm/logs/deploy_run2_balanced.log
#   tail -f genetic_algorithm/logs/deploy_run3_full.log
#
# ============================================================================

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

CONFIG_DIR="genetic_algorithm/config"
LOG_DIR="genetic_algorithm/logs"
OUTPUT_DIR="genetic_algorithm/output"

# Create directories
mkdir -p "$LOG_DIR"
mkdir -p "$OUTPUT_DIR"

# Summary file
SUMMARY_FILE="$LOG_DIR/deploy_runs_summary.txt"

# Determine starting run
START_FROM="${1:-1}"
if [[ ! "$START_FROM" =~ ^[123]$ ]]; then
    echo -e "${RED}ERROR:${NC} Invalid run number. Use 1, 2, or 3"
    echo "Usage: bash genetic_algorithm/scripts/run_deploy_sequential.sh [1|2|3]"
    exit 1
fi

# Banner
clear
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo -e "  ${BOLD}${CYAN}DEPLOYMENT RUNS - SEQUENTIAL EXECUTION${NC}"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo -e "${BLUE}Project root:${NC} $PROJECT_ROOT"
echo -e "${BLUE}Start from:${NC} Run $START_FROM"
echo -e "${BLUE}Estimated time:${NC} 8-10 hours total"
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo ""

# Activate virtualenv if available
VENV_DIR="$PROJECT_ROOT/.venv"
if [ -f "$VENV_DIR/bin/activate" ]; then
    echo -e "${GREEN}✓${NC} Activating virtual environment"
    source "$VENV_DIR/bin/activate"
else
    echo -e "${YELLOW}⚠${NC} No virtual environment found, using system Python"
fi
echo ""

# Check freqtrade availability
if ! command -v freqtrade &> /dev/null; then
    echo -e "${RED}✗ ERROR:${NC} freqtrade command not found"
    echo "Please install freqtrade or activate the correct virtual environment"
    exit 1
fi
echo -e "${GREEN}✓${NC} freqtrade found: $(which freqtrade)"
echo ""

# Initialize summary file
if [ "$START_FROM" = "1" ]; then
    cat > "$SUMMARY_FILE" << EOF
═══════════════════════════════════════════════════════════════
  DEPLOYMENT RUNS - EXECUTION SUMMARY
═══════════════════════════════════════════════════════════════

Started: $(date '+%Y-%m-%d %H:%M:%S')
Machine: $(hostname)

RUN CONFIGURATIONS:
  Run 1: Basic Exploration (no anti-overfitting)
  Run 2: Balanced Validation (walk-forward + holdout)
  Run 3: Full Robustness (regime-aware + all features)

───────────────────────────────────────────────────────────────

EOF
else
    echo "" >> "$SUMMARY_FILE"
    echo "═══════════════════════════════════════════════════════════════" >> "$SUMMARY_FILE"
    echo "RESUMED from Run $START_FROM — $(date '+%Y-%m-%d %H:%M:%S')" >> "$SUMMARY_FILE"
    echo "═══════════════════════════════════════════════════════════════" >> "$SUMMARY_FILE"
    echo "" >> "$SUMMARY_FILE"
fi

OVERALL_START=$(date +%s)

# Function to run a single evolution
run_evolution() {
    local run_num="$1"
    local run_name="$2"
    local config_file="$3"
    local log_file="$4"
    
    echo ""
    echo "═══════════════════════════════════════════════════════════════"
    echo -e "  ${BOLD}${CYAN}RUN $run_num: $run_name${NC}"
    echo "═══════════════════════════════════════════════════════════════"
    echo ""
    echo -e "${BLUE}Config:${NC} $config_file"
    echo -e "${BLUE}Log:${NC} $log_file"
    echo -e "${BLUE}Started:${NC} $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
    
    local start_ts=$(date +%s)
    
    # Run with tee to capture output to both terminal and log file
    python genetic_algorithm/run_ga.py \
        --config "$config_file" \
        --visualize \
        --yes \
        2>&1 | tee "$log_file"
    
    local exit_code=${PIPESTATUS[0]}
    local end_ts=$(date +%s)
    local elapsed=$(( end_ts - start_ts ))
    local hours=$(( elapsed / 3600 ))
    local minutes=$(( (elapsed % 3600) / 60 ))
    local seconds=$(( elapsed % 60 ))
    
    local status="SUCCESS"
    local status_symbol="✓"
    local status_color="$GREEN"
    
    if [ "$exit_code" -ne 0 ]; then
        status="FAILED"
        status_symbol="✗"
        status_color="$RED"
    fi
    
    # Update summary file
    cat >> "$SUMMARY_FILE" << EOF
RUN $run_num: $run_name
  Status:    $status (exit code: $exit_code)
  Duration:  ${hours}h ${minutes}m ${seconds}s
  Config:    $config_file
  Log:       $log_file
  Completed: $(date '+%Y-%m-%d %H:%M:%S')

EOF
    
    echo ""
    echo "───────────────────────────────────────────────────────────────"
    echo -e "  ${status_color}${status_symbol} Run $run_num: $status${NC} (${hours}h ${minutes}m ${seconds}s)"
    echo "───────────────────────────────────────────────────────────────"
    echo ""
    
    # Exit if run failed
    if [ "$exit_code" -ne 0 ]; then
        echo -e "${RED}ERROR:${NC} Run $run_num failed with exit code $exit_code"
        echo "Check log file: $log_file"
        echo ""
        echo "To resume from the next run, use:"
        echo "  bash genetic_algorithm/scripts/run_deploy_sequential.sh $((run_num + 1))"
        echo ""
        exit $exit_code
    fi
    
    # Brief pause between runs
    if [ $run_num -lt 3 ]; then
        echo "Pausing 30 seconds before next run..."
        sleep 30
    fi
}

# Execute runs
if [ "$START_FROM" -le 1 ]; then
    run_evolution \
        1 \
        "Basic Exploration" \
        "$CONFIG_DIR/ga_config_deploy_run1_basic.yaml" \
        "$LOG_DIR/deploy_run1_basic.log"
fi

if [ "$START_FROM" -le 2 ]; then
    run_evolution \
        2 \
        "Balanced Validation" \
        "$CONFIG_DIR/ga_config_deploy_run2_balanced.yaml" \
        "$LOG_DIR/deploy_run2_balanced.log"
fi

if [ "$START_FROM" -le 3 ]; then
    run_evolution \
        3 \
        "Full Robustness" \
        "$CONFIG_DIR/ga_config_deploy_run3_full.yaml" \
        "$LOG_DIR/deploy_run3_full.log"
fi

# Overall completion
OVERALL_END=$(date +%s)
TOTAL_ELAPSED=$(( OVERALL_END - OVERALL_START ))
TOTAL_HOURS=$(( TOTAL_ELAPSED / 3600 ))
TOTAL_MIN=$(( (TOTAL_ELAPSED % 3600) / 60 ))
TOTAL_SEC=$(( TOTAL_ELAPSED % 60 ))

# Final summary
cat >> "$SUMMARY_FILE" << EOF
═══════════════════════════════════════════════════════════════
  FINAL SUMMARY
═══════════════════════════════════════════════════════════════

Total Duration: ${TOTAL_HOURS}h ${TOTAL_MIN}m ${TOTAL_SEC}s
Completed:      $(date '+%Y-%m-%d %H:%M:%S')

OUTPUT LOCATIONS:
  Run 1: $OUTPUT_DIR/deploy_run1_basic/
  Run 2: $OUTPUT_DIR/deploy_run2_balanced/
  Run 3: $OUTPUT_DIR/deploy_run3_full/

HALL OF FAME:
  Run 1: $OUTPUT_DIR/deploy_run1_basic/hall_of_fame.json
  Run 2: $OUTPUT_DIR/deploy_run2_balanced/hall_of_fame.json
  Run 3: $OUTPUT_DIR/deploy_run3_full/hall_of_fame.json

LOG FILES:
  Run 1: $LOG_DIR/deploy_run1_basic.log
  Run 2: $LOG_DIR/deploy_run2_balanced.log
  Run 3: $LOG_DIR/deploy_run3_full.log

═══════════════════════════════════════════════════════════════

NEXT STEPS:

1. Review the hall of fame strategies from each run
2. Compare performance across the three approaches
3. Backtest the best strategies on holdout periods
4. Consider paper trading before live deployment

ANALYSIS COMMANDS:

  # View hall of fame
  cat $OUTPUT_DIR/deploy_run1_basic/hall_of_fame.json | jq
  cat $OUTPUT_DIR/deploy_run2_balanced/hall_of_fame.json | jq
  cat $OUTPUT_DIR/deploy_run3_full/hall_of_fame.json | jq

  # Compare fitness progression
  ls -la $OUTPUT_DIR/deploy_run*/fitness_*.png

═══════════════════════════════════════════════════════════════
EOF

# Display final summary
clear
echo ""
cat "$SUMMARY_FILE"
echo ""
echo -e "${GREEN}${BOLD}✓ ALL DEPLOYMENT RUNS COMPLETED SUCCESSFULLY!${NC}"
echo ""
echo -e "Full summary saved to: ${CYAN}$SUMMARY_FILE${NC}"
echo ""
