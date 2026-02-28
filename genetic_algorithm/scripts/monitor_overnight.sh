#!/usr/bin/env bash
# ============================================================================
# monitor_overnight.sh — Quick status check for the overnight GA run
#
# Usage: bash genetic_algorithm/scripts/monitor_overnight.sh
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GA_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
cd "$GA_ROOT"

PID_FILE="genetic_algorithm/logs/overnight_pid.txt"
STDOUT_LOG="genetic_algorithm/logs/overnight_stdout.log"
LOG_DIR="genetic_algorithm/logs"

echo "============================================================"
echo "  GA Overnight Run — Status Report"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"
echo ""

# --- 1. Process Status ---
echo "--- Process Status ---"
if [[ -f "$PID_FILE" ]]; then
    PID=$(cat "$PID_FILE")
    if ps -p "$PID" > /dev/null 2>&1; then
        ELAPSED=$(ps -o etime= -p "$PID" | xargs)
        echo "✅ Running (PID $PID, elapsed: $ELAPSED)"
    else
        echo "⚠️  Process $PID is NOT running (completed or crashed)"
        # Check exit status from log
        if [[ -f "$STDOUT_LOG" ]]; then
            LAST_LINE=$(tail -1 "$STDOUT_LOG" 2>/dev/null || echo "")
            echo "   Last stdout line: $LAST_LINE"
        fi
    fi
else
    echo "❌ No PID file found — run hasn't been started"
fi
echo ""

# --- 2. Find the latest evolution log ---
echo "--- Latest Log File ---"
LATEST_LOG=$(ls -t "$LOG_DIR"/overnight_evolution*.log "$LOG_DIR"/ga_run_*.log 2>/dev/null | head -1)
if [[ -z "$LATEST_LOG" ]]; then
    LATEST_LOG=$(ls -t "$LOG_DIR"/*.log 2>/dev/null | head -1)
fi

if [[ -n "$LATEST_LOG" ]]; then
    echo "Log: $LATEST_LOG"
    LOG_SIZE=$(du -h "$LATEST_LOG" | cut -f1)
    LOG_LINES=$(wc -l < "$LATEST_LOG")
    echo "Size: $LOG_SIZE ($LOG_LINES lines)"
else
    echo "❌ No log files found"
    exit 1
fi
echo ""

# --- 3. Current Generation ---
echo "--- Evolution Progress ---"
LAST_STEP=$(grep -n "\[STEP\]\|Generation\|=== Generation" "$LATEST_LOG" 2>/dev/null | tail -3)
if [[ -n "$LAST_STEP" ]]; then
    echo "$LAST_STEP"
else
    echo "No generation markers found yet"
fi
echo ""

# --- 4. Best Fitness ---
echo "--- Best Fitness So Far ---"
BEST=$(grep -i "\[NEW BEST\]\|new best\|Best fitness\|best_fitness" "$LATEST_LOG" 2>/dev/null | tail -5)
if [[ -n "$BEST" ]]; then
    echo "$BEST"
else
    echo "No best fitness updates found yet"
fi
echo ""

# --- 5. Diversity ---
echo "--- Diversity ---"
DIVERSITY=$(grep -i "\[DIVERSITY\]\|diversity\|genetic_diversity" "$LATEST_LOG" 2>/dev/null | tail -3)
if [[ -n "$DIVERSITY" ]]; then
    echo "$DIVERSITY"
else
    echo "No diversity info found"
fi
echo ""

# --- 6. Errors & Warnings ---
echo "--- Errors & Warnings ---"
ERROR_COUNT=$(grep -ci "ERROR\|Exception\|Traceback\|CRITICAL" "$LATEST_LOG" 2>/dev/null || echo "0")
WARN_COUNT=$(grep -ci "WARNING\|WARN" "$LATEST_LOG" 2>/dev/null || echo "0")
echo "Errors: $ERROR_COUNT | Warnings: $WARN_COUNT"

if [[ "$ERROR_COUNT" -gt 0 ]]; then
    echo ""
    echo "Recent errors:"
    grep -i "ERROR\|Exception\|Traceback" "$LATEST_LOG" 2>/dev/null | tail -5
fi
echo ""

# --- 7. Checkpoints ---
echo "--- Checkpoints ---"
CHECKPOINT_FILE="genetic_algorithm/data/checkpoints/latest_checkpoint.json"
if [[ -f "$CHECKPOINT_FILE" ]]; then
    CP_SIZE=$(du -h "$CHECKPOINT_FILE" | cut -f1)
    CP_TIME=$(stat -c '%y' "$CHECKPOINT_FILE" 2>/dev/null | cut -d'.' -f1)
    echo "Latest checkpoint: $CP_SIZE (saved at $CP_TIME)"
else
    echo "No checkpoint file found"
fi
echo ""

# --- 8. Output Files ---
echo "--- Output Files ---"
OUTPUT_DIR="genetic_algorithm/output"
if [[ -d "$OUTPUT_DIR" ]]; then
    STRATEGY_COUNT=$(find "$OUTPUT_DIR" -name "strategy_rank*.py" -newer "$PID_FILE" 2>/dev/null | wc -l || echo "0")
    SUMMARY_COUNT=$(find "$OUTPUT_DIR" -name "ga_summary_*.txt" -newer "$PID_FILE" 2>/dev/null | wc -l || echo "0")
    PLOT_COUNT=$(find "$OUTPUT_DIR" -name "*.png" -newer "$PID_FILE" 2>/dev/null | wc -l || echo "0")
    echo "New strategies: $STRATEGY_COUNT | Summaries: $SUMMARY_COUNT | Plots: $PLOT_COUNT"
fi
echo ""

# --- 9. Resource Usage ---
echo "--- Resource Usage ---"
if [[ -f "$PID_FILE" ]]; then
    PID=$(cat "$PID_FILE")
    if ps -p "$PID" > /dev/null 2>&1; then
        CPU_MEM=$(ps -p "$PID" -o %cpu,%mem,rss --no-headers 2>/dev/null || echo "N/A")
        echo "CPU% / MEM% / RSS(KB): $CPU_MEM"
        # Check children (parallel workers)
        CHILD_COUNT=$(pgrep -P "$PID" 2>/dev/null | wc -l || echo "0")
        echo "Worker processes: $CHILD_COUNT"
    fi
fi
echo ""

# --- 10. Holdout Penalty Impact ---
echo "--- Holdout Penalty Impact ---"
if [[ -n "$LATEST_LOG" ]]; then
    PENALTY_LINES=$(grep -i "holdout_penalty\|penalty_mult\|fitness.*penaliz" "$LATEST_LOG" 2>/dev/null | tail -5)
    if [[ -n "$PENALTY_LINES" ]]; then
        PENALIZED_COUNT=$(grep -ci "holdout_penalty" "$LATEST_LOG" 2>/dev/null || echo "0")
        echo "Holdout penalties applied: $PENALIZED_COUNT instances"
        echo "$PENALTY_LINES"
    else
        echo "No holdout penalty activity (disabled or no degradation)"
    fi
fi
echo ""

# --- 11. LLM Activity ---
echo "--- LLM Strategy Designer ---"
if [[ -n "$LATEST_LOG" ]]; then
    LLM_LINES=$(grep -i "\[LLM\]" "$LATEST_LOG" 2>/dev/null | tail -8)
    if [[ -n "$LLM_LINES" ]]; then
        echo "$LLM_LINES"
    else
        echo "No LLM activity (disabled or no api_key set)"
    fi
fi
echo ""

# --- 12. Walk-Forward Status ---
echo "--- Walk-Forward Status ---"
if [[ -n "$LATEST_LOG" ]]; then
    WF_LINES=$(grep -i "partial.credit\|wf_fallback\|walk.forward\|adaptive_min_trades\|walk-forward" "$LATEST_LOG" 2>/dev/null | tail -5)
    if [[ -n "$WF_LINES" ]]; then
        echo "$WF_LINES"
    else
        echo "No walk-forward activity (disabled or no log entries)"
    fi
fi
echo ""

# --- 13. Last 10 Log Lines ---
echo "--- Recent Log Tail ---"
tail -10 "$LATEST_LOG" 2>/dev/null
echo ""
echo "============================================================"
echo "  Run again: bash $0"
echo "============================================================"
