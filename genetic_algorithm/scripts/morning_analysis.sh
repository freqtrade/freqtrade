#!/usr/bin/env bash
# ============================================================================
# morning_analysis.sh — Comprehensive post-run analysis for overnight GA run
#
# Run this in the morning to assess results and diagnose issues.
# Usage: bash genetic_algorithm/scripts/morning_analysis.sh
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GA_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
cd "$GA_ROOT"

PID_FILE="genetic_algorithm/logs/overnight_pid.txt"
STDOUT_LOG="genetic_algorithm/logs/overnight_stdout.log"
LOG_DIR="genetic_algorithm/logs"
OUTPUT_DIR="genetic_algorithm/output"
HOF_FILE="genetic_algorithm/data/hall_of_fame/hall_of_fame.json"
CHECKPOINT_FILE="genetic_algorithm/data/checkpoints/latest_checkpoint.json"

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║           OVERNIGHT GA RUN — MORNING ANALYSIS                 ║"
echo "║           $(date '+%Y-%m-%d %H:%M:%S')                              ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# ═══════════════════════════════════════════════════════════
# 1. COMPLETION STATUS
# ═══════════════════════════════════════════════════════════
echo "═══════════════════════════════════════════════════════════"
echo "  1. COMPLETION STATUS"
echo "═══════════════════════════════════════════════════════════"

if [[ -f "$PID_FILE" ]]; then
    PID=$(cat "$PID_FILE")
    if ps -p "$PID" > /dev/null 2>&1; then
        ELAPSED=$(ps -o etime= -p "$PID" | xargs)
        echo "⏳ STILL RUNNING (PID $PID, elapsed: $ELAPSED)"
        echo "   Consider waiting or checking monitor_overnight.sh"
    else
        echo "✅ Process completed (PID $PID)"
    fi
else
    echo "❌ No PID file found"
fi

# Check for completion markers in log
LATEST_LOG=$(ls -t "$LOG_DIR"/overnight_evolution*.log "$LOG_DIR"/ga_run_*.log 2>/dev/null | head -1 || true)
if [[ -n "$LATEST_LOG" ]]; then
    if grep -qi "evolution complete\|Evolution finished\|final report\|Summary Report" "$LATEST_LOG" 2>/dev/null; then
        echo "✅ Evolution completed successfully (found completion marker in log)"
    elif grep -qi "Traceback\|CRITICAL\|fatal" "$LATEST_LOG" 2>/dev/null; then
        echo "❌ CRASHED — found error in log"
    fi
fi

# Check stdout log for Python errors
if [[ -f "$STDOUT_LOG" ]]; then
    if grep -qi "Traceback\|Error\|Exception" "$STDOUT_LOG" 2>/dev/null; then
        echo ""
        echo "⚠️  Errors found in stdout log:"
        grep -A 3 "Traceback\|Error\|Exception" "$STDOUT_LOG" 2>/dev/null | tail -20
    fi
fi
echo ""

# ═══════════════════════════════════════════════════════════
# 2. SUMMARY REPORT
# ═══════════════════════════════════════════════════════════
echo "═══════════════════════════════════════════════════════════"
echo "  2. TOP STRATEGIES (Summary Report)"
echo "═══════════════════════════════════════════════════════════"

LATEST_SUMMARY=$(ls -t "$OUTPUT_DIR"/ga_summary_*.txt "$OUTPUT_DIR"/overnight_run/ga_summary_*.txt 2>/dev/null | head -1 || true)
if [[ -n "$LATEST_SUMMARY" ]]; then
    echo "Report: $LATEST_SUMMARY"
    echo "---"
    cat "$LATEST_SUMMARY"
else
    echo "❌ No summary report found"
    echo "   Evolution may not have completed."
fi
echo ""

# ═══════════════════════════════════════════════════════════
# 3. HOLDOUT VALIDATION
# ═══════════════════════════════════════════════════════════
echo "═══════════════════════════════════════════════════════════"
echo "  3. HOLDOUT VALIDATION (Out-of-Sample Test)"
echo "═══════════════════════════════════════════════════════════"

if [[ -n "$LATEST_LOG" ]]; then
    HOLDOUT=$(grep -i "\[HOLDOUT\]\|holdout\|out-of-sample\|degradation" "$LATEST_LOG" 2>/dev/null | grep -iv "config\|loading\|enabled" | tail -15)
    if [[ -n "$HOLDOUT" ]]; then
        echo "$HOLDOUT"
        echo ""
        echo "Interpretation:"
        echo "  - Degradation < 30%: ✅ Strategy is ROBUST"
        echo "  - Degradation 30-50%: ⚠️ Moderate overfitting"
        echo "  - Degradation > 50%: ❌ Severe overfitting"
    else
        echo "No holdout results found (may not have been enabled or run not complete)"
    fi
fi
echo ""

# ═══════════════════════════════════════════════════════════
# 4. MONTE CARLO ROBUSTNESS
# ═══════════════════════════════════════════════════════════
echo "═══════════════════════════════════════════════════════════"
echo "  4. MONTE CARLO ROBUSTNESS"
echo "═══════════════════════════════════════════════════════════"

if [[ -n "$LATEST_LOG" ]]; then
    MC=$(grep -i "monte.carlo\|robustness\|permutation\|bootstrap" "$LATEST_LOG" 2>/dev/null | grep -iv "config\|loading\|enabled" | tail -15)
    if [[ -n "$MC" ]]; then
        echo "$MC"
        echo ""
        echo "Interpretation:"
        echo "  - Robustness ≥ 0.80: ✅ Strategy is robust"
        echo "  - Robustness 0.60-0.79: ⚠️ Moderately robust"
        echo "  - Robustness < 0.60: ❌ Fragile strategy (likely overfit)"
    else
        echo "No Monte Carlo results found"
    fi
fi
echo ""

# ═══════════════════════════════════════════════════════════
# 5. EVOLUTION CONVERGENCE
# ═══════════════════════════════════════════════════════════
echo "═══════════════════════════════════════════════════════════"
echo "  5. EVOLUTION CONVERGENCE"
echo "═══════════════════════════════════════════════════════════"

if [[ -n "$LATEST_LOG" ]]; then
    # Best fitness progression
    echo "Best fitness progression:"
    grep -i "\[NEW BEST\]\|new best\|best fitness\|best_fitness" "$LATEST_LOG" 2>/dev/null | tail -10
    echo ""

    # Convergence info
    CONVERGENCE=$(grep -i "convergence\|patience\|early.stop\|no_improvement" "$LATEST_LOG" 2>/dev/null | tail -5)
    if [[ -n "$CONVERGENCE" ]]; then
        echo "Convergence:"
        echo "$CONVERGENCE"
    fi

    # Final generation reached
    LAST_GEN=$(grep -oP "Generation\s+\K\d+" "$LATEST_LOG" 2>/dev/null | tail -1 || echo "?")
    echo "Last generation reached: $LAST_GEN"
fi
echo ""

# ═══════════════════════════════════════════════════════════
# 6. DIVERSITY ANALYSIS
# ═══════════════════════════════════════════════════════════
echo "═══════════════════════════════════════════════════════════"
echo "  6. DIVERSITY ANALYSIS"
echo "═══════════════════════════════════════════════════════════"

if [[ -n "$LATEST_LOG" ]]; then
    DIVERSITY=$(grep -i "\[DIVERSITY\]\|diversity\|genetic_diversity\|sharing" "$LATEST_LOG" 2>/dev/null | tail -8)
    if [[ -n "$DIVERSITY" ]]; then
        echo "$DIVERSITY"
        echo ""
        echo "Interpretation:"
        echo "  - Diversity > 0.12: ✅ Good population diversity"
        echo "  - Diversity 0.05-0.12: ⚠️ Low diversity (may need more immigrants)"
        echo "  - Diversity < 0.05: ❌ Premature convergence"
    else
        echo "No diversity data found"
    fi
fi
echo ""

# ═══════════════════════════════════════════════════════════
# 7. FEATURE IMPORTANCE
# ═══════════════════════════════════════════════════════════
echo "═══════════════════════════════════════════════════════════"
echo "  7. FEATURE IMPORTANCE"
echo "═══════════════════════════════════════════════════════════"

if [[ -n "$LATEST_LOG" ]]; then
    FI=$(grep -i "\[FEATURE.IMPORTANCE\]\|feature.importance\|top indicator\|indicator.*score\|condition.*score" "$LATEST_LOG" 2>/dev/null | tail -20)
    if [[ -n "$FI" ]]; then
        echo "$FI"
    else
        echo "No feature importance data found"
    fi
fi
echo ""

# ═══════════════════════════════════════════════════════════
# 8. PARSIMONY (Strategy Simplification)
# ═══════════════════════════════════════════════════════════
echo "═══════════════════════════════════════════════════════════"
echo "  8. PARSIMONY (Strategy Simplification)"
echo "═══════════════════════════════════════════════════════════"

if [[ -n "$LATEST_LOG" ]]; then
    PARSIMONY=$(grep -i "\[PARSIMONY\]\|parsimony\|simplified\|complexity.*reduced" "$LATEST_LOG" 2>/dev/null | tail -10)
    if [[ -n "$PARSIMONY" ]]; then
        echo "$PARSIMONY"
    else
        echo "No parsimony data found"
    fi
fi
echo ""

# ═══════════════════════════════════════════════════════════
# 9. REGIME-AWARE PERFORMANCE
# ═══════════════════════════════════════════════════════════
echo "═══════════════════════════════════════════════════════════"
echo "  9. REGIME-AWARE PERFORMANCE"
echo "═══════════════════════════════════════════════════════════"

if [[ -n "$LATEST_LOG" ]]; then
    REGIME=$(grep -i "regime\|bullish\|bearish\|sideways" "$LATEST_LOG" 2>/dev/null | grep -iv "config\|loading\|enabled\|method" | tail -15)
    if [[ -n "$REGIME" ]]; then
        echo "$REGIME"
    else
        echo "No regime performance data found"
    fi
fi
echo ""

# ═══════════════════════════════════════════════════════════
# 10. HALL OF FAME
# ═══════════════════════════════════════════════════════════
echo "═══════════════════════════════════════════════════════════"
echo "  10. HALL OF FAME"
echo "═══════════════════════════════════════════════════════════"

if [[ -f "$HOF_FILE" ]]; then
    ENTRY_COUNT=$(python3 -c "import json; data=json.load(open('$HOF_FILE')); print(len(data.get('strategies', data) if isinstance(data, dict) else data))" 2>/dev/null || echo "?")
    echo "Total hall of fame entries: $ENTRY_COUNT"
    echo ""
    # Show top 5 by fitness
    python3 -c "
import json
with open('$HOF_FILE') as f:
    data = json.load(f)
strategies = data.get('strategies', data) if isinstance(data, dict) else data
if isinstance(strategies, list):
    strategies.sort(key=lambda x: x.get('fitness', 0), reverse=True)
    print(f'Top 5 Hall of Fame entries:')
    print(f'{\"Rank\":<6}{\"Fitness\":<10}{\"Profit\":<12}{\"Sharpe\":<10}{\"Win%\":<10}{\"Trades\":<8}')
    print('-' * 56)
    for i, s in enumerate(strategies[:5]):
        m = s.get('metrics', s)
        print(f'{i+1:<6}{s.get(\"fitness\",0):<10.4f}{m.get(\"profit_percent\",m.get(\"profit\",0)):<12.2f}{m.get(\"sharpe_ratio\",0):<10.2f}{m.get(\"win_rate\",0)*100:<10.1f}{m.get(\"total_trades\",m.get(\"num_trades\",0)):<8}')
" 2>/dev/null || echo "Could not parse hall of fame"
else
    echo "No hall of fame file found"
fi
echo ""

# ═══════════════════════════════════════════════════════════
# 11. ERRORS & WARNINGS
# ═══════════════════════════════════════════════════════════
echo "═══════════════════════════════════════════════════════════"
echo "  11. ERRORS & WARNINGS"
echo "═══════════════════════════════════════════════════════════"

if [[ -n "$LATEST_LOG" ]]; then
    ERROR_COUNT=$(grep -ci "ERROR" "$LATEST_LOG" 2>/dev/null || echo "0")
    WARN_COUNT=$(grep -ci "WARNING" "$LATEST_LOG" 2>/dev/null || echo "0")
    EXCEPTION_COUNT=$(grep -ci "Exception\|Traceback" "$LATEST_LOG" 2>/dev/null || echo "0")

    echo "Errors: $ERROR_COUNT | Warnings: $WARN_COUNT | Exceptions: $EXCEPTION_COUNT"

    if [[ "$ERROR_COUNT" -gt 0 || "$EXCEPTION_COUNT" -gt 0 ]]; then
        echo ""
        echo "Unique error messages:"
        grep -i "ERROR\|Exception" "$LATEST_LOG" 2>/dev/null | sed 's/.*ERROR - //' | sort -u | head -10
    fi

    if [[ "$WARN_COUNT" -gt 0 ]]; then
        echo ""
        echo "Unique warning messages (top 10):"
        grep -i "WARNING" "$LATEST_LOG" 2>/dev/null | sed 's/.*WARNING - //' | sort | uniq -c | sort -rn | head -10
    fi
fi
echo ""

# ═══════════════════════════════════════════════════════════
# 12. ADAPTIVE MUTATION HISTORY
# ═══════════════════════════════════════════════════════════
echo "═══════════════════════════════════════════════════════════"
echo "  12. ADAPTIVE MUTATION"
echo "═══════════════════════════════════════════════════════════"

if [[ -n "$LATEST_LOG" ]]; then
    MUTATION=$(grep -i "mutation.*rate\|adaptive.*mutation\|mutation.*increased\|mutation.*reset" "$LATEST_LOG" 2>/dev/null | tail -10)
    if [[ -n "$MUTATION" ]]; then
        echo "$MUTATION"
    else
        echo "No adaptive mutation data found"
    fi
fi
echo ""

# ═══════════════════════════════════════════════════════════
# 13. CHECKPOINT / RESUME CAPABILITY
# ═══════════════════════════════════════════════════════════
echo "═══════════════════════════════════════════════════════════"
echo "  13. CHECKPOINT STATUS"
echo "═══════════════════════════════════════════════════════════"

if [[ -f "$CHECKPOINT_FILE" ]]; then
    CP_SIZE=$(du -h "$CHECKPOINT_FILE" | cut -f1)
    CP_TIME=$(stat -c '%y' "$CHECKPOINT_FILE" 2>/dev/null | cut -d'.' -f1)
    CP_GEN=$(python3 -c "import json; print(json.load(open('$CHECKPOINT_FILE')).get('generation', '?'))" 2>/dev/null || echo "?")
    echo "Latest checkpoint: Generation $CP_GEN ($CP_SIZE, saved $CP_TIME)"
    echo "Resume command:"
    echo "  python genetic_algorithm/run_ga.py --config genetic_algorithm/config/ga_config_overnight.yaml --resume --yes"
else
    echo "No checkpoint file found"
fi
echo ""

# ═══════════════════════════════════════════════════════════
# 14. GENERATED STRATEGY FILES
# ═══════════════════════════════════════════════════════════
echo "═══════════════════════════════════════════════════════════"
echo "  14. GENERATED STRATEGIES"
echo "═══════════════════════════════════════════════════════════"

STRAT_FILES=$(find "$OUTPUT_DIR" -name "strategy_rank*.py" -newer "$LOG_DIR" 2>/dev/null | sort)
if [[ -n "$STRAT_FILES" ]]; then
    echo "$STRAT_FILES"
else
    echo "No strategy files found in $OUTPUT_DIR"
    # Check all output dirs
    ALL_STRATS=$(find "$OUTPUT_DIR" -name "strategy_rank*.py" 2>/dev/null | sort | tail -10)
    if [[ -n "$ALL_STRATS" ]]; then
        echo "Previous strategy files:"
        echo "$ALL_STRATS"
    fi
fi
echo ""

# ═══════════════════════════════════════════════════════════
# 15. VISUALIZATION OUTPUTS
# ═══════════════════════════════════════════════════════════
echo "═══════════════════════════════════════════════════════════"
echo "  15. PLOTS & VISUALIZATIONS"
echo "═══════════════════════════════════════════════════════════"

PLOTS=$(find "$OUTPUT_DIR" -name "*.png" 2>/dev/null | sort)
if [[ -n "$PLOTS" ]]; then
    echo "Generated plots:"
    echo "$PLOTS"
else
    echo "No plot files found"
fi
echo ""

# ═══════════════════════════════════════════════════════════
# 16. WALK-FORWARD PARTIAL CREDIT
# ═══════════════════════════════════════════════════════════
echo "═══════════════════════════════════════════════════════════"
echo "  16. WALK-FORWARD PARTIAL CREDIT"
echo "═══════════════════════════════════════════════════════════"

if [[ -n "$LATEST_LOG" ]]; then
    WF=$(grep -i "partial.credit\|adaptive_min_trades\|wf_fallback\|train_trade_credit\|proportional" "$LATEST_LOG" 2>/dev/null | tail -10)
    if [[ -n "$WF" ]]; then
        echo "$WF"
        echo ""
        FALLBACK_COUNT=$(grep -ci "wf_fallback" "$LATEST_LOG" 2>/dev/null || echo "0")
        echo "Walk-forward fallbacks (all windows failed): $FALLBACK_COUNT"
    else
        echo "No walk-forward partial credit data found"
    fi
fi
echo ""

# ═══════════════════════════════════════════════════════════
# 17. LLM STRATEGY DESIGNER
# ═══════════════════════════════════════════════════════════
echo "═══════════════════════════════════════════════════════════"
echo "  17. LLM STRATEGY DESIGNER"
echo "═══════════════════════════════════════════════════════════"

if [[ -n "$LATEST_LOG" ]]; then
    LLM=$(grep -i "\[LLM\]" "$LATEST_LOG" 2>/dev/null | tail -15)
    if [[ -n "$LLM" ]]; then
        echo "$LLM"
        echo ""
        LLM_SEEDS=$(grep -ci "LLM.*seed" "$LATEST_LOG" 2>/dev/null || echo "0")
        LLM_IMMIGRANTS=$(grep -ci "LLM.*immigrant" "$LATEST_LOG" 2>/dev/null || echo "0")
        echo "LLM seeds: $LLM_SEEDS | LLM immigrants: $LLM_IMMIGRANTS"
    else
        echo "No LLM activity in this run (disabled or no API key)"
    fi
fi
echo ""

# ═══════════════════════════════════════════════════════════
# 18. OVERFIT COMPOSITE SCORING
# ═══════════════════════════════════════════════════════════
echo "═══════════════════════════════════════════════════════════"
echo "  18. OVERFIT COMPOSITE SCORING"
echo "═══════════════════════════════════════════════════════════"

if [[ -n "$LATEST_LOG" ]]; then
    OVERFIT=$(grep -i "composite\|OVERFIT\|SAFE\|WARNING.*overfit\|weighted.*score\|holdout_penalty" "$LATEST_LOG" 2>/dev/null | grep -iv "config\|loading\|enabled" | tail -15)
    if [[ -n "$OVERFIT" ]]; then
        echo "$OVERFIT"
        echo ""
        OVERFIT_COUNT=$(grep -ci "classification.*OVERFIT\|→ OVERFIT\|: OVERFIT" "$LATEST_LOG" 2>/dev/null || echo "0")
        SAFE_COUNT=$(grep -ci "classification.*SAFE\|→ SAFE\|: SAFE" "$LATEST_LOG" 2>/dev/null || echo "0")
        echo "Classified: $SAFE_COUNT SAFE | $OVERFIT_COUNT OVERFIT"
    else
        echo "No composite overfit scoring data found"
    fi
fi
echo ""

# ═══════════════════════════════════════════════════════════
# VERDICT
# ═══════════════════════════════════════════════════════════
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  QUICK VERDICT                                                ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

if [[ -n "$LATEST_SUMMARY" ]]; then
    echo "✅ Strategies generated — check summary report above"
else
    echo "⚠️  No summary report — evolution may not have completed"
fi

if [[ -n "$LATEST_LOG" ]]; then
    ERROR_COUNT=$(grep -ci "ERROR\|Exception\|Traceback" "$LATEST_LOG" 2>/dev/null || echo "0")
    if [[ "$ERROR_COUNT" -eq 0 ]]; then
        echo "✅ No errors in evolution log"
    else
        echo "❌ $ERROR_COUNT errors found — check section 11 above"
    fi
fi

echo ""
echo "Next steps:"
echo "  1. Review top strategies in section 2"
echo "  2. Check holdout degradation in section 3 (< 30% = robust)"
echo "  3. Check Monte Carlo robustness in section 4 (≥ 0.80 = robust)"
echo "  4. If errors occurred, check section 11 for details"
echo "  5. To continue evolution: python genetic_algorithm/run_ga.py --resume --config genetic_algorithm/config/ga_config_overnight.yaml --yes"
echo "  6. To visualize a strategy: python genetic_algorithm/visualize_strategy.py <strategy_file>"
echo ""
