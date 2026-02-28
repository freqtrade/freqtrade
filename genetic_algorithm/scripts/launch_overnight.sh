#!/usr/bin/env bash
# ============================================================================
# launch_overnight.sh — Start the overnight GA evolution run
#
# Usage: bash genetic_algorithm/scripts/launch_overnight.sh
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GA_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
cd "$GA_ROOT"

CONFIG="genetic_algorithm/config/ga_config_overnight.yaml"
PID_FILE="genetic_algorithm/logs/overnight_pid.txt"
STDOUT_LOG="genetic_algorithm/logs/overnight_stdout.log"

echo "============================================================"
echo "  OVERNIGHT GA EVOLUTION — LAUNCH"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"
echo ""

# Check if already running
if [[ -f "$PID_FILE" ]]; then
    OLD_PID=$(cat "$PID_FILE")
    if ps -p "$OLD_PID" > /dev/null 2>&1; then
        echo "⚠️  A GA run is already active (PID $OLD_PID)"
        echo "   Use: bash genetic_algorithm/scripts/monitor_overnight.sh"
        echo "   Kill it first: kill $OLD_PID"
        exit 1
    fi
fi

# Verify config exists
if [[ ! -f "$CONFIG" ]]; then
    echo "❌ Config not found: $CONFIG"
    exit 1
fi

# Verify data exists
DATA_DIR="user_data/data/binance"
MISSING=0
for pair in BTC_USDT ETH_USDT SOL_USDT BNB_USDT XRP_USDT; do
    for tf in 5m 15m 1h; do
        if [[ ! -f "$DATA_DIR/${pair}-${tf}.feather" ]]; then
            echo "❌ Missing data: $DATA_DIR/${pair}-${tf}.feather"
            MISSING=1
        fi
    done
done
if [[ "$MISSING" -eq 1 ]]; then
    echo ""
    echo "Run data download first:"
    echo "  freqtrade download-data --exchange binance --pairs BTC/USDT ETH/USDT SOL/USDT BNB/USDT XRP/USDT --timeframes 5m 15m 1h --timerange 20230101-20260228 --trading-mode spot --data-format-ohlcv feather --prepend"
    exit 1
fi
echo "✅ All 15 data files present"

# Create output directories
mkdir -p genetic_algorithm/output/overnight_run/plots
mkdir -p genetic_algorithm/logs

# Show config summary
echo ""
echo "Configuration: $CONFIG"
echo "  Population: 120 | Generations: 40"
echo "  Pairs: BTC/USDT, ETH/USDT, SOL/USDT, BNB/USDT, XRP/USDT"
echo "  Timerange: 20230101-20260228 (~38 months)"
echo "  Regime-aware: ON (sma_adx, harmonic mean)"
echo "  Holdout validation: ON (15% out-of-sample)"
echo "  Monte Carlo: ON (50 permutations)"
echo "  Parsimony: ON | Feature importance: ON"
echo "  Adaptive mutation: ON | Fitness sharing: ON"
echo "  Checkpoints: Every 3 generations"
echo "  Estimated runtime: 4-5 hours"
echo ""

# Launch
echo "Launching evolution..."
nohup python genetic_algorithm/run_ga.py \
    --config "$CONFIG" \
    --visualize --no-interactive --yes \
    > "$STDOUT_LOG" 2>&1 &

PID=$!
echo "$PID" > "$PID_FILE"

echo "✅ Started! PID: $PID"
echo ""
echo "Monitor progress:"
echo "  bash genetic_algorithm/scripts/monitor_overnight.sh"
echo ""
echo "Morning analysis:"
echo "  bash genetic_algorithm/scripts/morning_analysis.sh"
echo ""
echo "View live log:"
echo "  tail -f $STDOUT_LOG"
echo ""
echo "Stop if needed:"
echo "  kill $PID"
echo ""
echo "Resume after crash:"
echo "  python genetic_algorithm/run_ga.py --config $CONFIG --resume --yes"
echo ""
echo "============================================================"
echo "  You can safely close this terminal now."
echo "============================================================"
