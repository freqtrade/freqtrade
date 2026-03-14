#!/usr/bin/env bash
# Wait for A1 to finish, then run A2 and A3 sequentially
set -uo pipefail
cd /home/kali/trading/freqtradeForkGA

echo "Waiting for A1 (PID $1) to finish..."
while kill -0 "$1" 2>/dev/null; do
    sleep 30
    echo "  $(date '+%H:%M:%S') — A1 still running..."
done

echo "A1 finished at $(date '+%Y-%m-%d %H:%M:%S')"
echo "Sleeping 10s before A2..."
sleep 10

echo "Starting A2 (5m)..."
./run_phase1_sequential.sh a2
echo "A2 finished at $(date '+%Y-%m-%d %H:%M:%S')"
sleep 10

echo "Starting A3 (1h)..."
./run_phase1_sequential.sh a3
echo "A3 finished at $(date '+%Y-%m-%d %H:%M:%S')"

echo "ALL PHASE 1 RUNS COMPLETE at $(date '+%Y-%m-%d %H:%M:%S')"
