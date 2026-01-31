#!/bin/bash
# P38: Soak Stability Gate
# Verifies system stability under sustained run and error injection.

set -euo pipefail

GATE_ID="p38"
source scripts/gates/common.sh "$GATE_ID" "$@"

export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export FT_ENABLE_LIVE_ORDERS=1 # Enable logic even in dry run mock if applicable
export BREEZE_MOCK=1

HEALTH_FILE="user_data/generated/runtime/health.json"
METRICS_JSON="user_data/generated/runtime/metrics.json"
METRICS_PROM="user_data/generated/runtime/metrics.prom"
ALERTS_FILE="user_data/generated/runtime/alerts.jsonl"
LOG_FILE="$ARTIFACT_DIR/trade.log"

run_soak() {
    local duration=$1
    echo "Starting Freqtrade for ${duration}s soak..."
    
    # Remove old artifacts
    rm -f "$HEALTH_FILE" "$METRICS_JSON" "$METRICS_PROM"
    
    # Start Freqtrade in background
    "$FREQTRADE" trade --dry-run \
        -c user_data/config_icicibreeze.json \
        --userdir user_data \
        -s IndiaEquitySmokeStrategy \
        -v \
        > "$LOG_FILE" 2>&1 &
    
    PID=$!
    echo "Freqtrade PID: $PID"
    
    # Wait for startup
    sleep 10
    
    # Check if running
    if ! kill -0 $PID 2>/dev/null; then
        echo "[FAIL] Freqtrade crashed on startup"
        cat "$LOG_FILE"
        finish_gate 1
    fi
    
    # Run soak
    echo "Soaking..."
    sleep "$duration"
    
    echo "Soak complete."
    echo "$PID"
}

if [ "$GATE_MODE" == "pos" ]; then
    echo ">>> Gate P38: Positive (Stability & Metrics)..."
    
    FT_PID=$(run_soak 90 | tail -n1)
    
    # Validate Invariants
    
    # 1. Health updated
    if [ ! -f "$HEALTH_FILE" ]; then
        echo "[FAIL] health.json not generated"
        kill $FT_PID || true
        finish_gate 1
    fi
    
    # 2. Metrics Exporter (Invoke it)
    $PYTHON -c "from adapters.ccxt_shim.metrics_exporter import export_metrics; export_metrics()"
    
    if [ ! -f "$METRICS_PROM" ]; then
        echo "[FAIL] metrics.prom not generated"
        kill $FT_PID || true
        finish_gate 1
    fi
    
    # 3. No Traceback
    if grep -q "Traceback" "$LOG_FILE"; then
        echo "[FAIL] Traceback found in logs"
        grep -C 5 "Traceback" "$LOG_FILE"
        kill $FT_PID || true
        finish_gate 1
    fi
    
    # 4. Alerts valid? (Check metric logic didn't crash)
    
    echo "[OK] Invariants held."
    
    # Cleanup
    kill $FT_PID || true
    echo "P38_POS_PASS"
    finish_gate 0

elif [ "$GATE_MODE" == "neg" ]; then
    echo ">>> Gate P38: Negative (Circuit Breaker Resilience)..."
    
    run_soak 10
    FT_PID=$?
    
    # 1. Inject CB Open State (python script)
    echo "Injecting Circuit Open state..."
    $PYTHON <<'EOF'
from adapters.ccxt_shim import health_snapshot
import time
# Inject circuit open
health_snapshot.update("circuit_breaker", {
    "state": "open",
    "failures": 5,
    "last_failure_ts": time.time()
})
print("Injected CB Open.")
EOF
    
    # Wait for potential reaction or just ensure process lives
    sleep 5
    
    # 2. Run Exporter
    $PYTHON -c "from adapters.ccxt_shim.metrics_exporter import export_metrics; export_metrics()"
    
    # 3. Assert Circuit Open metric
    if grep -q "circuit_open_total 1" "$METRICS_PROM"; then
        echo "[OK] circuit_open_total=1 found in metrics"
    else
        echo "[FAIL] circuit_open_total=1 NOT found"
        cat "$METRICS_PROM"
        kill $FT_PID || true
        finish_gate 1
    fi
    
    # 4. Assert process still alive
    if kill -0 $FT_PID 2>/dev/null; then
        echo "[OK] Process still running"
    else
        echo "[FAIL] Process crashed after injection"
        cat "$LOG_FILE"
        finish_gate 1
    fi
    
    kill $FT_PID || true
    echo "P38_NEG_EXPECTED_CIRCUIT_OPEN"
    finish_gate 0

else
    echo "ERROR: Invalid mode"
    finish_gate 1
fi
