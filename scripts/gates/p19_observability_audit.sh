#!/bin/bash
# P19 Observability Audit Gate
# Verifies:
# 1. State transition markers in Dry-Run logs.
# 2. Stacktrace presence for repo-owned exceptions.
# 3. Static analysis for exception logging compliance.

set -euo pipefail

GATE_ID="p19"
source scripts/gates/common.sh "$GATE_ID" "$@"

export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

echo ">>> Gate P19: Observability Audit... ($GATE_MODE)"

if [ "$GATE_MODE" == "pos" ]; then
    # 1. Static Scan
    echo "1. Running Static Analysis (Exception Logging)..."
    $PYTHON scripts/ops/p19_scan_exc_logging.py || finish_gate $?

    # 2. Traceback Verification
    echo "2. Verifying Traceback Capture in Logs..."
    LOG_OUT=$($PYTHON scripts/p19_raise_and_log.py 2>&1)

    if echo "$LOG_OUT" | grep -q "p19_intentional_error_for_traceback_verification"; then
        echo "[OK] Found exception message"
    else
        echo "[FAIL] Exception message missing"
        finish_gate 1
    fi

    if echo "$LOG_OUT" | grep -q "Traceback (most recent call last)"; then
        echo "[OK] Found Traceback marker"
    else
        echo "[FAIL] Traceback missing in logs"
        echo "Output was:"
        echo "$LOG_OUT"
        finish_gate 1
    fi

    # 3. State Transition Marker in Dry-Run
    echo "3. Verifying State Transition Marker..."
    # Run a quick smoke strategy dry-run
    DRY_RUN_LOG="$ARTIFACT_DIR/p19_dry_run.log"
    # Remove if exists
    rm -f "$DRY_RUN_LOG"

    # Run for 20s to ensure startup (timeout 20s)
    # We ignore the exit code because we expect to kill it or it might time out
    timeout 20s "$FREQTRADE" trade --dry-run -c user_data/config_icicibreeze.json --userdir user_data -s IndiaEquitySmokeStrategy -vv > "$DRY_RUN_LOG" 2>&1 || true

    if grep -q "Changing state to: RUNNING" "$DRY_RUN_LOG"; then
        echo "[OK] Found 'Changing state to: RUNNING'"
    else
        echo "[FAIL] State transition marker missing"
        # echo last 50 lines for debugging
        tail -n 50 "$DRY_RUN_LOG"
        finish_gate 1
    fi

    echo ">>> Gate P19: SUCCESS"
    finish_gate 0

elif [ "$GATE_MODE" == "neg" ]; then
    echo "No negative path defined for P19 yet."
    echo "P19 Observability Audit passed (neg - skipped)"
    finish_gate 0

else
    echo "ERROR: Invalid mode $GATE_MODE"
    finish_gate 1
fi
