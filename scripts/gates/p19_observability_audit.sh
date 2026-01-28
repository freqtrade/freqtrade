#!/bin/bash
# P19 Observability Audit Gate
# Verifies:
# 1. State transition markers in Dry-Run logs.
# 2. Stacktrace presence for repo-owned exceptions.
# 3. Static analysis for exception logging compliance.

set -e
source .venv/bin/activate

echo ">>> Gate P19: Observability Audit..."

# 1. Static Scan
echo "1. Running Static Analysis (Exception Logging)..."
python3 scripts/ops/p19_scan_exc_logging.py

# 2. Traceback Verification
echo "2. Verifying Traceback Capture in Logs..."
LOG_OUT=$(python3 scripts/p19_raise_and_log.py 2>&1)

if echo "$LOG_OUT" | grep -q "p19_intentional_error_for_traceback_verification"; then
    echo "[OK] Found exception message"
else
    echo "[FAIL] Exception message missing"
    exit 1
fi

if echo "$LOG_OUT" | grep -q "Traceback (most recent call last)"; then
    echo "[OK] Found Traceback marker"
else
    echo "[FAIL] Traceback missing in logs"
    echo "Output was:"
    echo "$LOG_OUT"
    exit 1
fi

# 3. State Transition Marker in Dry-Run
echo "3. Verifying State Transition Marker..."
# Run a quick smoke strategy dry-run
DRY_RUN_LOG="/tmp/p19_dry_run.log"
# Remove if exists
rm -f "$DRY_RUN_LOG"

# Run for 20s to ensure startup (timeout 20s)
timeout 20s freqtrade trade --dry-run -c user_data/config_icicibreeze.json --userdir user_data -s IndiaEquitySmokeStrategy -vv > "$DRY_RUN_LOG" 2>&1 || true

if grep -q "Changing state to: RUNNING" "$DRY_RUN_LOG"; then
    echo "[OK] Found 'Changing state to: RUNNING'"
else
    echo "[FAIL] State transition marker missing"
    # echo last 50 lines for debugging
    tail -n 50 "$DRY_RUN_LOG"
    exit 1
fi

echo ">>> Gate P19: SUCCESS"
