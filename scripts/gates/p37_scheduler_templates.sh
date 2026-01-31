#!/bin/bash
# P37: Scheduler Templates & Locking Gate
# Verifies locking utility and systemd templates.

set -euo pipefail

GATE_ID="p37"
source scripts/gates/common.sh "$GATE_ID" "$@"

export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
WITH_LOCK="scripts/ops/with_lock.py"
SYSTEMD_DIR="docs/ops/systemd"

if [ "$GATE_MODE" == "pos" ]; then
    echo ">>> Gate P37: Positive (Locking & Templates)..."
    
    # 1. Unit Tests
    echo "1. Running Unit Tests..."
    $PYTHON -m pytest -q tests/test_p37_with_lock.py || finish_gate 1

    # 2. Template Verification
    echo "2. Verifying Systemd Templates..."
    REQUIRED_FILES=(
        "p25_security_master.service"
        "p25_security_master.timer"
        "p33_backup.service"
        "p33_backup.timer"
    )
    
    for f in "${REQUIRED_FILES[@]}"; do
        if [ ! -f "$SYSTEMD_DIR/$f" ]; then
            echo "[FAIL] Missing template: $f"
            finish_gate 1
        fi
    done
    
    # Check assertions
    if grep -q "ExecStart" "$SYSTEMD_DIR"/*.service; then
        echo "[OK] Services contain ExecStart"
    else
        echo "[FAIL] ExecStart missing"
        finish_gate 1
    fi
    
    if grep -q "OnCalendar" "$SYSTEMD_DIR"/*.timer; then
        echo "[OK] Timers contain OnCalendar"
    else
        echo "[FAIL] OnCalendar missing"
        finish_gate 1
    fi

    echo "P37_POS_PASS"
    finish_gate 0

elif [ "$GATE_MODE" == "neg" ]; then
    echo ">>> Gate P37: Negative (Lock Contention)..."
    
    LOCK_FILE="/tmp/p37_test.lock"
    
    # 1. Start holder
    $PYTHON "$WITH_LOCK" --lock "$LOCK_FILE" --cmd "sleep 3" &
    PID=$!
    
    # Wait for start
    sleep 1
    
    # 2. Try to acquire
    echo "Attempting to acquire held lock..."
    set +e
    $PYTHON "$WITH_LOCK" --lock "$LOCK_FILE" --cmd "echo should_fail"
    RET=$?
    set -e
    
    wait $PID
    
    if [ "$RET" -eq 1 ]; then
        echo "[OK] Lock acquisition failed as expected (Exit Code 1)"
        echo "P37_NEG_EXPECTED_BLOCK"
        finish_gate 0
    else
        echo "[FAIL] Lock acquisition succeeded unexpected (Exit Code $RET)"
        finish_gate 1
    fi

else
    echo "ERROR: Invalid mode"
    finish_gate 1
fi
