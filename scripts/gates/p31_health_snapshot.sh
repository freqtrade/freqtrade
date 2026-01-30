#!/bin/bash
set -e

# P31 Health Snapshot Gate
# Verifies that health snapshot logic persists state correctly and handles corruption.

MODE="pos"
for arg in "$@"; do
    case $arg in
        --mode=*)
            MODE="${arg#*=}"
            ;;
        pos|neg)
            MODE="$arg"
            ;;
    esac
done

HEALTH_FILE="user_data/generated/runtime/health.json"

if [ "$MODE" == "pos" ]; then
    echo ">>> Gate P31: Positive (Persistence & Validity)..."
    
    # Resolve binaries
    if [ -f ".venv/bin/pytest" ]; then
        PYTEST=".venv/bin/pytest"
        PYTHON=".venv/bin/python3"
    else
        PYTEST="pytest"
        PYTHON="python3"
    fi

    # 1. Run Unit Tests for detailed logic
    if $PYTEST tests/test_p31_health_snapshot.py; then
        echo "[OK] Unit Tests Passed."
    else
        echo "[FAIL] Unit Tests Failed."
        exit 1
    fi

    # 2. Integration Check (Simulate usage)
    echo ">>> Simulating integration usage..."
    $PYTHON -c "from adapters.ccxt_shim import health_snapshot; health_snapshot.update('policy_block'); print('Updated.')"
    
    if [ ! -f "$HEALTH_FILE" ]; then
        echo "[FAIL] Health file not created at $HEALTH_FILE"
        exit 1
    fi
    
    # 3. Check JSON validity and Content
    $PYTHON -c "import json; d=json.load(open('$HEALTH_FILE')); print('Counts:', d['counters']['policy_blocks']); assert d['counters']['policy_blocks'] >= 1"
    
    echo "P31_POS_PASS"

elif [ "$MODE" == "neg" ]; then
    echo ">>> Gate P31: Negative (Corruption Recovery)..."
    
     # Resolve binaries
    if [ -f ".venv/bin/python3" ]; then
        PYTHON=".venv/bin/python3"
    else
        PYTHON="python3"
    fi
    
    # 1. Corrupt the file
    echo "{ broken_json " > "$HEALTH_FILE"
    
    # 2. Run recovery test
    # Attempts to load -> should return empty dict or handle safely without crashing
    if $PYTHON -c "from adapters.ccxt_shim import health_snapshot; d=health_snapshot.load(); print('Recovered:', d); assert isinstance(d, dict)"; then
        echo "[OK] Recovered from corruption."
        echo "P31_NEG_EXPECTED_RECOVERY"
    else
        echo "[FAIL] Crashed on corruption."
        exit 1
    fi

else
    echo "Unknown mode: $MODE"
    exit 1
fi
