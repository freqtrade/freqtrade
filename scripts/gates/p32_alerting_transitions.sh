#!/bin/bash
set -e

# P32 Alerting Transitions Gate
# Verifies that alerts are triggered and suppressed correctly.

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

# Resolve binaries
if [ -f ".venv/bin/pytest" ]; then
    PYTEST=".venv/bin/pytest"
    PYTHON=".venv/bin/python3"
else
    PYTEST="pytest"
    PYTHON="python3"
fi

if [ "$MODE" == "pos" ]; then
    echo ">>> Gate P32: Positive (Trigger & Log Format)..."
    
    # 1. Run Unit Tests
    if $PYTEST tests/test_p32_alerting_transitions.py; then
        echo "[OK] Unit Tests Passed."
    else
        echo "[FAIL] Unit Tests Failed."
        exit 1
    fi
    
    # 2. Integration Check
    # Verify that trigger() function works
    $PYTHON -c "from adapters.ccxt_shim import alerts; alerts.trigger('GATE_POS', 'Integration Test Alert')"
    echo "P32_POS_PASS"

elif [ "$MODE" == "neg" ]; then
    echo ">>> Gate P32: Negative (Suppression)..."
    
    # 1. Trigger multiple alerts and verify suppression logic via python script
    $PYTHON <<EOF
from adapters.ccxt_shim import alerts
import logging
from unittest.mock import MagicMock

# Setup Mock Logger
mock_logger = MagicMock()
alerts.logger = mock_logger

# Reset
alerts.AlertManager._instance = None
mgr = alerts.AlertManager.get_instance()
mgr._suppression_window = 1000 # Long window

# Fire 1
mgr.alert("NEG_CAT", "Msg 1")
# Fire 2
mgr.alert("NEG_CAT", "Msg 2")

assert mock_logger.error.call_count == 1, f"Expected 1 alert, got {mock_logger.error.call_count}"
print("P32_NEG_SUPPRESSION_VERIFIED")
EOF

else
    echo "Unknown mode: $MODE"
    exit 1
fi
