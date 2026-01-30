#!/bin/bash
set -euo pipefail

# P34 Circuit Breaker Persistence Gate
# Verifies that CB state survives restarts via health persistence.

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
if [ -f ".venv/bin/python3" ]; then
    PYTHON=".venv/bin/python3"
else
    PYTHON="python3"
fi

HEALTH_FILE="user_data/generated/runtime/health.json"

if [ "$MODE" == "pos" ]; then
    echo ">>> Gate P34: Positive (Persistence)..."
    
    # 1. Reset Health file
    echo "{}" > "$HEALTH_FILE"
    
    # 2. Trigger CB Trip via Python
    $PYTHON <<'EOF'
from adapters.ccxt_shim import degraded_mode, health_snapshot
import time

msg = "P34 Test Failure"
guard = degraded_mode.DegradedModeGuard()
# Force failures to threshold
guard.failure_threshold = 3
guard.record_failure(Exception(msg))
guard.record_failure(Exception(msg))
guard.record_failure(Exception(msg))

# Verify persisted
state = health_snapshot.load()
cb = state.get("circuit_breaker", {})
assert cb.get("tripped") is True, "CB not persisted as tripped"
assert cb.get("failures") == 3, "Failures count wrong"
print("Triggered and Persisted successfully.")
EOF
    
    # 3. Simulate Restart (New Process)
    $PYTHON <<'EOF'
from adapters.ccxt_shim import degraded_mode
import time

# Init new guard
guard = degraded_mode.DegradedModeGuard()

# Should have loaded state
assert guard.failures == 3, f"Failures not restored: {guard.failures}"
# Check if Degraded (failures >= threshold)
assert guard.is_degraded() is True, "Guard not degraded after restart"
print("Restored successfully.")
EOF
    
    echo "P34_POS_PASS"

elif [ "$MODE" == "neg" ]; then
    echo ">>> Gate P34: Negative (Expiry/Cleanup)..."
    
    # 1. Manually inject EXPIRED trip state
    $PYTHON <<'EOF'
from adapters.ccxt_shim import health_snapshot
import time

# Inject state from 1 hour ago (default window usually 60s)
old_ts = time.time() - 3600 
health_snapshot.update("circuit_breaker", {
    "tripped": True,
    "tripped_at": old_ts,
    "failures": 3
})
print(f"Injected expired state: {old_ts}")
EOF

    # 2. Restart and Verify NOT degraded
    $PYTHON <<'EOF'
from adapters.ccxt_shim import degraded_mode
guard = degraded_mode.DegradedModeGuard()
# Should ignore expired state (logic updated to check time)
# Note: My implementation checked `now - tripped_at < failure_window * 10`
# 1 hour (3600) > 60 * 10 (600). So it should expire.
assert guard.failures == 0, f"Failures should be 0, got {guard.failures}"
assert guard.is_degraded() is False, "Guard should NOT be degraded"
print("Expired state ignored.")
EOF

    echo "P34_NEG_EXPIRY_PASS"

else
    echo "Unknown mode: $MODE"
    exit 1
fi
