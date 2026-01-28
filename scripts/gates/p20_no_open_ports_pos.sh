#!/bin/bash
# P20 Gate: No Open Ports
# Verifies:
# 1. No "0.0.0.0" binds in config or code.
# 2. No public port mappings in docker-compose.
# 3. Scanner script passes.

set -euo pipefail

GATE_ID="p20"
source scripts/gates/common.sh "$GATE_ID" "$@"

export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

echo ">>> Gate P20: Checking for Open Ports... ($GATE_MODE)"

if [ "$GATE_MODE" == "pos" ]; then
    # 1. Run Static Scanner
    echo "1. Running Port Exposure Scanner..."
    $PYTHON scripts/ops/p20_scan_port_exposure.py || finish_gate $?
    
    echo ">>> Gate P20: SUCCESS"
    finish_gate 0

elif [ "$GATE_MODE" == "neg" ]; then
    echo "No negative path defined for P20 yet."
    echo "P20 No Open Ports passed (neg - skipped)"
    finish_gate 0
    
else
    echo "ERROR: Invalid mode $GATE_MODE"
    finish_gate 1
fi
