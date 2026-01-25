#!/bin/bash
# P06 Green Gate Wrapper
# Links scripts/green_gate.sh into the gate framework

GATE_ID="p06"
source scripts/gates/common.sh "$GATE_ID"

echo "Executing scripts/green_gate.sh..."
# Export OUT_DIR so green_gate.sh writes to the correct artifact dir
export OUT_DIR="$OUT_DIR"
bash scripts/green_gate.sh || finish_gate $?

finish_gate 0
