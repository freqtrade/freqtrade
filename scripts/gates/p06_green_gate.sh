#!/bin/bash
# P06 Green Gate Wrapper
# Links scripts/green_gate.sh into the gate framework
set -euo pipefail

GATE_ID="p06"
source scripts/gates/common.sh "$GATE_ID"

echo "Executing scripts/green_gate.sh..."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OUT_DIR="$ARTIFACT_DIR"
bash scripts/green_gate.sh || finish_gate $?

finish_gate 0
