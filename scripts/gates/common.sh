#!/bin/bash
# Common utility for acceptance gates
set -euo pipefail

GATE_ID=$1
if [ -z "$GATE_ID" ]; then
    echo "ERROR: GATE_ID not provided to common.sh"
    exit 1
fi

# Support RUN_ID passed from orchestrator or default to local run
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
export RUN_ID

# Define artifact directory under the run ID
ARTIFACT_DIR="user_data/generated/accept_runs/$RUN_ID/gates/$GATE_ID"
mkdir -p "$ARTIFACT_DIR"
GATE_LOG="$ARTIFACT_DIR/gate.log"

# Define OUT_DIR for legacy compatibility/internal use
OUT_DIR="$ARTIFACT_DIR"

# Preflight checks
require_cmd() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "ERROR: Required command '$1' not found."
        exit 1
    fi
}

require_timeout() {
    require_cmd timeout
}

# Setup Python
export PYTHON=".venv/bin/python"
if [ ! -f "$PYTHON" ]; then
    echo "ERROR: $PYTHON not found. Activate a venv first."
    exit 1
fi

require_cmd jq
require_cmd "$PYTHON"

# Function to write status.json and exit
finish_gate() {
    EXIT_CODE=$1
    STATUS="PASS"
    if [ "$EXIT_CODE" -ne 0 ]; then
        STATUS="FAIL"
    fi
    TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    
    # Write rich status.json
    cat <<EOF > "$ARTIFACT_DIR/status.json"
{
    "gate_id": "$GATE_ID",
    "run_id": "$RUN_ID",
    "status": "$STATUS",
    "exit_code": $EXIT_CODE,
    "timestamp": "$TIMESTAMP",
    "artifact_dir": "$ARTIFACT_DIR",
    "gate_log": "$GATE_LOG"
}
EOF
    
    echo "=== Gate $GATE_ID Result: $STATUS ==="
    echo "GATE_RESULT=$STATUS ARTIFACTS=$ARTIFACT_DIR"
    exit "$EXIT_CODE"
}

# Redirect all output to log file and console
exec > >(tee -a "$GATE_LOG") 2>&1

echo "=== Starting Gate: $GATE_ID ==="
echo "Run ID: $RUN_ID"
echo "Artifact Directory: $ARTIFACT_DIR"
echo "Timestamp: $(date)"
