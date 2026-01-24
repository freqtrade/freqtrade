#!/bin/bash
# Common utility for acceptance gates

GATE_ID=$1
if [ -z "$GATE_ID" ]; then
    echo "ERROR: GATE_ID not provided to common.sh"
    exit 1
fi

ARTIFACT_DIR="user_data/generated/gates/$GATE_ID"
mkdir -p "$ARTIFACT_DIR"
GATE_LOG="$ARTIFACT_DIR/gate.log"

# Setup Python
export PYTHON=".venv/bin/python"
if [ ! -f "$PYTHON" ]; then
    PYTHON="python"
fi

# Function to write status.json and exit
finish_gate() {
    EXIT_CODE=$1
    STATUS="passed"
    if [ "$EXIT_CODE" -ne 0 ]; then
        STATUS="failed"
    fi
    TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    
    # Write status.json
    cat <<EOF > "$ARTIFACT_DIR/status.json"
{
    "gate_id": "$GATE_ID",
    "status": "$STATUS",
    "exit_code": $EXIT_CODE,
    "timestamp": "$TIMESTAMP"
}
EOF
    
    echo "Gate $GATE_ID $STATUS with exit code $EXIT_CODE"
    exit "$EXIT_CODE"
}

# Redirect all output to log file and console
exec > >(tee -a "$GATE_LOG") 2>&1

echo "=== Starting Gate: $GATE_ID ==="
echo "Artifact Directory: $ARTIFACT_DIR"
echo "Timestamp: $(date)"
