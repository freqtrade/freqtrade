#!/bin/bash
# Common utility for acceptance gates

GATE_ID=$1
if [ -z "$GATE_ID" ]; then
    echo "ERROR: GATE_ID not provided to common.sh"
    exit 1
fi

# Support OUT_DIR passed from orchestrator or default
OUT_DIR="${OUT_DIR:-user_data/generated/gates/$GATE_ID}"
mkdir -p "$OUT_DIR"
GATE_LOG="$OUT_DIR/gate.log"

# Setup Python
export PYTHON=".venv/bin/python"
if [ ! -f "$PYTHON" ]; then
    echo "ERROR: $PYTHON not found. Activate a venv first."
    exit 1
fi

# Function to write status.json and exit
finish_gate() {
    EXIT_CODE=$1
    STATUS="PASS"
    if [ "$EXIT_CODE" -ne 0 ]; then
        STATUS="FAIL"
    fi
    TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    
    # Write status.json
    cat <<EOF > "$OUT_DIR/status.json"
{
    "gate_id": "$GATE_ID",
    "status": "$STATUS",
    "exit_code": $EXIT_CODE,
    "timestamp": "$TIMESTAMP"
}
EOF
    
    echo "=== Gate $GATE_ID Result: $STATUS ==="
    echo "GATE_RESULT=$STATUS ARTIFACTS=$OUT_DIR"
    exit "$EXIT_CODE"
}

# Redirect all output to log file and console
exec > >(tee -a "$GATE_LOG") 2>&1

echo "=== Starting Gate: $GATE_ID ==="
echo "Artifact Directory: $OUT_DIR"
echo "Timestamp: $(date)"
