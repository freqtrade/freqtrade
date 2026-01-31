#!/bin/bash
# Common utility for acceptance gates
set -euo pipefail

GATE_ID="$1"
if [ -z "$GATE_ID" ]; then
    echo "ERROR: GATE_ID not provided to common.sh"
    exit 1
fi
shift

# Default Mode
GATE_MODE="pos"

# Parse arguments passed to common.sh (which should be forwarded from the gate script)
for arg in "$@"; do
    case $arg in
        --mode=*)
        GATE_MODE="${arg#*=}"
        shift
        ;;
    esac
done

if [[ "$GATE_MODE" != "pos" && "$GATE_MODE" != "neg" ]]; then
    echo "ERROR: Invalid mode '$GATE_MODE'. Use 'pos' or 'neg'."
    exit 1
fi

# Support RUN_ID passed from orchestrator or default to local run
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
export RUN_ID

# Define artifact directory under the run ID, separated by mode
# e.g. .../gates/p14_pos or .../gates/p14_neg
ARTIFACT_DIR="generated/accept_runs/$RUN_ID/gates/${GATE_ID}_${GATE_MODE}"
mkdir -p "$ARTIFACT_DIR"
GATE_LOG="$ARTIFACT_DIR/gate.log"

# Define OUT_DIR for legacy compatibility/internal use
OUT_DIR="$ARTIFACT_DIR"

echo "=== Starting Gate: $GATE_ID (Mode: $GATE_MODE) ==="

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

# Setup Python and Path
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
PATH="$PWD/.venv/bin:$PATH"
export PATH

export PYTHON=".venv/bin/python"
if [ ! -f "$PYTHON" ]; then
    echo "ERROR: $PYTHON not found. Activate a venv first."
    exit 1
fi
export FREQTRADE=".venv/bin/freqtrade"
if [ ! -f "$FREQTRADE" ]; then
    echo "ERROR: $FREQTRADE not found. Ensure freqtrade is installed in the venv."
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
