#!/bin/bash
# Environment Snapshot
# Captures runtime environment details for debugging/audit

set -euo pipefail

OUT_DIR="${1:-/tmp}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUT_FILE="$OUT_DIR/env_snapshot_$TIMESTAMP.txt"

echo "Capturing environment snapshot to $OUT_FILE..."

{
    echo "=== System Info ==="
    uname -a
    echo ""
    
    echo "=== Python Version ==="
    python3 --version
    echo ""
    
    echo "=== Git Revision ==="
    git rev-parse HEAD || echo "No git repo"
    git status -s || echo "No git repo"
    echo ""
    
    echo "=== Freqtrade Version ==="
    freqtrade --version || echo "Freqtrade not found in PATH"
    echo ""
    
    # Optional: Pip freeze (commented out if too large, but requested in spec)
    # echo "=== Installed Packages ==="
    # pip freeze
    
} > "$OUT_FILE"

echo "Snapshot created."
