#!/bin/bash
# Ports Snapshot
# Captures listening ports

set -euo pipefail

OUT_DIR="${1:-/tmp}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUT_FILE="$OUT_DIR/ports_snapshot_$TIMESTAMP.txt"

echo "Capturing ports snapshot to $OUT_FILE..."

{
    echo "=== Listening Ports ==="
    # try ss, fallback to netstat
    if command -v ss >/dev/null; then
        ss -lntup
    elif command -v netstat >/dev/null; then
        netstat -lntup
    else
        echo "Neither ss nor netstat found."
    fi
} > "$OUT_FILE"

echo "Snapshot created."
