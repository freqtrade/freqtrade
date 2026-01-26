#!/bin/bash
# File Permissions Audit
# Ensures sensitive files are not world-readable/writable

set -euo pipefail

EXIT_CODE=0
echo "Starting File Permissions Audit..."

# Sensitive paths to check if they exist
CHECKS=(
    "secrets"
    "user_data/secrets"
    "deploy/env/.env"
)

for PATH_TO_CHECK in "${CHECKS[@]}"; do
    if [ -e "$PATH_TO_CHECK" ]; then
        echo "Checking $PATH_TO_CHECK..."
        
        # Check world readable
        if [ -n "$(find "$PATH_TO_CHECK" -perm -o=r)" ]; then
            echo "FAIL: $PATH_TO_CHECK is world-readable!"
            EXIT_CODE=1
        fi
        
        # Check world writable
        if [ -n "$(find "$PATH_TO_CHECK" -perm -o=w)" ]; then
            echo "FAIL: $PATH_TO_CHECK is world-writable!"
            EXIT_CODE=1
        fi
        
        # Check group writable
        if [ -n "$(find "$PATH_TO_CHECK" -perm -g=w)" ]; then
            echo "FAIL: $PATH_TO_CHECK is group-writable!"
            EXIT_CODE=1
        fi
    else
        echo "Skip: $PATH_TO_CHECK (not found)"
    fi
done

# General sweep for any world-writable file in user_data
if [ -d "user_data" ]; then
    WORLD_WRITABLE=$(find user_data -type f -perm -o=w)
    if [ -n "$WORLD_WRITABLE" ]; then
        echo "FAIL: Found world-writable files in user_data:"
        echo "$WORLD_WRITABLE"
        EXIT_CODE=1
    fi
fi

if [ $EXIT_CODE -eq 0 ]; then
    echo "Permissions audit passed."
else
    echo "Permissions audit FAILED."
fi

exit $EXIT_CODE
