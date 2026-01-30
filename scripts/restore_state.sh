#!/bin/bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <backup_file_path>"
    exit 1
fi

BACKUP_FILE="$1"

# Scripts dir
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$(dirname "$DIR")"
USER_DATA="$ROOT_DIR/user_data"

if [ ! -f "$BACKUP_FILE" ]; then
    echo "Error: Backup file not found: $BACKUP_FILE"
    exit 1
fi

echo ">>> Restoring from: $BACKUP_FILE"

# Temp dir
RESTORE_TMP="$ROOT_DIR/user_data_restore_tmp"
rm -rf "$RESTORE_TMP"
mkdir -p "$RESTORE_TMP"

# Extract
echo "Extracting..."
if ! tar -xzf "$BACKUP_FILE" -C "$RESTORE_TMP"; then
    echo "Error: Extraction failed. Archive might be corrupt."
    rm -rf "$RESTORE_TMP"
    exit 1
fi

# Verify Structure (Expect 'user_data' inside tmp if we archived relative to root)
# My backup command used -C ROOT_DIR user_data, so it contains 'user_data/...' top level
if [ ! -d "$RESTORE_TMP/user_data" ]; then
    echo "Error: Invalid backup structure. 'user_data' not found in root of archive."
    rm -rf "$RESTORE_TMP"
    exit 1
fi

# Atomic Swap
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_OLD="$USER_DATA.old_$TIMESTAMP"

echo "Swapping directories..."
if [ -d "$USER_DATA" ]; then
    mv "$USER_DATA" "$BACKUP_OLD"
fi

mv "$RESTORE_TMP/user_data" "$USER_DATA"
rm -rf "$RESTORE_TMP"

echo "Restore Complete."
echo "Previous state saved to: $BACKUP_OLD"
