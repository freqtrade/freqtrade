#!/bin/bash
set -euo pipefail

# Scripts dir
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$(dirname "$DIR")"
USER_DATA="$ROOT_DIR/user_data"
BACKUP_DIR="$USER_DATA/backups"

mkdir -p "$BACKUP_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="$BACKUP_DIR/backup_$TIMESTAMP.tar.gz"

echo ">>> Creating Backup: $BACKUP_FILE"

# Tar command
# Excludes: logs, backups itself, plotting, hyperopt
tar -czf "$BACKUP_FILE" \
    -C "$ROOT_DIR" \
    --exclude="user_data/logs" \
    --exclude="user_data/backups" \
    --exclude="user_data/hyperopt_results" \
    --exclude="user_data/plot" \
    --exclude="user_data/*.sqlite-wal" \
    --exclude="user_data/*.sqlite-shm" \
    user_data

echo "Backup Created Successfully."
echo "PATH=$BACKUP_FILE"
