#!/bin/bash
set -euo pipefail

# P33 Backup & Restore Gate
# Verifies data durability via backup/restore scripts.

MODE="pos"
for arg in "$@"; do
    case $arg in
        --mode=*)
            MODE="${arg#*=}"
            ;;
        pos|neg)
            MODE="$arg"
            ;;
    esac
done

USER_DATA="user_data"
TEST_FILE="$USER_DATA/generated/p33_marker"

# Ensure dirs exist
mkdir -p "$USER_DATA/generated"

if [ "$MODE" == "pos" ]; then
    echo ">>> Gate P33: Positive (Backup & Restore Flow)..."
    
    # 1. Setup State
    echo "P33_MARKER_DATA" > "$TEST_FILE"
    echo "Created marker: $TEST_FILE"
    
    # 2. Run Backup
    echo "Running Backup..."
    BACKUP_OUT=$(bash scripts/backup_state.sh)
    echo "$BACKUP_OUT"
    
    # Extract path from output (PATH=...)
    BACKUP_PATH=$(echo "$BACKUP_OUT" | grep "PATH=" | cut -d= -f2)
    
    if [ ! -f "$BACKUP_PATH" ]; then
        echo "[FAIL] Backup file not created: $BACKUP_PATH"
        exit 1
    fi
    
    # 3. Destroy State
    echo "Destroying marker..."
    rm "$TEST_FILE"
    if [ -f "$TEST_FILE" ]; then
        echo "[FAIL] Failed to delete marker."
        exit 1
    fi
    
    # 4. Run Restore
    echo "Running Restore..."
    bash scripts/restore_state.sh "$BACKUP_PATH"
    
    # 5. Verify State
    if [ -f "$TEST_FILE" ]; then
        CONTENT=$(cat "$TEST_FILE")
        if [ "$CONTENT" == "P33_MARKER_DATA" ]; then
            echo "[OK] Marker restored successfully."
            echo "P33_POS_PASS"
        else
            echo "[FAIL] Marker content mismatch: $CONTENT"
            exit 1
        fi
    else
        echo "[FAIL] Marker file missing after restore."
        exit 1
    fi

elif [ "$MODE" == "neg" ]; then
    echo ">>> Gate P33: Negative (Corrupt Archive)..."
    
    # 1. Setup State
    echo "ORIGINAL_STATE" > "$TEST_FILE"
    
    # 2. Create Fake Corrupt Archive
    mkdir -p "$USER_DATA/backups"
    CORRUPT_FILE="$USER_DATA/backups/corrupt.tar.gz"
    echo "NOT_A_GZIP" > "$CORRUPT_FILE"
    
    # 3. Attempt Restore (Should Fail)
    if bash scripts/restore_state.sh "$CORRUPT_FILE"; then
        echo "[FAIL] Restore succeeded on corrupt file!"
        exit 1
    else
        echo "[OK] Restore failed as expected."
    fi
    
    # 4. Verify State Untouched (Rollback check)
    if [ -f "$TEST_FILE" ]; then
        echo "[OK] Original state preserved."
        echo "P33_NEG_SAFE_FAIL"
    else
        echo "[FAIL] Original state destroyed on failure!"
        exit 1
    fi

else
    echo "Unknown mode: $MODE"
    exit 1
fi
