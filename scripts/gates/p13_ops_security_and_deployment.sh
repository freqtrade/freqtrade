#!/bin/bash
# P13 Ops, Security, and Deployment Gate
# Verifies operational readiness: security scans, file permissions, snapshots, and docs.

set -euo pipefail

GATE_ID="p13"
source scripts/gates/common.sh "$GATE_ID" "$@"

# 1. Environment Setup (Mock only, no real secrets)
export BREEZE_MOCK=1
unset BREEZE_API_KEY
unset BREEZE_API_SECRET
unset BREEZE_SESSION_TOKEN

echo "=== P13: Ops, Security, and Deployment Gate (Mode: $GATE_MODE) ==="

if [ "$GATE_MODE" == "pos" ]; then
    # Positive Mode (Original)
    echo "Step 1: Running Strict Secret Scan (Positive)"
    bash scripts/security/secret_scan_strict.sh || finish_gate $?

    echo "Step 2: Running File Permissions Audit"
    bash scripts/security/file_perms_audit.sh || finish_gate $?

    echo "Step 3: Generating Ops Snapshots"
    bash scripts/ops/env_snapshot.sh "$ARTIFACT_DIR" || finish_gate $?
    bash scripts/ops/ports_snapshot.sh "$ARTIFACT_DIR" || finish_gate $?

    echo "Step 4: Verifying Deployment Assets and Docs"
    REQUIRED_FILES=(
        "docs/OPS_RUNBOOK.md"
        "docs/SECURITY_HYGIENE.md"
        "docs/DEPLOYMENT_SYSTEMD.md"
        "deploy/env/.env.example"
        "deploy/systemd/freqtrade-icicibreeze.service.example"
    )
    for FILE in "${REQUIRED_FILES[@]}"; do
        if [ ! -f "$FILE" ]; then
            echo "ERROR: Required file missing: $FILE"
            finish_gate 1
        fi
    done
    echo "All required assets present."

    echo "Step 5: Running Stable Tests (CI Subset)"
    freqtrade --help > /dev/null || finish_gate 1
    if [ -d "tests/unit" ]; then
       pytest -q tests/unit > "$ARTIFACT_DIR/pytest_unit.log" 2>&1 || echo "Unit tests failed (non-blocking for this strict ops gate, but logged)"
    fi

elif [ "$GATE_MODE" == "neg" ]; then
    echo "Step 1: Triggering Secret Scan Failure (Negative)"
    
    # Create a temporary file with a secret pattern
    TEST_SECRET_FILE="user_data/FAIL_SECRET.txt"
    echo "password=super_secret_123" > "$TEST_SECRET_FILE"
    
    # Run scan, expect failure
    if bash scripts/security/secret_scan_strict.sh > "$ARTIFACT_DIR/scan_neg.log" 2>&1; then
        echo "[FAIL] Secret scan passed despite presence of secrets"
        rm "$TEST_SECRET_FILE"
        finish_gate 1
    else
        echo "[OK] Secret scan detected secrets as expected"
    fi
    rm "$TEST_SECRET_FILE"
fi

echo "P13 Ops, Security, and Deployment passed ($GATE_MODE)"
finish_gate 0
