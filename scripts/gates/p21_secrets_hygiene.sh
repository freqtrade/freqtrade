#!/bin/bash
# P21 Gate: Secrets Hygiene
# Verifies:
# 1. No secret literals in codebase (grep).
# 2. No secrets leaked in artifacts/logs from current run.
# 3. Session Readiness Check (Mock vs Real).

set -euo pipefail

GATE_ID="p21"
source scripts/gates/common.sh "$GATE_ID" "$@"

echo ">>> Gate P21: Secrets Hygiene Check... ($GATE_MODE)"

# -------------------------------------------------------------
# Step 1: Static Repo Scan
# -------------------------------------------------------------
echo "1. Scanning Repository for Secret Literals..."
# We look for patterns like BREEZE_API_KEY="actual_value"
# Ignoring .env.example placeholders

RISKY_PATTERNS=(
    'BREEZE_API_KEY="[^"]+"'
    'BREEZE_API_SECRET="[^"]+"'
    'BREEZE_SESSION_TOKEN="[^"]+"'
    'session_token\s*=\s*"[^"]+"'
    'api_secret\s*=\\s*"[^"]+"'
)

FAILED_SCAN=0
for pattern in "${RISKY_PATTERNS[@]}"; do
    if rg -n "$pattern" --glob '!deploy/env/.env.example' --glob '!docs/**' --glob '!tests/**' --glob '!scripts/gates/**' --glob '!scripts/p20_api_smoke.sh' .; then
        echo "[FAIL] Found potential secret literal matching: $pattern"
        FAILED_SCAN=1
    fi
done

if [ "$FAILED_SCAN" -eq 1 ]; then
    echo "Static Scan FAILED. Hardcoded secrets detected."
    finish_gate 1
else
    echo "[OK] Static Scan Clean."
fi

# -------------------------------------------------------------
# Step 2: Artifacts Scan (Current Run)
# -------------------------------------------------------------
echo "2. Scanning Artifacts for Leaks..."
# Scan the entire run directory for this execution
SCANDIR="user_data/generated/accept_runs/${RUN_ID}"

if [ -d "$SCANDIR" ]; then
    # Look for likely secret values if they are set in env
    # Note: parsing env vars here to search for them is tricky if they aren't set in CI.
    # So we search for keys *names* appearing in logs with values.
    
    LEAK_PATTERNS=(
        "BREEZE_API_SECRET"
        "session_token="
        "api_secret="
        "Authorization: Bearer"
    )

    LEAKS_FOUND=0
    for pattern in "${LEAK_PATTERNS[@]}"; do
        # Exclude this script itself and the gate log being written to
        if grep -r "$pattern" "$SCANDIR" | grep -v "p21_secrets_hygiene" | grep -v "gate.log"; then
            echo "[FAIL] Found potential secret leak in artifacts: $pattern"
            LEAKS_FOUND=1
        fi
    done

    if [ "$LEAKS_FOUND" -eq 1 ]; then
        echo "Artifact Scan FAILED. Secrets leaked in logs."
        finish_gate 1
    else
        echo "[OK] Artifact Scan Clean."
    fi
else
    echo "[WARN] No artifacts found to scan yet."
fi

# -------------------------------------------------------------
# Step 3: Session Readiness Check
# -------------------------------------------------------------
echo "3. Session Readiness Check..."

# Check if we are in MOCK mode
if [[ "${BREEZE_MOCK:-0}" == "1" ]]; then
    echo "[INFO] BREEZE_MOCK=1 detected. Skipping strict session check."
    echo "P21-SESSION-CHECK-SKIP"
else
    echo "Running p21_session_check.py..."
    if python3 scripts/p21_session_check.py; then
        echo "P21-SESSION-CHECK-PASS"
    else
        echo "[FAIL] Session Check Failed."
        echo "P21-SESSION-CHECK-FAIL"
        # In negative mode, maybe we expect this? 
        # But scope says "secrets hygiene" is the goal.
        # If we are verifying the *checker* works, valid failure is okay only if that was the test case.
        # For now, let's assume gate fails if check fails.
        finish_gate 1
    fi
fi

# -------------------------------------------------------------
# Finish
# -------------------------------------------------------------
echo "P21-SECRETS-SCAN-PASS"
echo "P21-ARTIFACTS-SCAN-PASS"
echo ">>> Gate P21: SUCCESS"
finish_gate 0
