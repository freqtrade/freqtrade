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
# We use PCRE (-P) to use negative lookaheads.
# We want to match: VARIABLE="something"
# BUT NOT if "something" starts with: your_, test_, mock_, <, ${, ""
# And not empty string.

RISKY_PATTERNS=(
    'BREEZE_API_KEY\s*=\s*"(?!your_|test_|mock_|EXAMPLE_|<|\$|\"\")'
    'BREEZE_API_SECRET\s*=\s*"(?!your_|test_|mock_|EXAMPLE_|<|\$|\"\")'
    'BREEZE_SESSION_TOKEN\s*=\s*"(?!your_|test_|mock_|EXAMPLE_|<|\$|\"\")'
)

FAILED_SCAN=0
for pattern in "${RISKY_PATTERNS[@]}"; do
    # -P for PCRE, -n for line number
    if rg -P -n "$pattern" --glob '!deploy/env/.env.example' --glob '!docs/**' --glob '!tests/**' --glob '!scripts/p20_api_smoke.sh' --glob '!scripts/gates/p21_secrets_hygiene.sh' .; then
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
SCANDIR="generated/accept_runs/${RUN_ID}"

if [ -d "$SCANDIR" ]; then
    LEAK_PATTERNS=(
        "BREEZE_API_SECRET"
        "session_token="
        "api_secret="
        "Authorization: Bearer"
        "BREEZE_API_KEY="
    )
    # Note: Scanning for broad regex tokens like [A-Za-z0-9]{20,} matches too many false positives (filenames, run IDs).
    # We stick to explicit key=value leak detection for now as per strict signal-to-noise requirements.

    LEAKS_FOUND=0
    
    # 1. Scan gate logs and text files
    # Exclude the gate log itself from reporting hits (it logs the patterns it searches for)
    # We use find to be specific about targets
    
    # Files to scan: all .log and .txt files in the run dir
    find "$SCANDIR" -type f \( -name "*.log" -o -name "*.txt" \) ! -name "gate.log" -print0 | while IFS= read -r -d '' file; do
        for pattern in "${LEAK_PATTERNS[@]}"; do
            if grep -q "$pattern" "$file"; then
                echo "[FAIL] Potentially leaked secret in $file matching: $pattern"
                LEAKS_FOUND=1
            fi
        done
    done
    
    # Check if subshell detected leaks (this variable update won't persist if piped, so we used explicit loop)
    # Re-verify leaks found logic if needed, but the find loop above runs in subshell? 
    # Actually while loop with pipe runs in subshell. Correct approach:
    
    if grep -rE "Authorization: Bearer|api_secret=|session_token=|BREEZE_API_SECRET=" "$SCANDIR" | grep -v "p21_secrets_hygiene" | grep -v "gate.log"; then
         echo "[FAIL] Found potential secret leaks via grep scan."
         LEAKS_FOUND=1
    fi

    if [ "$LEAKS_FOUND" -eq 1 ]; then
        echo "Artifact Scan FAILED. Secrets leaked in logs."
        finish_gate 1
    else
        echo "[OK] Artifact Scan Clean."
    fi
else
    echo "[FAIL] No artifacts found to scan at $SCANDIR. This indicates a pipeline config error."
    finish_gate 1
fi

# -------------------------------------------------------------
# Step 3: Session Readiness Check
# -------------------------------------------------------------
echo "3. Session Readiness Check..."

# Check if we are in MOCK mode
if [[ "${BREEZE_MOCK:-0}" == "1" ]]; then
    echo "[INFO] BREEZE_MOCK=1 detected. Skipping strict session check."
    echo "P21-SESSION-CHECK-SKIP"
# Check if credentials present
elif [ -z "${BREEZE_API_KEY:-}" ] || [ -z "${BREEZE_API_SECRET:-}" ] || [ -z "${BREEZE_SESSION_TOKEN:-}" ]; then
    echo "[INFO] Credentials missing. Skipping session check."
    echo "P21-SESSION-CHECK-SKIP"
else
    echo "Running p21_session_check.py..."
    if python3 scripts/p21_session_check.py; then
        echo "P21-SESSION-CHECK-PASS"
    else
        echo "[FAIL] Session Check Failed."
        echo "P21-SESSION-CHECK-FAIL"
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
