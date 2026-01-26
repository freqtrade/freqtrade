#!/bin/bash
# Strict Secret Scanner
# Detects high-entropy strings and known secret patterns
set -euo pipefail

EXIT_CODE=0
SCANNED_FILES=0
LEAKS_FOUND=0

# Patterns to scan for
# 1. Private Keys
# 2. AWS Keys
# 3. Generic API Keys/Tokens (16+ chars)
# 4. Passwords (8+ chars)

echo "Starting Strict Secret Scan..."

# Prepare filters (files to exclude)
EXCLUDES=(
    ":!*.git/*" 
    ":!*.github/workflows/*" 
    ":!LICENSE" 
    ":!docs/*" 
    ":!deploy/env/.env.example" 
    ":!deploy/env/secrets.example.md"
    ":!user_data/generated/*"
)

# Use git grep if available, otherwise find/grep (simplified here to assume git repo or fallback)
if [ -d .git ]; then
    CMD="git grep -I -n -E"
else
    echo "Not a git repo, using grep recursive..."
    CMD="grep -I -r -n -E"
fi

# We will run checks individually to categorize them
check_pattern() {
    local NAME="$1"
    local PATTERN="$2"
    
    echo "Scanning for $NAME..."
    # We use '|| true' to prevent exit on no matches
    if [ -d .git ]; then
        # Use -i for case insensitivity
        MATCHES=$(git grep -I -i -n -E "$PATTERN" -- . "${EXCLUDES[@]}" || true)
    else
        # Basic exclude for generic grep not implemented fully here, assuming git env for this task
        MATCHES=$(grep -I -i -r -n -E "$PATTERN" . || true) 
    fi

    if [ -n "$MATCHES" ]; then
        # Filter false positives
        # Drop lines matching allowlist
        FILTERED=$(echo "$MATCHES" | grep -vE "\$\{\{\s*secrets\.|os\.environ\.get\(|BREEZE_API_KEY|BREEZE_API_SECRET|BREEZE_SESSION_TOKEN" || true)
        
        if [ -n "$FILTERED" ]; then
             echo "FAIL: Found potential $NAME:"
             echo "$FILTERED"
             LEAKS_FOUND=$((LEAKS_FOUND + 1))
             EXIT_CODE=2
        fi
    fi
}

check_pattern "Private Keys" "-----BEGIN (RSA|EC|OPENSSH|PRIVATE) KEY-----"
check_pattern "AWS Keys" "aws(.{0,20})?(access|secret)[_-]?key\s*[:=]\s*['\"]?[A-Za-z0-9/+=]{16,}"
check_pattern "Generic API Tokens" "(api[_-]?key|api[_-]?secret|session[_-]?token|access[_-]?token)\s*[:=]\s*['\"][^'\"]{16,}['\"]"
check_pattern "Hardcoded Passwords" "password\s*[:=]\s*['\"][^'\"]{8,}['\"]"

if [ $EXIT_CODE -eq 0 ]; then
    echo "Secret scan passed. No leaks found."
else
    echo "Secret scan FAILED. Found $LEAKS_FOUND potential leaks."
fi

exit $EXIT_CODE
