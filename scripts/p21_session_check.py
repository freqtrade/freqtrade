#!/usr/bin/env python3
"""
P21 Session Readiness Checker
Validates that necessary Breeze credentials are present in the environment
and conform to basic format requirements (length, no whitespace),
without ever printing the secrets themselves.

Exit Codes:
0: Success - All credentials present and valid format.
1: Invalid Format - Credentials present but malformed.
2: Missing Credentials - One or more required environment variables are missing.
"""

import os
import sys

REQUIRED_VARS = ["BREEZE_API_KEY", "BREEZE_API_SECRET", "BREEZE_SESSION_TOKEN"]


def redact(value):
    """Returns a redacted string showing only length."""
    if not value:
        return "<MISSING>"
    return f"<PRESENT, length={len(value)}>"


def check_format(name, value):
    """
    Checks basic format rules.
    Returns error message if invalid, None otherwise.
    """
    if not value:
        return f"{name} is empty."

    if len(value.strip()) != len(value):
        return f"{name} contains leading/trailing whitespace."

    if any(c.isspace() for c in value):
        return f"{name} contains internal whitespace."

    if name in ["BREEZE_API_KEY", "BREEZE_API_SECRET"]:
        if len(value) < 6:
            return f"{name} is too short (min 6 chars)."

    return None


def main():
    missing = []
    errors = []

    print("P21: Checking Session Credentials...")

    # 1. Check Presence
    env_state = {}
    for var in REQUIRED_VARS:
        val = os.environ.get(var)
        if val is None:
            missing.append(var)
            env_state[var] = "<MISSING>"
        else:
            env_state[var] = list(val)  # Temporary list for format check, never printed

    if missing:
        print(f"FAILED: Missing environment variables: {', '.join(missing)}")
        sys.exit(2)

    # 2. Check Format
    for var in REQUIRED_VARS:
        val_str = os.environ.get(var)
        err = check_format(var, val_str)
        if err:
            errors.append(err)
            print(f"  {var}: [INVALID] {err}")
        else:
            print(f"  {var}: [OK] {redact(val_str)}")

    if errors:
        print(f"\nFAILED: {len(errors)} format errors validation failed.")
        sys.exit(1)

    print("\nSUCCESS: All Breeze credentials present and formatted correctly.")
    sys.exit(0)


if __name__ == "__main__":
    main()
