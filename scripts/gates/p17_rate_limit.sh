#!/bin/bash
set -e
source scripts/gates/common.sh

# P17 Rate Limit Gate
# Verifies that shim enforces rate limits in both sleep (default) and block modes.

echo "=========================================================="
echo "GATE: P17 Rate Limiter"
echo "=========================================================="

echo "1. Verify Block Mode (Pytest Integration w/ Env)"
# Uses tests/exchange/test_icicibreeze_rate_limit_applied.py
# Env vars set in the test fixture, but we can double check logic here if needed?
# Actually the test file is self-contained. Is that enough?
# Plan says: "Run pytest tests with specific failure expectations"
# Our integration test file forces block mode.
pytest -v tests/exchange/test_icicibreeze_rate_limit_applied.py
echo "   [+] Block Mode Verified"

echo "2. Verify Sleep Mode (Unit Test)"
pytest -v tests/test_rate_limiter_sleep_mode.py
echo "   [+] Sleep Mode Verified"

echo "----------------------------------------------------------"
echo "GATE P17-RateLimit PASSED"
echo "----------------------------------------------------------"
