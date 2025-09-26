#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
LOG=.pytest_ext.log
XML=.pytest_ext.xml
: > "$LOG"
if [ -d .venv ]; then source .venv/bin/activate; fi
python3 -m pytest -q \
  tests/test_feature_store.py \
  tests/test_ob_collector_ws.py \
  tests/test_strategy_features.py \
  --junitxml="$XML" 2>&1 | tee "$LOG" || true
# Print brief summary
TOTAL=$(grep -Eo "[0-9]+ passed|[0-9]+ failed|[0-9]+ skipped|[0-9]+ xfailed|[0-9]+ xpassed|[0-9]+ warnings" "$LOG" | tr '\n' ' ')
echo "SUMMARY: ${TOTAL}"
