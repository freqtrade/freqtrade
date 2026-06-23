#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

./.venv/bin/python user_data/strategy_research/run_research_agent.py "$@"
