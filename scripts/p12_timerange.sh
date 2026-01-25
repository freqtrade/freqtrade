#!/bin/bash
# scripts/p12_timerange.sh
# Multi-purpose timerange helper for P12
set -euo pipefail

PAIR=${1:-"RELIANCE/INR"}
TF=${2:-"5m"}
DATADIR=${3:-"user_data/data/icicibreeze"}

# Call the python helper
python3 scripts/p12_timerange_from_data.py --pair "$PAIR" --tf "$TF" --datadir "$DATADIR"
