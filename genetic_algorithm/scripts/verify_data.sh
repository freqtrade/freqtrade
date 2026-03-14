#!/usr/bin/env bash
# ============================================================================
# Data Verification and Update Script
# ============================================================================
# 
# PURPOSE:
#   Verify that required data exists for the deployment runs and update it
#   if necessary. Ensures data coverage for timerange 20230101-20260228.
#
# USAGE:
#   bash genetic_algorithm/scripts/verify_data.sh
#
# ============================================================================

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PAIRS=("BTC/USDT" "ETH/USDT" "SOL/USDT" "BNB/USDT")
TIMEFRAMES=("5m" "15m" "1h" "4h")
EXCHANGE="binance"
TIMERANGE_START="20230101"
TIMERANGE_END="20260309"  # Current date
TIMERANGE="${TIMERANGE_START}-${TIMERANGE_END}"

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Data Verification for Deployment Runs"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# Navigate to project root
cd "$(dirname "$0")/../.."
PROJECT_ROOT="$(pwd)"

echo -e "${BLUE}Project root:${NC} $PROJECT_ROOT"
echo ""

# Activate virtualenv if available
VENV_DIR="$PROJECT_ROOT/.venv"
if [ -d "$VENV_DIR" ]; then
    echo -e "${GREEN}✓${NC} Activating virtual environment: $VENV_DIR"
    source "$VENV_DIR/bin/activate"
else
    echo -e "${YELLOW}⚠${NC} No virtual environment found at $VENV_DIR"
    echo "  Continuing with system Python..."
fi
echo ""

# Check if freqtrade is available
if ! command -v freqtrade &> /dev/null; then
    echo -e "${RED}✗ ERROR:${NC} freqtrade command not found"
    echo "  Please install freqtrade or activate the correct virtual environment"
    exit 1
fi

echo -e "${GREEN}✓${NC} freqtrade found: $(which freqtrade)"
echo ""

# Display current data status
echo "───────────────────────────────────────────────────────────────"
echo "  Current Data Status"
echo "───────────────────────────────────────────────────────────────"
echo ""

freqtrade list-data --exchange "$EXCHANGE" --show-timerange || true
echo ""

# Check data directory
DATA_DIR="$PROJECT_ROOT/user_data/data/$EXCHANGE"
echo -e "${BLUE}Data directory:${NC} $DATA_DIR"
echo ""

if [ ! -d "$DATA_DIR" ]; then
    echo -e "${YELLOW}⚠${NC} Data directory does not exist, creating..."
    mkdir -p "$DATA_DIR"
fi

# Check for missing data files
echo "───────────────────────────────────────────────────────────────"
echo "  Checking Required Data Files"
echo "───────────────────────────────────────────────────────────────"
echo ""

MISSING_FILES=0
EXISTING_FILES=0

for pair in "${PAIRS[@]}"; do
    pair_filename="${pair//\//_}"
    echo -e "${BLUE}Pair: $pair${NC}"
    
    for tf in "${TIMEFRAMES[@]}"; do
        filename="${DATA_DIR}/${pair_filename}-${tf}.feather"
        
        if [ -f "$filename" ]; then
            file_size=$(du -h "$filename" | cut -f1)
            echo -e "  ${GREEN}✓${NC} ${tf}: exists (${file_size})"
            ((EXISTING_FILES++))
        else
            echo -e "  ${RED}✗${NC} ${tf}: MISSING"
            ((MISSING_FILES++))
        fi
    done
    echo ""
done

echo "───────────────────────────────────────────────────────────────"
echo -e "  Summary: ${GREEN}${EXISTING_FILES} files exist${NC}, ${RED}${MISSING_FILES} missing${NC}"
echo "───────────────────────────────────────────────────────────────"
echo ""

# Download/update data
if [ $MISSING_FILES -gt 0 ]; then
    echo -e "${YELLOW}⚠ Missing data files detected${NC}"
    echo ""
    read -p "Download missing data? [Y/n] " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]] || [[ -z $REPLY ]]; then
        DOWNLOAD=true
    else
        DOWNLOAD=false
    fi
else
    echo -e "${GREEN}✓ All required data files exist${NC}"
    echo ""
    read -p "Update data to current date? [Y/n] " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]] || [[ -z $REPLY ]]; then
        DOWNLOAD=true
    else
        DOWNLOAD=false
    fi
fi

if [ "$DOWNLOAD" = true ]; then
    echo ""
    echo "───────────────────────────────────────────────────────────────"
    echo "  Downloading/Updating Data"
    echo "───────────────────────────────────────────────────────────────"
    echo ""
    echo -e "${BLUE}Timerange:${NC} $TIMERANGE"
    echo -e "${BLUE}Pairs:${NC} ${PAIRS[*]}"
    echo -e "${BLUE}Timeframes:${NC} ${TIMEFRAMES[*]}"
    echo -e "${BLUE}Exchange:${NC} $EXCHANGE"
    echo ""
    
    # Build pairs string for command
    PAIRS_STR="${PAIRS[@]}"
    TIMEFRAMES_STR="${TIMEFRAMES[@]}"
    
    # Download data
    echo "Executing freqtrade download-data..."
    echo ""
    
    freqtrade download-data \
        --exchange "$EXCHANGE" \
        --pairs $PAIRS_STR \
        --timeframes $TIMEFRAMES_STR \
        --timerange "$TIMERANGE" \
        --trading-mode spot \
        --data-format-ohlcv feather \
        --prepend
    
    echo ""
    echo -e "${GREEN}✓ Data download/update completed${NC}"
else
    echo ""
    echo -e "${YELLOW}⚠ Skipping data download${NC}"
fi

echo ""
echo "───────────────────────────────────────────────────────────────"
echo "  Final Data Status"
echo "───────────────────────────────────────────────────────────────"
echo ""

freqtrade list-data --exchange "$EXCHANGE" --show-timerange

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo -e "  ${GREEN}Data verification complete!${NC}"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "Next steps:"
echo "  1. Review the data coverage above"
echo "  2. Run the deployment script:"
echo "     bash genetic_algorithm/scripts/run_deploy_sequential.sh"
echo ""
