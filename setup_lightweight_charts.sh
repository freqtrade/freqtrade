#!/bin/bash

# Lightweight Charts Integration Setup Script
# This script automates the integration of Lightweight Charts into FreqUI

set -e  # Exit on error

echo "=================================="
echo "Lightweight Charts Setup Script"
echo "=================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
FREQTRADE_DIR="$SCRIPT_DIR"
WORKSPACE_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
FREQUI_DIR="${WORKSPACE_DIR}/frequi"
FREQUI_REPO="https://github.com/freqtrade/frequi.git"

# Check if we're in the right directory
if [ ! -f "$FREQTRADE_DIR/docs/LIGHTWEIGHT_CHARTS_INTEGRATION.md" ]; then
    echo -e "${RED}Error: This doesn't appear to be the Freqtrade directory${NC}"
    echo -e "${RED}Please run this script from the Freqtrade root directory${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Found Freqtrade directory${NC}"

# Step 1: Clone FreqUI if not exists
echo ""
echo "Step 1: Checking for FreqUI repository..."
if [ -d "$FREQUI_DIR" ]; then
    echo -e "${YELLOW}! FreqUI directory already exists at $FREQUI_DIR${NC}"
    read -p "Do you want to pull latest changes? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        cd "$FREQUI_DIR"
        git pull origin main || git pull origin master
        echo -e "${GREEN}✓ Pulled latest FreqUI changes${NC}"
    fi
else
    echo "Cloning FreqUI repository..."
    cd "$WORKSPACE_DIR"
    git clone "$FREQUI_REPO"
    echo -e "${GREEN}✓ Cloned FreqUI repository${NC}"
fi

cd "$FREQUI_DIR"

# Step 2: Install dependencies
echo ""
echo "Step 2: Installing FreqUI dependencies..."
if [ ! -d "node_modules" ]; then
    npm install
    echo -e "${GREEN}✓ Installed npm dependencies${NC}"
else
    echo -e "${YELLOW}! node_modules already exists, skipping npm install${NC}"
    read -p "Run npm install anyway? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        npm install
    fi
fi

# Step 3: Install lightweight-charts
echo ""
echo "Step 3: Installing lightweight-charts..."
if npm list lightweight-charts >/dev/null 2>&1; then
    echo -e "${YELLOW}! lightweight-charts already installed${NC}"
    npm list lightweight-charts
else
    npm install lightweight-charts
    echo -e "${GREEN}✓ Installed lightweight-charts${NC}"
fi

# Step 4: Create directories
echo ""
echo "Step 4: Creating necessary directories..."
mkdir -p src/components/charts
mkdir -p src/utils
mkdir -p src/views
echo -e "${GREEN}✓ Created directories${NC}"

# Step 5: Copy example files
echo ""
echo "Step 5: Copying example files from Freqtrade to FreqUI..."

# Function to copy file with backup
copy_with_backup() {
    local source=$1
    local dest=$2
    local filename=$(basename "$dest")

    if [ -f "$dest" ]; then
        echo -e "${YELLOW}! File $filename already exists${NC}"
        read -p "Overwrite? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            cp "$dest" "${dest}.backup.$(date +%Y%m%d_%H%M%S)"
            cp "$source" "$dest"
            echo -e "${GREEN}✓ Copied $filename (backup created)${NC}"
        else
            echo -e "${YELLOW}- Skipped $filename${NC}"
        fi
    else
        cp "$source" "$dest"
        echo -e "${GREEN}✓ Copied $filename${NC}"
    fi
}

# Copy Vue components
copy_with_backup "$FREQTRADE_DIR/docs/examples/LightweightChart.vue" "$FREQUI_DIR/src/components/charts/LightweightChart.vue"
copy_with_backup "$FREQTRADE_DIR/docs/examples/TradingChart.vue" "$FREQUI_DIR/src/components/charts/TradingChart.vue"

# Copy utility files
copy_with_backup "$FREQTRADE_DIR/docs/examples/chartDataTransformer.ts" "$FREQUI_DIR/src/utils/chartDataTransformer.ts"
copy_with_backup "$FREQTRADE_DIR/docs/examples/chartAnnotations.ts" "$FREQUI_DIR/src/utils/chartAnnotations.ts"

# Copy example view (optional)
echo ""
read -p "Copy the example ChartView.vue to src/views/? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    copy_with_backup "$FREQTRADE_DIR/docs/examples/ChartView.vue" "$FREQUI_DIR/src/views/ChartView.vue"
fi

# Step 6: Summary
echo ""
echo "=================================="
echo -e "${GREEN}Setup Complete!${NC}"
echo "=================================="
echo ""
echo "Files copied to:"
echo "  - src/components/charts/LightweightChart.vue"
echo "  - src/components/charts/TradingChart.vue"
echo "  - src/utils/chartDataTransformer.ts"
echo "  - src/utils/chartAnnotations.ts"
echo ""
echo "Next steps:"
echo "  1. Review the integration guide:"
echo "     ${FREQTRADE_DIR}/docs/LIGHTWEIGHT_CHARTS_INTEGRATION.md"
echo ""
echo "  2. Start the FreqUI development server:"
echo "     cd ${FREQUI_DIR}"
echo "     npm run dev"
echo ""
echo "  3. Test the components in your browser"
echo ""
echo "  4. Integrate into your existing views"
echo ""
echo -e "${YELLOW}Note: You may need to update your router configuration${NC}"
echo -e "${YELLOW}to add the new ChartView if you copied it.${NC}"
echo ""

# Step 7: Offer to start dev server
read -p "Start FreqUI development server now? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "Starting development server..."
    echo "Press Ctrl+C to stop"
    echo ""
    npm run dev
else
    echo ""
    echo -e "${GREEN}All done! Run 'npm run dev' when ready.${NC}"
fi
