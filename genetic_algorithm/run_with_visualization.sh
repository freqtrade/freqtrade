#!/bin/bash
# Quick start script for Genetic Algorithm with Live Visualization
# This script runs the GA with visualization enabled by default

echo "=========================================="
echo "Starting Genetic Algorithm Evolution"
echo "With Live Visualization"
echo "=========================================="
echo ""

# Check if dependencies are installed
if ! python3 -c "import matplotlib" 2>/dev/null; then
    echo "⚠️  Warning: Visualization dependencies not found"
    echo ""
    echo "Please run the setup script first:"
    echo "  ./genetic_algorithm/setup_ga.sh"
    echo ""
    echo "Or install manually:"
    echo "  pip install -r genetic_algorithm/requirements.txt"
    echo ""
    read -p "Do you want to continue without visualization? (y/N) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
    echo ""
    echo "Running WITHOUT visualization..."
    python3 genetic_algorithm/run_ga.py
else
    echo "✓ Visualization dependencies found"
    echo ""
    echo "Starting GA with live visualization..."
    echo "This will show real-time plots of:"
    echo "  - Fitness evolution over generations"
    echo "  - Population diversity"
    echo "  - Performance metrics (profit, Sharpe, win rate, drawdown)"
    echo "  - Fitness distribution"
    echo ""
    python3 genetic_algorithm/run_ga.py --visualize
fi
