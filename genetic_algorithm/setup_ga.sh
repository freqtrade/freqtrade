#!/bin/bash
# Setup script for Genetic Algorithm
# Installs all required dependencies for running the GA with visualization

set -e  # Exit on error

echo "=========================================="
echo "Setting up Genetic Algorithm Environment"
echo "=========================================="
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
    echo "❌ Error: Python is not installed"
    echo "Please install Python 3.8 or higher first"
    exit 1
fi

# Use python3 if available, otherwise python
PYTHON_CMD="python3"
if ! command -v python3 &> /dev/null; then
    PYTHON_CMD="python"
fi

echo "✓ Found Python: $($PYTHON_CMD --version)"
echo ""

# Check if pip is available
if ! $PYTHON_CMD -m pip --version &> /dev/null; then
    echo "❌ Error: pip is not installed"
    echo "Please install pip first"
    exit 1
fi

echo "✓ Found pip: $($PYTHON_CMD -m pip --version)"
echo ""

# Install GA requirements
echo "Installing Genetic Algorithm dependencies..."
echo ""

if [ -f "genetic_algorithm/requirements.txt" ]; then
    $PYTHON_CMD -m pip install -r genetic_algorithm/requirements.txt
    echo ""
    echo "✓ GA dependencies installed successfully"
else
    echo "❌ Error: genetic_algorithm/requirements.txt not found"
    echo "Please run this script from the repository root directory"
    exit 1
fi

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "You can now run the Genetic Algorithm with:"
echo ""
echo "  # Run with live visualization (recommended):"
echo "  python genetic_algorithm/run_ga.py --visualize"
echo ""
echo "  # Run without visualization:"
echo "  python genetic_algorithm/run_ga.py"
echo ""
echo "  # Test visualization first:"
echo "  python genetic_algorithm/test_visualization.py"
echo ""
echo "For more information, see:"
echo "  genetic_algorithm/README.md"
echo "  genetic_algorithm/RUN_GA_GUIDE.md"
echo ""
