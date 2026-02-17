# Setup script for Genetic Algorithm (Windows PowerShell)
# Installs all required dependencies for running the GA with visualization

$ErrorActionPreference = "Stop"

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Setting up Genetic Algorithm Environment" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

# Check if Python is available
$pythonCmd = $null
if (Get-Command python -ErrorAction SilentlyContinue) {
    $pythonCmd = "python"
} elseif (Get-Command python3 -ErrorAction SilentlyContinue) {
    $pythonCmd = "python3"
} else {
    Write-Host "❌ Error: Python is not installed" -ForegroundColor Red
    Write-Host "Please install Python 3.8 or higher first" -ForegroundColor Red
    exit 1
}

$pythonVersion = & $pythonCmd --version
Write-Host "✓ Found Python: $pythonVersion" -ForegroundColor Green
Write-Host ""

# Check if pip is available
try {
    $pipVersion = & $pythonCmd -m pip --version
    Write-Host "✓ Found pip: $pipVersion" -ForegroundColor Green
    Write-Host ""
} catch {
    Write-Host "❌ Error: pip is not installed" -ForegroundColor Red
    Write-Host "Please install pip first" -ForegroundColor Red
    exit 1
}

# Install GA requirements
Write-Host "Installing Genetic Algorithm dependencies..." -ForegroundColor Yellow
Write-Host ""

if (Test-Path "genetic_algorithm/requirements.txt") {
    & $pythonCmd -m pip install -r genetic_algorithm/requirements.txt
    Write-Host ""
    Write-Host "✓ GA dependencies installed successfully" -ForegroundColor Green
} else {
    Write-Host "❌ Error: genetic_algorithm/requirements.txt not found" -ForegroundColor Red
    Write-Host "Please run this script from the repository root directory" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Setup Complete!" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "You can now run the Genetic Algorithm with:" -ForegroundColor Green
Write-Host ""
Write-Host "  # Run with live visualization (recommended):" -ForegroundColor White
Write-Host "  python genetic_algorithm/run_ga.py --visualize" -ForegroundColor Yellow
Write-Host ""
Write-Host "  # Run without visualization:" -ForegroundColor White
Write-Host "  python genetic_algorithm/run_ga.py" -ForegroundColor Yellow
Write-Host ""
Write-Host "  # Test visualization first:" -ForegroundColor White
Write-Host "  python genetic_algorithm/test_visualization.py" -ForegroundColor Yellow
Write-Host ""
Write-Host "For more information, see:" -ForegroundColor White
Write-Host "  genetic_algorithm/README.md" -ForegroundColor Yellow
Write-Host "  genetic_algorithm/RUN_GA_GUIDE.md" -ForegroundColor Yellow
Write-Host ""
