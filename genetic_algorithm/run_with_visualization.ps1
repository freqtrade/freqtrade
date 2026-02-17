# Quick start script for Genetic Algorithm with Live Visualization (Windows)
# This script runs the GA with visualization enabled by default

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Starting Genetic Algorithm Evolution" -ForegroundColor Cyan
Write-Host "With Live Visualization" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

# Determine Python command
$pythonCmd = "python"
if (Get-Command python3 -ErrorAction SilentlyContinue) {
    $pythonCmd = "python3"
}

# Check if dependencies are installed
try {
    & $pythonCmd -c "import matplotlib" 2>$null
    $hasMatplotlib = $true
} catch {
    $hasMatplotlib = $false
}

if (-not $hasMatplotlib) {
    Write-Host "⚠️  Warning: Visualization dependencies not found" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Please run the setup script first:" -ForegroundColor White
    Write-Host "  .\genetic_algorithm\setup_ga.ps1" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Or install manually:" -ForegroundColor White
    Write-Host "  pip install -r genetic_algorithm/requirements.txt" -ForegroundColor Yellow
    Write-Host ""
    
    $continue = Read-Host "Do you want to continue without visualization? (y/N)"
    if ($continue -notmatch "^[Yy]$") {
        exit 1
    }
    
    Write-Host ""
    Write-Host "Running WITHOUT visualization..." -ForegroundColor Yellow
    & $pythonCmd genetic_algorithm/run_ga.py
} else {
    Write-Host "✓ Visualization dependencies found" -ForegroundColor Green
    Write-Host ""
    Write-Host "Starting GA with live visualization..." -ForegroundColor Green
    Write-Host "This will show real-time plots of:" -ForegroundColor White
    Write-Host "  - Fitness evolution over generations" -ForegroundColor Cyan
    Write-Host "  - Population diversity" -ForegroundColor Cyan
    Write-Host "  - Performance metrics (profit, Sharpe, win rate, drawdown)" -ForegroundColor Cyan
    Write-Host "  - Fitness distribution" -ForegroundColor Cyan
    Write-Host ""
    & $pythonCmd genetic_algorithm/run_ga.py --visualize
}
