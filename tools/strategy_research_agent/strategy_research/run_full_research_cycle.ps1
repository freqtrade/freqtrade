param(
    [switch]$SkipAuxFetch
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Resolve-RepoRoot {
    $current = (Resolve-Path $PSScriptRoot).Path
    while ($current) {
        if ((Test-Path -LiteralPath (Join-Path $current ".git")) -and (Test-Path -LiteralPath (Join-Path $current "user_data"))) {
            return $current
        }
        $parent = Split-Path -Parent $current
        if ($parent -eq $current) {
            break
        }
        $current = $parent
    }
    throw "Could not locate repository root from $PSScriptRoot"
}

function Resolve-Python {
    param([string]$RepoRoot)
    if ($env:PYTHON) {
        return $env:PYTHON
    }
    $venvPython = Join-Path $RepoRoot ".venv\Scripts\python.exe"
    if (Test-Path -LiteralPath $venvPython) {
        return $venvPython
    }
    return "python"
}

function Invoke-Step {
    param(
        [Parameter(Mandatory = $true)][string]$Title,
        [Parameter(Mandatory = $true)][string]$Python,
        [Parameter(Mandatory = $true)][string[]]$Arguments
    )

    Write-Host "== $Title =="
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Title failed with exit code $LASTEXITCODE"
    }
}

$RepoRoot = Resolve-RepoRoot
Set-Location -LiteralPath $RepoRoot

$Python = Resolve-Python -RepoRoot $RepoRoot
$offlinePath = Join-Path $RepoRoot "user_data\offline_exchange"
if ([string]::IsNullOrWhiteSpace($env:PYTHONPATH)) {
    $env:PYTHONPATH = $offlinePath
} else {
    $env:PYTHONPATH = "$offlinePath;$env:PYTHONPATH"
}

Invoke-Step -Title "Update BTC/ETH 1m futures OHLCV" -Python $Python -Arguments @("user_data\download_binance_um_1m.py", "--incremental")

if (-not $SkipAuxFetch) {
    Invoke-Step -Title "Fetch futures funding/mark aux data" -Python $Python -Arguments @("user_data\strategy_research\fetch_futures_aux_data.py")
}

Invoke-Step -Title "Convert futures aux data" -Python $Python -Arguments @("user_data\strategy_research\convert_aux_to_freqtrade_futures.py")
Invoke-Step -Title "Audit futures cost data" -Python $Python -Arguments @("user_data\strategy_research\audit_futures_cost_data.py")
Invoke-Step -Title "Generate autonomous strategy hypotheses" -Python $Python -Arguments @("user_data\strategy_research\autonomous_strategy_lab.py")
Invoke-Step -Title "Build experiment matrix" -Python $Python -Arguments @("user_data\strategy_research\build_experiment_matrix.py")

Invoke-Step -Title "Run autonomous strategy smoke" -Python $Python -Arguments @(
    "user_data\strategy_research\run_research_agent.py",
    "--experiment",
    "user_data\strategy_research\experiments\autonomous_strategy_experiment.json",
    "--timerange",
    "20260101-20260201"
)
$index = Get-Content -Raw -Encoding UTF8 -LiteralPath "user_data\strategy_research\reports\agent_report_index.json" | ConvertFrom-Json
$autonomousReport = $index.latest_report.path

Invoke-Step -Title "Generate failure-driven strategy iterations" -Python $Python -Arguments @(
    "user_data\strategy_research\strategy_iteration_engine.py",
    "--report",
    $autonomousReport
)
Invoke-Step -Title "Run iterative strategy smoke" -Python $Python -Arguments @(
    "user_data\strategy_research\run_research_agent.py",
    "--experiment",
    "user_data\strategy_research\experiments\iterative_strategy_experiment.json",
    "--timerange",
    "20260101-20260201"
)
$index = Get-Content -Raw -Encoding UTF8 -LiteralPath "user_data\strategy_research\reports\agent_report_index.json" | ConvertFrom-Json
$iterativeReport = $index.latest_report.path

Invoke-Step -Title "Build walk-forward validation experiment" -Python $Python -Arguments @(
    "user_data\strategy_research\walk_forward_validator.py",
    "build",
    "--source",
    "iterative",
    "--limit",
    "6"
)
Invoke-Step -Title "Run walk-forward validation" -Python $Python -Arguments @(
    "user_data\strategy_research\run_research_agent.py",
    "--experiment",
    "user_data\strategy_research\experiments\walk_forward_validation_experiment.json"
)
$index = Get-Content -Raw -Encoding UTF8 -LiteralPath "user_data\strategy_research\reports\agent_report_index.json" | ConvertFrom-Json
$walkForwardReport = $index.latest_report.path
Invoke-Step -Title "Summarize walk-forward validation" -Python $Python -Arguments @(
    "user_data\strategy_research\walk_forward_validator.py",
    "summarize",
    "--report",
    $walkForwardReport
)

Invoke-Step -Title "Run base-cost matrix" -Python $Python -Arguments @(
    "user_data\strategy_research\run_research_agent.py",
    "--experiment",
    "user_data\strategy_research\experiments\candidate_regime_matrix_base_cost.json"
)
$index = Get-Content -Raw -Encoding UTF8 -LiteralPath "user_data\strategy_research\reports\agent_report_index.json" | ConvertFrom-Json
$baseReport = $index.latest_report.path

Invoke-Step -Title "Run stress-cost matrix" -Python $Python -Arguments @(
    "user_data\strategy_research\run_research_agent.py",
    "--experiment",
    "user_data\strategy_research\experiments\candidate_regime_matrix_stress_cost.json"
)
$index = Get-Content -Raw -Encoding UTF8 -LiteralPath "user_data\strategy_research\reports\agent_report_index.json" | ConvertFrom-Json
$stressReport = $index.latest_report.path

Invoke-Step -Title "Summarize matrix" -Python $Python -Arguments @(
    "user_data\strategy_research\summarize_matrix.py",
    "--report",
    $baseReport,
    "--report",
    $stressReport
)
Invoke-Step -Title "Build scorecards" -Python $Python -Arguments @("user_data\strategy_research\analyze_strategy_research.py")
Invoke-Step -Title "Analyze trade behavior" -Python $Python -Arguments @("user_data\strategy_research\analyze_trade_behavior.py")
Invoke-Step -Title "Plan behavior-driven experiments" -Python $Python -Arguments @("user_data\strategy_research\plan_behavior_experiments.py")
Invoke-Step -Title "Run promotion gate" -Python $Python -Arguments @("user_data\strategy_research\promotion_gate.py")
Invoke-Step -Title "Build research agenda" -Python $Python -Arguments @("user_data\strategy_research\research_agenda.py")
Invoke-Step -Title "Refresh dashboard" -Python $Python -Arguments @("user_data\strategy_research\run_research_agent.py", "--skip-backtests")

Write-Host "Research cycle complete."
Write-Host "Autonomous:   $autonomousReport"
Write-Host "Iterative:    $iterativeReport"
Write-Host "Walk-forward: $walkForwardReport"
Write-Host "Base report:   $baseReport"
Write-Host "Stress report: $stressReport"
Write-Host "Hypotheses:    user_data\strategy_research\experiments\autonomous_hypothesis_ledger.md"
Write-Host "Iterations:    user_data\strategy_research\experiments\iterative_hypothesis_ledger.md"
Write-Host "Walk-Fwd:      user_data\strategy_research\walk_forward_summaries\latest_walk_forward_summary.md"
Write-Host "Summary:       user_data\strategy_research\matrix_summaries\latest_matrix_summary.md"
Write-Host "Assessment:    user_data\strategy_research\strategy_assessments\latest_strategy_assessment.md"
Write-Host "Behavior:      user_data\strategy_research\trade_behavior\latest_trade_behavior.md"
Write-Host "BehaviorEx:    user_data\strategy_research\behavior_experiments\latest_behavior_experiment_plan.md"
Write-Host "Promotion:     user_data\strategy_research\promotion_reports\latest_promotion_report.md"
Write-Host "Agenda:        user_data\strategy_research\research_agendas\latest_research_agenda.md"
Write-Host "Dashboard:     user_data\strategy_research\dashboard\index.html"
