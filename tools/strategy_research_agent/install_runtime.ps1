Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$AgentRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Resolve-Path (Join-Path $AgentRoot "..\..")
$SourceStrategyRoot = Join-Path $AgentRoot "strategy_research"
$TargetStrategyRoot = Join-Path $RepoRoot "user_data\strategy_research"
$SourceGeneratedStrategies = Join-Path $AgentRoot "strategies\research_generated"
$TargetGeneratedStrategies = Join-Path $RepoRoot "user_data\strategies\research_generated"

$ExcludedTopDirs = @(
    "__pycache__",
    "reports",
    "dashboard",
    "cost_adjustments",
    "cost_audits",
    "matrix_summaries",
    "walk_forward_summaries",
    "promotion_candidates",
    "promotion_blocks",
    "promotion_reports",
    "research_agendas",
    "agenda_runs",
    "trade_behavior",
    "behavior_experiments",
    "failure_attribution",
    "strategy_library",
    "data_updates",
    "strategy_assessments"
)
$StateDirsWithJson = @("candidates", "rejected", "watchlist")
$ExcludedRuntimeExperimentFiles = @(
    "experiments/autonomous_hypothesis_ledger.md",
    "experiments/autonomous_strategy_experiment.json",
    "experiments/autonomous_strategy_registry.json",
    "experiments/iterative_hypothesis_ledger.md",
    "experiments/iterative_strategy_experiment.json",
    "experiments/iterative_strategy_registry.json",
    "experiments/behavior_experiment_hypothesis_ledger.md",
    "experiments/behavior_experiment_strategy_experiment.json",
    "experiments/behavior_experiment_strategy_registry.json",
    "experiments/walk_forward_validation_experiment.json"
)
$ExcludedSourceStateDirs = @("sources/inbox", "sources/reviews", "sources/translation_drafts")

function Ensure-Directory {
    param([Parameter(Mandatory = $true)][string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) {
        New-Item -ItemType Directory -Force -Path $Path | Out-Null
    }
}

function Copy-AgentTree {
    param(
        [Parameter(Mandatory = $true)][string]$Source,
        [Parameter(Mandatory = $true)][string]$Destination
    )

    Ensure-Directory -Path $Destination
    $sourceRoot = (Resolve-Path $Source).Path
    Get-ChildItem -LiteralPath $sourceRoot -Recurse -Force | ForEach-Object {
        $relative = $_.FullName.Substring($sourceRoot.Length).TrimStart("\", "/")
        if ([string]::IsNullOrWhiteSpace($relative)) {
            return
        }

        $normalized = $relative -replace "\\", "/"
        $parts = $normalized -split "/"
        if ($parts.Count -gt 0 -and $ExcludedTopDirs -contains $parts[0]) {
            return
        }
        foreach ($excluded in $ExcludedSourceStateDirs) {
            if ($normalized -eq $excluded -or $normalized.StartsWith("$excluded/")) {
                return
            }
        }
        if ($parts.Count -gt 0 -and $StateDirsWithJson -contains $parts[0] -and $_.Extension -eq ".json") {
            return
        }
        if ($ExcludedRuntimeExperimentFiles -contains $normalized) {
            return
        }
        if ($_.Extension -eq ".pyc") {
            return
        }

        $target = Join-Path $Destination $relative
        if ($_.PSIsContainer) {
            Ensure-Directory -Path $target
        } else {
            Ensure-Directory -Path (Split-Path -Parent $target)
            Copy-Item -LiteralPath $_.FullName -Destination $target -Force
        }
    }
}

function Copy-GeneratedStrategies {
    param(
        [Parameter(Mandatory = $true)][string]$Source,
        [Parameter(Mandatory = $true)][string]$Destination
    )

    Ensure-Directory -Path $Destination
    Get-ChildItem -LiteralPath $Source -Force | ForEach-Object {
        if ($_.Name -eq "__pycache__" -or $_.Extension -eq ".pyc") {
            return
        }
        $target = Join-Path $Destination $_.Name
        Copy-Item -LiteralPath $_.FullName -Destination $target -Recurse -Force
    }
}

Ensure-Directory -Path $TargetStrategyRoot
Ensure-Directory -Path $TargetGeneratedStrategies

Copy-AgentTree -Source $SourceStrategyRoot -Destination $TargetStrategyRoot
Copy-GeneratedStrategies -Source $SourceGeneratedStrategies -Destination $TargetGeneratedStrategies
Copy-Item -LiteralPath (Join-Path $AgentRoot "download_binance_um_1m.py") -Destination (Join-Path $RepoRoot "user_data\download_binance_um_1m.py") -Force

Write-Host "Installed strategy research agent runtime files into user_data/."
