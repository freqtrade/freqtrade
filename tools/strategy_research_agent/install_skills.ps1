Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$AgentRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Resolve-Path (Join-Path $AgentRoot "..\..")
$SourceRoot = Join-Path $RepoRoot "tools\strategy_research_agent\skills"

if ($env:CODEX_AGENT_SKILLS_DIR) {
    $TargetRoot = $env:CODEX_AGENT_SKILLS_DIR
} else {
    $TargetRoot = Join-Path $HOME ".agents\skills"
}

if (-not (Test-Path -LiteralPath $SourceRoot)) {
    throw "Missing source skill directory: $SourceRoot"
}

if (-not (Test-Path -LiteralPath $TargetRoot)) {
    New-Item -ItemType Directory -Force -Path $TargetRoot | Out-Null
}

Get-ChildItem -LiteralPath $SourceRoot -Directory | ForEach-Object {
    $target = Join-Path $TargetRoot $_.Name
    if (Test-Path -LiteralPath $target) {
        Remove-Item -LiteralPath $target -Recurse -Force
    }
    Copy-Item -LiteralPath $_.FullName -Destination $target -Recurse -Force
    Write-Host "Installed skill: $($_.Name) -> $target"
}

Write-Host "Strategy research skills installed into $TargetRoot"
