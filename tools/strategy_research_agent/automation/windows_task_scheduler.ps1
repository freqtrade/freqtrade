param(
    [ValidateSet("install", "uninstall", "status")]
    [string]$Action = "install"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$AgentRoot = Resolve-Path (Join-Path $ScriptRoot "..")
$RepoRoot = Resolve-Path (Join-Path $AgentRoot "..\..")
$InstallScript = Join-Path $AgentRoot "install_runtime.ps1"
$WeeklyKnowledgeScript = Join-Path $RepoRoot "user_data\strategy_research\weekly_external_knowledge_update.py"
$WeeklyKnowledgeTaskName = "Freqtrade Strategy Research Weekly Knowledge"

function Install-AgentRuntime {
    & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $InstallScript
    if ($LASTEXITCODE -ne 0) {
        throw "Runtime install failed with exit code $LASTEXITCODE"
    }
}

function Register-ResearchTasks {
    Install-AgentRuntime

    $weeklyKnowledgeAction = New-ScheduledTaskAction `
        -Execute "powershell.exe" `
        -Argument "-NoProfile -ExecutionPolicy Bypass -Command `".\.venv\Scripts\python.exe '$WeeklyKnowledgeScript' --with-bilibili`"" `
        -WorkingDirectory $RepoRoot
    $weeklyKnowledgeTrigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday -At 10:30am
    Register-ScheduledTask `
        -TaskName $WeeklyKnowledgeTaskName `
        -Action $weeklyKnowledgeAction `
        -Trigger $weeklyKnowledgeTrigger `
        -Description "Refresh external knowledge, rebuild knowledge graph, research memory, and consolidation policy." `
        -Force | Out-Null

    Write-Host "Installed Windows scheduled tasks:"
    Write-Host "- $WeeklyKnowledgeTaskName"
}

function Unregister-ResearchTasks {
    foreach ($taskName in @($WeeklyKnowledgeTaskName)) {
        $task = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
        if ($null -ne $task) {
            Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
            Write-Host "Uninstalled $taskName"
        } else {
            Write-Host "Not installed: $taskName"
        }
    }
}

function Show-ResearchTaskStatus {
    foreach ($taskName in @($WeeklyKnowledgeTaskName)) {
        $task = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
        if ($null -eq $task) {
            Write-Host "== $taskName =="
            Write-Host "not installed"
            continue
        }
        $info = Get-ScheduledTaskInfo -TaskName $taskName
        Write-Host "== $taskName =="
        Write-Host "State: $($task.State)"
        Write-Host "LastRunTime: $($info.LastRunTime)"
        Write-Host "LastTaskResult: $($info.LastTaskResult)"
        Write-Host "NextRunTime: $($info.NextRunTime)"
    }
}

switch ($Action) {
    "install" { Register-ResearchTasks }
    "uninstall" { Unregister-ResearchTasks }
    "status" { Show-ResearchTaskStatus }
}
