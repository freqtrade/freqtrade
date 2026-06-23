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
$RuntimeScript = Join-Path $RepoRoot "user_data\strategy_research\run_full_research_cycle.ps1"
$DailyTaskName = "Freqtrade Strategy Research Daily"
$WeeklyTaskName = "Freqtrade Strategy Research Weekly Aux"

function Install-AgentRuntime {
    & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $InstallScript
    if ($LASTEXITCODE -ne 0) {
        throw "Runtime install failed with exit code $LASTEXITCODE"
    }
}

function Register-ResearchTasks {
    Install-AgentRuntime

    $dailyAction = New-ScheduledTaskAction `
        -Execute "powershell.exe" `
        -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$RuntimeScript`" -SkipAuxFetch" `
        -WorkingDirectory $RepoRoot
    $dailyTrigger = New-ScheduledTaskTrigger -Daily -At 8:30am
    Register-ScheduledTask `
        -TaskName $DailyTaskName `
        -Action $dailyAction `
        -Trigger $dailyTrigger `
        -Description "Run local Freqtrade strategy research cycle without aux data download." `
        -Force | Out-Null

    $weeklyAction = New-ScheduledTaskAction `
        -Execute "powershell.exe" `
        -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$RuntimeScript`"" `
        -WorkingDirectory $RepoRoot
    $weeklyTrigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Sunday -At 9:15am
    Register-ScheduledTask `
        -TaskName $WeeklyTaskName `
        -Action $weeklyAction `
        -Trigger $weeklyTrigger `
        -Description "Run local Freqtrade strategy research cycle with funding/mark aux data refresh." `
        -Force | Out-Null

    Write-Host "Installed Windows scheduled tasks:"
    Write-Host "- $DailyTaskName"
    Write-Host "- $WeeklyTaskName"
}

function Unregister-ResearchTasks {
    foreach ($taskName in @($DailyTaskName, $WeeklyTaskName)) {
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
    foreach ($taskName in @($DailyTaskName, $WeeklyTaskName)) {
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
