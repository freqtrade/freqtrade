# ============================================================================
# Distributed Strategy Migration — PowerShell version for Windows laptop
# ============================================================================
# Bidirectional strategy exchange with the server via SSH (SCP).
#   1. PUSH: local outgoing_migrants/ → remote incoming_migrants/
#   2. PULL: remote outgoing_migrants/ → local incoming_migrants/
#
# Usage:
#   .\scripts\distribute_migrate.ps1                   # run once
#   .\scripts\distribute_migrate.ps1 -Daemon           # run continuously
#   .\scripts\distribute_migrate.ps1 -Daemon -Interval 60
#   .\scripts\distribute_migrate.ps1 -Status           # show status
# ============================================================================

param(
    [switch]$Daemon,
    [switch]$Status,
    [switch]$PushOnly,
    [switch]$PullOnly,
    [int]$Interval = 30
)

$ErrorActionPreference = "Continue"

# ── Configuration ──
$RepoDir = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$OutgoingDir = Join-Path $RepoDir "genetic_algorithm\data\outgoing_migrants"
$IncomingDir = Join-Path $RepoDir "genetic_algorithm\data\incoming_migrants"
$SentDir = Join-Path $OutgoingDir ".sent"
$PulledDir = Join-Path $IncomingDir ".pulled_log"
$LogDir = Join-Path $RepoDir "genetic_algorithm\logs"
$LogFile = Join-Path $LogDir "distribute_migrate.log"

# Configure via environment variables: REMOTE_HOST, REMOTE_USER, REMOTE_REPO
if (-not $env:REMOTE_HOST -or -not $env:REMOTE_USER) {
    Write-Host "ERROR: Set REMOTE_HOST and REMOTE_USER environment variables." -ForegroundColor Red
    Write-Host "  Example: `$env:REMOTE_HOST='192.168.1.100'; `$env:REMOTE_USER='user'" -ForegroundColor Yellow
    exit 1
}
$RemoteHost = $env:REMOTE_HOST
$RemoteUser = $env:REMOTE_USER
$RemoteRepo = if ($env:REMOTE_REPO) { $env:REMOTE_REPO } else { (ssh ${env:REMOTE_USER}@${env:REMOTE_HOST} 'pwd') + "/freqtradeForkGA" }
$RemoteIncoming = "$RemoteRepo/genetic_algorithm/data/incoming_migrants"
$RemoteOutgoing = "$RemoteRepo/genetic_algorithm/data/outgoing_migrants"

$SshOpts = @("-o", "ConnectTimeout=5", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new")
$Remote = "${RemoteUser}@${RemoteHost}"

# ── Setup dirs ──
foreach ($d in @($OutgoingDir, $IncomingDir, $SentDir, $PulledDir, $LogDir)) {
    if (-not (Test-Path $d)) { New-Item -ItemType Directory -Path $d -Force | Out-Null }
}

function Write-Log {
    param([string]$Level, [string]$Message)
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $line = "[$ts] [$Level] $Message"
    Add-Content -Path $LogFile -Value $line -ErrorAction SilentlyContinue
    switch ($Level) {
        "ERROR" { Write-Host $line -ForegroundColor Red }
        "WARN"  { Write-Host $line -ForegroundColor Yellow }
        default { Write-Host $line -ForegroundColor Green }
    }
}

function Test-SshConnection {
    $result = ssh @SshOpts $Remote "echo ok" 2>$null
    return ($result -eq "ok")
}

function Push-Migrants {
    $files = Get-ChildItem -Path $OutgoingDir -Filter "*.json" -File -ErrorAction SilentlyContinue
    if (-not $files -or $files.Count -eq 0) { return }

    Write-Log "INFO" "[PUSH] Found $($files.Count) outgoing migrant file(s) to send"
    ssh @SshOpts $Remote "mkdir -p '$RemoteIncoming'" 2>$null

    $sent = 0
    foreach ($f in $files) {
        $result = scp @SshOpts -q $f.FullName "${Remote}:${RemoteIncoming}/$($f.Name)" 2>$null
        if ($LASTEXITCODE -eq 0) {
            Move-Item -Path $f.FullName -Destination (Join-Path $SentDir $f.Name) -Force
            $sent++
            Write-Log "INFO" "[PUSH]   -> Sent $($f.Name)"
        } else {
            Write-Log "ERROR" "[PUSH]   x Failed to send $($f.Name)"
        }
    }

    if ($sent -gt 0) {
        Write-Log "INFO" "[PUSH] Sent $sent/$($files.Count) files to $Remote"
    }

    # Cleanup old .sent (keep last 50)
    $sentFiles = Get-ChildItem -Path $SentDir -Filter "*.json" -File -ErrorAction SilentlyContinue | Sort-Object LastWriteTime
    if ($sentFiles -and $sentFiles.Count -gt 50) {
        $sentFiles | Select-Object -First ($sentFiles.Count - 50) | Remove-Item -Force
    }
}

function Pull-Migrants {
    $remoteFiles = ssh @SshOpts $Remote "find '$RemoteOutgoing' -maxdepth 1 -name '*.json' -printf '%f\n' 2>/dev/null" 2>$null
    if (-not $remoteFiles) { return }

    $fileList = $remoteFiles -split "`n" | Where-Object { $_.Trim() -ne "" }
    if ($fileList.Count -eq 0) { return }

    Write-Log "INFO" "[PULL] Found $($fileList.Count) file(s) on remote to pull"

    $pulled = 0
    foreach ($fname in $fileList) {
        $fname = $fname.Trim()
        if (-not $fname) { continue }

        $markerPath = Join-Path $PulledDir $fname
        if (Test-Path $markerPath) { continue }

        $localPath = Join-Path $IncomingDir $fname
        scp @SshOpts -q "${Remote}:${RemoteOutgoing}/${fname}" $localPath 2>$null
        if ($LASTEXITCODE -eq 0) {
            New-Item -Path $markerPath -ItemType File -Force | Out-Null
            ssh @SshOpts $Remote "rm -f '${RemoteOutgoing}/${fname}'" 2>$null
            $pulled++
            Write-Log "INFO" "[PULL]   <- Pulled $fname"
        } else {
            Write-Log "ERROR" "[PULL]   x Failed to pull $fname"
        }
    }

    if ($pulled -gt 0) {
        Write-Log "INFO" "[PULL] Pulled $pulled files from $Remote"
    }

    # Cleanup old markers (keep last 100)
    $markers = Get-ChildItem -Path $PulledDir -Filter "*.json" -File -ErrorAction SilentlyContinue | Sort-Object LastWriteTime
    if ($markers -and $markers.Count -gt 100) {
        $markers | Select-Object -First ($markers.Count - 100) | Remove-Item -Force
    }
}

function Show-Status {
    $inCount = (Get-ChildItem -Path $IncomingDir -Filter "*.json" -File -ErrorAction SilentlyContinue).Count
    $outCount = (Get-ChildItem -Path $OutgoingDir -Filter "*.json" -File -ErrorAction SilentlyContinue).Count
    $sentCount = (Get-ChildItem -Path $SentDir -Filter "*.json" -File -ErrorAction SilentlyContinue).Count
    $pulledCount = (Get-ChildItem -Path $PulledDir -Filter "*.json" -File -ErrorAction SilentlyContinue).Count

    Write-Host "`n=== Distributed Migration Status ===" -ForegroundColor Cyan
    Write-Host "  This machine:  $env:COMPUTERNAME (laptop)"
    Write-Host "  Remote:        $Remote"
    Write-Host ""
    Write-Host "  Local:" -ForegroundColor White
    Write-Host "    Incoming (waiting for GA):  $inCount files"
    Write-Host "    Outgoing (waiting to send): $outCount files"
    Write-Host "    Sent (archived):            $sentCount files"
    Write-Host "    Pulled (from remote):       $pulledCount files"
    Write-Host ""

    if (Test-SshConnection) {
        Write-Host "  SSH connectivity:  OK" -ForegroundColor Green
        $remoteIn = ssh @SshOpts $Remote "find '$RemoteIncoming' -maxdepth 1 -name '*.json' 2>/dev/null | wc -l" 2>$null
        $remoteOut = ssh @SshOpts $Remote "find '$RemoteOutgoing' -maxdepth 1 -name '*.json' 2>/dev/null | wc -l" 2>$null
        Write-Host "  Remote:" -ForegroundColor White
        Write-Host "    Incoming (waiting for GA):  $remoteIn files"
        Write-Host "    Outgoing (waiting to pull): $remoteOut files"
    } else {
        Write-Host "  SSH connectivity:  FAILED" -ForegroundColor Red
    }
    Write-Host ""
}

# ── Main ──

if ($Status) {
    Show-Status
    return
}

if (-not (Test-SshConnection)) {
    Write-Log "ERROR" "Cannot reach $Remote via SSH BatchMode. Set up SSH keys first."
    return
}

if ($Daemon) {
    Write-Log "INFO" "Starting migration daemon (poll every ${Interval}s, push+pull)"
    Write-Log "INFO" "  Local:  $env:COMPUTERNAME | Remote: $Remote"
    Write-Log "INFO" "  Press Ctrl+C to stop"

    try {
        while ($true) {
            if (Test-SshConnection) {
                if (-not $PullOnly) { Push-Migrants }
                if (-not $PushOnly) { Pull-Migrants }
            } else {
                Write-Log "WARN" "Remote unreachable, will retry in ${Interval}s"
            }
            Start-Sleep -Seconds $Interval
        }
    } catch {
        Write-Log "INFO" "Daemon shutting down..."
    }
} else {
    if (-not $PullOnly) { Push-Migrants }
    if (-not $PushOnly) { Pull-Migrants }
    Write-Log "INFO" "Exchange complete."
}
