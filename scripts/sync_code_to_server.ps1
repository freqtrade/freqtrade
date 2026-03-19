# ============================================================================
# Sync Code to Server — Push local code changes to remote server
# ============================================================================
# Uses rsync over SSH to sync the GA code. Excludes data, logs, venv, etc.
#
# Usage:
#   .\scripts\sync_code_to_server.ps1              # dry-run (preview)
#   .\scripts\sync_code_to_server.ps1 -Apply       # actually sync
#   .\scripts\sync_code_to_server.ps1 -GitSync     # git commit+push, then pull on server
# ============================================================================

param(
    [switch]$Apply,
    [switch]$GitSync,
    [switch]$Force
)

$ErrorActionPreference = "Stop"

$RepoDir = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
# Configure via environment variables: REMOTE_HOST, REMOTE_USER, REMOTE_REPO
if (-not $env:REMOTE_HOST -or -not $env:REMOTE_USER -or -not $env:REMOTE_REPO) {
    Write-Host "ERROR: Set REMOTE_HOST, REMOTE_USER, REMOTE_REPO environment variables." -ForegroundColor Red
    exit 1
}
$RemoteHost = $env:REMOTE_HOST
$RemoteUser = $env:REMOTE_USER
$RemoteRepo = $env:REMOTE_REPO
$Remote = "${RemoteUser}@${RemoteHost}"
$SshOpts = @("-o", "ConnectTimeout=5", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new")

Write-Host "`n=== Code Sync to Server ===" -ForegroundColor Cyan

# Test SSH connectivity
Write-Host "Testing SSH connection..." -NoNewline
$sshTest = ssh @SshOpts $Remote "echo ok" 2>$null
if ($sshTest -ne "ok") {
    Write-Host " FAILED" -ForegroundColor Red
    Write-Host "Cannot connect to $Remote. Ensure SSH keys are set up."
    exit 1
}
Write-Host " OK" -ForegroundColor Green

if ($GitSync) {
    # ── Git-based sync ──
    Write-Host "`nUsing git-based sync..." -ForegroundColor Yellow

    # Check local status
    Push-Location $RepoDir
    try {
        $status = git status --porcelain 2>$null
        $branch = git branch --show-current 2>$null

        if ($status) {
            Write-Host "`nLocal changes detected:" -ForegroundColor Yellow
            git status --short
            Write-Host ""

            if (-not $Force) {
                $response = Read-Host "Commit and push these changes? [y/N]"
                if ($response -notmatch '^[Yy]') {
                    Write-Host "Aborted." -ForegroundColor Yellow
                    return
                }
            }

            $msg = "sync: update from laptop $(Get-Date -Format 'yyyy-MM-dd HH:mm')"
            git add -A
            git commit -m $msg
            Write-Host "Committed: $msg" -ForegroundColor Green
        }

        # Push
        Write-Host "Pushing to origin/$branch..." -ForegroundColor Yellow
        git push origin $branch
        Write-Host "Push complete." -ForegroundColor Green

        # Pull on server
        Write-Host "Pulling on server..." -ForegroundColor Yellow
        $pullResult = ssh @SshOpts $Remote "cd $RemoteRepo && git fetch origin && git reset --hard origin/$branch 2>&1"
        Write-Host $pullResult
        Write-Host "Server synced." -ForegroundColor Green

        # Verify
        $serverCommit = ssh @SshOpts $Remote "cd $RemoteRepo && git log --oneline -1 2>/dev/null"
        $localCommit = git log --oneline -1
        Write-Host "`nLocal:  $localCommit"
        Write-Host "Server: $serverCommit"
        if ($serverCommit -eq $localCommit) {
            Write-Host "Commits match!" -ForegroundColor Green
        } else {
            Write-Host "WARNING: Commits don't match!" -ForegroundColor Red
        }
    } finally {
        Pop-Location
    }
} else {
    # ── rsync-based sync (files listed explicitly) ──
    Write-Host "`nUsing SCP-based selective sync..." -ForegroundColor Yellow

    # Sync these directories/files
    $syncItems = @(
        "genetic_algorithm/core/",
        "genetic_algorithm/config/",
        "genetic_algorithm/run_ga.py",
        "genetic_algorithm/__init__.py",
        "scripts/ga_distributed_monitor.sh",
        "scripts/distribute_migrate.sh",
        "scripts/ga_distributed_monitor.ps1",
        "scripts/distribute_migrate.ps1"
    )

    $synced = 0
    $failed = 0

    foreach ($item in $syncItems) {
        $localPath = Join-Path $RepoDir $item
        $remotePath = "$RemoteRepo/$item"

        if (-not (Test-Path $localPath)) {
            Write-Host "  SKIP $item (not found locally)" -ForegroundColor DarkGray
            continue
        }

        $isDir = (Get-Item $localPath).PSIsContainer

        if ($Apply) {
            try {
                if ($isDir) {
                    # For directories, ensure remote dir exists then copy contents
                    ssh @SshOpts $Remote "mkdir -p $remotePath" 2>$null
                    $localFiles = Get-ChildItem -Path $localPath -File -Recurse
                    foreach ($f in $localFiles) {
                        $relPath = $f.FullName.Substring($localPath.Length).TrimStart('\', '/').Replace('\', '/')
                        $remoteFilePath = "$remotePath$relPath"
                        $remoteDir = $remoteFilePath -replace '/[^/]+$', ''
                        ssh @SshOpts $Remote "mkdir -p $remoteDir" 2>$null
                        scp -q @SshOpts $f.FullName "${Remote}:${remoteFilePath}" 2>$null
                    }
                    $synced++
                    Write-Host "  SYNC $item ($($localFiles.Count) files)" -ForegroundColor Green
                } else {
                    $remoteDir = $remotePath -replace '/[^/]+$', ''
                    ssh @SshOpts $Remote "mkdir -p $remoteDir" 2>$null
                    scp -q @SshOpts $localPath "${Remote}:${remotePath}" 2>$null
                    $synced++
                    Write-Host "  SYNC $item" -ForegroundColor Green
                }
            } catch {
                $failed++
                Write-Host "  FAIL $item : $_" -ForegroundColor Red
            }
        } else {
            # Dry run - show what would be synced
            if ($isDir) {
                $fileCount = (Get-ChildItem -Path $localPath -File -Recurse | Measure-Object).Count
                Write-Host "  WOULD SYNC $item ($fileCount files)" -ForegroundColor DarkYellow
            } else {
                $size = [math]::Round((Get-Item $localPath).Length / 1KB, 1)
                Write-Host "  WOULD SYNC $item (${size}KB)" -ForegroundColor DarkYellow
            }
        }
    }

    if ($Apply) {
        Write-Host "`nSynced: $synced  Failed: $failed" -ForegroundColor $(if ($failed -eq 0) { "Green" } else { "Yellow" })
    } else {
        Write-Host "`nDry run complete. Use -Apply to actually sync." -ForegroundColor Yellow
        Write-Host "Or use -GitSync for git-based sync (recommended)." -ForegroundColor DarkGray
    }
}
