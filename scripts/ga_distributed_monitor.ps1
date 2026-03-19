# ============================================================================
# Distributed GA Monitor — Real-time dashboard for laptop + server
# ============================================================================
# Shows: processes, generation progress, fitness, migration status, connection
#
# Usage:
#   .\scripts\ga_distributed_monitor.ps1              # one-shot dashboard
#   .\scripts\ga_distributed_monitor.ps1 -Watch       # auto-refresh every 15s
#   .\scripts\ga_distributed_monitor.ps1 -Watch -Interval 30
#   .\scripts\ga_distributed_monitor.ps1 -Compact     # less verbose
# ============================================================================

param(
    [switch]$Watch,
    [switch]$Compact,
    [int]$Interval = 15
)

$ErrorActionPreference = "Continue"

# ── Config ──
$RepoDir = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$LogDir = Join-Path $RepoDir "genetic_algorithm\logs"
$IncomingDir = Join-Path $RepoDir "genetic_algorithm\data\incoming_migrants"
$OutgoingDir = Join-Path $RepoDir "genetic_algorithm\data\outgoing_migrants"

# Configure via environment variables: REMOTE_HOST, REMOTE_USER, REMOTE_REPO
if (-not $env:REMOTE_HOST -or -not $env:REMOTE_USER -or -not $env:REMOTE_REPO) {
    Write-Host "WARN: Set REMOTE_HOST, REMOTE_USER, REMOTE_REPO env vars for remote monitoring." -ForegroundColor Yellow
}
$RemoteHost = $env:REMOTE_HOST
$RemoteUser = $env:REMOTE_USER
$RemoteRepo = $env:REMOTE_REPO
$Remote = "${RemoteUser}@${RemoteHost}"
$SshOpts = @("-o", "ConnectTimeout=3", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new")

# ── Helpers ──
function Write-Header {
    param([string]$Text)
    $line = "=" * 70
    Write-Host ""
    Write-Host $line -ForegroundColor Cyan
    Write-Host "  $Text" -ForegroundColor White
    Write-Host $line -ForegroundColor Cyan
}

function Write-Section {
    param([string]$Text)
    Write-Host ""
    Write-Host "--- $Text ---" -ForegroundColor Yellow
}

function Format-Elapsed {
    param([datetime]$Start)
    $span = (Get-Date) - $Start
    if ($span.TotalHours -ge 1) {
        return "{0:N0}h {1:N0}m" -f [math]::Floor($span.TotalHours), $span.Minutes
    }
    return "{0:N0}m {1:N0}s" -f [math]::Floor($span.TotalMinutes), $span.Seconds
}

function Get-LogProgress {
    param([string]$LogPath)
    if (-not (Test-Path $LogPath)) { return $null }
    $info = Get-Item $LogPath
    $lines = Get-Content $LogPath -Tail 200 -ErrorAction SilentlyContinue

    # Extract generation progress
    $genLines = $lines | Where-Object { $_ -match 'Creating generation (\d+)' }
    $lastGen = 0
    if ($genLines) {
        if ($genLines[-1] -match 'Creating generation (\d+)') {
            $lastGen = [int]$Matches[1]
        }
    }

    # Count unique generation numbers to estimate island progress
    $genCounts = @{}
    foreach ($l in $genLines) {
        if ($l -match 'Creating generation (\d+)') {
            $g = $Matches[1]
            if (-not $genCounts.ContainsKey($g)) { $genCounts[$g] = 0 }
            $genCounts[$g]++
        }
    }

    # Extract best fitness
    $bestLines = $lines | Where-Object { $_ -match 'NEW BEST.*fitness=([\d.]+)' }
    $bestFitness = $null
    $bestProfit = $null
    if ($bestLines) {
        if ($bestLines[-1] -match 'fitness=([\d.]+)\s+profit=([-\d.]+)') {
            $bestFitness = $Matches[1]
            $bestProfit = $Matches[2]
        } elseif ($bestLines[-1] -match 'fitness=([\d.]+)') {
            $bestFitness = $Matches[1]
        }
    }

    # Extract eval progress
    $evalLines = $lines | Where-Object { $_ -match '\[EVAL\] Complete.*succeeded.*avg profit' }
    $lastEval = $null
    if ($evalLines) { $lastEval = $evalLines[-1] }

    # Island summary lines
    $islandLines = $lines | Where-Object { $_ -match 'best=[\d.]+ avg=[\d.]+' }
    $islandSummary = @()
    if ($islandLines) {
        # Get the most recent entry per island
        $islandMap = @{}
        foreach ($l in $islandLines) {
            if ($l -match '\[(island_\S+)\s*\]\s*best=([\d.]+)\s+avg=([\d.]+)\s+diversity=([\d.]+)') {
                $islandMap[$Matches[1]] = @{
                    Best = $Matches[2]
                    Avg = $Matches[3]
                    Diversity = $Matches[4]
                }
            }
        }
        $islandSummary = $islandMap
    }

    # Ext migration lines
    $migLines = $lines | Where-Object { $_ -match 'EXT-MIGRATION' }

    return @{
        FileSize = $info.Length
        LastModified = $info.LastWriteTime
        LastGen = $lastGen
        GenCounts = $genCounts
        BestFitness = $bestFitness
        BestProfit = $bestProfit
        LastEval = $lastEval
        IslandSummary = $islandSummary
        MigrationEvents = $migLines.Count
        LastLogLine = if ($lines) { $lines[-1] } else { "(empty)" }
    }
}

function Show-Dashboard {
    Clear-Host
    $now = Get-Date -Format "yyyy-MM-dd HH:mm:ss"

    Write-Host ""
    Write-Host "  DISTRIBUTED GA MONITOR" -ForegroundColor Magenta -NoNewline
    Write-Host "  |  $now" -ForegroundColor DarkGray
    Write-Host ("  " + "=" * 66) -ForegroundColor DarkGray

    # ── LAPTOP SECTION ──
    Write-Header "LAPTOP ($env:COMPUTERNAME)"

    # Process check
    $pyProcs = Get-Process python -ErrorAction SilentlyContinue
    $gaProc = $pyProcs | Where-Object { $_.CPU -gt 10 } | Sort-Object CPU -Descending | Select-Object -First 1

    if ($gaProc) {
        $elapsed = Format-Elapsed $gaProc.StartTime
        $ramMB = [math]::Round($gaProc.WorkingSet64 / 1MB)
        Write-Host "  Process: " -NoNewline; Write-Host "RUNNING" -ForegroundColor Green -NoNewline
        Write-Host "  PID=$($gaProc.Id)  CPU=$([math]::Round($gaProc.CPU))s  RAM=${ramMB}MB  Uptime=$elapsed"
    } else {
        Write-Host "  Process: " -NoNewline; Write-Host "NOT RUNNING" -ForegroundColor Red
    }

    # Log progress
    $laptopLog = Join-Path $LogDir "distributed_laptop_err.log"
    $progress = Get-LogProgress $laptopLog
    if ($progress) {
        $modAge = [math]::Round(((Get-Date) - $progress.LastModified).TotalMinutes)
        $sizeKB = [math]::Round($progress.FileSize / 1KB)

        Write-Section "Evolution Progress"
        Write-Host "  Log: ${sizeKB}KB, last updated ${modAge}m ago"

        if ($progress.LastGen -gt 0) {
            Write-Host "  Generation: " -NoNewline
            Write-Host "$($progress.LastGen)/20" -ForegroundColor Green -NoNewline
            $pct = [math]::Round(($progress.LastGen / 20) * 100)
            $bar = "[" + ("#" * [math]::Floor($pct / 5)) + ("-" * (20 - [math]::Floor($pct / 5))) + "]"
            Write-Host "  $bar ${pct}%"
        } else {
            Write-Host "  Generation: " -NoNewline
            Write-Host "0/20 (initial evaluation in progress)" -ForegroundColor DarkYellow
        }

        if ($progress.BestFitness) {
            Write-Host "  Best fitness: " -NoNewline
            Write-Host "$($progress.BestFitness)" -ForegroundColor Green -NoNewline
            if ($progress.BestProfit) {
                $profColor = if ([double]$progress.BestProfit -ge 0) { "Green" } else { "Red" }
                Write-Host "  profit=$($progress.BestProfit)%" -ForegroundColor $profColor
            } else { Write-Host "" }
        }

        if ($progress.IslandSummary.Count -gt 0 -and -not $Compact) {
            Write-Section "Island Status"
            foreach ($island in ($progress.IslandSummary.GetEnumerator() | Sort-Object Name)) {
                $name = $island.Key.PadRight(28)
                $b = $island.Value.Best
                $a = $island.Value.Avg
                $d = $island.Value.Diversity
                Write-Host "  $name best=$b  avg=$a  div=$d"
            }
        }

        if ($progress.MigrationEvents -gt 0) {
            Write-Host "  External migrations: $($progress.MigrationEvents) events"
        }

        if (-not $Compact) {
            Write-Section "Last Log Entry"
            $lastLine = $progress.LastLogLine
            if ($lastLine.Length -gt 100) { $lastLine = $lastLine.Substring(0, 100) + "..." }
            Write-Host "  $lastLine" -ForegroundColor DarkGray
        }
    } else {
        Write-Host "  No log file found" -ForegroundColor DarkYellow
    }

    # Migration status
    Write-Section "Migration (Local)"
    $inCount = (Get-ChildItem -Path $IncomingDir -Filter "*.json" -File -ErrorAction SilentlyContinue | Measure-Object).Count
    $outCount = (Get-ChildItem -Path $OutgoingDir -Filter "*.json" -File -ErrorAction SilentlyContinue | Measure-Object).Count
    $sentCount = (Get-ChildItem -Path (Join-Path $OutgoingDir ".sent") -Filter "*.json" -File -ErrorAction SilentlyContinue | Measure-Object).Count
    $pulledCount = (Get-ChildItem -Path (Join-Path $IncomingDir ".pulled_log") -Filter "*.json" -File -ErrorAction SilentlyContinue | Measure-Object).Count

    # Daemon check
    $daemonLog = Join-Path $LogDir "distribute_migrate.log"
    $daemonRunning = $false
    if (Test-Path $daemonLog) {
        $daemonMod = (Get-Item $daemonLog).LastWriteTime
        $daemonAge = [math]::Round(((Get-Date) - $daemonMod).TotalMinutes)
        if ($daemonAge -lt 2) { $daemonRunning = $true }
    }

    Write-Host "  Daemon: " -NoNewline
    if ($daemonRunning) {
        Write-Host "RUNNING" -ForegroundColor Green -NoNewline
        Write-Host " (last activity ${daemonAge}m ago)"
    } else {
        Write-Host "STOPPED" -ForegroundColor Red -NoNewline
        Write-Host " -- Start with: .\scripts\distribute_migrate.ps1 -Daemon"
    }
    Write-Host "  Incoming: $inCount  Outgoing: $outCount  Sent: $sentCount  Pulled: $pulledCount"

    # ── SERVER SECTION ──
    Write-Header "SERVER ($RemoteHost)"

    $sshOk = $false
    $sshResult = ssh @SshOpts $Remote "echo ok" 2>$null
    if ($sshResult -eq "ok") { $sshOk = $true }

    if (-not $sshOk) {
        Write-Host "  SSH: " -NoNewline; Write-Host "UNREACHABLE" -ForegroundColor Red
        Write-Host "  Cannot connect to $Remote"
    } else {
        Write-Host "  SSH: " -NoNewline; Write-Host "CONNECTED" -ForegroundColor Green

        # Get all server info via server-side probe script
        # Step 1: Create a local temp probe script
        $probeContent = @"
#!/bin/bash
cd $RemoteRepo
echo '~~PROCS~~'
ps -u $RemoteUser -o pid,pcpu,pmem,etime,args --sort=-pcpu 2>/dev/null | grep run_ga.py | grep -v grep
echo '~~LOAD~~'
cat /proc/loadavg | cut -d' ' -f1-3
echo '~~MEM~~'
free -m | awk 'NR==2{printf "%d/%dMB (%d%%)\n", `$3, `$2, `$3/`$2*100}'
echo '~~GA_LOGS~~'
for f in genetic_algorithm/logs/*.log logs/*.log; do
  [ -f "`$f" ] || continue
  local_name=`$(basename "`$f")
  sz=`$(stat --format='%s' "`$f" 2>/dev/null || echo 0)
  mod=`$(stat --format='%Y' "`$f" 2>/dev/null || echo 0)
  gen=`$(grep -oP 'GENERATION \K\d+' "`$f" 2>/dev/null | tail -1)
  total=`$(grep -oP 'GENERATION \d+/\K\d+' "`$f" 2>/dev/null | tail -1)
  best=`$(grep -oP 'fitness=\K[\d.]+' "`$f" 2>/dev/null | sort -rn | head -1)
  complete=`$(grep -c 'GA RUN COMPLETE' "`$f" 2>/dev/null || echo 0)
  echo "LOG|`$local_name|`$sz|`$mod|`${gen:-0}|`${total:-0}|`${best:-0}|`$complete"
done
echo '~~MIGRATE~~'
echo "in=`$(find genetic_algorithm/data/incoming_migrants -maxdepth 1 -name '*.json' 2>/dev/null | wc -l)"
echo "out=`$(find genetic_algorithm/data/outgoing_migrants -maxdepth 1 -name '*.json' 2>/dev/null | wc -l)"
echo "sent=`$(find genetic_algorithm/data/outgoing_migrants/.sent -name '*.json' 2>/dev/null | wc -l)"
echo "pulled=`$(find genetic_algorithm/data/incoming_migrants/.pulled_log -name '*.json' 2>/dev/null | wc -l)"
echo '~~GIT~~'
git log --oneline -1 2>/dev/null
echo '~~BRANCH~~'
git rev-parse --abbrev-ref HEAD 2>/dev/null
"@
        $localProbe = Join-Path $env:TEMP "ga_monitor_probe.sh"
        [System.IO.File]::WriteAllText($localProbe, $probeContent.Replace("`r`n", "`n"))
        $remoteProbe = "/tmp/ga_monitor_probe.sh"
        scp -q @SshOpts $localProbe "${Remote}:${remoteProbe}" 2>$null
        $serverInfo = ssh @SshOpts $Remote "bash $remoteProbe; rm -f $remoteProbe" 2>$null

        if (-not $serverInfo) {
            Write-Host "  Failed to get server info" -ForegroundColor Red
        } else {
            $lines = $serverInfo -split "`n"
            $section = ""
            $procLines = @(); $loadLine = ""; $memLine = ""
            $gaLogEntries = @()
            $migrateLines = @(); $gitLine = ""; $branchLine = ""

            foreach ($l in $lines) {
                $l = $l.Trim()
                if ($l -match '^~~(\w+)~~$') { $section = $Matches[1]; continue }
                switch ($section) {
                    "PROCS" { if ($l) { $procLines += $l } }
                    "LOAD" { if ($l) { $loadLine = $l } }
                    "MEM" { if ($l) { $memLine = $l } }
                    "GA_LOGS" { if ($l -match '^LOG\|') { $gaLogEntries += $l } }
                    "MIGRATE" { if ($l) { $migrateLines += $l } }
                    "GIT" { if ($l) { $gitLine = $l } }
                    "BRANCH" { if ($l) { $branchLine = $l } }
                }
            }

            # Resources
            Write-Host "  Load: $loadLine  |  Memory: $memLine"
            Write-Host "  Branch: $branchLine  |  Commit: $gitLine"

            # Processes
            Write-Section "Server Processes"
            if ($procLines.Count -eq 0) {
                Write-Host "  No GA processes running" -ForegroundColor Red
            }
            foreach ($p in $procLines) {
                if ($p -match '(\d+)\s+([\d.]+)\s+([\d.]+)\s+(\S+)\s+.*--config[= ](\S+)') {
                    $procPid = $Matches[1]; $cpu = $Matches[2]; $mem = $Matches[3]
                    $etime = $Matches[4]; $config = Split-Path $Matches[5] -Leaf
                    $config = $config -replace '\.yaml$', ''

                    # Shorten config name
                    if ($config.Length -gt 40) { $config = $config.Substring(0, 40) + "..." }

                    $cpuColor = if ([double]$cpu -gt 50) { "Green" } else { "Yellow" }
                    Write-Host "  PID=$procPid " -NoNewline
                    Write-Host "CPU=${cpu}%" -ForegroundColor $cpuColor -NoNewline
                    Write-Host " MEM=${mem}% Uptime=$etime" -NoNewline
                    Write-Host "  $config" -ForegroundColor Cyan
                }
            }

            # GA Run Logs (generic — discovers all log files)
            Write-Section "Server GA Runs"
            if ($gaLogEntries.Count -eq 0) {
                Write-Host "  No GA log files found" -ForegroundColor DarkYellow
            }
            foreach ($entry in $gaLogEntries) {
                $parts = $entry -split '\|'
                if ($parts.Count -ge 8) {
                    $logName = $parts[1]; $sizeKB = [math]::Round([long]$parts[2] / 1024)
                    $modEpoch = [long]$parts[3]; $gen = [int]$parts[4]; $totalGen = [int]$parts[5]
                    $bestFit = $parts[6]; $isComplete = [int]$parts[7]

                    $modTime = (Get-Date "1970-01-01").AddSeconds($modEpoch).ToLocalTime()
                    $modAge = [math]::Round(((Get-Date) - $modTime).TotalMinutes)
                    $shortName = $logName -replace '\.log$', ''

                    if ($isComplete -gt 0) {
                        Write-Host "  $($shortName.PadRight(40)) " -NoNewline
                        Write-Host "COMPLETE" -ForegroundColor Green -NoNewline
                        Write-Host "  best=$bestFit  (${sizeKB}KB)"
                    } elseif ($totalGen -gt 0 -and $gen -gt 0) {
                        $pct = [math]::Round(($gen / $totalGen) * 100)
                        $bar = "[" + ("#" * [math]::Floor($pct / 5)) + ("-" * (20 - [math]::Floor($pct / 5))) + "]"
                        Write-Host "  $($shortName.PadRight(40)) " -NoNewline
                        Write-Host "Gen ${gen}/${totalGen} $bar ${pct}%" -ForegroundColor Green -NoNewline
                        Write-Host "  best=$bestFit  (${modAge}m ago)"
                    } else {
                        Write-Host "  $($shortName.PadRight(40)) " -NoNewline
                        Write-Host "Starting..." -ForegroundColor DarkYellow -NoNewline
                        Write-Host "  (${sizeKB}KB, ${modAge}m ago)"
                    }
                }
            }

            # Server migration
            Write-Section "Migration (Server)"
            $sin = 0; $sout = 0; $ssent = 0; $spulled = 0
            foreach ($ml in $migrateLines) {
                if ($ml -match 'in=(\d+)') { $sin = [int]$Matches[1] }
                if ($ml -match 'out=(\d+)') { $sout = [int]$Matches[1] }
                if ($ml -match 'sent=(\d+)') { $ssent = [int]$Matches[1] }
                if ($ml -match 'pulled=(\d+)') { $spulled = [int]$Matches[1] }
            }
            Write-Host "  Incoming: $sin  Outgoing: $sout  Sent: $ssent  Pulled: $spulled"
        }
    }

    # ── SYNC STATUS ──
    Write-Header "CONNECTIVITY SUMMARY"
    $sshStatus = if ($sshOk) { "OK" } else { "FAILED" }
    $sshColor = if ($sshOk) { "Green" } else { "Red" }

    Write-Host "  SSH Laptop -> Server: " -NoNewline
    Write-Host $sshStatus -ForegroundColor $sshColor
    Write-Host "  Migration daemon: " -NoNewline
    if ($daemonRunning) {
        Write-Host "ACTIVE" -ForegroundColor Green
    } else {
        Write-Host "INACTIVE" -ForegroundColor Red
    }

    # Total migration counts
    $totalExchanged = $sentCount + $pulledCount
    Write-Host "  Total strategies exchanged: $totalExchanged ($sentCount sent, $pulledCount pulled)"

    Write-Host ""
    if ($Watch) {
        Write-Host "  Auto-refreshing every ${Interval}s. Press Ctrl+C to stop." -ForegroundColor DarkGray
    } else {
        Write-Host "  Run with -Watch for auto-refresh. -Compact for less detail." -ForegroundColor DarkGray
    }
    Write-Host ""
}

# ── Main ──
if ($Watch) {
    try {
        while ($true) {
            Show-Dashboard
            Start-Sleep -Seconds $Interval
        }
    } catch {
        Write-Host "`nMonitor stopped." -ForegroundColor Yellow
    }
} else {
    Show-Dashboard
}
