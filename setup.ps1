# Set the console color to Matrix theme and clear the console
$host.UI.RawUI.BackgroundColor = "Black"
$host.UI.RawUI.ForegroundColor = "Green"
Clear-Host

# Set the log file path and initialize variables
$LogFilePath = Join-Path $env:TEMP "script_log.txt"
$ProjectDir = "G:\Downloads\GitHub\freqtrade"
$RequirementFiles = @("requirements.txt", "requirements-dev.txt", "requirements-hyperopt.txt", "requirements-freqai.txt", "requirements-plot.txt")
$VenvName = ".venv"

function Write-Log {
  param (
    [string]$Message,
    [string]$Level = 'INFO'
  )
  switch ($Level) {
    'INFO' { Write-Host $Message -ForegroundColor Green }
    'WARNING' { Write-Host $Message -ForegroundColor Yellow }
    'ERROR' { Write-Host $Message -ForegroundColor Red }
    'PROMPT' { Write-Host $Message -ForegroundColor Cyan }
  }
  "${Level}: $Message" | Out-File $LogFilePath -Append
}

function Get-UserSelection {
  param (
    [string]$prompt,
    [string[]]$options,
    [string]$defaultChoice = 'A'
  )
  
  Write-Log "$prompt`n" -Level 'PROMPT'
  for ($i = 0; $i -lt $options.Length; $i++) {
    Write-Log "$([char](65 + $i)). $($options[$i])" -Level 'PROMPT'
  }
  
  Write-Log "`nSelect one or more options by typing the corresponding letters, separated by commas." -Level 'PROMPT'
  $inputPath = Read-Host 
  if ([string]::IsNullOrEmpty($inputPath)) {
    $inputPath = $defaultChoice
  }
  
  # Ensure $inputPath is treated as a string and split it by commas
  $inputPath = [string]$inputPath
  $selections = $inputPath.Split(',') | ForEach-Object {
    $_.Trim().ToUpper()
  }
  
  # Convert each selection from letter to index
  $indices = $selections | ForEach-Object {
    if ($_ -match '^[A-Z]$') {
      # Ensure the input is a single uppercase letter
      [int][char]$_ - [int][char]'A'
    }
    else {
      Write-Log "Invalid input: $_. Please enter letters between A and Z." -Level 'ERROR'
      continue
    }
  }
  
  return $indices
}

function Exit-Script {
  param (
    [int]$exitCode,
    [bool]$isSubShell = $true
  )

  if ($OldVirtualPath) {
    $env:PATH = $OldVirtualPath
  }

  # Check if the script is exiting with an error and it's not a subshell
  if ($exitCode -ne 0 -and $isSubShell) {
    Write-Log "Script failed. Would you like to open the log file? (Y/N)" -Level 'PROMPT'
    $openLog = Read-Host
    if ($openLog -eq 'Y' -or $openLog -eq 'y') {
      Start-Process notepad.exe -ArgumentList $LogFilePath
    }
  }

  Write-Log "Press any key to exit..."
  $host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown") | Out-Null
  exit $exitCode
}

# Function to handle installation and conflict resolution
function Install-And-Resolve {
  param ([string]$InputPath)
  if (-not $InputPath) {
    Write-Log "ERROR: No input provided for installation." -Level 'ERROR'
    Exit-Script -exitCode 1
  }
  Write-Log "Installing $InputPath..."
  $installCmd = if (Test-Path $InputPath) { $VenvPip + @('install', '-r', $InputPath) } else { $VenvPip + @('install', $InputPath) }
  $output = & $installCmd[0] $installCmd[1..$installCmd.Length] 2>&1
  $output | Out-File $LogFilePath -Append
  if ($LASTEXITCODE -ne 0) {
    Write-Log "Conflict detected, attempting to resolve..." -Level 'ERROR'
    & $VenvPip[0] $VenvPip[1..$VenvPip.Length] 'check' | Out-File "conflicts.txt"
    Exit-Script -exitCode 1
  }
}

# Function to get the installed Python version tag for wheel compatibility
function Get-PythonVersionTag {
  $pythonVersion = & python -c "import sys; print(f'cp{sys.version_info.major}{sys.version_info.minor}')"
  $architecture = & python -c "import platform; print('win_amd64' if platform.machine().endswith('64') else 'win32')"
  return "$pythonVersion-$architecture"
}

# Check for admin privileges and elevate if necessary
if (-NOT ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
  Write-Log "Requesting administrative privileges..." -Level 'ERROR'
  Start-Process PowerShell -ArgumentList "-File `"$PSCommandPath`"" -Verb RunAs
  Exit-Script -exitCode 1 -isSubShell $false
}

# Log file setup
"Admin rights confirmed" | Out-File $LogFilePath -Append
"Starting the script operations..." | Out-File $LogFilePath -Append

# Navigate to the project directory
Set-Location -Path $ProjectDir
"Current directory: $(Get-Location)" | Out-File $LogFilePath -Append

# Define the path to the Python executable in the virtual environment
$VenvPython = Join-Path $ProjectDir "$VenvName\Scripts\python.exe"

# Check if the virtual environment exists, if not, create it
if (-Not (Test-Path $VenvPython)) {
  Write-Log "Virtual environment not found. Creating virtual environment..." -IsError $false
  python -m venv "$VenvName"
  if (-Not (Test-Path $VenvPython)) {
    Write-Log "Failed to create virtual environment." -Level 'ERROR'
    Exit-Script -exitCode 1
  }
  Write-Log "Virtual environment created successfully." -IsError $false
}

# Define the pip command using the Python executable
$VenvPip = @($VenvPython, '-m', 'pip')

# Activate the virtual environment
$OldVirtualPath = $env:PATH
$env:PATH = "$ProjectDir\$VenvName\Scripts;$env:PATH"

# Ensure setuptools is installed using the virtual environment's pip
Write-Log "Ensuring setuptools is installed..."
& $VenvPip[0] $VenvPip[1..$VenvPip.Length] 'install', '-v', 'setuptools' | Out-File $LogFilePath -Append 2>&1
if ($LASTEXITCODE -ne 0) {
  Write-Log "Failed to install setuptools." -Level 'ERROR'
  Exit-Script -exitCode 1
}

# Pull latest updates
Write-Log "Pulling latest updates..."
& git pull | Out-File $LogFilePath -Append 2>&1
if ($LASTEXITCODE -ne 0) {
  Write-Log "Failed to pull updates from Git." -Level 'ERROR'
  Exit-Script -exitCode 1
}

# Install TA-Lib using the virtual environment's pip
Write-Log "Installing TA-Lib using virtual environment's pip..."
& $VenvPip[0] $VenvPip[1..$VenvPip.Length] 'install', '--find-links=build_helpers\', '--prefer-binary', 'TA-Lib' | Out-File $LogFilePath -Append 2>&1

# Present options for requirement files
$selectedIndices = Get-UserSelection -prompt "Select which requirement files to install:" -options $RequirementFiles -defaultChoice 'A'

# Install selected requirement files
foreach ($index in $selectedIndices) {
  if ($index -lt 0 -or $index -ge $RequirementFiles.Length) {
    Write-Log "Invalid selection index: $index" -Level 'ERROR'
    continue
  }
  
  $filePath = Join-Path $ProjectDir $RequirementFiles[$index]
  if (Test-Path $filePath) {
    Install-And-Resolve $filePath
  }
  else {
    Write-Log "Requirement file not found: $filePath" -Level 'ERROR'
    Exit-Script -exitCode 1
  }
}

# Install freqtrade from setup using the virtual environment's Python
Write-Log "Installing freqtrade from setup..."
$setupInstallCommand = "$VenvPython -m pip install -e ."
Invoke-Expression $setupInstallCommand | Out-File $LogFilePath -Append 2>&1
if ($LASTEXITCODE -ne 0) {
  Write-Log "Failed to install freqtrade." -Level 'ERROR'
  Exit-Script -exitCode 1
}

# Ask if the user wants to install the UI
$uiOptions = @("Yes", "No")
$installUI = Get-UserSelection -prompt "Do you want to install the freqtrade UI?" -options $uiOptions -defaultChoice 'B'

if ($uiOptions[$installUI] -eq "Yes") {
  # Install freqtrade UI using the virtual environment's install-ui command
  Write-Log "Installing freqtrade UI..."
  & $VenvPython 'freqtrade', 'install-ui' | Out-File $LogFilePath -Append 2>&1
  if ($LASTEXITCODE -ne 0) {
    Write-Log "Failed to install freqtrade UI." -Level 'ERROR'
    Exit-Script -exitCode 1
  }
}

Write-Log "Update complete!"
Exit-Script -exitCode 0
