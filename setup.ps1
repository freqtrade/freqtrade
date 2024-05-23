Clear-Host

# Set the log file path and initialize variables
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$Global:LogFilePath = Join-Path $env:TEMP "script_log_$timestamp.txt"

$RequirementFiles = @("requirements.txt", "requirements-dev.txt", "requirements-hyperopt.txt", "requirements-freqai.txt", "requirements-freqai-rl.txt", "requirements-plot.txt")
$VenvName = ".venv"
$VenvDir = Join-Path $PSScriptRoot $VenvName

function Write-Log {
  param (
    [string]$Message,
    [string]$Level = 'INFO'
  )

  if (-not (Test-Path -Path $LogFilePath)) {
    New-Item -ItemType File -Path $LogFilePath -Force | Out-Null
  }
  
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
    [string]$Prompt,
    [string[]]$Options,
    [string]$DefaultChoice = 'A',
    [bool]$AllowMultipleSelections = $true
  )
  
  Write-Log "$Prompt`n" -Level 'PROMPT'
  for ($i = 0; $i -lt $Options.Length; $i++) {
    Write-Log "$([char](65 + $i)). $($Options[$i])" -Level 'PROMPT'
  }
  
  if ($AllowMultipleSelections) {
    Write-Log "`nSelect one or more options by typing the corresponding letters, separated by commas." -Level 'PROMPT'
  }
  else {
    Write-Log "`nSelect an option by typing the corresponding letter." -Level 'PROMPT'
  }
  
  $userInput = Read-Host
  if ([string]::IsNullOrEmpty($userInput)) {
    $userInput = $DefaultChoice
  }
  
  if ($AllowMultipleSelections) {
    # Ensure $userInput is treated as a string and split it by commas
    $userInput = [string]$userInput
    $selections = $userInput.Split(',') | ForEach-Object {
      $_.Trim().ToUpper()
    }
    
    # Convert each selection from letter to index and validate
    $selectedIndices = @()
    foreach ($selection in $selections) {
      if ($selection -match '^[A-Z]$') {
        $index = [int][char]$selection - [int][char]'A'
        if ($index -ge 0 -and $index -lt $Options.Length) {
          $selectedIndices += $index
        }
        else {
          Write-Log "Invalid input: $selection. Please enter letters within the valid range of options." -Level 'ERROR'
          return -1
        }
      }
      else {
        Write-Log "Invalid input: $selection. Please enter letters between A and Z." -Level 'ERROR'
        return -1
      }
    }
    
    return $selectedIndices
  }
  else {
    # Convert the selection from letter to index and validate
    if ($userInput -match '^[A-Z]$') {
      $selectedIndex = [int][char]$userInput - [int][char]'A'
      if ($selectedIndex -ge 0 -and $selectedIndex -lt $Options.Length) {
        return $selectedIndex
      }
      else {
        Write-Log "Invalid input: $userInput. Please enter a letter within the valid range of options." -Level 'ERROR'
        return -1
      }
    }
    else {
      Write-Log "Invalid input: $userInput. Please enter a letter between A and Z." -Level 'ERROR'
      return -1
    }
  }
}

function Exit-Script {
  param (
    [int]$ExitCode,
    [bool]$WaitForKeypress = $true
  )

  if ($Global:OldVirtualPath) {
    $env:PATH = $Global:OldVirtualPath
  }

  if ($ExitCode -ne 0) {
    Write-Log "Script failed. Would you like to open the log file? (Y/N)" -Level 'PROMPT'
    $openLog = Read-Host
    if ($openLog -eq 'Y' -or $openLog -eq 'y') {
      Start-Process notepad.exe -ArgumentList $LogFilePath
    }
  }
  elseif ($WaitForKeypress) {
    Write-Log "Press any key to exit..."
    $host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown") | Out-Null
  }

  return $ExitCode
}

# Function to install requirements
function Install-Requirements {
  param ([string]$RequirementsPath)
  
  if (-not $RequirementsPath) {
    Write-Log "No requirements path provided for installation." -Level 'ERROR'
    Exit-Script -ExitCode 1
  }

  Write-Log "Installing requirements from $RequirementsPath..."
  $installCmd = if (Test-Path $RequirementsPath) { & $VenvPip install -r $RequirementsPath } else { & $VenvPip install $RequirementsPath }
  $output = & $installCmd[0] $installCmd[1..$installCmd.Length] 2>&1
  $output | Out-File $LogFilePath -Append
  if ($LASTEXITCODE -ne 0) {
    Write-Log "Conflict detected. Exiting now..." -Level 'ERROR'
    Exit-Script -ExitCode 1
  }
}

function Test-PythonExecutable {
  param(
    [string]$PythonExecutable
  )

  $pythonCmd = Get-Command $PythonExecutable -ErrorAction SilentlyContinue
  if ($pythonCmd) {
    $command = "$($pythonCmd.Source) --version 2>&1"
    $versionOutput = Invoke-Expression $command
    if ($LASTEXITCODE -eq 0) {
      $version = $versionOutput | Select-String -Pattern "Python (\d+\.\d+\.\d+)" | ForEach-Object { $_.Matches.Groups[1].Value }
      Write-Log "Python version $version found using executable '$PythonExecutable'."
      return $true
    }
    else {
      Write-Log "Python executable '$PythonExecutable' not working correctly." -Level 'ERROR'
      return $false
    }
  }
  else {
    Write-Log "Python executable '$PythonExecutable' not found." -Level 'ERROR'
    return $false
  }
}

function Find-PythonExecutable {

  $pythonExecutables = @("python", "python3", "python3.9", "python3.10", "python3.11", "C:\Python39\python.exe", "C:\Python310\python.exe", "C:\Python311\python.exe")
  
  foreach ($executable in $pythonExecutables) {
    if (Test-PythonExecutable -PythonExecutable $executable) {
      return $executable
    }
  }

  return $null
}

# Function to get the list of requirements from a file
function Get-Requirements($file) {
  $requirements = @()
  $lines = Get-Content $file
  foreach ($line in $lines) {
    if ($line.StartsWith("-r ")) {
      $nestedFile = $line.Substring(3).Trim()
      $nestedFilePath = Join-Path (Split-Path $file -Parent) $nestedFile
      if (Test-Path $nestedFilePath) {
        $requirements += Get-Requirements $nestedFilePath
      }
    }
    elseif (-not $line.StartsWith("#")) {
      $requirements += $line
    }
  }
  return $requirements
}

function Main {
  # Exit on lower versions than Python 3.9 or when Python executable not found
  $pythonExecutable = Find-PythonExecutable
  if ($null -eq $pythonExecutable) {
    Write-Host "Error: No suitable Python executable found. Please ensure that Python 3.9 or higher is installed and available in the system PATH."
    Exit 1
  }

  "Starting the < operations..." | Out-File $LogFilePath -Append

  # Navigate to the project directory
  Set-Location -Path $PSScriptRoot
  "Current directory: $(Get-Location)" | Out-File $LogFilePath -Append

  # Define the path to the Python executable in the virtual environment
  $VenvPython = "$VenvDir\Scripts\python.exe"

  # Check if the virtual environment exists, if not, create it
  if (-Not (Test-Path $VenvPython)) {
    Write-Log "Virtual environment not found. Creating virtual environment..." -Level 'ERROR'
    & $pythonExecutable -m venv "$VenvName"
    if (-Not (Test-Path $VenvPython)) {
      Write-Log "Failed to create virtual environment." -Level 'ERROR'
      Exit-Script -exitCode 1
    }
    Write-Log "Virtual environment created successfully." -Level 'ERROR'
  }

  # Activate the virtual environment
  $Global:OldVirtualPath = $env:PATH
  $env:PATH = "$VenvDir\Scripts;$env:PATH"

  # Pull latest updates
  Write-Log "Pulling latest updates..."
  & "C:\Program Files\Git\cmd\git.exe" pull | Out-File $LogFilePath -Append 2>&1
  if ($LASTEXITCODE -ne 0) {
    Write-Log "Failed to pull updates from Git." -Level 'ERROR'
    Exit-Script -exitCode 1
  }

  if (-not (Test-Path "$VenvDir\Lib\site-packages\talib")) {
    # Install TA-Lib using the virtual environment's pip
    Write-Log "Installing TA-Lib using virtual environment's pip..."
    & $VenvPython -m pip install --find-links=build_helpers\ --prefer-binary TA-Lib | Out-File $LogFilePath -Append 2>&1
  }

  # Present options for requirement files
  $selectedIndices = Get-UserSelection -prompt "Select which requirement files to install:" -options $RequirementFiles -defaultChoice 'A'

  # Cache the selected requirement files
  $selectedRequirementFiles = @()
  foreach ($index in $selectedIndices) {
    if ($index -lt 0 -or $index -ge $RequirementFiles.Length) {
      Write-Log "Invalid selection. Please try again using the allowed letters." -Level 'ERROR'
      continue
    }
  
    $filePath = Join-Path $PSScriptRoot $RequirementFiles[$index]
    if (Test-Path $filePath) {
      $selectedRequirementFiles += $filePath
    }
    else {
      Write-Log "Requirement file not found: $filePath" -Level 'ERROR'
      Exit-Script -exitCode 1
    }
  }

  # Merge the contents of the selected requirement files
  if ($selectedRequirementFiles.Count -gt 0) {
    Write-Log "Merging selected requirement files..."
    $mergedDependencies = @()
    foreach ($file in $selectedRequirementFiles) {
      $mergedDependencies += Get-Requirements $file
    }
    # Avoid duplicate dependencies
    $mergedDependencies = $mergedDependencies | Select-Object -Unique

    # Create a temporary directory
    $tempDir = Join-Path $env:TEMP "freqtrade_requirements"
    New-Item -ItemType Directory -Path $tempDir -Force | Out-Null

    # Create a temporary file with the merged dependencies
    $tempFile = Join-Path $tempDir "requirements.txt"
    $mergedDependencies | Out-File $tempFile

    # Install the merged dependencies
    Write-Log "Installing merged dependencies..."
    & $VenvPython -m pip install -r $tempFile
    if ($LASTEXITCODE -ne 0) {
      Write-Log "Failed to install merged dependencies. Exiting now..." -Level 'ERROR'
      Exit-Script -exitCode 1
    }

    # Remove the temporary directory
    Remove-Item $tempDir -Recurse -Force
  }

  # Install freqtrade from setup using the virtual environment's Python
  Write-Log "Installing freqtrade from setup..."
  $setupInstallCommand = "$VenvPython -m pip install -e ."
  Invoke-Expression $setupInstallCommand | Out-File $LogFilePath -Append 2>&1
  if ($LASTEXITCODE -ne 0) {
    Write-Log "Failed to install freqtrade." -Level 'ERROR'
    Exit-Script -exitCode 1
  }

  $uiOptions = @("Yes", "No")
  $installUI = Get-UserSelection -prompt "Do you want to install the freqtrade UI?" -options $uiOptions -defaultChoice 'B' -allowMultipleSelections $false

  if ($installUI -eq 0) {
    # User selected "Yes"
    # Install freqtrade UI using the virtual environment's install-ui command
    Write-Log "Installing freqtrade UI..."
    & $VenvPython 'freqtrade', 'install-ui' | Out-File $LogFilePath -Append 2>&1
    if ($LASTEXITCODE -ne 0) {
      Write-Log "Failed to install freqtrade UI." -Level 'ERROR'
      Exit-Script -exitCode 1
    }
  }
  elseif ($installUI -eq 1) {
    # User selected "No"
    # Skip installing freqtrade UI
    Write-Log "Skipping freqtrade UI installation."
  }
  else {
    # Invalid selection
    # Handle the error case
    Write-Log "Invalid selection for freqtrade UI installation." -Level 'ERROR'
    Exit-Script -exitCode 1
  }
  
  Write-Log "Update complete!"
  Exit-Script -exitCode 0
}

# Call the Main function
Main