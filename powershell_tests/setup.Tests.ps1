# Ensure the latest version of Pester is installed and imported
if (-not (Get-Module -ListAvailable -Name Pester | Where-Object { $_.Version -ge [version]"5.3.1" })) {
  Install-Module -Name Pester -Force -Scope CurrentUser -SkipPublisherCheck
}

Import-Module -Name Pester -MinimumVersion 5.3.1

# Describe block to contain all tests and setup
Describe "Setup and Tests" {

  BeforeAll {
    # Construct the absolute path to setup.ps1
    $setupScriptPath = Join-Path -Path (Get-Location) -ChildPath "setup.ps1"

    # Check if the setup script exists
    if (-Not (Test-Path -Path $setupScriptPath)) {
      Write-Host "Error: setup.ps1 script not found at path: $setupScriptPath"
      exit 1
    }

    # Load the script to test
    . $setupScriptPath
  }

  Context "Write-Log Tests" -Tag "Unit" {
    It "should write INFO level log" {
      $logFilePath = Join-Path $env:TEMP "script_log.txt"
      Remove-Item $logFilePath -ErrorAction SilentlyContinue

      Write-Log -Message "Test Info Message" -Level "INFO"

      $logContent = Get-Content $logFilePath
      $logContent | Should -Contain "INFO: Test Info Message"
    }

    It "should write ERROR level log" {
      $logFilePath = Join-Path $env:TEMP "script_log.txt"
      Remove-Item $logFilePath -ErrorAction SilentlyContinue

      Write-Log -Message "Test Error Message" -Level "ERROR"

      $logContent = Get-Content $logFilePath
      $logContent | Should -Contain "ERROR: Test Error Message"
    }
  }

  Context "Get-UserSelection Tests" -Tag "Unit" {
    It "should handle valid input correctly" {
      Mock Read-Host { return "A,B,C" }
    
      $options = @("Option1", "Option2", "Option3")
      $indices = Get-UserSelection -prompt "Select options" -options $options -defaultChoice "A"
    
      $indices | Should -Be @(0, 1, 2)
    }

    It "should return indices for selected options" {
      Mock Read-Host { return "a,b" }
        
      $options = @("Option1", "Option2", "Option3")
      $indices = Get-UserSelection -prompt "Select options" -options $options

      $indices | Should -Be @(0, 1)
    }

    It "should return default choice if no input" {
      Mock Read-Host { return "" }

      $options = @("Option1", "Option2", "Option3")
      $indices = Get-UserSelection -prompt "Select options" -options $options -defaultChoice "C"

      $indices | Should -Be @(2)
    }

    It "should handle mixed valid and invalid input correctly" {
      Mock Read-Host { return "A,X,B,Y,C,Z" }
    
      $options = @("Option1", "Option2", "Option3")
      $indices = Get-UserSelection -prompt "Select options" -options $options -defaultChoice "A"
    
      $indices | Should -Be @(0, 1, 2)
    }

    It "should handle invalid input gracefully" {
      Mock Read-Host { return "x,y,z" }
    
      $options = @("Option1", "Option2", "Option3")
      $indices = Get-UserSelection -prompt "Select options" -options $options -defaultChoice "A"
    
      $indices | Should -Be @()
    }
    
    It "should handle input without whitespace" {
      Mock Read-Host { return "a,b,c" }

      $options = @("Option1", "Option2", "Option3")
      $indices = Get-UserSelection -prompt "Select options" -options $options

      $indices | Should -Be @(0, 1, 2)
    }
  }

  Context "Exit-Script Tests" -Tag "Unit" {
    BeforeAll {
      # Set environment variables for the test
      $global:OldVirtualPath = "C:\old\path"
      $global:LogFilePath = "C:\path\to\logfile.log"
    }
  
    BeforeEach {
      Mock Write-Log {}
      Mock Start-Process {}
      Mock Read-Host { return "Y" }

      # Backup the original PATH
      $global:OriginalPath = $env:PATH
    }
  
    AfterEach {
      # Restore the original PATH
      $env:PATH = $OriginalPath
    }
  
    It "should exit with the given exit code without waiting for key press" {
      $exitCode = Exit-Script -exitCode 0 -isSubShell $true -waitForKeypress $false
      $exitCode | Should -Be 0
    }
    
    It "should prompt to open log file on error" {
      Exit-Script -exitCode 1 -isSubShell $true -waitForKeypress $false
      Assert-MockCalled Read-Host -Exactly 1
      Assert-MockCalled Start-Process -Exactly 1
    }
  
    It "should restore the environment path if OldVirtualPath is set" {
      # Set a different PATH to simulate the change
      $env:PATH = "C:\new\path"
      Exit-Script -exitCode 0 -isSubShell $true -waitForKeypress $false
      $env:PATH | Should -Be "C:\old\path"
    }
  }

  Context "Get-PythonVersionTag Tests" -Tag "Unit" {
    It "should return the correct Python version tag" {
      Mock Invoke-Expression { param($cmd) return "cp39-win_amd64" }

      $tag = Get-PythonVersionTag
      $tag | Should -Be "cp39-win_amd64"
    }
  }
}
