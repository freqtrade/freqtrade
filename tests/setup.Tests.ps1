# Ensure the specific version 5.3.1 of Pester is installed and imported
$requiredVersion = [version]"5.3.1"
$installedModule = Get-Module -ListAvailable -Name Pester
if (-not ($installedModule) -or ($installedModule.Version -lt $requiredVersion)) {
  Install-Module -Name Pester -RequiredVersion $requiredVersion -Force -Scope CurrentUser -SkipPublisherCheck
}

Import-Module -Name Pester -MinimumVersion 5.3.1

# Describe block to contain all tests and setup
Describe "Setup and Tests" {
  BeforeAll {
    # Construct the absolute path to setup.ps1
    $setupScriptPath = Join-Path $PSScriptRoot "..\setup.ps1"
    
    # Check if the setup script exists
    if (-Not (Test-Path -Path $setupScriptPath)) {
      Write-Host "Error: setup.ps1 script not found at path: $setupScriptPath"
      exit 1
    }

    # Mock main to prevent it from running
    Mock Main {}
    
    . $setupScriptPath
  }

  Context "Write-Log Tests" -Tag "Unit" {
    It "should write INFO level log" {
      Remove-Item $Global:LogFilePath -ErrorAction SilentlyContinue
    
      Write-Log -Message "Test Info Message" -Level "INFO"
    
      $Global:LogFilePath | Should -Exist
      $logContent = Get-Content $Global:LogFilePath
      $logContent | Should -Contain "INFO: Test Info Message"
    }    

    It "should write ERROR level log" {
      $Global:LogFilePath = Join-Path $env:TEMP "script_log.txt"
      Remove-Item $Global:LogFilePath -ErrorAction SilentlyContinue

      Write-Log -Message "Test Error Message" -Level "ERROR"

      $logContent = Get-Content $Global:LogFilePath
      $logContent | Should -Contain "ERROR: Test Error Message"
    }
  }

  Describe "Get-UserSelection Tests" {
    Context "Valid input" {
      It "Should return the correct index for a valid single selection" {
        $options = @("Option1", "Option2", "Option3")
        Mock Read-Host { return "B" }
        $result = Get-UserSelection -prompt "Select an option" -options $options
        $result | Should -Be 1
      }

      It "Should return the default choice when no input is provided" {
        $options = @("Option1", "Option2", "Option3")
        Mock Read-Host { return "" }
        $result = Get-UserSelection -prompt "Select an option" -options $options -defaultChoice "C"
        $result | Should -Be 2
      }
    }

    Context "Invalid input" {
      It "Should return -1 for an invalid letter selection" {
        $options = @("Option1", "Option2", "Option3")
        Mock Read-Host { return "X" }
        $result = Get-UserSelection -prompt "Select an option" -options $options
        $result | Should -Be -1
      }

      It "Should return -1 for a selection outside the valid range" {
        $options = @("Option1", "Option2", "Option3")
        Mock Read-Host { return "D" }
        $result = Get-UserSelection -prompt "Select an option" -options $options
        $result | Should -Be -1
      }

      It "Should return -1 for a non-letter input" {
        $options = @("Option1", "Option2", "Option3")
        Mock Read-Host { return "1" }
        $result = Get-UserSelection -prompt "Select an option" -options $options
        $result | Should -Be -1
      }

      It "Should return -1 for mixed valid and invalid input" {
        Mock Read-Host { return "A,X,B,Y,C,Z" }
        $options = @("Option1", "Option2", "Option3")
        $indices = Get-UserSelection -prompt "Select options" -options $options -defaultChoice "A"
        $indices | Should -Be -1
      }
    }

    Context "Multiple selections" {
      It "Should handle valid input correctly" {
        Mock Read-Host { return "A,B,C" }
        $options = @("Option1", "Option2", "Option3")
        $indices = Get-UserSelection -prompt "Select options" -options $options -defaultChoice "A"
        $indices | Should -Be @(0, 1, 2)
      }

      It "Should return indices for selected options" {
        Mock Read-Host { return "a,b" }
        $options = @("Option1", "Option2", "Option3")
        $indices = Get-UserSelection -prompt "Select options" -options $options
        $indices | Should -Be @(0, 1)
      }

      It "Should return default choice if no input" {
        Mock Read-Host { return "" }
        $options = @("Option1", "Option2", "Option3")
        $indices = Get-UserSelection -prompt "Select options" -options $options -defaultChoice "C"
        $indices | Should -Be @(2)
      }

      It "Should handle invalid input gracefully" {
        Mock Read-Host { return "x,y,z" }
        $options = @("Option1", "Option2", "Option3")
        $indices = Get-UserSelection -prompt "Select options" -options $options -defaultChoice "A"
        $indices | Should -Be -1
      }

      It "Should handle input without whitespace" {
        Mock Read-Host { return "a,b,c" }
        $options = @("Option1", "Option2", "Option3")
        $indices = Get-UserSelection -prompt "Select options" -options $options
        $indices | Should -Be @(0, 1, 2)
      }
    }
  }

  Describe "Exit-Script Tests" -Tag "Unit" {
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
      $Global:OldVirtualPath = $env:PATH
      Exit-Script -exitCode 0 -isSubShell $true -waitForKeypress $false
      $env:PATH | Should -Be "C:\new\path"
    }
  }

  Context 'Find-PythonExecutable' {
    It 'Returns the first valid Python executable' {
      Mock Test-PythonExecutable { $true } -ParameterFilter { $PythonExecutable -eq 'python' }
      $result = Find-PythonExecutable
      $result | Should -Be 'python'
    }

    It 'Returns null if no valid Python executable is found' {
      Mock Test-PythonExecutable { $false }
      $result = Find-PythonExecutable
      $result | Should -Be $null
    }
  }
}
