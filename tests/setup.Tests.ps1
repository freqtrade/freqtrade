
Describe "Setup and Tests" {
  BeforeAll {
    # Setup variables
    $SetupScriptPath = Join-Path $PSScriptRoot "..\setup.ps1"
    $Global:LogFilePath = Join-Path $env:TEMP "script_log.txt"

    # Check if the setup script exists
    if (-Not (Test-Path -Path $SetupScriptPath)) {
      Write-Host "Error: setup.ps1 script not found at path: $SetupScriptPath"
      exit 1
    }

    # Mock main to prevent it from running
    Mock Main {}

    . $SetupScriptPath
  }

  Context "Write-Log Tests" -Tag "Unit" {
    It "should write INFO level log" {
      if (Test-Path $Global:LogFilePath){
        Remove-Item $Global:LogFilePath -ErrorAction SilentlyContinue
      }

      Write-Log -Message "Test Info Message" -Level "INFO"
      $Global:LogFilePath | Should -Exist

      $LogContent = Get-Content $Global:LogFilePath
      $LogContent | Should -Contain "INFO: Test Info Message"
    }

    It "should write ERROR level log" {
      if (Test-Path $Global:LogFilePath){
        Remove-Item $Global:LogFilePath -ErrorAction SilentlyContinue
      }

      Write-Log -Message "Test Error Message" -Level "ERROR"
      $Global:LogFilePath | Should -Exist

      $LogContent = Get-Content $Global:LogFilePath
      $LogContent | Should -Contain "ERROR: Test Error Message"
    }
  }

  Describe "Get-UserSelection Tests" {
    Context "Valid input" {
      It "Should return the correct index for a valid single selection" {
        $Options = @("Option1", "Option2", "Option3")
        Mock Read-Host { return "B" }
        $Result = Get-UserSelection -prompt "Select an option" -options $Options
        $Result | Should -Be 1
      }

      It "Should return the correct index for a valid single selection" {
        $Options = @("Option1", "Option2", "Option3")
        Mock Read-Host { return "b" }
        $Result = Get-UserSelection -prompt "Select an option" -options $Options
        $Result | Should -Be 1
      }

      It "Should return the default choice when no input is provided" {
        $Options = @("Option1", "Option2", "Option3")
        Mock Read-Host { return "" }
        $Result = Get-UserSelection -prompt "Select an option" -options $Options -defaultChoice "C"
        $Result | Should -Be 2
      }
    }

    Context "Invalid input" {
      It "Should return -1 for an invalid letter selection" {
        $Options = @("Option1", "Option2", "Option3")
        Mock Read-Host { return "X" }
        $Result = Get-UserSelection -prompt "Select an option" -options $Options
        $Result | Should -Be -1
      }

      It "Should return -1 for a selection outside the valid range" {
        $Options = @("Option1", "Option2", "Option3")
        Mock Read-Host { return "D" }
        $Result = Get-UserSelection -prompt "Select an option" -options $Options
        $Result | Should -Be -1
      }

      It "Should return -1 for a non-letter input" {
        $Options = @("Option1", "Option2", "Option3")
        Mock Read-Host { return "1" }
        $Result = Get-UserSelection -prompt "Select an option" -options $Options
        $Result | Should -Be -1
      }

      It "Should return -1 for mixed valid and invalid input" {
        Mock Read-Host { return "A,X,B,Y,C,Z" }
        $Options = @("Option1", "Option2", "Option3")
        $Indices = Get-UserSelection -prompt "Select options" -options $Options -defaultChoice "A"
        $Indices | Should -Be -1
      }
    }

    Context "Multiple selections" {
      It "Should handle valid input correctly" {
        Mock Read-Host { return "A, B, C" }
        $Options = @("Option1", "Option2", "Option3")
        $Indices = Get-UserSelection -prompt "Select options" -options $Options -defaultChoice "A"
        $Indices | Should -Be @(0, 1, 2)
      }

      It "Should handle valid input without whitespace correctly" {
        Mock Read-Host { return "A,B,C" }
        $Options = @("Option1", "Option2", "Option3")
        $Indices = Get-UserSelection -prompt "Select options" -options $Options -defaultChoice "A"
        $Indices | Should -Be @(0, 1, 2)
      }

      It "Should return indices for selected options" {
        Mock Read-Host { return "a,b" }
        $Options = @("Option1", "Option2", "Option3")
        $Indices = Get-UserSelection -prompt "Select options" -options $Options
        $Indices | Should -Be @(0, 1)
      }

      It "Should return default choice if no input" {
        Mock Read-Host { return "" }
        $Options = @("Option1", "Option2", "Option3")
        $Indices = Get-UserSelection -prompt "Select options" -options $Options -defaultChoice "C"
        $Indices | Should -Be @(2)
      }

      It "Should handle invalid input gracefully" {
        Mock Read-Host { return "x,y,z" }
        $Options = @("Option1", "Option2", "Option3")
        $Indices = Get-UserSelection -prompt "Select options" -options $Options -defaultChoice "A"
        $Indices | Should -Be -1
      }

      It "Should handle input without whitespace" {
        Mock Read-Host { return "a,b,c" }
        $Options = @("Option1", "Option2", "Option3")
        $Indices = Get-UserSelection -prompt "Select options" -options $Options
        $Indices | Should -Be @(0, 1, 2)
      }
    }
  }

  Describe "Exit-Script Tests" -Tag "Unit" {
    BeforeEach {
      Mock Write-Log {}
      Mock Start-Process {}
      Mock Read-Host { return "Y" }
    }

    It "should exit with the given exit code without waiting for key press" {
      $ExitCode = Exit-Script -ExitCode 0 -isSubShell $true -waitForKeypress $false
      $ExitCode | Should -Be 0
    }

    It "should prompt to open log file on error" {
      Exit-Script -ExitCode 1 -isSubShell $true -waitForKeypress $false
      Assert-MockCalled Read-Host -Exactly 1
      Assert-MockCalled Start-Process -Exactly 1
    }
  }

  Context 'Find-PythonExecutable' {
    It 'Returns the first valid Python executable' {
      Mock Test-PythonExecutable { $true } -ParameterFilter { $PythonExecutable -eq 'python' }
      $Result = Find-PythonExecutable
      $Result | Should -Be 'python'
    }

    It 'Returns null if no valid Python executable is found' {
      Mock Test-PythonExecutable { $false }
      $Result = Find-PythonExecutable
      $Result | Should -Be $null
    }
  }
}
