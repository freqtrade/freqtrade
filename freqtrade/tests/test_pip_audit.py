"""
Run pip audit to check for known security vulnerabilities in installed packages.
Original Idea and base for this implementation by Michael Kennedy's blog:
https://mkennedy.codes/posts/python-supply-chain-security-made-easy/
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest


IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"


# Skip this test in github actions - github issues a security warning on it's own.
# This is to detect local transient dependencies.
@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Skip pip-audit in GitHub Actions")
def test_pip_audit_no_vulnerabilities():
    """
    Run pip-audit to check for known security vulnerabilities.

    This test will fail if any vulnerabilities are detected in the installed packages.

    Note: CVE-2025-53000 (nbconvert Windows vulnerability) is ignored as it only affects
    Windows platforms and is a known acceptable risk for this project.
    """
    # Get the project root directory
    project_root = Path(__file__).parent.parent
    command = [
        sys.executable,
        "-m",
        "pip_audit",
        # "--format=json",
        "--progress-spinner=off",
        "--ignore-vuln",
        "CVE-2025-53000",
        "--skip-editable",
    ]

    # Run pip-audit with JSON output for easier parsing
    try:
        result = subprocess.run(
            command,
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=120,  # 2 minute timeout
        )
    except subprocess.TimeoutExpired:
        pytest.fail("pip-audit command timed out after 120 seconds")
    except FileNotFoundError:
        pytest.fail("pip-audit not installed or not accessible")

    # Check if pip-audit found any vulnerabilities
    if result.returncode != 0:
        # pip-audit returns non-zero when vulnerabilities are found
        error_output = result.stdout + "\n" + result.stderr

        # Check if it's an actual vulnerability vs an error
        if "vulnerabilities found" in error_output.lower() or '"dependencies"' in result.stdout:
            pytest.fail(
                f"pip-audit detected security vulnerabilities!\n\n"
                f"Output:\n{result.stdout}\n\n"
                f"Please review and update vulnerable packages.\n"
                f"Run manually with: {' '.join(command)}"
            )
        else:
            # Some other error occurred
            pytest.fail(
                f"pip-audit failed to run properly:\n\nReturn code: {result.returncode}\n"
                f"Output: {error_output}\n"
            )

    # Success - no vulnerabilities found
    assert result.returncode == 0, "pip-audit should return 0 when no vulnerabilities are found"


def test_pip_audit_runs_successfully():
    """
    Verify that pip-audit can run successfully (even if vulnerabilities are found).

    This is a smoke test to ensure pip-audit is properly installed and functional.
    """
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip_audit", "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0, f"pip-audit --version failed: {result.stderr}"
        assert "pip-audit" in result.stdout.lower(), "pip-audit version output unexpected"
    except FileNotFoundError:
        pytest.fail("pip-audit not installed")
    except subprocess.TimeoutExpired:
        pytest.fail("pip-audit --version timed out")
