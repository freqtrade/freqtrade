import os
import subprocess
import time
import pytest
from pathlib import Path

WITH_LOCK_SCRIPT = "scripts/ops/with_lock.py"


def test_lock_acquisition(tmp_path):
    """Test standard acquisition."""
    lock_file = tmp_path / "test.lock"
    cmd = [os.sys.executable, WITH_LOCK_SCRIPT, "--lock", str(lock_file), "--cmd", "echo hello"]

    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0
    assert "Lock acquired" in result.stderr
    assert "hello" in result.stdout


def test_lock_contention(tmp_path):
    """Test that second instance fails."""
    lock_file = tmp_path / "contend.lock"

    # Start first process that holds lock for 2 seconds
    p1 = subprocess.Popen(
        [os.sys.executable, WITH_LOCK_SCRIPT, "--lock", str(lock_file), "--cmd", "sleep 2"],
        stderr=subprocess.PIPE,
        stdout=subprocess.PIPE,
    )

    # Give it time to start and acquire
    time.sleep(0.5)

    # Start second process trying to acquire same lock
    # It should fail immediately (LOCK_NB)
    start_time = time.time()
    result = subprocess.run(
        [os.sys.executable, WITH_LOCK_SCRIPT, "--lock", str(lock_file), "--cmd", "echo fail"],
        capture_output=True,
        text=True,
    )
    end_time = time.time()

    # Assert p2 failed
    assert result.returncode == 1
    assert "Could not acquire lock" in result.stderr

    # Assert it was fast (non-blocking)
    assert (end_time - start_time) < 1.0

    # Cleanup p1
    p1.wait()


def test_lock_independence(tmp_path):
    """Test that different locks don't block each other."""
    lock1 = tmp_path / "L1.lock"
    lock2 = tmp_path / "L2.lock"

    # Hold L1
    p1 = subprocess.Popen(
        [os.sys.executable, WITH_LOCK_SCRIPT, "--lock", str(lock1), "--cmd", "sleep 2"]
    )
    time.sleep(0.5)

    # Acquire L2 - should succeed
    result = subprocess.run(
        [os.sys.executable, WITH_LOCK_SCRIPT, "--lock", str(lock2), "--cmd", "echo success"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    p1.wait()
