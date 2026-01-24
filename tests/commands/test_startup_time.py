import subprocess
import sys
import time

from tests.conftest import is_arm, is_mac


MAXIMUM_STARTUP_TIME = 0.7 if is_mac() and not is_arm(True) else 0.5


def test_startup_time():
    # command to run
    cmd = [sys.executable, "-m", "freqtrade", "-h"]
    # warm up to generate pyc
    subprocess.run(cmd)

    start = time.time()
    subprocess.run(cmd)
    elapsed = time.time() - start
    assert elapsed < MAXIMUM_STARTUP_TIME, (
        "The startup time is too long, try to use lazy import in the command entry function"
        f" (maximum {MAXIMUM_STARTUP_TIME}s, got {elapsed}s)"
    )
