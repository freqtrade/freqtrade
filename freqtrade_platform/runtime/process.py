"""Process manager for OS-level subprocess execution and monitoring."""

from __future__ import annotations

import os
import signal
import subprocess
import time
from pathlib import Path


class RuntimeProcessHandle:
    """Handle for managing a running subprocess."""

    def __init__(self, process: subprocess.Popen, log_file_path: Path | None = None) -> None:
        self.process = process
        self.pid = process.pid
        self.log_file_path = log_file_path

    def poll(self) -> int | None:
        """Poll the underlying process return code. Returns None if still running."""
        return self.process.poll()

    def is_running(self) -> bool:
        """Check if process is currently running."""
        return self.poll() is None

    def confirm_startup(self, check_window_secs: float = 0.5) -> bool:
        """Confirm that the process did not exit immediately during initial startup window."""
        start_time = time.time()
        while time.time() - start_time < check_window_secs:
            if self.poll() is not None:
                return False
            time.sleep(0.05)
        return self.poll() is None

    def stop(self, timeout: float = 5.0) -> int:
        """Gracefully stop process using SIGINT/SIGTERM, or terminate if non-responsive."""
        if not self.is_running():
            return self.process.returncode or 0

        try:
            self.process.send_signal(signal.SIGINT)
            self.process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                self.process.terminate()
                self.process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=2.0)

        return self.process.returncode or 0


class RuntimeProcessManager:
    """Launches, polls, and terminates real OS subprocesses for Strategy Runtime instances."""

    def __init__(self) -> None:
        self._processes: dict[str, RuntimeProcessHandle] = {}

    def start_process(
        self,
        runtime_id: str,
        cmd: list[str],
        cwd: Path | str | None = None,
        env: dict[str, str] | None = None,
        stdout_path: Path | str | None = None,
    ) -> RuntimeProcessHandle:
        if runtime_id in self._processes and self._processes[runtime_id].is_running():
            raise RuntimeError(f"Runtime process for {runtime_id} is already running (PID {self._processes[runtime_id].pid})")

        merged_env = os.environ.copy()
        if env:
            merged_env.update(env)

        stdout_file = None
        if stdout_path:
            stdout_path = Path(stdout_path)
            stdout_path.parent.mkdir(parents=True, exist_ok=True)
            stdout_file = open(stdout_path, "a", encoding="utf-8")

        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd) if cwd else None,
            env=merged_env,
            stdout=stdout_file or subprocess.DEVNULL,
            stderr=subprocess.STDOUT if stdout_file else subprocess.DEVNULL,
            start_new_session=True,
        )

        handle = RuntimeProcessHandle(proc, log_file_path=Path(stdout_path) if stdout_path else None)
        self._processes[runtime_id] = handle
        return handle

    def get_handle(self, runtime_id: str) -> RuntimeProcessHandle | None:
        return self._processes.get(runtime_id)

    def is_running(self, runtime_id: str) -> bool:
        handle = self.get_handle(runtime_id)
        return handle.is_running() if handle else False

    def poll(self, runtime_id: str) -> int | None:
        handle = self.get_handle(runtime_id)
        return handle.poll() if handle else None

    def stop_process(self, runtime_id: str, timeout: float = 5.0) -> int | None:
        handle = self.get_handle(runtime_id)
        if not handle:
            return None
        exit_code = handle.stop(timeout=timeout)
        return exit_code

    def cleanup_finished(self) -> list[str]:
        """Check all processes, return list of runtime_ids that have exited."""
        finished = []
        for runtime_id, handle in list(self._processes.items()):
            if not handle.is_running():
                finished.append(runtime_id)
        return finished
