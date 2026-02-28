"""
Key Listener — Non-blocking single-keypress reader for view-mode switching.

Runs as a daemon thread.  Works on Linux/macOS (termios/tty).
Gracefully disables itself when stdin is not a TTY (e.g. piped input).
"""

import atexit
import sys
import threading
from typing import Callable, Optional


class KeyListener:
    """
    Daemon thread that listens for single keypresses on stdin.

    Args:
        on_key: Callback ``(key: str) -> None`` invoked on each keypress.
    """

    def __init__(self, on_key: Callable[[str], None]):
        self._on_key = on_key
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._old_settings = None
        self._available = False

    # ------------------------------------------------------------------
    def start(self) -> None:
        """Start the listener thread (no-op if stdin is not a TTY)."""
        if not sys.stdin.isatty():
            return

        try:
            import termios  # noqa: F401 — availability check
        except ImportError:
            return  # Not available on this platform

        self._available = True
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="KeyListener")
        self._thread.start()
        atexit.register(self.stop)

    # ------------------------------------------------------------------
    def stop(self) -> None:
        """Signal the thread to stop and restore terminal settings."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=0.5)
            self._thread = None
        self._restore_terminal()

    # ------------------------------------------------------------------
    @property
    def is_active(self) -> bool:
        return self._available and self._thread is not None and self._thread.is_alive()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    def _run(self) -> None:
        import termios
        import tty

        fd = sys.stdin.fileno()
        try:
            self._old_settings = termios.tcgetattr(fd)
            tty.setcbreak(fd)  # character-at-a-time, no echo
        except termios.error:
            self._available = False
            return

        try:
            while not self._stop_event.is_set():
                # Use select with a short timeout to allow clean shutdown
                import select
                rlist, _, _ = select.select([sys.stdin], [], [], 0.2)
                if rlist:
                    ch = sys.stdin.read(1)
                    if ch:
                        try:
                            self._on_key(ch.lower())
                        except Exception:
                            pass  # Don't crash the listener
        finally:
            self._restore_terminal()

    def _restore_terminal(self) -> None:
        if self._old_settings is not None:
            try:
                import termios
                termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self._old_settings)
            except Exception:
                pass
            self._old_settings = None
