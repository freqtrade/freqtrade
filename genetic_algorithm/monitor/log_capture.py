"""
Log Capture Handler — Captures log records for the terminal monitor Logs view.

Attaches to the 'GeneticAlgorithm' logger (and optionally the root logger)
and stores formatted log lines in a bounded deque for display by
TerminalMonitor in VIEW_LOGS mode.
"""

import logging
from collections import deque
from typing import List


class MonitorLogHandler(logging.Handler):
    """
    Logging handler that captures records into a bounded buffer.

    Thread-safe via the built-in ``logging.Handler`` lock.

    Attributes:
        max_lines: Maximum number of formatted lines to retain.
    """

    def __init__(self, max_lines: int = 200, level: int = logging.DEBUG):
        super().__init__(level=level)
        self.max_lines = max_lines
        self._buffer: deque = deque(maxlen=max_lines)
        self.setFormatter(
            logging.Formatter("%(asctime)s │ %(levelname)-7s │ %(message)s", datefmt="%H:%M:%S")
        )

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            self._buffer.append((record.levelno, msg))
        except Exception:
            self.handleError(record)

    def get_lines(self, last_n: int | None = None) -> List[str]:
        """Return the last *last_n* formatted log lines (all if None)."""
        self.acquire()
        try:
            lines = list(self._buffer)
        finally:
            self.release()
        if last_n is not None:
            lines = lines[-last_n:]
        return lines

    def clear(self) -> None:
        self.acquire()
        try:
            self._buffer.clear()
        finally:
            self.release()

    def emit_text(self, message: str, level: str = "info") -> None:
        """Inject a pre-formatted text message into the buffer.

        Useful for programmatic messages that bypass the logging framework
        (e.g. ``on_log`` / ``on_error`` monitor callbacks).
        """
        level_map = {
            "debug": logging.DEBUG,
            "info": logging.INFO,
            "warning": logging.WARNING,
            "error": logging.ERROR,
            "critical": logging.CRITICAL,
        }
        record = logging.LogRecord(
            name="Monitor", level=level_map.get(level.lower(), logging.INFO),
            pathname="", lineno=0, msg=message, args=(), exc_info=None,
        )
        self.emit(record)
