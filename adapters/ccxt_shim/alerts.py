import logging
import time
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class AlertManager:
    _instance = None

    def __init__(self, now_fn=None):
        self._last_alert_ts: Dict[str, float] = {}
        self._suppression_window = 60.0  # Seconds
        self._now_fn = now_fn or time.time

    @classmethod
    def get_instance(cls, now_fn=None):
        if cls._instance is None:
            cls._instance = cls(now_fn)
        # If singleton exists, we don't re-init with new now_fn, logic assumes it's set once or reset manually for tests.
        return cls._instance

    def alert(self, category: str, message: str, priority: str = "HIGH"):
        """
        now = self._now_fn()
        last_ts = self._last_alert_ts.get(category, 0.0)

        if now - last_ts < self._suppression_window:
            # Suppressed
            return

        self._last_alert_ts[category] = now

        prefix = f"[ALERT:{priority}]"
        log_msg = f"{prefix} [{category}] {message}"

        if priority == "HIGH":
            logger.error(log_msg)
        else:
            logger.warning(log_msg)


# Global Helper
def trigger(category: str, message: str, priority: str = "HIGH"):
    AlertManager.get_instance().alert(category, message, priority)
