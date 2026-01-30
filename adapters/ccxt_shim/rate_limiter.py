import logging
import os
import time
from typing import Any

from freqtrade.exceptions import OperationalException

logger = logging.getLogger(__name__)


class RateLimiter:
    """
    Token Bucket Rate Limiter for BreezeCCXT shim.
    Enforces API limits locally to prevent 429s or to provide deterministic
    blocking for testing.
    """

    def __init__(self, now_fn=None, sleep_fn=None):
        self._now = now_fn or time.time
        self._sleep = sleep_fn or time.sleep

        # Configuration via Environment Variables (Shim standard)
        self.enabled = os.environ.get("FT_RATE_LIMIT_DISABLE", "0") != "1"

        # Limit Configuration
        # Default: 100 requests per minute
        self.per_minute = int(os.environ.get("FT_RATE_LIMIT_PER_MINUTE", "100"))

        # Mode: 'sleep' (production default) or 'block' (testing/gates)
        self.mode = os.environ.get("FT_RATE_LIMIT_MODE", "sleep").lower()

        # Token Bucket State
        self.capacity = self.per_minute
        self.tokens = float(self.capacity)
        self.last_refill = self._now()

        # Refill rate: tokens per second
        # If per_minute=60, rate=1.0 token/sec
        self.refill_rate = self.per_minute / 60.0

        if self.enabled:
            logger.info(
                f"RateLimiter initialized: {self.per_minute}/min, "
                f"Rate: {self.refill_rate:.4f} tps, Mode: {self.mode}"
            )
        else:
            logger.warning("RateLimiter is DISABLED via FT_RATE_LIMIT_DISABLE")

    def _refill(self):
        now = self._now()
        elapsed = now - self.last_refill

        if elapsed > 0:
            added = elapsed * self.refill_rate
            self.tokens = min(self.capacity, self.tokens + added)
            self.last_refill = now

    def allow(self, op: str, cost: int = 1) -> None:
        """
        Check if operation is allowed.
        Consumes tokens if available.
        If not available:
          - mode='sleep': Sleeps until tokens available
          - mode='block': Raises OperationalException
        """
        if not self.enabled:
            return

        self._refill()

        if self.tokens >= cost:
            self.tokens -= cost
            self._log_usage(op, cost)
            return

        # Not enough tokens
        if self.mode == "block":
            self._raise_block(op, cost)
        else:
            self._sleep_until_allowed(op, cost)

    def _raise_block(self, op: str, cost: int):
        # Stable token for acceptance gates
        logger.warning(f"RATE_LIMIT_BLOCK: op={op} cost={cost} remaining={self.tokens:.2f}")
        raise OperationalException(f"rate_limit_block: {op}")

    def _sleep_until_allowed(self, op: str, cost: int):
        needed = cost - self.tokens
        # Time required to refill 'needed' tokens
        sleep_time = needed / self.refill_rate

        # Safety bound (prevent infinite sleeps if config is bad)
        if sleep_time > 60:
            logger.warning(f"RateLimiter request to sleep {sleep_time:.2f}s clamped to 60s")
            sleep_time = 60

        logger.info(
            f"RATE_LIMIT_SLEEP: op={op} cost={cost} remaining={self.tokens:.2f} sleep={sleep_time:.3f}s"
        )
        self._sleep(sleep_time)

        # After sleep, refill and consume
        self._refill()
        self.tokens -= cost
        self._log_usage(op, cost)

    def _log_usage(self, op: str, cost: int):
        # Debug level to avoid spam, unless approaching limit?
        # For P17 verification, might want distinct logs.
        # Keeping it DEBUG for high volume, INFO for block/sleep.
        logger.debug(f"RATE_LIMIT: op={op} cost={cost} remaining={self.tokens:.2f}")

    def stats(self) -> dict[str, Any]:
        self._refill()
        return {
            "enabled": self.enabled,
            "mode": self.mode,
            "tokens": self.tokens,
            "capacity": self.capacity,
        }
