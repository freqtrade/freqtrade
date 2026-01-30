import logging
import os
import time
from typing import Optional
from freqtrade.exceptions import OperationalException

logger = logging.getLogger(__name__)


class DegradedModeGuard:
    """
    Degraded Mode (Circuit Breaker) for BreezeCCXT shim.
    blocks order entries when in degraded state.
    """

    def __init__(self):
        # Configuration
        self.degraded = os.environ.get("FT_DEGRADED_MODE", "0") == "1"
        self.block_entries = os.environ.get("FT_DEGRADED_BLOCK_ENTRIES", "1") == "1"

        # Internal state for auto-trigger (optional for P17, but good to have foundation)
        self.failures = 0
        self.last_failure_ts = 0
        self.failure_threshold = 3
        self.failure_window = 60

        if self.degraded:
            logger.warning("DEGRADED MODE FORCED via FT_DEGRADED_MODE=1")
        else:
            # P34: Load Persistence
            try:
                from adapters.ccxt_shim import health_snapshot

                state = health_snapshot.load()
                cb = state.get("circuit_breaker", {})
                if cb.get("tripped"):
                    tripped_at = cb.get("tripped_at", 0)
                    now = time.time()
                    # Check if still valid (using failure_window or separate cooldown)
                    # For now, using failure_window logic (if it was recent enough to still be relevant?)
                    # Actually, if it tripped, it should stay tripped until explicitly reset or timeout.
                    # Let's say we honor it if within 2 * failure_window (example)
                    if now - tripped_at < (self.failure_window * 10):
                        self.failures = cb.get("failures", self.failure_threshold)
                        self.last_failure_ts = tripped_at
                        logger.warning(
                            f"DegradedModeGuard: Restored TRIPPED state from snapshot (ts={tripped_at})"
                        )
            except Exception as e:
                logger.warning(f"DegradedModeGuard: Failed to load persistence: {e}")

    def record_failure(self, exc: Exception) -> None:
        """
        Record a network/API failure.
        """
        now = time.time()
        # Reset if outside window
        if now - self.last_failure_ts > self.failure_window:
            self.failures = 0

        # Check for Policy Blocks (Safety checks shouldn't trigger circuit breaker)
        from adapters.ccxt_shim.policy_codes import is_safety_block

        if is_safety_block(str(exc)):
            logger.info(f"DegradedModeGuard: Ignoring policy block: {exc}")
            return

        self.failures += 1
        self.last_failure_ts = now
        logger.warning(
            f"DegradedModeGuard: Failure recorded ({self.failures}/{self.failure_threshold}) - {exc}"
        )

        if self.failures >= self.failure_threshold:
            from adapters.ccxt_shim.alerts import trigger

            trigger("DEGRADED_ENTER", f"Circuit Breaker Tripped. Last Error: {exc}")

            # P34 Persistence
            try:
                from adapters.ccxt_shim import health_snapshot

                health_snapshot.update(
                    "circuit_breaker",
                    {"tripped": True, "tripped_at": now, "failures": self.failures},
                )
            except Exception as e:
                logger.warning(f"DegradedModeGuard: Failed to persist failure: {e}")

        # Auto-trigger if enabled (P17 focus is mostly on Forced Mode, but logic handles it)
        # Note: Auto-trigger logic not fully enforced via env var in P17 plan, relies on forced mode primarily.
        # But we can expose state.

    def record_success(self) -> None:
        """
        Record a success. Could heal degraded mode if we had auto-healing.
        """
        if self.failures > 0:
            self.failures = 0
            # If we were auto-degraded, we could recover here.
            # But forced mode sticks.

    def is_degraded(self) -> bool:
        return self.degraded or (self.failures >= self.failure_threshold)

    def assert_can_order(self, side: str, symbol: str) -> None:
        """
        Check if order is allowed.
        Entry orders blocked in degraded mode.
        Exit orders allowed (safety).
        """
        if not self.is_degraded():
            return

        # If degraded, checking logic
        is_entry = (
            side.lower() == "buy"
        )  # Simplified assumptions for Shim: Buy=Entry, Sell=Exit usually.
        # But for Shorting? Shim context is usually Long-Only India Equity?
        # Let's stick to Plan: "if degraded and side=='buy' (entry): block"

        if is_entry and self.block_entries:
            logger.warning(f"DEGRADED_BLOCK entry: symbol={symbol} side={side}")
            raise OperationalException(f"degraded_block: {side} {symbol}")

        # Sells/Exits allowed
        if side.lower() == "sell":
            logger.info(f"Degraded Mode: Allowing exit order for {symbol}")
