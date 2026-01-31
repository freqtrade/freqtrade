"""
Market Hours Guard for ICICI Breeze Shim.

Enforces NSE trading hours (09:15 - 15:30 IST) at the shim boundary.
Provides deterministic overrides for testing.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, time, timedelta, timezone

from freqtrade.exceptions import OperationalException

logger = logging.getLogger(__name__)

# NSE Market Hours (IST)
MARKET_OPEN = time(9, 15)
MARKET_CLOSE = time(15, 30)

# IST Offset (+05:30)
IST_OFFSET = timezone(timedelta(hours=5, minutes=30))


class MarketHoursGuard:
    """
    Guard to block entry orders outside of NSE trading hours.
    """

    def __init__(self):
        self._force_open = False
        self._force_closed = False
        self._reload_overrides()

    def _reload_overrides(self):
        """Reload overrides from environment variables."""
        self._force_open = os.environ.get("FT_FORCE_MARKET_OPEN") == "1"
        self._force_closed = os.environ.get("FT_FORCE_MARKET_CLOSED") == "1"

    def is_market_open(self, now: datetime | None = None) -> bool:
        """
        Check if market is currently open.

        Args:
            now: Optional datetime to check against (defaults to current UTC converted to IST).
                 Can be naive (assumed UTC) or aware.

        Returns:
            bool: True if open, False if closed.
        """
        self._reload_overrides()

        if self._force_open:
            return True
        if self._force_closed:
            return False

        if now is None:
            # Check for deterministic time injection (Mock/Backtest)
            injected_time = os.environ.get("FT_IST_NOW")
            if injected_time:
                try:
                    now = datetime.fromisoformat(injected_time)
                except ValueError:
                    logger.warning(
                        f"Invalid FT_IST_NOW format: {injected_time}. Using system time."
                    )
                    now = datetime.now(timezone.utc)
            else:
                now = datetime.now(timezone.utc)

        # Ensure we are in IST
        if now.tzinfo is None:
            # Assume UTC if naive
            now = now.replace(tzinfo=timezone.utc)

        # Convert to IST
        now_ist = now.astimezone(IST_OFFSET)

        # 1. Check Weekend (Mon=0, Sun=6)
        # 5=Sat, 6=Sun
        if now_ist.weekday() >= 5:
            return False

        # 2. Check Time
        current_time = now_ist.time()
        if MARKET_OPEN <= current_time < MARKET_CLOSE:
            return True

        return False

    def assert_can_create_order(self, side: str, symbol: str):
        """
        Assert that an order can be created.
        Blocks 'buy' orders (entries) if market is closed.
        Allows 'sell' orders (exits) always for safety.

        Args:
            side: 'buy' or 'sell'
            symbol: Pair symbol

        Raises:
            Exception: If blocked (CCXT-style message pattern).
        """
        if side.lower() == "buy":
            # Entry logic - requires market open
            if not self.is_market_open():
                msg = f"market_closed: blocking entry order (buy) for {symbol} outside NSE hours"
                logger.info(
                    {
                        "event": "market_hours_block",
                        "action": "create_order",
                        "side": side,
                        "symbol": symbol,
                        "reason": msg,
                    }
                )
                # Use a standard Exception that Freqtrade caught handle, or specific CCXT error if imported
                # Using generic Exception with specific string that Freqtrade might log
                raise OperationalException(f"market_hours_block:{msg}")

        # Sell/Exits are always allowed (read-only + close position)

    def assert_can_cancel_order(self, order_id: str, symbol: str):
        """
        Policy: Allow cancels always? Or only during hours?
        Requirements say: "cancel/edit: block by default (unless explicitly allowed later)"
        "block by default... if market_closed"

        Wait, requirements say: "cancel/edit: block by default" under 'decision_rule -> if market_closed'.
        """
        if not self.is_market_open():
            # Strict safety or loose?
            # Requirement: "block by default"
            msg = f"market_closed: blocking cancel order {order_id} for {symbol} outside NSE hours"
            logger.info(
                {
                    "event": "market_hours_block",
                    "action": "cancel_order",
                    "order_id": order_id,
                    "symbol": symbol,
                    "reason": msg,
                }
            )
            raise OperationalException(f"market_hours_block:{msg}")

    def assert_can_edit_order(self, order_id: str, symbol: str):
        """
        Policy: Block edits outside market hours.
        """
        if not self.is_market_open():
            msg = f"market_closed: blocking edit order {order_id} for {symbol} outside NSE hours"
            logger.info(
                {
                    "event": "market_hours_block",
                    "action": "edit_order",
                    "order_id": order_id,
                    "symbol": symbol,
                    "reason": msg,
                }
            )
            raise OperationalException(f"market_hours_block:{msg}")
