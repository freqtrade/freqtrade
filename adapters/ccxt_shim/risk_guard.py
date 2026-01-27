import logging
import os
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from freqtrade.exceptions import OperationalException

logger = logging.getLogger(__name__)


class RiskGuard:
    def __init__(self, config: dict[str, Any]):
        self.config = config.get("risk_guard", {})
        self.enabled = self.config.get("enabled", True)
        self.max_trades_per_day = self.config.get("max_trades_per_day", 10)
        self.max_open_positions = self.config.get("max_open_positions", 1)
        self.green_day_lock_pct = self.config.get("green_day_profit_lock_pct", 1.0)
        self.cutoff_time_str = self.config.get("intraday_entry_cutoff_ist", "15:05")

        self.spread_guard = self.config.get("spread_guard", {})
        self.spread_enabled = self.spread_guard.get("enabled", True)
        self.max_spread_pct = self.spread_guard.get("max_spread_pct", 0.40)

        self.allow_exits_when_blocked = self.config.get("allow_exits_when_blocked", True)

        # In-memory state (non-persistent for P15)
        self.daily_trades_count = 0
        self.ist_tz = ZoneInfo("Asia/Kolkata")
        self.last_reset_date = self.get_now_ist().strftime("%Y-%m-%d")

    def get_now_ist(self) -> datetime:
        # P15: Allow forcing time via ENV for deterministic testing
        forced_now = os.environ.get("FT_IST_NOW")
        if forced_now:
            try:
                # Expect ISO format: "2026-01-26T15:10:00+05:30"
                return datetime.fromisoformat(forced_now)
            except ValueError:
                logger.warning(f"Invalid FT_IST_NOW format: {forced_now}. Using system time.")

        return datetime.now(self.ist_tz)

    def _reset_daily_counters_if_needed(self, now_ist: datetime):
        today_str = now_ist.strftime("%Y-%m-%d")
        if today_str != self.last_reset_date:
            logger.info(f"RiskGuard: Resetting daily counters. New Day: {today_str}")
            self.daily_trades_count = 0
            self.last_reset_date = today_str

    def should_block_entry(self, symbol: str, side: str, price_surface: dict) -> tuple[bool, str]:
        """
        Check if entry should be blocked.
        Returns: (blocked: bool, reason: str)
        """
        if not self.enabled:
            return False, ""

        # P15: Pnl blocking logic is placeholder until Ledger implementation
        # Can force block via env for testing
        if os.environ.get("FT_FORCE_RISK_BLOCK"):
            return True, "force_block_env"

        # 1. Check Side (Exits may be exempt)
        is_entry = side.lower() == "buy"
        if not is_entry and self.allow_exits_when_blocked:
            return False, ""

        now_ist = self.get_now_ist()
        self._reset_daily_counters_if_needed(now_ist)

        # 2. Max Trades Per Day
        if self.daily_trades_count >= self.max_trades_per_day:
            return True, "max_trades_per_day"

        # 3. Intraday Cutoff
        current_time_str = now_ist.strftime("%H:%M")
        if current_time_str >= self.cutoff_time_str:
            return True, "intraday_cutoff"

        # 4. Spread Guard (Entries Only)
        if is_entry and self.spread_enabled and price_surface:
            bid = price_surface.get("bid", 0.0)
            ask = price_surface.get("ask", 0.0)
            if bid > 0 and ask > 0:
                mid = (bid + ask) / 2
                spread_pct = ((ask - bid) / mid) * 100
                if spread_pct > self.max_spread_pct:
                    return True, "spread_guard"

        return False, ""

    def record_trade_attempt(self, symbol: str, side: str):
        """
        Record a successful trade submission to update counters.
        """
        if not self.enabled:
            return

        is_entry = side.lower() == "buy"
        if is_entry:
            self.daily_trades_count += 1
            logger.info(
                f"RiskGuard: Trade recorded. Daily Count: {self.daily_trades_count}/{self.max_trades_per_day}"
            )
