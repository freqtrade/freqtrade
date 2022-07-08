# flake8: noqa: E501

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

from freqtrade.constants import LongShort
from freqtrade.plugins.protections import IProtection, ProtectionReturn


logger = logging.getLogger(__name__)


class TradingTime(IProtection):

    has_global_stop: bool = True
    has_local_stop: bool = False

    def __init__(self, config: Dict[str, Any], protection_config: Dict[str, Any]) -> None:
        super().__init__(config, protection_config)
        self._start_time = datetime.strptime(protection_config.get('start_time', '00:00'), "%H:%M")
        self._end_time = datetime.strptime(protection_config.get('end_time', '23:59'), "%H:%M")
        now = datetime.now()

        self._update_trading_period(now)

    def _update_trading_period(self, now: datetime) -> None:
        self.trade_start, self.trade_end = (
                now.replace(hour=self._start_time.hour, minute=self._start_time.minute, second=0),
                now.replace(hour=self._end_time.hour, minute=self._end_time.minute, second=0)
            )

        if now < self.trade_start:
            self.next_trading_day = self.trade_start
        else:
            self.next_trading_day = self.trade_start + timedelta(days=1)


    def short_desc(self) -> str:
        """
        Short method description - used for startup-messages
        """
        return (f"{self.name} - Limit trading to time period")

    def _reason(self) -> str:
        """
        LockReason to use
        """
        return (f'Current time is not in Allowed time period {self.trade_start}-{self.trade_end}'
                f'trading locked until {self.next_trading_day}.')

    def _allowed_trading_period(self, date_now: datetime) -> Optional[ProtectionReturn]:
        """
        Evaluate recent trades for drawdown ...
        """
        now = date_now
        self._update_trading_period(now)

        if not (self.trade_start < now < self.trade_end):
            return ProtectionReturn(
                lock=True,
                until=self.next_trading_day,
                reason=self._reason()
            )

        return None

    def global_stop(self, date_now: datetime, side: LongShort) -> Optional[ProtectionReturn]:
        """
        Stops trading (position entering) for all pairs
        This must evaluate to true for the whole period of the "cooldown period".
        :return: Tuple of [bool, until, reason].
            If true, all pairs will be locked with <reason> until <until>
        """
        return self._allowed_trading_period(date_now)

    def stop_per_pair(
            self, pair: str, date_now: datetime, side: LongShort) -> Optional[ProtectionReturn]:
        """
        Stops trading (position entering) for this pair
        This must evaluate to true for the whole period of the "cooldown period".
        :return: Tuple of [bool, until, reason].
            If true, this pair will be locked with <reason> until <until>
        """
        return None
