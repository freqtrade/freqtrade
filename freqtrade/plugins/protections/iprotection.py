import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Union

from freqtrade.constants import Config, LongShort
from freqtrade.exchange import timeframe_to_minutes
from freqtrade.misc import plural
from freqtrade.mixins import LoggingMixin
from freqtrade.persistence import LocalTrade


logger = logging.getLogger(__name__)


@dataclass
class ProtectionReturn:
    lock: bool
    until: datetime
    reason: Optional[str]
    lock_side: str = "*"


class IProtection(LoggingMixin, ABC):
    # Can globally stop the bot
    has_global_stop: bool = False
    # Can stop trading for one pair
    has_local_stop: bool = False

    def __init__(self, config: Config, protection_config: Dict[str, Any]) -> None:
        self._config = config
        self._protection_config = protection_config
        self._stop_duration_candles: Optional[int] = None
        self._stop_duration: int = 0
        self._lookback_period_candles: Optional[int] = None
        self._unlock_at: Optional[datetime] = None

        tf_in_min = timeframe_to_minutes(config["timeframe"])
        if "stop_duration_candles" in protection_config:
            self._stop_duration_candles = int(protection_config.get("stop_duration_candles", 1))
            self._stop_duration = tf_in_min * self._stop_duration_candles
        elif "unlock_at" in protection_config:
            self._unlock_at = self.calculate_unlock_at()
        else:
            self._stop_duration = int(protection_config.get("stop_duration", 60))

        if "lookback_period_candles" in protection_config:
            self._lookback_period_candles = int(protection_config.get("lookback_period_candles", 1))
            self._lookback_period = tf_in_min * self._lookback_period_candles
        else:
            self._lookback_period_candles = None
            self._lookback_period = int(protection_config.get("lookback_period", 60))

        LoggingMixin.__init__(self, logger)

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @property
    def stop_duration_str(self) -> str:
        """
        Output configured stop duration in either candles or minutes
        """
        if self._stop_duration_candles:
            return (
                f"{self._stop_duration_candles} "
                f"{plural(self._stop_duration_candles, 'candle', 'candles')}"
            )
        else:
            return f"{self._stop_duration} {plural(self._stop_duration, 'minute', 'minutes')}"

    @property
    def lookback_period_str(self) -> str:
        """
        Output configured lookback period in either candles or minutes
        """
        if self._lookback_period_candles:
            return (
                f"{self._lookback_period_candles} "
                f"{plural(self._lookback_period_candles, 'candle', 'candles')}"
            )
        else:
            return f"{self._lookback_period} {plural(self._lookback_period, 'minute', 'minutes')}"

    @property
    def unlock_at_str(self) -> Union[str, None]:
        """
        Output configured unlock time
        """
        if self._unlock_at:
            return self._unlock_at.strftime("%H:%M")
        return None

    @property
    def unlock_reason_time_element(self) -> str:
        """
        Output configured unlock time or stop duration
        """
        if self.unlock_at_str is not None:
            return f"until {self.unlock_at_str}"
        else:
            return f"for {self.stop_duration_str}"

    def calculate_unlock_at(self) -> datetime:
        """
        Calculate and update the unlock time based on the unlock at config.
        """
        now_time = datetime.now(timezone.utc)
        unlock_at = datetime.strptime(
            str(self._protection_config.get("unlock_at")), "%H:%M"
        ).replace(day=now_time.day, year=now_time.year, month=now_time.month)

        if unlock_at.time() < now_time.time():
            unlock_at = unlock_at.replace(day=now_time.day + 1)

        return unlock_at.replace(tzinfo=timezone.utc)

    @abstractmethod
    def short_desc(self) -> str:
        """
        Short method description - used for startup-messages
        -> Please overwrite in subclasses
        """

    @abstractmethod
    def global_stop(self, date_now: datetime, side: LongShort) -> Optional[ProtectionReturn]:
        """
        Stops trading (position entering) for all pairs
        This must evaluate to true for the whole period of the "cooldown period".
        """

    @abstractmethod
    def stop_per_pair(
        self, pair: str, date_now: datetime, side: LongShort
    ) -> Optional[ProtectionReturn]:
        """
        Stops trading (position entering) for this pair
        This must evaluate to true for the whole period of the "cooldown period".
        :return: Tuple of [bool, until, reason].
            If true, this pair will be locked with <reason> until <until>
        """

    @staticmethod
    def calculate_lock_end(trades: List[LocalTrade], stop_minutes: int) -> datetime:
        """
        Get lock end time
        """
        max_date: datetime = max([trade.close_date for trade in trades if trade.close_date])
        # coming from Database, tzinfo is not set.
        if max_date.tzinfo is None:
            max_date = max_date.replace(tzinfo=timezone.utc)

        until = max_date + timedelta(minutes=stop_minutes)
        return until
