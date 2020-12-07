
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from freqtrade.exchange import timeframe_to_minutes
from freqtrade.mixins import LoggingMixin
from freqtrade.persistence import Trade


logger = logging.getLogger(__name__)

ProtectionReturn = Tuple[bool, Optional[datetime], Optional[str]]


class IProtection(LoggingMixin, ABC):

    # Can globally stop the bot
    has_global_stop: bool = False
    # Can stop trading for one pair
    has_local_stop: bool = False

    def __init__(self, config: Dict[str, Any], protection_config: Dict[str, Any]) -> None:
        self._config = config
        self._protection_config = protection_config
        tf_in_min = timeframe_to_minutes(config['timeframe'])
        if 'stop_duration_candles' in protection_config:
            self._stop_duration = (tf_in_min * protection_config.get('stop_duration_candles'))
        else:
            self._stop_duration = protection_config.get('stop_duration', 60)
        if 'lookback_period_candles' in protection_config:
            self._lookback_period = tf_in_min * protection_config.get('lookback_period_candles', 60)
        else:
            self._lookback_period = protection_config.get('lookback_period', 60)

        LoggingMixin.__init__(self, logger)

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    def short_desc(self) -> str:
        """
        Short method description - used for startup-messages
        -> Please overwrite in subclasses
        """

    @abstractmethod
    def global_stop(self, date_now: datetime) -> ProtectionReturn:
        """
        Stops trading (position entering) for all pairs
        This must evaluate to true for the whole period of the "cooldown period".
        """

    @abstractmethod
    def stop_per_pair(self, pair: str, date_now: datetime) -> ProtectionReturn:
        """
        Stops trading (position entering) for this pair
        This must evaluate to true for the whole period of the "cooldown period".
        :return: Tuple of [bool, until, reason].
            If true, this pair will be locked with <reason> until <until>
        """

    @staticmethod
    def calculate_lock_end(trades: List[Trade], stop_minutes: int) -> datetime:
        """
        Get lock end time
        """
        max_date: datetime = max([trade.close_date for trade in trades])
        # comming from Database, tzinfo is not set.
        if max_date.tzinfo is None:
            max_date = max_date.replace(tzinfo=timezone.utc)

        until = max_date + timedelta(minutes=stop_minutes)

        return until
