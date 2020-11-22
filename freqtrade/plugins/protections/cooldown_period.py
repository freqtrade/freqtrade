
import logging
from datetime import datetime, timedelta
from typing import Any, Dict

from freqtrade.persistence import Trade
from freqtrade.plugins.protections import IProtection, ProtectionReturn


logger = logging.getLogger(__name__)


class CooldownPeriod(IProtection):

    # Can globally stop the bot
    has_global_stop: bool = False
    # Can stop trading for one pair
    has_local_stop: bool = True

    def __init__(self, config: Dict[str, Any], protection_config: Dict[str, Any]) -> None:
        super().__init__(config, protection_config)

        self._stop_duration = protection_config.get('stop_duration', 60)

    def _reason(self) -> str:
        """
        LockReason to use
        """
        return (f'Cooldown period for {self._stop_duration} min.')

    def short_desc(self) -> str:
        """
        Short method description - used for startup-messages
        """
        return (f"{self.name} - Cooldown period of {self._stop_duration} min.")

    def _cooldown_period(self, pair: str, date_now: datetime, ) -> ProtectionReturn:
        """
        Get last trade for this pair
        """
        look_back_until = date_now - timedelta(minutes=self._stop_duration)
        filters = [
            Trade.is_open.is_(False),
            Trade.close_date > look_back_until,
            Trade.pair == pair,
        ]
        trade = Trade.get_trades(filters).first()
        if trade:
            self.log_once(f"Cooldown for {pair} for {self._stop_duration}.", logger.info)
            until = self.calculate_lock_end([trade], self._stop_duration)

            return True, until, self._reason()

        return False, None, None

    def global_stop(self, date_now: datetime) -> ProtectionReturn:
        """
        Stops trading (position entering) for all pairs
        This must evaluate to true for the whole period of the "cooldown period".
        :return: Tuple of [bool, until, reason].
            If true, all pairs will be locked with <reason> until <until>
        """
        # Not implemented for cooldown period.
        return False, None, None

    def stop_per_pair(self, pair: str, date_now: datetime) -> ProtectionReturn:
        """
        Stops trading (position entering) for this pair
        This must evaluate to true for the whole period of the "cooldown period".
        :return: Tuple of [bool, until, reason].
            If true, this pair will be locked with <reason> until <until>
        """
        return self._cooldown_period(pair, date_now)
