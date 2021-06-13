
import logging
from datetime import datetime, timedelta

from freqtrade.persistence import Trade
from freqtrade.plugins.protections import IProtection, ProtectionReturn


logger = logging.getLogger(__name__)


class CooldownPeriod(IProtection):

    has_global_stop: bool = False
    has_local_stop: bool = True

    def _reason(self) -> str:
        """
        LockReason to use
        """
        return (f'Cooldown period for {self.stop_duration_str}.')

    def short_desc(self) -> str:
        """
        Short method description - used for startup-messages
        """
        return (f"{self.name} - Cooldown period of {self.stop_duration_str}.")

    def _cooldown_period(self, pair: str, date_now: datetime, ) -> ProtectionReturn:
        """
        Get last trade for this pair
        """
        look_back_until = date_now - timedelta(minutes=self._stop_duration)
        # filters = [
        #     Trade.is_open.is_(False),
        #     Trade.close_date > look_back_until,
        #     Trade.pair == pair,
        # ]
        # trade = Trade.get_trades(filters).first()
        trades = Trade.get_trades_proxy(pair=pair, is_open=False, close_date=look_back_until)
        if trades:
            # Get latest trade
            # Ignore type error as we know we only get closed trades.
            trade = sorted(trades, key=lambda t: t.close_date)[-1]  # type: ignore
            self.log_once(f"Cooldown for {pair} for {self.stop_duration_str}.", logger.info)
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
