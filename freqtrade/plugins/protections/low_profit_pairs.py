
import logging
from datetime import datetime, timedelta
from typing import Any, Dict

from freqtrade.persistence import Trade
from freqtrade.plugins.protections import IProtection, ProtectionReturn


logger = logging.getLogger(__name__)


class LowProfitPairs(IProtection):

    def __init__(self, config: Dict[str, Any], protection_config: Dict[str, Any]) -> None:
        super().__init__(config, protection_config)

        self._lookback_period = protection_config.get('lookback_period', 60)
        self._trade_limit = protection_config.get('trade_limit', 1)
        self._stop_duration = protection_config.get('stop_duration', 60)
        self._required_profit = protection_config.get('required_profit', 0.0)

    def short_desc(self) -> str:
        """
        Short method description - used for startup-messages
        """
        return (f"{self.name} - Low Profit Protection, locks pairs with "
                f"profit < {self._required_profit} within {self._lookback_period} minutes.")

    def _reason(self, profit: float) -> str:
        """
        LockReason to use
        """
        return (f'{profit} < {self._required_profit} in {self._lookback_period} min, '
                f'locking for {self._stop_duration} min.')

    def _low_profit(self, date_now: datetime, pair: str) -> ProtectionReturn:
        """
        Evaluate recent trades for pair
        """
        look_back_until = date_now - timedelta(minutes=self._lookback_period)
        filters = [
            Trade.is_open.is_(False),
            Trade.close_date > look_back_until,
        ]
        if pair:
            filters.append(Trade.pair == pair)
        trades = Trade.get_trades(filters).all()
        if len(trades) < self._trade_limit:
            # Not enough trades in the relevant period
            return False, None, None

        profit = sum(trade.close_profit for trade in trades)
        if profit < self._required_profit:
            self.log_on_refresh(
                logger.info,
                f"Trading for {pair} stopped due to {profit:.2f} < {self._required_profit} "
                f"within {self._lookback_period} minutes.")
            until = self.calculate_lock_end(trades, self._stop_duration)

            return True, until, self._reason(profit)

        return False, None, None

    def global_stop(self, date_now: datetime) -> ProtectionReturn:
        """
        Stops trading (position entering) for all pairs
        This must evaluate to true for the whole period of the "cooldown period".
        :return: Tuple of [bool, until, reason].
            If true, all pairs will be locked with <reason> until <until>
        """
        return False, None, None

    def stop_per_pair(self, pair: str, date_now: datetime) -> ProtectionReturn:
        """
        Stops trading (position entering) for this pair
        This must evaluate to true for the whole period of the "cooldown period".
        :return: Tuple of [bool, until, reason].
            If true, this pair will be locked with <reason> until <until>
        """
        return self._low_profit(date_now, pair=pair)
