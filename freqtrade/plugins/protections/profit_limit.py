
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

from freqtrade.constants import Config, LongShort
from freqtrade.persistence import Trade
from freqtrade.plugins.protections import IProtection, ProtectionReturn


logger = logging.getLogger(__name__)


class ProfitLimit(IProtection):

    has_global_stop: bool = True
    has_local_stop: bool = False

    def __init__(self, config: Config, protection_config: Dict[str, Any]) -> None:
        super().__init__(config, protection_config)

        self._trade_limit = protection_config.get('trade_limit', 1)
        self._required_profit = protection_config.get('profit_limit', 1.0)

    def short_desc(self) -> str:
        """
        Short method description - used for startup-messages
        """
        return (f"{self.name} - Profit Limit Protection, locks all pairs when "
                f"profit > {self._required_profit} within {self.lookback_period_str}.")

    def _reason(self, profit: float) -> str:
        """
        LockReason to use
        """
        return (f'{profit} > {self._required_profit} in {self.lookback_period_str}, '
                f'locking for {self.stop_duration_str}.')

    def _limit_profit(
            self, date_now: datetime) -> Optional[ProtectionReturn]:
        """
        Evaluate recent trades for pair
        """
        look_back_until = date_now - timedelta(minutes=self._lookback_period)

        trades = Trade.get_trades_proxy(is_open=False, close_date=look_back_until)

        if len(trades) < self._trade_limit:
            # Not enough trades in the relevant period
            return None

        profit_sum = sum(trade.close_profit_abs for trade in trades if trade.close_profit_abs)
        stake_sum = sum(trade.stake_amount for trade in trades)
        profit_ratio = profit_sum / stake_sum

        if profit_ratio >= self._required_profit:
            self.log_once(
                f"Trading stopped due to {profit_ratio:.2f} >= {self._required_profit} "
                f"within {self._lookback_period} minutes.", logger.info)
            until = self.calculate_lock_end(trades, self._stop_duration)

            return ProtectionReturn(
                lock=True,
                until=until,
                reason=self._reason(profit_ratio)
            )

        return None

    def global_stop(self, date_now: datetime, side: LongShort) -> Optional[ProtectionReturn]:
        """
        Stops trading (position entering) for all pairs
        This must evaluate to true for the whole period of the "cooldown period".
        :return: Tuple of [bool, until, reason].
            If true, all pairs will be locked with <reason> until <until>
        """
        return self._limit_profit(date_now)

    def stop_per_pair(
            self, pair: str, date_now: datetime, side: LongShort) -> Optional[ProtectionReturn]:
        """
        Stops trading (position entering) for this pair
        This must evaluate to true for the whole period of the "cooldown period".
        :return: Tuple of [bool, until, reason].
            If true, this pair will be locked with <reason> until <until>
        """
        return None
