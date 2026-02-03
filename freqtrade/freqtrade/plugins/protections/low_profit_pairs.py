import logging
from datetime import datetime, timedelta
from typing import Any

from freqtrade.constants import Config, LongShort
from freqtrade.persistence import Trade
from freqtrade.plugins.protections import IProtection, ProtectionReturn


logger = logging.getLogger(__name__)


class LowProfitPairs(IProtection):
    has_global_stop: bool = False
    has_local_stop: bool = True

    def __init__(self, config: Config, protection_config: dict[str, Any]) -> None:
        super().__init__(config, protection_config)

        self._trade_limit = protection_config.get("trade_limit", 1)
        self._required_profit = protection_config.get("required_profit", 0.0)
        self._only_per_side = protection_config.get("only_per_side", False)

    def short_desc(self) -> str:
        """
        Short method description - used for startup-messages
        """
        return (
            f"{self.name} - Low Profit Protection, locks pairs with "
            f"profit < {self._required_profit} within {self.lookback_period_str}."
        )

    def _reason(self, profit: float) -> str:
        """
        LockReason to use
        """
        return (
            f"{profit} < {self._required_profit} in {self.lookback_period_str}, "
            f"locking {self.unlock_reason_time_element}."
        )

    def _low_profit(
        self, date_now: datetime, pair: str, side: LongShort
    ) -> ProtectionReturn | None:
        """
        Evaluate recent trades for pair
        """
        look_back_until = date_now - timedelta(minutes=self._lookback_period)
        # filters = [
        #     Trade.is_open.is_(False),
        #     Trade.close_date > look_back_until,
        # ]
        # if pair:
        #     filters.append(Trade.pair == pair)

        trades = Trade.get_trades_proxy(pair=pair, is_open=False, close_date=look_back_until)
        # trades = Trade.get_trades(filters).all()
        if len(trades) < self._trade_limit:
            # Not enough trades in the relevant period
            return None

        profit = sum(
            trade.close_profit
            for trade in trades
            if trade.close_profit and (not self._only_per_side or trade.trade_direction == side)
        )
        if profit < self._required_profit:
            self.log_once(
                f"Trading for {pair} stopped due to {profit:.2f} < {self._required_profit} "
                f"within {self._lookback_period} minutes.",
                logger.info,
            )
            until = self.calculate_lock_end(trades)

            return ProtectionReturn(
                lock=True,
                until=until,
                reason=self._reason(profit),
                lock_side=(side if self._only_per_side else "*"),
            )

        return None

    def global_stop(self, date_now: datetime, side: LongShort) -> ProtectionReturn | None:
        """
        Stops trading (position entering) for all pairs
        This must evaluate to true for the whole period of the "cooldown period".
        :return: Tuple of [bool, until, reason].
            If true, all pairs will be locked with <reason> until <until>
        """
        return None

    def stop_per_pair(
        self, pair: str, date_now: datetime, side: LongShort
    ) -> ProtectionReturn | None:
        """
        Stops trading (position entering) for this pair
        This must evaluate to true for the whole period of the "cooldown period".
        :return: Tuple of [bool, until, reason].
            If true, this pair will be locked with <reason> until <until>
        """
        return self._low_profit(date_now, pair=pair, side=side)
