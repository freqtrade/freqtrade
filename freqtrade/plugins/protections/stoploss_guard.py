
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

from freqtrade.constants import Config, LongShort
from freqtrade.enums import ExitType
from freqtrade.persistence import Trade
from freqtrade.plugins.protections import IProtection, ProtectionReturn


logger = logging.getLogger(__name__)


class StoplossGuard(IProtection):

    has_global_stop: bool = True
    has_local_stop: bool = True

    def __init__(self, config: Config, protection_config: Dict[str, Any]) -> None:
        super().__init__(config, protection_config)

        self._trade_limit = protection_config.get('trade_limit', 10)
        self._disable_global_stop = protection_config.get('only_per_pair', False)
        self._only_per_side = protection_config.get('only_per_side', False)
        self._profit_limit = protection_config.get('required_profit', 0.0)

    def short_desc(self) -> str:
        """
        Short method description - used for startup-messages
        """
        return (f"{self.name} - Frequent Stoploss Guard, {self._trade_limit} stoplosses "
                f"with profit < {self._profit_limit:.2%} within {self.lookback_period_str}.")

    def _reason(self) -> str:
        """
        LockReason to use
        """
        return (f'{self._trade_limit} stoplosses in {self._lookback_period} min, '
                f'locking for {self._stop_duration} min.')

    def _stoploss_guard(self, date_now: datetime, pair: Optional[str],
                        side: LongShort) -> Optional[ProtectionReturn]:
        """
        Evaluate recent trades
        """
        look_back_until = date_now - timedelta(minutes=self._lookback_period)

        trades1 = Trade.get_trades_proxy(pair=pair, is_open=False, close_date=look_back_until)
        trades = [trade for trade in trades1 if (str(trade.exit_reason) in (
            ExitType.TRAILING_STOP_LOSS.value, ExitType.STOP_LOSS.value,
            ExitType.STOPLOSS_ON_EXCHANGE.value, ExitType.LIQUIDATION.value)
            and trade.close_profit and trade.close_profit < self._profit_limit)]

        if self._only_per_side:
            # Long or short trades only
            trades = [trade for trade in trades if trade.trade_direction == side]

        if len(trades) < self._trade_limit:
            return None

        self.log_once(f"Trading stopped due to {self._trade_limit} "
                      f"stoplosses within {self._lookback_period} minutes.", logger.info)
        until = self.calculate_lock_end(trades, self._stop_duration)
        return ProtectionReturn(
            lock=True,
            until=until,
            reason=self._reason(),
            lock_side=(side if self._only_per_side else '*')
            )

    def global_stop(self, date_now: datetime, side: LongShort) -> Optional[ProtectionReturn]:
        """
        Stops trading (position entering) for all pairs
        This must evaluate to true for the whole period of the "cooldown period".
        :return: Tuple of [bool, until, reason].
            If true, all pairs will be locked with <reason> until <until>
        """
        if self._disable_global_stop:
            return None
        return self._stoploss_guard(date_now, None, side)

    def stop_per_pair(
            self, pair: str, date_now: datetime, side: LongShort) -> Optional[ProtectionReturn]:
        """
        Stops trading (position entering) for this pair
        This must evaluate to true for the whole period of the "cooldown period".
        :return: Tuple of [bool, until, reason].
            If true, this pair will be locked with <reason> until <until>
        """
        return self._stoploss_guard(date_now, pair, side)
