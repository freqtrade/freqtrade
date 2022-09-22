
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

import pandas as pd

from freqtrade.constants import Config, LongShort
from freqtrade.data.metrics import calculate_max_drawdown
from freqtrade.persistence import Trade
from freqtrade.plugins.protections import IProtection, ProtectionReturn


logger = logging.getLogger(__name__)


class MaxDrawdown(IProtection):

    has_global_stop: bool = True
    has_local_stop: bool = False

    def __init__(self, config: Config, protection_config: Dict[str, Any]) -> None:
        super().__init__(config, protection_config)

        self._trade_limit = protection_config.get('trade_limit', 1)
        self._max_allowed_drawdown = protection_config.get('max_allowed_drawdown', 0.0)
        # TODO: Implement checks to limit max_drawdown to sensible values

    def short_desc(self) -> str:
        """
        Short method description - used for startup-messages
        """
        return (f"{self.name} - Max drawdown protection, stop trading if drawdown is > "
                f"{self._max_allowed_drawdown} within {self.lookback_period_str}.")

    def _reason(self, drawdown: float) -> str:
        """
        LockReason to use
        """
        return (f'{drawdown} passed {self._max_allowed_drawdown} in {self.lookback_period_str}, '
                f'locking for {self.stop_duration_str}.')

    def _max_drawdown(self, date_now: datetime) -> Optional[ProtectionReturn]:
        """
        Evaluate recent trades for drawdown ...
        """
        look_back_until = date_now - timedelta(minutes=self._lookback_period)

        trades = Trade.get_trades_proxy(is_open=False, close_date=look_back_until)

        trades_df = pd.DataFrame([trade.to_json() for trade in trades])

        if len(trades) < self._trade_limit:
            # Not enough trades in the relevant period
            return None

        # Drawdown is always positive
        try:
            # TODO: This should use absolute profit calculation, considering account balance.
            drawdown, _, _, _, _, _ = calculate_max_drawdown(trades_df, value_col='close_profit')
        except ValueError:
            return None

        if drawdown > self._max_allowed_drawdown:
            self.log_once(
                f"Trading stopped due to Max Drawdown {drawdown:.2f} > {self._max_allowed_drawdown}"
                f" within {self.lookback_period_str}.", logger.info)
            until = self.calculate_lock_end(trades, self._stop_duration)

            return ProtectionReturn(
                lock=True,
                until=until,
                reason=self._reason(drawdown),
            )

        return None

    def global_stop(self, date_now: datetime, side: LongShort) -> Optional[ProtectionReturn]:
        """
        Stops trading (position entering) for all pairs
        This must evaluate to true for the whole period of the "cooldown period".
        :return: Tuple of [bool, until, reason].
            If true, all pairs will be locked with <reason> until <until>
        """
        return self._max_drawdown(date_now)

    def stop_per_pair(
            self, pair: str, date_now: datetime, side: LongShort) -> Optional[ProtectionReturn]:
        """
        Stops trading (position entering) for this pair
        This must evaluate to true for the whole period of the "cooldown period".
        :return: Tuple of [bool, until, reason].
            If true, this pair will be locked with <reason> until <until>
        """
        return None
