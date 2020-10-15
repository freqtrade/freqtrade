
import logging
from datetime import datetime, timedelta
from typing import Any, Dict

from sqlalchemy import and_, or_

from freqtrade.persistence import Trade
from freqtrade.plugins.protections import IProtection
from freqtrade.strategy.interface import SellType


logger = logging.getLogger(__name__)


class StoplossGuard(IProtection):

    def __init__(self, config: Dict[str, Any], protection_config: Dict[str, Any]) -> None:
        super().__init__(config, protection_config)
        self._lookback_period = protection_config.get('lookback_period', 60)
        self._trade_limit = protection_config.get('trade_limit', 10)

    def short_desc(self) -> str:
        """
        Short method description - used for startup-messages
        """
        return f"{self.name} - Frequent Stoploss Guard"

    def _stoploss_guard(self, date_now: datetime, pair: str = None) -> bool:
        """
        Evaluate recent trades
        """
        look_back_until = date_now - timedelta(minutes=self._lookback_period)
        filters = [
            Trade.is_open.is_(False),
            Trade.close_date > look_back_until,
            or_(Trade.sell_reason == SellType.STOP_LOSS.value,
                and_(Trade.sell_reason == SellType.TRAILING_STOP_LOSS.value,
                     Trade.close_profit < 0))
        ]
        if pair:
            filters.append(Trade.pair == pair)
        trades = Trade.get_trades(filters).all()

        if len(trades) > self._trade_limit:
            self.log_on_refresh(logger.info, f"Trading stopped due to {self._trade_limit} "
                                f"stoplosses within {self._lookback_period} minutes.")
            return True

        return False

    def global_stop(self, date_now: datetime) -> bool:
        """
        Stops trading (position entering) for all pairs
        This must evaluate to true for the whole period of the "cooldown period".
        """
        return self._stoploss_guard(date_now, pair=None)
