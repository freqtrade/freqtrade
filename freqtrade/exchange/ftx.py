""" FTX exchange subclass """
import logging
from typing import Any, Dict

from freqtrade.exchange import Exchange

logger = logging.getLogger(__name__)


class Ftx(Exchange):

    _ft_has: Dict = {
        "ohlcv_candle_limit": 1500,
    }

    def market_is_tradable(self, market: Dict[str, Any]) -> bool:
        """
        Check if the market symbol is tradable by Freqtrade.
        Default checks + check if pair is spot pair (no futures trading yet).
        """
        parent_check = super().market_is_tradable(market)

        return (parent_check and
                market.get('spot', False) is True)
