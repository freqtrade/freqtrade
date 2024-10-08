"""HTX exchange subclass"""

import logging

from freqtrade.constants import BuySell
from freqtrade.exchange import Exchange
from freqtrade.exchange.exchange_types import FtHas


logger = logging.getLogger(__name__)


class Htx(Exchange):
    """
    HTX exchange class. Contains adjustments needed for Freqtrade to work
    with this exchange.
    """

    _ft_has: FtHas = {
        "stoploss_on_exchange": True,
        "stop_price_param": "stopPrice",
        "stop_price_prop": "stopPrice",
        "stoploss_order_types": {"limit": "stop-limit"},
        "ohlcv_candle_limit": 1000,
        "l2_limit_range": [5, 10, 20],
        "l2_limit_range_required": False,
        "ohlcv_candle_limit_per_timeframe": {
            "1w": 500,
            "1M": 500,
        },
        "trades_has_history": False,  # Endpoint doesn't have a "since" parameter
    }

    def _get_stop_params(self, side: BuySell, ordertype: str, stop_price: float) -> dict:
        params = self._params.copy()
        params.update(
            {
                "stopPrice": stop_price,
                "operator": "lte",
            }
        )
        return params
