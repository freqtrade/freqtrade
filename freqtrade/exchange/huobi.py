""" Huobi exchange subclass """
import logging
from typing import Dict

from freqtrade.constants import BuySell
from freqtrade.exchange import Exchange


logger = logging.getLogger(__name__)


class Huobi(Exchange):
    """
    Huobi exchange class. Contains adjustments needed for Freqtrade to work
    with this exchange.
    """

    _ft_has: Dict = {
        "stoploss_on_exchange": True,
        "stoploss_order_types": {"limit": "stop-limit"},
        "ohlcv_candle_limit": 1000,
        "l2_limit_range": [5, 10, 20],
        "l2_limit_range_required": False,
    }

    def _get_stop_params(self, side: BuySell, ordertype: str, stop_price: float) -> Dict:

        params = self._params.copy()
        params.update({
            "stopPrice": stop_price,
            "operator": "lte",
        })
        return params
