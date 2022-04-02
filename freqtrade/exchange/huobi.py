""" Huobi exchange subclass """
import logging
from typing import Dict

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

    def stoploss_adjust(self, stop_loss: float, order: Dict, side: str) -> bool:
        """
        Verify stop_loss against stoploss-order value (limit or price)
        Returns True if adjustment is necessary.
        """
        return order['type'] == 'stop' and stop_loss > float(order['stopPrice'])

    def _get_stop_params(self, ordertype: str, stop_price: float) -> Dict:

        params = self._params.copy()
        params.update({
            "stopPrice": stop_price,
            "operator": "lte",
        })
        return params
