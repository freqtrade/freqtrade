"""Kucoin exchange subclass."""
import logging
from typing import Dict

from freqtrade.exchange import Exchange


logger = logging.getLogger(__name__)


class Kucoin(Exchange):
    """Kucoin exchange class.

    Contains adjustments needed for Freqtrade to work with this exchange.

    Please note that this exchange is not included in the list of exchanges
    officially supported by the Freqtrade development team. So some features
    may still not work as expected.
    """

    _ft_has: Dict = {
        "stoploss_on_exchange": True,
        "stoploss_order_types": {"limit": "limit", "market": "market"},
        "l2_limit_range": [20, 100],
        "l2_limit_range_required": False,
        "order_time_in_force": ['gtc', 'fok', 'ioc'],
        "time_in_force_parameter": "timeInForce",
        "ohlcv_candle_limit": 1500,
    }

    def stoploss_adjust(self, stop_loss: float, order: Dict, side: str) -> bool:
        """
        Verify stop_loss against stoploss-order value (limit or price)
        Returns True if adjustment is necessary.
        """
        return order['info'].get('stop') is not None and stop_loss > float(order['stopPrice'])

    def _get_stop_params(self, ordertype: str, stop_price: float) -> Dict:

        params = self._params.copy()
        params.update({
            'stopPrice': stop_price,
            'stop': 'loss'
            })
        return params
