""" Gate.io exchange subclass """
import logging
from typing import Dict

from freqtrade.exceptions import OperationalException
from freqtrade.exchange import Exchange


logger = logging.getLogger(__name__)


class Gateio(Exchange):
    """
    Gate.io exchange class. Contains adjustments needed for Freqtrade to work
    with this exchange.

    Please note that this exchange is not included in the list of exchanges
    officially supported by the Freqtrade development team. So some features
    may still not work as expected.
    """

    _ft_has: Dict = {
        "ohlcv_candle_limit": 1000,
        "ohlcv_volume_currency": "quote",
        "stoploss_order_types": {"limit": "limit"},
        "stoploss_on_exchange": True,
    }

    def validate_ordertypes(self, order_types: Dict) -> None:
        super().validate_ordertypes(order_types)

        if any(v == 'market' for k, v in order_types.items()):
            raise OperationalException(
                f'Exchange {self.name} does not support market orders.')

    def fetch_stoploss_order(self, order_id: str, pair: str, params={}) -> Dict:
        return self.fetch_order(
            order_id=order_id,
            pair=pair,
            params={'stop': True}
        )

    def cancel_stoploss_order(self, order_id: str, pair: str, params={}) -> Dict:
        return self.cancel_order(
            order_id=order_id,
            pair=pair,
            params={'stop': True}
        )

    def stoploss_adjust(self, stop_loss: float, order: Dict) -> bool:
        """
        Verify stop_loss against stoploss-order value (limit or price)
        Returns True if adjustment is necessary.
        """
        return stop_loss > float(order['stopPrice'])
