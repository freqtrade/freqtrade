"""Kucoin exchange subclass."""
import logging
from typing import Dict

from freqtrade.constants import BuySell
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
        "order_time_in_force": ['GTC', 'FOK', 'IOC'],
        "ohlcv_candle_limit": 1500,
    }

    def _get_stop_params(self, side: BuySell, ordertype: str, stop_price: float) -> Dict:

        params = self._params.copy()
        params.update({
            'stopPrice': stop_price,
            'stop': 'loss'
            })
        return params

    def create_order(
            self,
            *,
            pair: str,
            ordertype: str,
            side: BuySell,
            amount: float,
            rate: float,
            leverage: float,
            reduceOnly: bool = False,
            time_in_force: str = 'GTC',
            ) -> Dict:

        res = super().create_order(
            pair=pair,
            ordertype=ordertype,
            side=side,
            amount=amount,
            rate=rate,
            leverage=leverage,
            reduceOnly=reduceOnly,
            time_in_force=time_in_force,
        )
        # Kucoin returns only the order-id.
        # ccxt returns status = 'closed' at the moment - which is information ccxt invented.
        # Since we rely on status heavily, we must set it to 'open' here.
        # ref: https://github.com/ccxt/ccxt/pull/16674, (https://github.com/ccxt/ccxt/pull/16553)
        if not self._config['dry_run']:
            res['type'] = ordertype
            res['status'] = 'open'
        return res
