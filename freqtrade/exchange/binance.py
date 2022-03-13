""" Binance exchange subclass """
import logging
from typing import Dict, List, Tuple

import arrow

from freqtrade.exchange import Exchange


logger = logging.getLogger(__name__)


class Binance(Exchange):

    _ft_has: Dict = {
        "stoploss_on_exchange": True,
        "stoploss_order_types": {"limit": "stop_loss_limit"},
        "order_time_in_force": ['gtc', 'fok', 'ioc'],
        "time_in_force_parameter": "timeInForce",
        "ohlcv_candle_limit": 1000,
        "trades_pagination": "id",
        "trades_pagination_arg": "fromId",
        "l2_limit_range": [5, 10, 20, 50, 100, 500, 1000],
    }

    def stoploss_adjust(self, stop_loss: float, order: Dict) -> bool:
        """
        Verify stop_loss against stoploss-order value (limit or price)
        Returns True if adjustment is necessary.
        """
        return order['type'] == 'stop_loss_limit' and stop_loss > float(order['info']['stopPrice'])

    async def _async_get_historic_ohlcv(self, pair: str, timeframe: str,
                                        since_ms: int, is_new_pair: bool = False,
                                        raise_: bool = False
                                        ) -> Tuple[str, str, List]:
        """
        Overwrite to introduce "fast new pair" functionality by detecting the pair's listing date
        Does not work for other exchanges, which don't return the earliest data when called with "0"
        """
        if is_new_pair:
            x = await self._async_get_candle_history(pair, timeframe, 0)
            if x and x[2] and x[2][0] and x[2][0][0] > since_ms:
                # Set starting date to first available candle.
                since_ms = x[2][0][0]
                logger.info(f"Candle-data for {pair} available starting with "
                            f"{arrow.get(since_ms // 1000).isoformat()}.")
        return await super()._async_get_historic_ohlcv(
            pair=pair, timeframe=timeframe, since_ms=since_ms, is_new_pair=is_new_pair,
            raise_=raise_)
