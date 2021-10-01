""" Bybit exchange subclass """
import logging
from typing import Dict, List

from freqtrade.exchange import Exchange


logger = logging.getLogger(__name__)


class Bybit(Exchange):
    """
    Bybit exchange class. Contains adjustments needed for Freqtrade to work
    with this exchange.

    Please note that this exchange is not included in the list of exchanges
    officially supported by the Freqtrade development team. So some features
    may still not work as expected.
    """

    _ft_has: Dict = {
        "ohlcv_candle_limit": 200,
    }

    funding_fee_times: List[int] = [0, 8, 16]  # hours of the day
