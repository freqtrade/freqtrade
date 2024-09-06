"""Bitvavo exchange subclass."""

import logging

from ccxt import DECIMAL_PLACES

from freqtrade.exchange import Exchange
from freqtrade.exchange.exchange_types import FtHas


logger = logging.getLogger(__name__)


class Bitvavo(Exchange):
    """Bitvavo exchange class.

    Contains adjustments needed for Freqtrade to work with this exchange.

    Please note that this exchange is not included in the list of exchanges
    officially supported by the Freqtrade development team. So some features
    may still not work as expected.
    """

    _ft_has: FtHas = {
        "ohlcv_candle_limit": 1440,
    }

    @property
    def precisionMode(self) -> int:
        """
        Exchange ccxt precisionMode
        Override due to https://github.com/ccxt/ccxt/issues/20408
        """
        return DECIMAL_PLACES
