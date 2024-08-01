"""Hyperliquid exchange subclass"""

import logging
from typing import Dict

from freqtrade.exchange import Exchange


logger = logging.getLogger(__name__)


class Hyperliquid(Exchange):
    """Hyperliquid exchange class.
    Contains adjustments needed for Freqtrade to work with this exchange.
    """

    _ft_has: Dict = {
        # Only the most recent 5000 candles are available according to the
        # exchange's API documentation.
        "ohlcv_has_history": False,
        "ohlcv_candle_limit": 5000,
        "trades_has_history": False,  # Trades endpoint doesn't seem available.
        "exchange_has_overrides": {"fetchTrades": False},
    }
