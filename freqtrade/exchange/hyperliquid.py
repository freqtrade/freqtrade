"""Hyperliquid exchange subclass"""

import logging

from freqtrade.enums import TradingMode
from freqtrade.exchange import Exchange
from freqtrade.exchange.exchange_types import FtHas


logger = logging.getLogger(__name__)


class Hyperliquid(Exchange):
    """Hyperliquid exchange class.
    Contains adjustments needed for Freqtrade to work with this exchange.
    """

    _ft_has: FtHas = {
        # Only the most recent 5000 candles are available according to the
        # exchange's API documentation.
        "ohlcv_has_history": False,
        "ohlcv_candle_limit": 5000,
        "trades_has_history": False,  # Trades endpoint doesn't seem available.
        "exchange_has_overrides": {"fetchTrades": False},
    }

    @property
    def _ccxt_config(self) -> dict:
        # Parameters to add directly to ccxt sync/async initialization.
        # ccxt defaults to swap mode.
        config = {}
        if self.trading_mode == TradingMode.SPOT:
            config.update({"options": {"defaultType": "spot"}})
        config.update(super()._ccxt_config)
        return config
