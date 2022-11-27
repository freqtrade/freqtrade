""" Bybit exchange subclass """
import logging
from typing import Dict, List, Tuple

from freqtrade.enums import MarginMode, TradingMode
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
        "ohlcv_candle_limit": 1000,
        "ccxt_futures_name": "linear",
        "ohlcv_has_history": False,
    }
    _ft_has_futures: Dict = {
        "ohlcv_has_history": True,
    }

    _supported_trading_mode_margin_pairs: List[Tuple[TradingMode, MarginMode]] = [
        # TradingMode.SPOT always supported and not required in this list
        # (TradingMode.FUTURES, MarginMode.CROSS),
        # (TradingMode.FUTURES, MarginMode.ISOLATED)
    ]

    @property
    def _ccxt_config(self) -> Dict:
        # Parameters to add directly to ccxt sync/async initialization.
        # ccxt defaults to swap mode.
        config = {}
        if self.trading_mode == TradingMode.SPOT:
            config.update({
                "options": {
                    "defaultType": "spot"
                }
            })
        config.update(super()._ccxt_config)
        return config
