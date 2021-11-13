""" Bybit exchange subclass """
import logging
from typing import Dict, List, Tuple

from freqtrade.enums import Collateral, TradingMode
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
        "ccxt_futures_name": "linear"
    }

    funding_fee_times: List[int] = [0, 8, 16]  # hours of the day

    _supported_trading_mode_collateral_pairs: List[Tuple[TradingMode, Collateral]] = [
        # TradingMode.SPOT always supported and not required in this list
        # (TradingMode.FUTURES, Collateral.CROSS),  # TODO-lev: Uncomment once supported
        # (TradingMode.FUTURES, Collateral.ISOLATED) # TODO-lev: Uncomment once supported
    ]
