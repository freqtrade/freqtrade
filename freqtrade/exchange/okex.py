import logging
from typing import Dict, List, Tuple

from freqtrade.enums import Collateral, TradingMode
from freqtrade.exchange import Exchange


logger = logging.getLogger(__name__)


class Okex(Exchange):
    """
    Okex exchange class. Contains adjustments needed for Freqtrade to work
    with this exchange.
    """

    _ft_has: Dict = {
        "ohlcv_candle_limit": 100,
    }
    funding_fee_times: List[int] = [0, 8, 16]  # hours of the day

    _supported_trading_mode_collateral_pairs: List[Tuple[TradingMode, Collateral]] = [
        # TradingMode.SPOT always supported and not required in this list
        # (TradingMode.MARGIN, Collateral.CROSS),  # TODO-lev: Uncomment once supported
        # (TradingMode.FUTURES, Collateral.CROSS),  # TODO-lev: Uncomment once supported
        # (TradingMode.FUTURES, Collateral.ISOLATED) # TODO-lev: Uncomment once supported
    ]
