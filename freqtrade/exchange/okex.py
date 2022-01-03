import logging
from typing import Dict, List, Tuple

from freqtrade.enums import Collateral, TradingMode
from freqtrade.exchange import Exchange


logger = logging.getLogger(__name__)


class Okex(Exchange):
    """Okex exchange class.

    Contains adjustments needed for Freqtrade to work with this exchange.
    """

    _ft_has: Dict = {
        "ohlcv_candle_limit": 100,
        "mark_ohlcv_timeframe": "4h",
        "funding_fee_timeframe": "8h",
    }

    _supported_trading_mode_collateral_pairs: List[Tuple[TradingMode, Collateral]] = [
        # TradingMode.SPOT always supported and not required in this list
        # (TradingMode.MARGIN, Collateral.CROSS),
        # (TradingMode.FUTURES, Collateral.CROSS),
        # (TradingMode.FUTURES, Collateral.ISOLATED)
    ]
