import logging
from typing import Dict, List, Tuple

from freqtrade.enums import MarginMode, TradingMode
from freqtrade.exchange import Exchange


logger = logging.getLogger(__name__)


class Okex(Exchange):
    """Okex exchange class.

    Contains adjustments needed for Freqtrade to work with this exchange.
    """

    _ft_has: Dict = {
        "ohlcv_candle_limit": 300,
        "mark_ohlcv_timeframe": "4h",
        "funding_fee_timeframe": "8h",
    }

    _supported_trading_mode_margin_pairs: List[Tuple[TradingMode, MarginMode]] = [
        # TradingMode.SPOT always supported and not required in this list
        # (TradingMode.MARGIN, MarginMode.CROSS),
        # (TradingMode.FUTURES, MarginMode.CROSS),
        # (TradingMode.FUTURES, MarginMode.ISOLATED)
    ]
