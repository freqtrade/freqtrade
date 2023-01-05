from typing import Dict, List, Optional, Tuple, TypedDict

from freqtrade.enums import CandleType


class Ticker(TypedDict):
    symbol: str
    ask: Optional[float]
    askVolume: Optional[float]
    bid: Optional[float]
    bidVolume: Optional[float]
    last: Optional[float]
    quoteVolume: Optional[float]
    baseVolume: Optional[float]
    # Several more - only listing required.


Tickers = Dict[str, Ticker]

# pair, timeframe, candleType, OHLCV, drop last?,
OHLCVResponse = Tuple[str, str, CandleType, List, bool]
