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
    percentage: Optional[float]
    # Several more - only listing required.


Tickers = Dict[str, Ticker]


class OrderBook(TypedDict):
    symbol: str
    bids: List[Tuple[float, float]]
    asks: List[Tuple[float, float]]
    timestamp: Optional[int]
    datetime: Optional[str]
    nonce: Optional[int]


class CcxtBalance(TypedDict):
    free: float
    used: float
    total: float


CcxtBalances = Dict[str, CcxtBalance]


class CcxtPosition(TypedDict):
    symbol: str
    side: str
    contracts: float
    leverage: float
    collateral: Optional[float]
    initialMargin: Optional[float]
    liquidationPrice: Optional[float]


# pair, timeframe, candleType, OHLCV, drop last?,
OHLCVResponse = Tuple[str, str, CandleType, List, bool]
