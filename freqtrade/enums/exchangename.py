from enum import Enum


class ExchangeName(Enum):
    """All the exchanges supported by freqtrade that support leverage"""

    BINANCE = "Binance"
    KRAKEN = "Kraken"
    FTX = "FTX"
    OTHER = None
