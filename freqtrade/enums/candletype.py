from enum import Enum


class CandleType(str, Enum):
    """Enum to distinguish candle types"""
    SPOT = "spot"
    SPOT_ = ""
    FUTURES = "futures"
    MARK = "mark"
    INDEX = "index"
    PREMIUMINDEX = "premiumIndex"
    # TODO-lev: not sure this belongs here, as the datatype is really different
    FUNDING_RATE = "funding_rate"
