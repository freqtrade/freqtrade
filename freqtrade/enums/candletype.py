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

    @classmethod
    def from_string(cls, value: str) -> 'CandleType':
        if not value:
            # Default to spot
            return CandleType.SPOT
        return CandleType(value)
