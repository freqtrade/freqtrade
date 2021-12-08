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

    @staticmethod
    def from_string(value: str) -> 'CandleType':
        if not value:
            # Default to spot
            return CandleType.SPOT_
        return CandleType(value)

    @staticmethod
    def get_default(trading_mode: str) -> 'CandleType':
        if trading_mode == 'futures':
            return CandleType.FUTURES
        # TODO-lev: The below should be SPOT, not SPOT_
        return CandleType.SPOT_
