from enum import Enum


class CandleType(str, Enum):
    """Enum to distinguish candle types"""
    SPOT = "spot"
    FUTURES = "futures"
    MARK = "mark"
    INDEX = "index"
    PREMIUMINDEX = "premiumIndex"

    # TODO: Could take up less memory if these weren't a CandleType
    FUNDING_RATE = "funding_rate"
    # BORROW_RATE = "borrow_rate"  # * unimplemented

    def __str__(self):
        return f"{self.name.lower()}"

    @staticmethod
    def from_string(value: str) -> 'CandleType':
        return CandleType.SPOT if not value else CandleType(value)

    @staticmethod
    def get_default(trading_mode: str) -> 'CandleType':
        return CandleType.FUTURES if trading_mode == 'futures' else CandleType.SPOT
