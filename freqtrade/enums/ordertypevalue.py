from enum import StrEnum


class OrderTypeValues(StrEnum):
    limit = "limit"
    market = "market"
    trailing_stop_market = "trailing_stop_market"
    trailing_stop = "trailing_stop"
