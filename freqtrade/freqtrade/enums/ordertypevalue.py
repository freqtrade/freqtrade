from enum import Enum


class OrderTypeValues(str, Enum):
    limit = "limit"
    market = "market"
