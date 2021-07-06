from enum import Enum, auto
from decimal import Decimal

one = Decimal(1.0)
four = Decimal(4.0)
twenty_four = Decimal(24.0)


class FunctionProxy:
    """Allow to mask a function as an Object."""

    def __init__(self, function):
        self.function = function

    def __call__(self, *args, **kwargs):
        return self.function(*args, **kwargs)


class InterestMode(Enum):
    """Equations to calculate interest"""

    # Interest_rate is per day, minimum time of 1 hour
    HOURSPERDAY = FunctionProxy(
        lambda borrowed, rate, hours: borrowed * rate * max(hours, one)/twenty_four
    )

    # Interest_rate is per 4 hours, minimum time of 4 hours
    HOURSPER4 = FunctionProxy(
        lambda borrowed, rate, hours: borrowed * rate * (1 + max(0, (hours-four)/four))
    )
