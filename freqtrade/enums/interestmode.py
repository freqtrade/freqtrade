from decimal import Decimal
from enum import Enum

from freqtrade.exceptions import OperationalException


one = Decimal(1.0)
four = Decimal(4.0)
twenty_four = Decimal(24.0)


class InterestMode(Enum):
    """Equations to calculate interest"""

    HOURSPERDAY = "HOURSPERDAY"
    HOURSPER4 = "HOURSPER4"  # Hours per 4 hour segment
    NONE = "NONE"

    def __call__(self, *args, **kwargs):

        borrowed, rate, hours = kwargs["borrowed"], kwargs["rate"], kwargs["hours"]

        if self.name == "HOURSPERDAY":
            return borrowed * rate * max(hours, one)/twenty_four
        elif self.name == "HOURSPER4":
            return borrowed * rate * (1 + max(0, (hours-four)/four))
        else:
            raise OperationalException("Leverage not available on this exchange with freqtrade")
