from decimal import Decimal
from enum import Enum
from math import ceil

from freqtrade.exceptions import OperationalException


one = Decimal(1.0)
four = Decimal(4.0)
twenty_four = Decimal(24.0)


class InterestMode(Enum):
    """Equations to calculate interest"""

    HOURSPERDAY = "HOURSPERDAY"
    HOURSPER4 = "HOURSPER4"  # Hours per 4 hour segment
    NONE = "NONE"

    def __call__(self, borrowed: Decimal, rate: Decimal, hours: Decimal):

        if self.name == "HOURSPERDAY":
            return borrowed * rate * ceil(hours)/twenty_four
        elif self.name == "HOURSPER4":
            # Rounded based on https://kraken-fees-calculator.github.io/
            return borrowed * rate * (1+ceil(hours/four))
        else:
            raise OperationalException("Leverage not available on this exchange with freqtrade")
