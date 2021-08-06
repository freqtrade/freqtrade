from decimal import Decimal
from math import ceil

from freqtrade.exceptions import OperationalException


one = Decimal(1.0)
four = Decimal(4.0)
twenty_four = Decimal(24.0)


def interest(
    exchange_name: str,
    borrowed: Decimal,
    rate: Decimal,
    hours: Decimal
) -> Decimal:
    """Equation to calculate interest on margin trades


    :param exchange_name: The exchanged being trading on
    :param borrowed: The amount of currency being borrowed
    :param rate: The rate of interest
    :param hours: The time in hours that the currency has been borrowed for

    Raises:
        OperationalException: Raised if freqtrade does
        not support margin trading for this exchange

    Returns: The amount of interest owed (currency matches borrowed)

    """
    exchange_name = exchange_name.lower()
    if exchange_name == "binance":
        return borrowed * rate * ceil(hours)/twenty_four
    elif exchange_name == "kraken":
        # Rounded based on https://kraken-fees-calculator.github.io/
        return borrowed * rate * (one+ceil(hours/four))
    elif exchange_name == "ftx":
        # TODO-lev: Add FTX interest formula
        raise OperationalException(f"Leverage not available on {exchange_name} with freqtrade")
    else:
        raise OperationalException(f"Leverage not available on {exchange_name} with freqtrade")
