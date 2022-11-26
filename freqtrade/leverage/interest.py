from math import ceil

from freqtrade.exceptions import OperationalException
from freqtrade.util import FtPrecise


one = FtPrecise(1.0)
four = FtPrecise(4.0)
twenty_four = FtPrecise(24.0)


def interest(
    exchange_name: str,
    borrowed: FtPrecise,
    rate: FtPrecise,
    hours: FtPrecise
) -> FtPrecise:
    """
    Equation to calculate interest on margin trades

    :param exchange_name: The exchanged being trading on
    :param borrowed: The amount of currency being borrowed
    :param rate: The rate of interest (i.e daily interest rate)
    :param hours: The time in hours that the currency has been borrowed for

    Raises:
        OperationalException: Raised if freqtrade does
        not support margin trading for this exchange

    Returns: The amount of interest owed (currency matches borrowed)
    """
    exchange_name = exchange_name.lower()
    if exchange_name == "binance":
        return borrowed * rate * FtPrecise(ceil(hours)) / twenty_four
    elif exchange_name == "kraken":
        # Rounded based on https://kraken-fees-calculator.github.io/
        return borrowed * rate * (one + FtPrecise(ceil(hours / four)))
    else:
        raise OperationalException(f"Leverage not available on {exchange_name} with freqtrade")
