from datetime import datetime
from typing import Optional

from freqtrade.exceptions import OperationalException


def funding_fees(
    exchange_name: str,
    pair: str,
    contract_size: float,
    open_date: datetime,
    close_date: datetime
    # index_price: float,
    # interest_rate: float
):
    """
        Equation to calculate funding_fees on futures trades

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
    # fees = 0
    if exchange_name == "binance":
        for timeslot in ["23:59:45", "07:59:45", "15:59:45"]:
            # for each day in close_date - open_date
            #   mark_price = mark_price at this time
            #   rate = rate at this time
            #   fees = fees + funding_fee(exchange_name, contract_size, mark_price, rate)
            # return fees
            return
    elif exchange_name == "kraken":
        raise OperationalException("Funding_fees has not been implemented for Kraken")
    elif exchange_name == "ftx":
        # for timeslot in every hour since open_date:
        #   mark_price = mark_price at this time
        #   fees = fees + funding_fee(exchange_name, contract_size, mark_price, rate)
        return
    else:
        raise OperationalException(f"Leverage not available on {exchange_name} with freqtrade")


def funding_fee(
    exchange_name: str,
    contract_size: float,
    mark_price: float,
    rate: Optional[float],
    # index_price: float,
    # interest_rate: float
):
    """
        Calculates a single funding fee
    """
    if exchange_name == "binance":
        assert isinstance(rate, float)
        nominal_value = mark_price * contract_size
        adjustment = nominal_value * rate
        return adjustment
    elif exchange_name == "kraken":
        raise OperationalException("Funding fee has not been implemented for kraken")
    elif exchange_name == "ftx":
        """
            Always paid in USD on FTX # TODO: How do we account for this
        """
        (contract_size * mark_price) / 24
        return
