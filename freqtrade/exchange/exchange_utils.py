"""
Exchange support utils
"""
from datetime import datetime, timedelta, timezone
from math import ceil
from typing import Any, Dict, List, Optional, Tuple

import ccxt
from ccxt import ROUND_DOWN, ROUND_UP, TICK_SIZE, TRUNCATE, decimal_to_precision

from freqtrade.exchange.common import BAD_EXCHANGES, EXCHANGE_HAS_OPTIONAL, EXCHANGE_HAS_REQUIRED
from freqtrade.util import FtPrecise


CcxtModuleType = Any


def is_exchange_known_ccxt(
        exchange_name: str, ccxt_module: Optional[CcxtModuleType] = None) -> bool:
    return exchange_name in ccxt_exchanges(ccxt_module)


def ccxt_exchanges(ccxt_module: Optional[CcxtModuleType] = None) -> List[str]:
    """
    Return the list of all exchanges known to ccxt
    """
    return ccxt_module.exchanges if ccxt_module is not None else ccxt.exchanges


def available_exchanges(ccxt_module: Optional[CcxtModuleType] = None) -> List[str]:
    """
    Return exchanges available to the bot, i.e. non-bad exchanges in the ccxt list
    """
    exchanges = ccxt_exchanges(ccxt_module)
    return [x for x in exchanges if validate_exchange(x)[0]]


def validate_exchange(exchange: str) -> Tuple[bool, str]:
    ex_mod = getattr(ccxt, exchange.lower())()
    if not ex_mod or not ex_mod.has:
        return False, ''
    missing = [k for k in EXCHANGE_HAS_REQUIRED if ex_mod.has.get(k) is not True]
    if missing:
        return False, f"missing: {', '.join(missing)}"

    missing_opt = [k for k in EXCHANGE_HAS_OPTIONAL if not ex_mod.has.get(k)]

    if exchange.lower() in BAD_EXCHANGES:
        return False, BAD_EXCHANGES.get(exchange.lower(), '')
    if missing_opt:
        return True, f"missing opt: {', '.join(missing_opt)}"

    return True, ''


def validate_exchanges(all_exchanges: bool) -> List[Tuple[str, bool, str]]:
    """
    :return: List of tuples with exchangename, valid, reason.
    """
    exchanges = ccxt_exchanges() if all_exchanges else available_exchanges()
    exchanges_valid = [
        (e, *validate_exchange(e)) for e in exchanges
    ]
    return exchanges_valid


def timeframe_to_seconds(timeframe: str) -> int:
    """
    Translates the timeframe interval value written in the human readable
    form ('1m', '5m', '1h', '1d', '1w', etc.) to the number
    of seconds for one timeframe interval.
    """
    return ccxt.Exchange.parse_timeframe(timeframe)


def timeframe_to_minutes(timeframe: str) -> int:
    """
    Same as timeframe_to_seconds, but returns minutes.
    """
    return ccxt.Exchange.parse_timeframe(timeframe) // 60


def timeframe_to_msecs(timeframe: str) -> int:
    """
    Same as timeframe_to_seconds, but returns milliseconds.
    """
    return ccxt.Exchange.parse_timeframe(timeframe) * 1000


def timeframe_to_prev_date(timeframe: str, date: Optional[datetime] = None) -> datetime:
    """
    Use Timeframe and determine the candle start date for this date.
    Does not round when given a candle start date.
    :param timeframe: timeframe in string format (e.g. "5m")
    :param date: date to use. Defaults to now(utc)
    :returns: date of previous candle (with utc timezone)
    """
    if not date:
        date = datetime.now(timezone.utc)

    new_timestamp = ccxt.Exchange.round_timeframe(timeframe, date.timestamp() * 1000,
                                                  ROUND_DOWN) // 1000
    return datetime.fromtimestamp(new_timestamp, tz=timezone.utc)


def timeframe_to_next_date(timeframe: str, date: Optional[datetime] = None) -> datetime:
    """
    Use Timeframe and determine next candle.
    :param timeframe: timeframe in string format (e.g. "5m")
    :param date: date to use. Defaults to now(utc)
    :returns: date of next candle (with utc timezone)
    """
    if not date:
        date = datetime.now(timezone.utc)
    new_timestamp = ccxt.Exchange.round_timeframe(timeframe, date.timestamp() * 1000,
                                                  ROUND_UP) // 1000
    return datetime.fromtimestamp(new_timestamp, tz=timezone.utc)


def date_minus_candles(
        timeframe: str, candle_count: int, date: Optional[datetime] = None) -> datetime:
    """
    subtract X candles from a date.
    :param timeframe: timeframe in string format (e.g. "5m")
    :param candle_count: Amount of candles to subtract.
    :param date: date to use. Defaults to now(utc)

    """
    if not date:
        date = datetime.now(timezone.utc)

    tf_min = timeframe_to_minutes(timeframe)
    new_date = timeframe_to_prev_date(timeframe, date) - timedelta(minutes=tf_min * candle_count)
    return new_date


def market_is_active(market: Dict) -> bool:
    """
    Return True if the market is active.
    """
    # "It's active, if the active flag isn't explicitly set to false. If it's missing or
    # true then it's true. If it's undefined, then it's most likely true, but not 100% )"
    # See https://github.com/ccxt/ccxt/issues/4874,
    # https://github.com/ccxt/ccxt/issues/4075#issuecomment-434760520
    return market.get('active', True) is not False


def amount_to_contracts(amount: float, contract_size: Optional[float]) -> float:
    """
    Convert amount to contracts.
    :param amount: amount to convert
    :param contract_size: contract size - taken from exchange.get_contract_size(pair)
    :return: num-contracts
    """
    if contract_size and contract_size != 1:
        return float(FtPrecise(amount) / FtPrecise(contract_size))
    else:
        return amount


def contracts_to_amount(num_contracts: float, contract_size: Optional[float]) -> float:
    """
    Takes num-contracts and converts it to contract size
    :param num_contracts: number of contracts
    :param contract_size: contract size - taken from exchange.get_contract_size(pair)
    :return: Amount
    """

    if contract_size and contract_size != 1:
        return float(FtPrecise(num_contracts) * FtPrecise(contract_size))
    else:
        return num_contracts


def amount_to_precision(amount: float, amount_precision: Optional[float],
                        precisionMode: Optional[int]) -> float:
    """
    Returns the amount to buy or sell to a precision the Exchange accepts
    Re-implementation of ccxt internal methods - ensuring we can test the result is correct
    based on our definitions.
    :param amount: amount to truncate
    :param amount_precision: amount precision to use.
                             should be retrieved from markets[pair]['precision']['amount']
    :param precisionMode: precision mode to use. Should be used from precisionMode
                          one of ccxt's DECIMAL_PLACES, SIGNIFICANT_DIGITS, or TICK_SIZE
    :return: truncated amount
    """
    if amount_precision is not None and precisionMode is not None:
        precision = int(amount_precision) if precisionMode != TICK_SIZE else amount_precision
        # precision must be an int for non-ticksize inputs.
        amount = float(decimal_to_precision(amount, rounding_mode=TRUNCATE,
                                            precision=precision,
                                            counting_mode=precisionMode,
                                            ))

    return amount


def amount_to_contract_precision(
        amount, amount_precision: Optional[float], precisionMode: Optional[int],
        contract_size: Optional[float]) -> float:
    """
    Returns the amount to buy or sell to a precision the Exchange accepts
    including calculation to and from contracts.
    Re-implementation of ccxt internal methods - ensuring we can test the result is correct
    based on our definitions.
    :param amount: amount to truncate
    :param amount_precision: amount precision to use.
                             should be retrieved from markets[pair]['precision']['amount']
    :param precisionMode: precision mode to use. Should be used from precisionMode
                          one of ccxt's DECIMAL_PLACES, SIGNIFICANT_DIGITS, or TICK_SIZE
    :param contract_size: contract size - taken from exchange.get_contract_size(pair)
    :return: truncated amount
    """
    if amount_precision is not None and precisionMode is not None:
        contracts = amount_to_contracts(amount, contract_size)
        amount_p = amount_to_precision(contracts, amount_precision, precisionMode)
        return contracts_to_amount(amount_p, contract_size)
    return amount


def price_to_precision(price: float, price_precision: Optional[float],
                       precisionMode: Optional[int]) -> float:
    """
    Returns the price rounded up to the precision the Exchange accepts.
    Partial Re-implementation of ccxt internal method decimal_to_precision(),
    which does not support rounding up
    TODO: If ccxt supports ROUND_UP for decimal_to_precision(), we could remove this and
    align with amount_to_precision().
    !!! Rounds up
    :param price: price to convert
    :param price_precision: price precision to use. Used from markets[pair]['precision']['price']
    :param precisionMode: precision mode to use. Should be used from precisionMode
                          one of ccxt's DECIMAL_PLACES, SIGNIFICANT_DIGITS, or TICK_SIZE
    :return: price rounded up to the precision the Exchange accepts

    """
    if price_precision is not None and precisionMode is not None:
        # price = float(decimal_to_precision(price, rounding_mode=ROUND,
        #                                    precision=price_precision,
        #                                    counting_mode=self.precisionMode,
        #                                    ))
        if precisionMode == TICK_SIZE:
            precision = FtPrecise(price_precision)
            price_str = FtPrecise(price)
            missing = price_str % precision
            if not missing == FtPrecise("0"):
                price = round(float(str(price_str - missing + precision)), 14)
        else:
            symbol_prec = price_precision
            big_price = price * pow(10, symbol_prec)
            price = ceil(big_price) / pow(10, symbol_prec)
    return price
