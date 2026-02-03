from datetime import timedelta

from numpy import isnan

from freqtrade.constants import DECIMAL_PER_COIN_FALLBACK, DECIMALS_PER_COIN


def decimals_per_coin(coin: str):
    """
    Helper method getting decimal amount for this coin
    example usage: f".{decimals_per_coin('USD')}f"
    :param coin: Which coin are we printing the price / value for
    """
    return DECIMALS_PER_COIN.get(coin, DECIMAL_PER_COIN_FALLBACK)


def strip_trailing_zeros(value: str) -> str:
    """
    Strip trailing zeros from a string
    :param value: Value to be stripped
    :return: Stripped value
    """
    return value.rstrip("0").rstrip(".")


def round_value(value: float | None, decimals: int, keep_trailing_zeros=False) -> str:
    """
    Round value to given decimals
    :param value: Value to be rounded
    :param decimals: Number of decimals to round to
    :param keep_trailing_zeros: Keep trailing zeros "222.200" vs. "222.2"
    :return: Rounded value as string
    """
    if value is None or isnan(value):
        return "N/A"
    val = f"{value:.{decimals}f}"
    if not keep_trailing_zeros:
        val = strip_trailing_zeros(val)
    return val


def fmt_coin(value: float, coin: str, show_coin_name=True, keep_trailing_zeros=False) -> str:
    """
    Format price value for this coin
    :param value: Value to be printed
    :param coin: Which coin are we printing the price / value for
    :param show_coin_name: Return string in format: "222.22 USDT" or "222.22"
    :param keep_trailing_zeros: Keep trailing zeros "222.200" vs. "222.2"
    :return: Formatted / rounded value (with or without coin name)
    """
    val = round_value(value, decimals_per_coin(coin), keep_trailing_zeros)
    if show_coin_name:
        val = f"{val} {coin}"

    return val


def fmt_coin2(
    value: float, coin: str, decimals: int = 8, *, show_coin_name=True, keep_trailing_zeros=False
) -> str:
    """
    Format price value for this coin. Should be preferred for rate formatting
    :param value: Value to be printed
    :param coin: Which coin are we printing the price / value for
    :param decimals: Number of decimals to round to
    :param show_coin_name: Return string in format: "222.22 USDT" or "222.22"
    :param keep_trailing_zeros: Keep trailing zeros "222.200" vs. "222.2"
    :return: Formatted / rounded value (with or without coin name)
    """
    val = round_value(value, decimals, keep_trailing_zeros)
    if show_coin_name:
        val = f"{val} {coin}"

    return val


def format_duration(td: timedelta) -> str:
    """
    Format a timedelta object to "XXd HH:MM" format
    :param td: Timedelta object to format
    :return: Formatted time string
    """
    d = td.days
    h, r = divmod(td.seconds, 3600)
    m, _ = divmod(r, 60)
    return f"{d}d {h:02d}:{m:02d}"


def format_pct(value: float | None) -> str:
    """
    Format a float value as percentage string with 2 decimals
    None and NaN values are formatted as "N/A"
    :param value: Float value to format
    :return: Formatted percentage string
    """
    if value is None or isnan(value):
        return "N/A"
    return f"{value:.2%}"
