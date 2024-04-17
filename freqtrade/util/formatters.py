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
    return value.rstrip('0').rstrip('.')


def round_value(value: float, decimals: int, keep_trailing_zeros=False) -> str:
    """
    Round value to given decimals
    :param value: Value to be rounded
    :param decimals: Number of decimals to round to
    :param keep_trailing_zeros: Keep trailing zeros "222.200" vs. "222.2"
    :return: Rounded value as string
    """
    val = f"{value:.{decimals}f}"
    if not keep_trailing_zeros:
        val = strip_trailing_zeros(val)
    return val


def fmt_coin(
        value: float, coin: str, show_coin_name=True, keep_trailing_zeros=False) -> str:
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
