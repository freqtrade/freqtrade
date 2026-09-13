"""
Per-CandleType column schemas for stored candle data.

Not every candle type is a candle. Funding rates for example carry exactly one value
per timestamp, so storing them in the full OHLCV shape wastes four columns per row and
gives them names that don't describe their content. Open interest carries two.

This module is the single authority for "which columns does this candle type have".
It is deliberately dependency-free (no pandas) so that `freqtrade.constants` can
re-export from it without pulling pandas into every import of the bot.
"""

from freqtrade.enums import CandleType


OHLCV_COLUMNS = ["date", "open", "high", "low", "close", "volume"]
_FUNDING_RATE_COLUMNS = ["date", "funding_rate"]
# Open interest is reported in the base currency ("amount") and the quote currency ("value").
# Which of the two an exchange fills depends on the exchange and the market - e.g. Bybit reports
# only "amount" on linear markets - so either column can legitimately be all-NaN.
_OPEN_INTEREST_COLUMNS = ["date", "open_interest_amount", "open_interest_value"]

_OHLCV_AGG = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "max"}

# Columns persisted per candle type. Types not listed here use OHLCV_COLUMNS.
# Every entry must start with "date" - it is mandatory for all candle types.
_CANDLE_TYPE_COLUMNS: dict[str, list[str]] = {
    CandleType.FUNDING_RATE: _FUNDING_RATE_COLUMNS,
    CandleType.OPEN_INTEREST: _OPEN_INTEREST_COLUMNS,
}

# Funding rates used to be stored as OHLCV candles, with the rate in "open".
# Old files are read through this mapping, and "open" is kept as an in-memory alias so
# existing strategies keep working - it is not written back to disk.
# This is the only alias there is - candle types added since are stored under their own names.
FUNDING_RATE_LEGACY_RENAME = {"open": "funding_rate"}

# Every column that can legitimately hold candle data. Used to keep such columns out of
# dtype downcasting.
ALL_CANDLE_VALUE_COLUMNS: frozenset[str] = frozenset(
    {*OHLCV_COLUMNS[1:], *_FUNDING_RATE_COLUMNS[1:], *_OPEN_INTEREST_COLUMNS[1:]}
)


def get_candle_columns(candle_type: CandleType | str | None) -> list[str]:
    """
    Columns persisted on disk for this candle type - always starting with "date".
    Accepts plain strings and None (DataProvider and informative pairs use "" for spot).
    :param candle_type: Candle type to use (spot, futures, funding_rate, ...)
    :return: List of column names
    """
    if not candle_type:
        return OHLCV_COLUMNS
    return _CANDLE_TYPE_COLUMNS.get(candle_type, OHLCV_COLUMNS)


def get_candle_dtypes(candle_type: CandleType | str | None) -> dict[str, str]:
    """
    astype() mapping for the value columns of this candle type.
    Some exchanges return ints for values TA-LIB expects to be floats.
    :param candle_type: Candle type to use (spot, futures, funding_rate, ...)
    :return: Mapping of column name to dtype, excluding "date"
    """
    return {col: "float" for col in get_candle_columns(candle_type)[1:]}


def get_candle_agg_dict(candle_type: CandleType | str | None) -> dict[str, str]:
    """
    groupby("date") aggregation used to eliminate duplicate candles.
    Single-value candle types have nothing to aggregate, so they take the first value.
    :param candle_type: Candle type to use (spot, futures, funding_rate, ...)
    :return: Mapping of column name to aggregation function
    """
    columns = get_candle_columns(candle_type)
    if columns is OHLCV_COLUMNS:
        return dict(_OHLCV_AGG)
    return {col: "first" for col in columns[1:]}


def candle_type_is_ohlcv(candle_type: CandleType | str | None) -> bool:
    """
    Whether this candle type uses the standard OHLCV column layout.
    Single-value types (e.g. funding rates) cannot be resampled or filled up like candles.
    :param candle_type: Candle type to use (spot, futures, funding_rate, ...)
    :return: True if the candle type stores regular OHLCV columns
    """
    return get_candle_columns(candle_type) is OHLCV_COLUMNS
