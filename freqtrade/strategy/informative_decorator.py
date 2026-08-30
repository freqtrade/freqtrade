from collections.abc import Callable
from dataclasses import dataclass, field
from time import monotonic
from typing import Any, NoReturn

from cachetools import TLRUCache
from pandas import DataFrame, isna

from freqtrade.candle_columns import get_candle_columns
from freqtrade.enums import CandleType
from freqtrade.exceptions import OperationalException
from freqtrade.exchange import timeframe_to_seconds
from freqtrade.strategy.strategy_helper import (
    _merge_prepared_informative_pair,
    _prepare_informative_pair,
    _PreparedInformative,
    merge_informative_pair,
)


PopulateIndicators = Callable[[Any, DataFrame, dict], DataFrame]


@dataclass(frozen=True, slots=True)
class InformativeCacheKey:
    callback: PopulateIndicators = field(repr=False)
    callback_name: str = field(compare=False, hash=False)
    asset: str
    informative_timeframe: str
    effective_timeframe: str
    candle_type: CandleType | None


_INFORMATIVE_DATE_MERGE = "date_merge"
_INFORMATIVE_CACHE_TTL_CANDLES = 2


@dataclass(slots=True)
class _InformativeCacheEntry:
    fingerprint: tuple[Any, ...]
    prepared: _PreparedInformative


class InformativeCache(TLRUCache[InformativeCacheKey, _InformativeCacheEntry]):
    """LRU cache whose entries expire two informative candles after their last update."""

    def __init__(self, maxsize: int, timer: Callable[[], float] = monotonic) -> None:
        super().__init__(maxsize=maxsize, ttu=self._time_to_use, timer=timer)

    @staticmethod
    def _time_to_use(key: InformativeCacheKey, _entry: _InformativeCacheEntry, now: float) -> float:
        # Using effective timeframe instead of informative timeframe because some informative candle
        # types (like funding_rate) have a different effective timeframe. On most cases, effective
        # timeframe is equal to informative timeframe.
        return now + (
            _INFORMATIVE_CACHE_TTL_CANDLES * timeframe_to_seconds(key.effective_timeframe)
        )


@dataclass
class InformativeData:
    asset: str | None
    timeframe: str
    fmt: str | Callable[[Any], str] | None
    ffill: bool
    candle_type: CandleType | None
    cache: bool = True


def informative(
    timeframe: str,
    asset: str = "",
    fmt: str | Callable[[Any], str] | None = None,
    *,
    candle_type: CandleType | str | None = None,
    ffill: bool = True,
    cache: bool = True,
) -> Callable[[PopulateIndicators], PopulateIndicators]:
    """
    A decorator for populate_indicators_Nn(self, dataframe, metadata), allowing these functions to
    define informative indicators.

    Example usage:

        @informative('1h')
        def populate_indicators_1h(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
            dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
            return dataframe

    :param timeframe: Informative timeframe. Must always be equal or higher than strategy timeframe.
    :param asset: Informative asset, for example BTC, BTC/USDT, ETH/BTC. Do not specify to use
                  current pair. Also supports limited pair format strings (see below)
    :param fmt: Column format (str) or column formatter (callable(name, asset, timeframe)). When not
    specified, defaults to:
    * {base}_{quote}_{column}_{timeframe} if asset is specified.
    * {column}_{timeframe} if asset is not specified.
    Pair format supports these format variables:
    * {base} - base currency in lower case, for example 'eth'.
    * {BASE} - same as {base}, except in upper case.
    * {quote} - quote currency in lower case, for example 'usdt'.
    * {QUOTE} - same as {quote}, except in upper case.
    Format string additionally supports this variables.
    * {asset} - full name of the asset, for example 'BTC/USDT'.
    * {column} - name of dataframe column.
    * {timeframe} - timeframe of informative dataframe.
    :param ffill: ffill dataframe after merging informative pair.
    :param candle_type: Candle type to use (spot, futures, funding_rate, ...)
        None or '' (the default) resolves to the trading mode's candle type.
        Attention: Availability for non-spot/futures candle-types across exchanges may vary.
           funding_rate candles only contain the "funding_rate" column (open for historic reasons)
           open_interest candles only contain the "open_interest_amount" and
           "open_interest_value" columns.
           All other columns will be missing from these dataframes.
    :param cache: Cache populated indicators in dry/live mode while the latest informative candle
                  remains unchanged. Entries expire after two effective informative timeframes
                  without an update. Disable for methods that use external state, have side effects,
                  or otherwise need to run for every base pair. Defaults to True.
    """
    _asset = asset
    _timeframe = timeframe
    _fmt = fmt
    _ffill = ffill
    _candle_type = CandleType.from_string(candle_type) if candle_type else None
    _cache = cache

    def decorator(fn: PopulateIndicators):
        informative_pairs = getattr(fn, "_ft_informative", [])
        informative_pairs.append(
            InformativeData(_asset, _timeframe, _fmt, _ffill, _candle_type, _cache)
        )
        setattr(fn, "_ft_informative", informative_pairs)  # noqa: B010
        return fn

    return decorator


def __get_pair_formats(market: dict[str, Any] | None) -> dict[str, str]:
    if not market:
        return {}
    base = market["base"]
    quote = market["quote"]
    return {
        "base": base.lower(),
        "BASE": base.upper(),
        "quote": quote.lower(),
        "QUOTE": quote.upper(),
    }


def _format_pair_name(config, pair: str, market: dict[str, Any] | None = None) -> str:
    return pair.format(
        stake_currency=config["stake_currency"],
        stake=config["stake_currency"],
        **__get_pair_formats(market),
    ).upper()


# NaN never compares equal to itself, so a NaN anywhere in the fingerprint would make every
# lookup a miss. Some candle types have legitimately empty columns - open interest is reported
# in the base currency, the quote currency, or both, depending on exchange and market - so a
# permanently NaN column must not silently disable the cache.
_MISSING = object()


def _informative_dataframe_fingerprint(
    dataframe: DataFrame, candle_type: CandleType | None
) -> tuple[Any, ...]:
    last_candle = dataframe.iloc[-1]
    values = (last_candle[column] for column in get_candle_columns(candle_type))
    return (len(dataframe), *(_MISSING if isna(value) else value for value in values))


def _raise_reserved_column_name() -> NoReturn:
    raise OperationalException(
        f"Column name '{_INFORMATIVE_DATE_MERGE}' is reserved for informative merging."
    )


def _format_informative_column(
    column: Any,
    formatter: Callable[..., str],
    fmt_args: dict[str, str],
    *,
    preserve_date_merge: bool = False,
) -> str:
    if preserve_date_merge and column == _INFORMATIVE_DATE_MERGE:
        return column
    formatted_column = formatter(column=column, **fmt_args)
    if formatted_column == _INFORMATIVE_DATE_MERGE:
        _raise_reserved_column_name()
    return formatted_column


def _get_populated_informative_dataframe(
    strategy,
    populate_indicators_fn: PopulateIndicators,
    dataframe: DataFrame,
    metadata: dict,
    cache_key: InformativeCacheKey,
    cache: InformativeCache | None,
    timeframe: str,
) -> DataFrame:
    """
    Populate the informative dataframe, reusing the cached result while it stays valid.
    The returned dataframe can be owned by the cache - callers must not modify it in place.
    This is safe because the dataframe is rewritten by the caller as part of the column rename
    and makes an additional .copy() call unnecessary
    """
    if _INFORMATIVE_DATE_MERGE in dataframe.columns:
        _raise_reserved_column_name()

    if cache is not None:
        fingerprint = _informative_dataframe_fingerprint(dataframe, cache_key.candle_type)
        cached: _InformativeCacheEntry | None = cache.get(cache_key)
        if cached is not None and cached.fingerprint == fingerprint:
            return cached.prepared.dataframe

    dataframe = populate_indicators_fn(strategy, dataframe, metadata)
    if _INFORMATIVE_DATE_MERGE in dataframe.columns:
        _raise_reserved_column_name()
    if cache is None:
        return dataframe

    prepared = _prepare_informative_pair(
        dataframe,
        strategy.timeframe,
        timeframe,
        append_timeframe=False,
        date_merge_column=_INFORMATIVE_DATE_MERGE,
    )
    cache[cache_key] = _InformativeCacheEntry(fingerprint, prepared)
    return prepared.dataframe


def _create_and_merge_informative_pair(
    strategy,
    dataframe: DataFrame,
    metadata: dict,
    inf_data: InformativeData,
    populate_indicators_fn: PopulateIndicators,
):
    if _INFORMATIVE_DATE_MERGE in dataframe.columns:
        _raise_reserved_column_name()

    asset = inf_data.asset or ""
    timeframe = inf_data.timeframe
    timeframe1 = inf_data.timeframe
    fmt = inf_data.fmt
    candle_type = inf_data.candle_type
    if candle_type == CandleType.FUNDING_RATE:
        timeframe1 = strategy.dp.get_funding_rate_timeframe()

    config = strategy.config

    if asset:
        # Insert stake currency if needed.
        market1 = strategy.dp.market(metadata["pair"])
        asset = _format_pair_name(config, asset, market1)
    else:
        # Not specifying an asset will define informative dataframe for current pair.
        asset = metadata["pair"]

    market = strategy.dp.market(asset)
    if market is None:
        raise OperationalException(f"Market {asset} is not available.")

    # Default format. This optimizes for the common case: informative pairs using same stake
    # currency. When quote currency matches stake currency, column name will omit base currency.
    # This allows easily reconfiguring strategy to use different base currency. In a rare case
    # where it is desired to keep quote currency in column name at all times user should specify
    # fmt='{base}_{quote}_{column}_{timeframe}' format or similar.
    if not fmt:
        fmt = "{column}_{timeframe}"  # Informatives of current pair
        if inf_data.asset:
            fmt = "{base}_{quote}_" + fmt  # Informatives of other pairs

    inf_metadata = {"pair": asset, "timeframe": timeframe}
    inf_dataframe = strategy.dp.get_pair_dataframe(asset, timeframe1, candle_type)
    if inf_dataframe.empty:
        raise ValueError(
            f"Informative dataframe for ({asset}, {timeframe1}, {candle_type}) is empty. "
            "Can't populate informative indicators."
        )
    cache: InformativeCache | None = (
        getattr(strategy, "_ft_informative_cache", None) if inf_data.cache else None
    )
    cache_key = InformativeCacheKey(
        callback=populate_indicators_fn,
        callback_name=populate_indicators_fn.__qualname__,
        asset=asset,
        informative_timeframe=timeframe,
        effective_timeframe=timeframe1,
        candle_type=candle_type,
    )
    inf_dataframe = _get_populated_informative_dataframe(
        strategy,
        populate_indicators_fn,
        inf_dataframe,
        inf_metadata,
        cache_key,
        cache,
        timeframe1,
    )

    formatter: Callable[..., str]
    if callable(fmt):
        formatter = fmt  # A custom user-specified formatter function.
    else:
        formatter = fmt.format  # A default string formatter.

    fmt_args: dict[str, str] = {
        **__get_pair_formats(market),
        "asset": asset,
        "timeframe": timeframe,
    }
    cache_enabled = cache is not None
    inf_dataframe = inf_dataframe.rename(
        columns=lambda column: _format_informative_column(
            column, formatter, fmt_args, preserve_date_merge=cache_enabled
        )
    )

    date_column = _format_informative_column("date", formatter, fmt_args)
    if date_column in dataframe.columns:
        raise OperationalException(
            f"Duplicate column name {date_column} exists in "
            f"dataframe! Ensure column names are unique!"
        )
    if cache_enabled:
        dataframe = _merge_prepared_informative_pair(
            dataframe,
            _PreparedInformative(inf_dataframe, _INFORMATIVE_DATE_MERGE),
            ffill=inf_data.ffill,
        )
    else:
        dataframe = merge_informative_pair(
            dataframe,
            inf_dataframe,
            strategy.timeframe,
            timeframe1,
            ffill=inf_data.ffill,
            append_timeframe=False,
            date_column=date_column,
        )
    return dataframe
