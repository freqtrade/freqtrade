from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from cachetools import LRUCache
from pandas import DataFrame

from freqtrade.constants import DEFAULT_DATAFRAME_COLUMNS
from freqtrade.enums import CandleType
from freqtrade.exceptions import OperationalException
from freqtrade.strategy.strategy_helper import (
    _merge_prepared_informative_pair,
    _prepare_informative_pair,
    _PreparedInformative,
    merge_informative_pair,
)


PopulateIndicators = Callable[[Any, DataFrame, dict], DataFrame]
InformativeCacheKey = tuple[PopulateIndicators, str, str, str, CandleType | None, str]

_INFORMATIVE_DATE_MERGE = object()


@dataclass(slots=True)
class _InformativeCacheEntry:
    fingerprint: tuple[Any, ...]
    prepared: _PreparedInformative


InformativeCache = LRUCache[InformativeCacheKey, _InformativeCacheEntry]


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
    :param candle_type: '', mark, index, premiumIndex, or funding_rate
    :param cache: Cache populated indicators in dry/live mode while the latest informative candle
                  remains unchanged. Disable for methods that use external state, have side effects,
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


def _informative_dataframe_fingerprint(dataframe: DataFrame) -> tuple[Any, ...]:
    last_candle = dataframe.iloc[-1]
    return (len(dataframe), *(last_candle[column] for column in DEFAULT_DATAFRAME_COLUMNS))


def _get_populated_informative_dataframe(
    strategy,
    populate_indicators_fn: PopulateIndicators,
    dataframe: DataFrame,
    metadata: dict,
    cache_key: InformativeCacheKey,
    cache: InformativeCache | None,
    timeframe: str,
) -> tuple[DataFrame, bool]:
    if cache is None:
        return populate_indicators_fn(strategy, dataframe, metadata), False

    fingerprint = _informative_dataframe_fingerprint(dataframe)
    cached: _InformativeCacheEntry | None = cache.get(cache_key)
    if cached is not None and cached.fingerprint == fingerprint:
        return cached.prepared.dataframe.copy(), True

    # Cached live data is borrowed read-only from DataProvider. Give strategy code an owned copy.
    dataframe = populate_indicators_fn(strategy, dataframe.copy(), metadata)
    prepared = _prepare_informative_pair(
        dataframe,
        strategy.timeframe,
        timeframe,
        append_timeframe=False,
        date_merge_column=_INFORMATIVE_DATE_MERGE,
    )
    cache[cache_key] = _InformativeCacheEntry(fingerprint, prepared)
    return prepared.dataframe.copy(), True


def _create_and_merge_informative_pair(
    strategy,
    dataframe: DataFrame,
    metadata: dict,
    inf_data: InformativeData,
    populate_indicators_fn: PopulateIndicators,
):
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
    cache: InformativeCache | None = (
        getattr(strategy, "_ft_informative_cache", None) if inf_data.cache else None
    )
    if cache is None:
        inf_dataframe = strategy.dp.get_pair_dataframe(asset, timeframe1, candle_type)
    else:
        inf_dataframe = strategy.dp.get_pair_dataframe(asset, timeframe1, candle_type, copy=False)
    if inf_dataframe.empty:
        raise ValueError(
            f"Informative dataframe for ({asset}, {timeframe1}, {candle_type}) is empty. "
            "Can't populate informative indicators."
        )
    cache_key: InformativeCacheKey = (
        populate_indicators_fn,
        asset,
        timeframe,
        timeframe1,
        candle_type,
        strategy.timeframe,
    )
    inf_dataframe, prepared = _get_populated_informative_dataframe(
        strategy,
        populate_indicators_fn,
        inf_dataframe,
        inf_metadata,
        cache_key,
        cache,
        timeframe1,
    )

    formatter: Any = None
    if callable(fmt):
        formatter = fmt  # A custom user-specified formatter function.
    else:
        formatter = fmt.format  # A default string formatter.

    fmt_args = {
        **__get_pair_formats(market),
        "asset": asset,
        "timeframe": timeframe,
    }
    inf_dataframe.rename(
        columns=lambda column: (
            column
            if prepared and column is _INFORMATIVE_DATE_MERGE
            else formatter(column=column, **fmt_args)
        ),
        inplace=True,
    )

    date_column = formatter(column="date", **fmt_args)
    if date_column in dataframe.columns:
        raise OperationalException(
            f"Duplicate column name {date_column} exists in "
            f"dataframe! Ensure column names are unique!"
        )
    if prepared:
        date_merge_column = "date_merge"
        while date_merge_column in dataframe.columns or date_merge_column in inf_dataframe.columns:
            date_merge_column = f"_{date_merge_column}"
        inf_dataframe.columns = [
            date_merge_column if column is _INFORMATIVE_DATE_MERGE else column
            for column in inf_dataframe.columns
        ]
        dataframe = _merge_prepared_informative_pair(
            dataframe,
            _PreparedInformative(inf_dataframe, date_merge_column),
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
