import logging
import operator
from datetime import datetime, timedelta
from pathlib import Path

from pandas import DataFrame, concat

from freqtrade.configuration import TimeRange
from freqtrade.constants import (
    DATETIME_PRINT_FORMAT,
    DL_DATA_TIMEFRAMES,
    DOCS_LINK,
    Config,
    ListPairsWithTimeframes,
    PairWithTimeframe,
)
from freqtrade.data.converter import (
    clean_ohlcv_dataframe,
    convert_trades_to_ohlcv,
    trades_df_remove_duplicates,
    trades_list_to_df,
)
from freqtrade.data.history.datahandlers import IDataHandler, get_datahandler
from freqtrade.enums import CandleType, TradingMode
from freqtrade.exceptions import OperationalException
from freqtrade.exchange import Exchange
from freqtrade.exchange.exchange_utils import date_minus_candles
from freqtrade.plugins.pairlist.pairlist_helpers import dynamic_expand_pairlist
from freqtrade.util import dt_now, dt_ts, format_ms_time, format_ms_time_det
from freqtrade.util.migrations import migrate_data
from freqtrade.util.progress_tracker import CustomProgress, retrieve_progress_tracker


logger = logging.getLogger(__name__)


def load_pair_history(
    pair: str,
    timeframe: str,
    datadir: Path,
    *,
    timerange: TimeRange | None = None,
    fill_up_missing: bool = True,
    drop_incomplete: bool = False,
    startup_candles: int = 0,
    data_format: str | None = None,
    data_handler: IDataHandler | None = None,
    candle_type: CandleType = CandleType.SPOT,
) -> DataFrame:
    """
    Load cached ohlcv history for the given pair.

    :param pair: Pair to load data for
    :param timeframe: Timeframe (e.g. "5m")
    :param datadir: Path to the data storage location.
    :param data_format: Format of the data. Ignored if data_handler is set.
    :param timerange: Limit data to be loaded to this timerange
    :param fill_up_missing: Fill missing values with "No action"-candles
    :param drop_incomplete: Drop last candle assuming it may be incomplete.
    :param startup_candles: Additional candles to load at the start of the period
    :param data_handler: Initialized data-handler to use.
                         Will be initialized from data_format if not set
    :param candle_type: Any of the enum CandleType (must match trading mode!)
    :return: DataFrame with ohlcv data, or empty DataFrame
    """
    data_handler = get_datahandler(datadir, data_format, data_handler)

    return data_handler.ohlcv_load(
        pair=pair,
        timeframe=timeframe,
        timerange=timerange,
        fill_missing=fill_up_missing,
        drop_incomplete=drop_incomplete,
        startup_candles=startup_candles,
        candle_type=candle_type,
    )


def load_data(
    datadir: Path,
    timeframe: str,
    pairs: list[str],
    *,
    timerange: TimeRange | None = None,
    fill_up_missing: bool = True,
    startup_candles: int = 0,
    fail_without_data: bool = False,
    data_format: str = "feather",
    candle_type: CandleType = CandleType.SPOT,
    user_futures_funding_rate: int | None = None,
) -> dict[str, DataFrame]:
    """
    Load ohlcv history data for a list of pairs.

    :param datadir: Path to the data storage location.
    :param timeframe: Timeframe (e.g. "5m")
    :param pairs: List of pairs to load
    :param timerange: Limit data to be loaded to this timerange
    :param fill_up_missing: Fill missing values with "No action"-candles
    :param startup_candles: Additional candles to load at the start of the period
    :param fail_without_data: Raise OperationalException if no data is found.
    :param data_format: Data format which should be used. Defaults to json
    :param candle_type: Any of the enum CandleType (must match trading mode!)
    :return: dict(<pair>:<Dataframe>)
    """
    result: dict[str, DataFrame] = {}
    if startup_candles > 0 and timerange:
        logger.debug(f"Using indicator startup period: {startup_candles} ...")

    data_handler = get_datahandler(datadir, data_format)

    for pair in pairs:
        hist = load_pair_history(
            pair=pair,
            timeframe=timeframe,
            datadir=datadir,
            timerange=timerange,
            fill_up_missing=fill_up_missing,
            startup_candles=startup_candles,
            data_handler=data_handler,
            candle_type=candle_type,
        )
        if not hist.empty:
            result[pair] = hist
        else:
            if candle_type is CandleType.FUNDING_RATE and user_futures_funding_rate is not None:
                logger.warning(f"{pair} using user specified [{user_futures_funding_rate}]")
            elif candle_type not in (CandleType.SPOT, CandleType.FUTURES):
                result[pair] = DataFrame(columns=["date", "open", "close", "high", "low", "volume"])

    if fail_without_data and not result:
        raise OperationalException("No data found. Terminating.")
    return result


def refresh_data(
    *,
    datadir: Path,
    timeframe: str,
    pairs: list[str],
    exchange: Exchange,
    data_format: str | None = None,
    timerange: TimeRange | None = None,
    candle_type: CandleType,
) -> None:
    """
    Refresh ohlcv history data for a list of pairs.

    :param datadir: Path to the data storage location.
    :param timeframe: Timeframe (e.g. "5m")
    :param pairs: List of pairs to load
    :param exchange: Exchange object
    :param data_format: dataformat to use
    :param timerange: Limit data to be loaded to this timerange
    :param candle_type: Any of the enum CandleType (must match trading mode!)
    """
    data_handler = get_datahandler(datadir, data_format)
    for pair in pairs:
        _download_pair_history(
            pair=pair,
            timeframe=timeframe,
            datadir=datadir,
            timerange=timerange,
            exchange=exchange,
            data_handler=data_handler,
            candle_type=candle_type,
        )


def _load_cached_data_for_updating(
    pair: str,
    timeframe: str,
    timerange: TimeRange | None,
    data_handler: IDataHandler,
    candle_type: CandleType,
    prepend: bool = False,
) -> tuple[DataFrame, int | None, int | None]:
    """
    Load cached data to download more data.
    If timerange is passed in, checks whether data from an before the stored data will be
    downloaded.
    If that's the case then what's available should be completely overwritten.
    Otherwise downloads always start at the end of the available data to avoid data gaps.
    Note: Only used by download_pair_history().
    """
    start = None
    end = None
    if timerange:
        if timerange.starttype == "date":
            start = timerange.startdt
        if timerange.stoptype == "date":
            end = timerange.stopdt

    # Intentionally don't pass timerange in - since we need to load the full dataset.
    data = data_handler.ohlcv_load(
        pair,
        timeframe=timeframe,
        timerange=None,
        fill_missing=False,
        drop_incomplete=True,
        warn_no_data=False,
        candle_type=candle_type,
    )
    if not data.empty:
        if prepend:
            end = data.iloc[0]["date"]
        else:
            if start and start < data.iloc[0]["date"]:
                # Earlier data than existing data requested, Update start date
                logger.info(
                    f"{pair}, {timeframe}, {candle_type}: "
                    f"Requested start date {start:{DATETIME_PRINT_FORMAT}} earlier than local "
                    f"data start date {data.iloc[0]['date']:{DATETIME_PRINT_FORMAT}}. "
                    f"Use `--prepend` to download data prior "
                    f"to {data.iloc[0]['date']:{DATETIME_PRINT_FORMAT}}, or "
                    "`--erase` to redownload all data."
                )
            start = data.iloc[-1]["date"]

    start_ms = int(start.timestamp() * 1000) if start else None
    end_ms = int(end.timestamp() * 1000) if end else None
    return data, start_ms, end_ms


def _download_pair_history(
    pair: str,
    *,
    datadir: Path,
    exchange: Exchange,
    timeframe: str = "5m",
    new_pairs_days: int = 30,
    data_handler: IDataHandler | None = None,
    timerange: TimeRange | None = None,
    candle_type: CandleType,
    erase: bool = False,
    prepend: bool = False,
    pair_candles: DataFrame | None = None,
) -> bool:
    """
    Download latest candles from the exchange for the pair and timeframe passed in parameters
    The data is downloaded starting from the last correct data that
    exists in a cache. If timerange starts earlier than the data in the cache,
    the full data will be redownloaded

    :param pair: pair to download
    :param timeframe: Timeframe (e.g "5m")
    :param timerange: range of time to download
    :param candle_type: Any of the enum CandleType (must match trading mode!)
    :param erase: Erase existing data
    :param pair_candles: Optional with "1 call" pair candles.
    :return: bool with success state
    """
    data_handler = get_datahandler(datadir, data_handler=data_handler)

    try:
        if erase:
            if data_handler.ohlcv_purge(pair, timeframe, candle_type=candle_type):
                logger.info(f"Deleting existing data for pair {pair}, {timeframe}, {candle_type}.")

        data, since_ms, until_ms = _load_cached_data_for_updating(
            pair,
            timeframe,
            timerange,
            data_handler=data_handler,
            candle_type=candle_type,
            prepend=prepend,
        )

        logger.info(
            f'Download history data for "{pair}", {timeframe}, '
            f"{candle_type} and store in {datadir}. "
            f"From {format_ms_time(since_ms) if since_ms else 'start'} to "
            f"{format_ms_time(until_ms) if until_ms else 'now'}"
        )

        logger.debug(
            "Current Start: %s",
            f"{data.iloc[0]['date']:{DATETIME_PRINT_FORMAT}}" if not data.empty else "None",
        )
        logger.debug(
            "Current End: %s",
            f"{data.iloc[-1]['date']:{DATETIME_PRINT_FORMAT}}" if not data.empty else "None",
        )
        # used to check if the passed in pair_candles (parallel downloaded) covers since_ms.
        # If we need more data, we have to fall back to the standard method.
        pair_candles_since_ms = (
            dt_ts(pair_candles.iloc[0]["date"])
            if pair_candles is not None and len(pair_candles.index) > 0
            else 0
        )
        if (
            pair_candles is None
            or len(pair_candles.index) == 0
            or data.empty
            or prepend is True
            or erase is True
            or pair_candles_since_ms > (since_ms if since_ms else 0)
        ):
            new_dataframe = exchange.get_historic_ohlcv(
                pair=pair,
                timeframe=timeframe,
                since_ms=(
                    since_ms
                    if since_ms
                    else int((datetime.now() - timedelta(days=new_pairs_days)).timestamp()) * 1000
                ),
                is_new_pair=data.empty,
                candle_type=candle_type,
                until_ms=until_ms if until_ms else None,
            )
            logger.info(
                f"Downloaded data for {pair}, {timeframe}, {candle_type} with length "
                f"{len(new_dataframe)}."
            )
        else:
            new_dataframe = pair_candles
            logger.info(
                f"Downloaded data for {pair}, {timeframe}, {candle_type} with length "
                f"{len(new_dataframe)}. Parallel Method."
            )

        if data.empty:
            data = new_dataframe
        else:
            # Run cleaning again to ensure there were no duplicate candles
            # Especially between existing and new data.
            data = clean_ohlcv_dataframe(
                concat([data, new_dataframe], axis=0),
                timeframe,
                pair,
                fill_missing=False,
                drop_incomplete=False,
            )

        logger.debug(
            "New Start: %s",
            f"{data.iloc[0]['date']:{DATETIME_PRINT_FORMAT}}" if not data.empty else "None",
        )
        logger.debug(
            "New End: %s",
            f"{data.iloc[-1]['date']:{DATETIME_PRINT_FORMAT}}" if not data.empty else "None",
        )

        data_handler.ohlcv_store(pair, timeframe, data=data, candle_type=candle_type)
        return True

    except Exception:
        logger.exception(
            f'Failed to download history data for pair: "{pair}", timeframe: {timeframe}.'
        )
        return False


def refresh_backtest_ohlcv_data(
    exchange: Exchange,
    *,
    pairs: list[str],
    timeframes: list[str],
    datadir: Path,
    trading_mode: str,
    timerange: TimeRange | None = None,
    new_pairs_days: int = 30,
    erase: bool = False,
    data_format: str | None = None,
    prepend: bool = False,
    progress_tracker: CustomProgress | None = None,
    candle_types: list[CandleType] | None = None,
    no_parallel_download: bool = False,
) -> list[str]:
    """
    Refresh stored ohlcv data for backtesting and hyperopt operations.
    Used by freqtrade download-data subcommand.
    :return: List of pairs that are not available.
    """
    progress_tracker = retrieve_progress_tracker(progress_tracker)

    pairs_not_available = []
    fast_candles: dict[PairWithTimeframe, DataFrame] = {}
    data_handler = get_datahandler(datadir, data_format)
    def_candletype = CandleType.SPOT if trading_mode != "futures" else CandleType.FUTURES
    if trading_mode != "futures":
        # Ignore user passed candle types for non-futures trading
        timeframes_with_candletype = [(tf, def_candletype) for tf in timeframes]
    else:
        # Filter out SPOT candle type for futures trading
        candle_types = (
            [ct for ct in candle_types if ct != CandleType.SPOT] if candle_types else None
        )
        fr_candle_type = CandleType.from_string(exchange.get_option("mark_ohlcv_price"))
        tf_funding_rate = exchange.get_option("funding_fee_timeframe")
        tf_mark = exchange.get_option("mark_ohlcv_timeframe")

        if candle_types:
            for ct in candle_types:
                exchange.verify_candle_type_support(ct)
            timeframes_with_candletype = [
                (tf, ct)
                for ct in candle_types
                for tf in timeframes
                if ct != CandleType.FUNDING_RATE
            ]
        else:
            # Default behavior
            timeframes_with_candletype = [(tf, def_candletype) for tf in timeframes]
            timeframes_with_candletype.append((tf_mark, fr_candle_type))
        if not candle_types or CandleType.FUNDING_RATE in candle_types:
            # All exchanges need FundingRate for futures trading.
            # The timeframe is aligned to the mark-price timeframe.
            timeframes_with_candletype.append((tf_funding_rate, CandleType.FUNDING_RATE))
    # Deduplicate list ...
    timeframes_with_candletype = list(dict.fromkeys(timeframes_with_candletype))
    logger.debug(
        "Downloading %s.", ", ".join(f'"{tf} {ct}"' for tf, ct in timeframes_with_candletype)
    )

    with progress_tracker as progress:
        timeframe_task = progress.add_task("Timeframe", total=len(timeframes_with_candletype))
        pair_task = progress.add_task("Downloading data...", total=len(pairs))

        for pair in pairs:
            progress.update(pair_task, description=f"Downloading {pair}")
            progress.update(timeframe_task, completed=0)

            if pair not in exchange.markets:
                pairs_not_available.append(f"{pair}: Pair not available on exchange.")
                logger.info(f"Skipping pair {pair}...")
                continue
            for timeframe, candle_type in timeframes_with_candletype:
                # Get fast candles via parallel method on first loop through per timeframe
                # and candle type. Downloads all the pairs in the list and stores them.
                # Also skips if only 1 pair/timeframe combination is scheduled for download.
                if (
                    not no_parallel_download
                    and (len(pairs) + len(timeframes)) > 2
                    and exchange.get_option("download_data_parallel_quick", True)
                    and (
                        ((pair, timeframe, candle_type) not in fast_candles)
                        and (erase is False)
                        and (prepend is False)
                    )
                ):
                    fast_candles.update(
                        _download_all_pairs_history_parallel(
                            exchange=exchange,
                            pairs=pairs,
                            timeframe=timeframe,
                            candle_type=candle_type,
                            timerange=timerange,
                        )
                    )

                # get the already downloaded pair candles if they exist
                pair_candles = fast_candles.pop((pair, timeframe, candle_type), None)

                progress.update(timeframe_task, description=f"Timeframe {timeframe} {candle_type}")
                logger.debug(f"Downloading pair {pair}, {candle_type}, interval {timeframe}.")
                _download_pair_history(
                    pair=pair,
                    datadir=datadir,
                    exchange=exchange,
                    timerange=timerange,
                    data_handler=data_handler,
                    timeframe=str(timeframe),
                    new_pairs_days=new_pairs_days,
                    candle_type=candle_type,
                    erase=erase,
                    prepend=prepend,
                    pair_candles=pair_candles,  # optional pass of dataframe of parallel candles
                )
                progress.update(timeframe_task, advance=1)

            progress.update(pair_task, advance=1)
            progress.update(timeframe_task, description="Timeframe")

    return pairs_not_available


def _download_all_pairs_history_parallel(
    exchange: Exchange,
    pairs: list[str],
    timeframe: str,
    candle_type: CandleType,
    timerange: TimeRange | None = None,
) -> dict[PairWithTimeframe, DataFrame]:
    """
    Allows to use the faster parallel async download method for many coins
    but only if the data is short enough to be retrieved in one call.
    Used by freqtrade download-data subcommand.
    :return: Candle pairs with timeframes
    """
    candles: dict[PairWithTimeframe, DataFrame] = {}
    since: int | None = None
    if timerange:
        if timerange.starttype == "date":
            since = timerange.startts * 1000

    candle_limit = exchange.ohlcv_candle_limit(timeframe, candle_type)
    one_call_min_time_dt = dt_ts(date_minus_candles(timeframe, candle_limit))
    # check if we can get all candles in one go, if so then we can download them in parallel
    if since is None or since > one_call_min_time_dt:
        logger.info(
            f"Downloading parallel candles for {timeframe} for all pairs"
            f" since {format_ms_time(since)}"
            if since
            else "."
        )
        needed_pairs: ListPairsWithTimeframes = [
            (p, timeframe, candle_type) for p in [p for p in pairs]
        ]
        candles = exchange.refresh_latest_ohlcv(needed_pairs, since_ms=since, cache=False)

    return candles


def _download_trades_history(
    exchange: Exchange,
    pair: str,
    *,
    new_pairs_days: int = 30,
    timerange: TimeRange | None = None,
    data_handler: IDataHandler,
    trading_mode: TradingMode,
) -> bool:
    """
    Download trade history from the exchange.
    Appends to previously downloaded trades data.
    """
    until = None
    since = 0
    if timerange:
        if timerange.starttype == "date":
            since = timerange.startts * 1000
        if timerange.stoptype == "date":
            until = timerange.stopts * 1000

    trades = data_handler.trades_load(pair, trading_mode)

    # TradesList columns are defined in constants.DEFAULT_TRADES_COLUMNS
    # DEFAULT_TRADES_COLUMNS: 0 -> timestamp
    # DEFAULT_TRADES_COLUMNS: 1 -> id

    if not trades.empty and since > 0 and (since + 1000) < trades.iloc[0]["timestamp"]:
        # since is before the first trade
        raise ValueError(
            f"Start {format_ms_time_det(since)} earlier than "
            f"available data ({format_ms_time_det(trades.iloc[0]['timestamp'])}). "
            f"Please use `--erase` if you'd like to redownload {pair}."
        )

    from_id = trades.iloc[-1]["id"] if not trades.empty else None
    if not trades.empty and since < trades.iloc[-1]["timestamp"]:
        # Reset since to the last available point
        # - 5 seconds (to ensure we're getting all trades)
        since = int(trades.iloc[-1]["timestamp"] - (5 * 1000))
        logger.info(
            f"Using last trade date -5s - Downloading trades for {pair} "
            f"since: {format_ms_time(since)}."
        )

    if not since:
        since = dt_ts(dt_now() - timedelta(days=new_pairs_days))

    logger.debug(
        "Current Start: %s",
        "None" if trades.empty else f"{trades.iloc[0]['date']:{DATETIME_PRINT_FORMAT}}",
    )
    logger.debug(
        "Current End: %s",
        "None" if trades.empty else f"{trades.iloc[-1]['date']:{DATETIME_PRINT_FORMAT}}",
    )
    logger.info(f"Current Amount of trades: {len(trades)}")

    new_trades = exchange.get_historic_trades(
        pair=pair,
        since=since,
        until=until,
        from_id=from_id,
    )
    new_trades_df = trades_list_to_df(new_trades[1])
    trades = concat([trades, new_trades_df], axis=0)
    # Remove duplicates to make sure we're not storing data we don't need
    trades = trades_df_remove_duplicates(trades)
    data_handler.trades_store(pair, trades, trading_mode)

    logger.debug(
        "New Start: %s",
        "None" if trades.empty else f"{trades.iloc[0]['date']:{DATETIME_PRINT_FORMAT}}",
    )
    logger.debug(
        "New End: %s",
        "None" if trades.empty else f"{trades.iloc[-1]['date']:{DATETIME_PRINT_FORMAT}}",
    )
    logger.info(f"New Amount of trades: {len(trades)}")
    return True


def refresh_backtest_trades_data(
    exchange: Exchange,
    pairs: list[str],
    datadir: Path,
    timerange: TimeRange,
    trading_mode: TradingMode,
    new_pairs_days: int = 30,
    erase: bool = False,
    data_format: str = "feather",
    progress_tracker: CustomProgress | None = None,
) -> list[str]:
    """
    Refresh stored trades data for backtesting and hyperopt operations.
    Used by freqtrade download-data subcommand.
    :return: List of pairs that are not available.
    """
    progress_tracker = retrieve_progress_tracker(progress_tracker)
    pairs_not_available = []
    data_handler = get_datahandler(datadir, data_format=data_format)
    with progress_tracker as progress:
        pair_task = progress.add_task("Downloading data...", total=len(pairs))
        for pair in pairs:
            progress.update(pair_task, description=f"Downloading trades [{pair}]")
            if pair not in exchange.markets:
                pairs_not_available.append(f"{pair}: Pair not available on exchange.")
                logger.info(f"Skipping pair {pair}...")
                continue

            if erase:
                if data_handler.trades_purge(pair, trading_mode):
                    logger.info(f"Deleting existing data for pair {pair}.")

            logger.info(f"Downloading trades for pair {pair}.")
            try:
                _download_trades_history(
                    exchange=exchange,
                    pair=pair,
                    new_pairs_days=new_pairs_days,
                    timerange=timerange,
                    data_handler=data_handler,
                    trading_mode=trading_mode,
                )
            except ValueError as e:
                pairs_not_available.append(f"{pair}: {str(e)}")
            except Exception:
                logger.exception(
                    f'Failed to download and store historic trades for pair: "{pair}". '
                )

            progress.update(pair_task, advance=1)

    return pairs_not_available


def get_timerange(data: dict[str, DataFrame]) -> tuple[datetime, datetime]:
    """
    Get the maximum common timerange for the given backtest data.

    :param data: dictionary with preprocessed backtesting data
    :return: tuple containing min_date, max_date
    """
    timeranges = [
        (frame["date"].min().to_pydatetime(), frame["date"].max().to_pydatetime())
        for frame in data.values()
    ]
    return (
        min(timeranges, key=operator.itemgetter(0))[0],
        max(timeranges, key=operator.itemgetter(1))[1],
    )


def validate_backtest_data(
    data: DataFrame, pair: str, min_date: datetime, max_date: datetime, timeframe_min: int
) -> bool:
    """
    Validates preprocessed backtesting data for missing values and shows warnings about it that.

    :param data: preprocessed backtesting data (as DataFrame)
    :param pair: pair used for log output.
    :param min_date: start-date of the data
    :param max_date: end-date of the data
    :param timeframe_min: Timeframe in minutes
    """
    # total difference in minutes / timeframe-minutes
    expected_frames = int((max_date - min_date).total_seconds() // 60 // timeframe_min)
    found_missing = False
    dflen = len(data)
    if dflen < expected_frames:
        found_missing = True
        logger.warning(
            "%s has missing frames: expected %s, got %s, that's %s missing values",
            pair,
            expected_frames,
            dflen,
            expected_frames - dflen,
        )
    return found_missing


def download_data_main(config: Config) -> None:
    from freqtrade.resolvers.exchange_resolver import ExchangeResolver

    exchange = ExchangeResolver.load_exchange(config, validate=False)

    download_data(config, exchange)


def download_data(
    config: Config,
    exchange: Exchange,
    *,
    progress_tracker: CustomProgress | None = None,
) -> None:
    """
    Download data function. Used from both cli and API.
    """
    exchange.validate_trading_mode_and_margin_mode(
        config.get("trading_mode", TradingMode.SPOT), None, allow_none_margin_mode=True
    )
    timerange = TimeRange()
    if "days" in config and config["days"] is not None:
        time_since = (datetime.now() - timedelta(days=config["days"])).strftime("%Y%m%d")
        timerange = TimeRange.parse_timerange(f"{time_since}-")

    if "timerange" in config:
        timerange = TimeRange.parse_timerange(config["timerange"])

    # Remove stake-currency to skip checks which are not relevant for datadownload
    config["stake_currency"] = ""

    pairs_not_available: list[str] = []

    available_pairs = [
        p
        for p in exchange.get_markets(
            tradable_only=True, active_only=not config.get("include_inactive")
        ).keys()
    ]

    expanded_pairs = dynamic_expand_pairlist(config, available_pairs)
    if "timeframes" not in config:
        config["timeframes"] = DL_DATA_TIMEFRAMES

    if len(expanded_pairs) == 0:
        logger.warning(
            "No pairs available for download. "
            "Please make sure you're using the correct Pair naming for your selected trade mode. \n"
            f"More info: {DOCS_LINK}/bot-basics/#pair-naming"
        )
        return

    logger.info(
        f"About to download pairs: {expanded_pairs}, "
        f"intervals: {config['timeframes']} to {config['datadir']}"
    )

    for timeframe in config["timeframes"]:
        exchange.validate_timeframes(timeframe)

    # Start downloading
    try:
        if config.get("download_trades"):
            if not exchange.get_option("trades_has_history", True):
                raise OperationalException(
                    f"Trade history not available for {exchange.name}. "
                    "You cannot use --dl-trades for this exchange."
                )
            pairs_not_available = refresh_backtest_trades_data(
                exchange,
                pairs=expanded_pairs,
                datadir=config["datadir"],
                timerange=timerange,
                new_pairs_days=config["new_pairs_days"],
                erase=bool(config.get("erase")),
                data_format=config["dataformat_trades"],
                trading_mode=config.get("trading_mode", TradingMode.SPOT),
                progress_tracker=progress_tracker,
            )

            if config.get("convert_trades") or not exchange.get_option("ohlcv_has_history", True):
                # Convert downloaded trade data to different timeframes
                # Only auto-convert for exchanges without historic klines

                convert_trades_to_ohlcv(
                    pairs=expanded_pairs,
                    timeframes=config["timeframes"],
                    datadir=config["datadir"],
                    timerange=timerange,
                    erase=bool(config.get("erase")),
                    data_format_ohlcv=config["dataformat_ohlcv"],
                    data_format_trades=config["dataformat_trades"],
                    candle_type=config.get("candle_type_def", CandleType.SPOT),
                )
        else:
            if not exchange.get_option("ohlcv_has_history", True):
                if not exchange.get_option("trades_has_history", True):
                    raise OperationalException(
                        f"Historic data not available for {exchange.name}. "
                        f"{exchange.name} does not support downloading trades or ohlcv data."
                    )
                else:
                    raise OperationalException(
                        f"Historic klines not available for {exchange.name}. "
                        "Please use `--dl-trades` instead for this exchange "
                        "(will unfortunately take a long time)."
                    )
            migrate_data(config, exchange)
            pairs_not_available = refresh_backtest_ohlcv_data(
                exchange,
                pairs=expanded_pairs,
                timeframes=config["timeframes"],
                datadir=config["datadir"],
                timerange=timerange,
                new_pairs_days=config["new_pairs_days"],
                erase=bool(config.get("erase")),
                data_format=config["dataformat_ohlcv"],
                trading_mode=config.get("trading_mode", "spot"),
                prepend=config.get("prepend_data", False),
                progress_tracker=progress_tracker,
                candle_types=config.get("candle_types"),
                no_parallel_download=config.get("no_parallel_download", False),
            )
    finally:
        if pairs_not_available:
            errors = "\n" + ("\n".join(pairs_not_available))
            logger.warning(
                f"Encountered a problem downloading the following pairs from {exchange.name}: "
                f"{errors}"
            )
