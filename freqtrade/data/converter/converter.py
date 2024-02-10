"""
Functions to convert data from one format to another
"""
import logging
import time
from typing import Dict, List

import numpy as np
import pandas as pd
from pandas import DataFrame, to_datetime

from freqtrade.constants import (DEFAULT_DATAFRAME_COLUMNS, DEFAULT_ORDERFLOW_COLUMNS,
                                 DEFAULT_TRADES_COLUMNS, Config)
from freqtrade.data.converter.trade_converter import trades_df_remove_duplicates
from freqtrade.enums import CandleType, TradingMode


logger = logging.getLogger(__name__)


def ohlcv_to_dataframe(ohlcv: list, timeframe: str, pair: str, *,
                       fill_missing: bool = True, drop_incomplete: bool = True) -> DataFrame:
    """
    Converts a list with candle (OHLCV) data (in format returned by ccxt.fetch_ohlcv)
    to a Dataframe
    :param ohlcv: list with candle (OHLCV) data, as returned by exchange.async_get_candle_history
    :param timeframe: timeframe (e.g. 5m). Used to fill up eventual missing data
    :param pair: Pair this data is for (used to warn if fillup was necessary)
    :param fill_missing: fill up missing candles with 0 candles
                         (see ohlcv_fill_up_missing_data for details)
    :param drop_incomplete: Drop the last candle of the dataframe, assuming it's incomplete
    :return: DataFrame
    """
    logger.debug(f"Converting candle (OHLCV) data to dataframe for pair {pair}.")
    cols = DEFAULT_DATAFRAME_COLUMNS
    df = DataFrame(ohlcv, columns=cols)

    df['date'] = to_datetime(df['date'], unit='ms', utc=True)

    # Some exchanges return int values for Volume and even for OHLC.
    # Convert them since TA-LIB indicators used in the strategy assume floats
    # and fail with exception...
    df = df.astype(dtype={'open': 'float', 'high': 'float', 'low': 'float', 'close': 'float',
                          'volume': 'float'})
    return clean_ohlcv_dataframe(df, timeframe, pair,
                                 fill_missing=fill_missing,
                                 drop_incomplete=drop_incomplete)


def _init_dataframe_with_trades_columns(dataframe: DataFrame):
    """
    Populates a dataframe with trades columns
    :param dataframe: Dataframe to populate
    """
    dataframe['trades'] = dataframe.apply(lambda _: [], axis=1)
    dataframe['orderflow'] = dataframe.apply(lambda _: {}, axis=1)
    dataframe['bid'] = np.nan
    dataframe['ask'] = np.nan
    dataframe['delta'] = np.nan
    dataframe['min_delta'] = np.nan
    dataframe['max_delta'] = np.nan
    dataframe['total_trades'] = np.nan
    dataframe['stacked_imbalances_bid'] = np.nan
    dataframe['stacked_imbalances_ask'] = np.nan


def _convert_timeframe_to_pandas_frequency(timeframe: str):
    # convert timeframe to format usable by pandas
    from freqtrade.exchange import timeframe_to_minutes
    timeframe_minutes = timeframe_to_minutes(timeframe)
    timeframe_frequency = f'{timeframe_minutes}min'
    return (timeframe_frequency, timeframe_minutes)


def _calculate_ohlcv_candle_start_and_end(df: DataFrame, timeframe: str):
    timeframe_frequency, timeframe_minutes = _convert_timeframe_to_pandas_frequency(
        timeframe)
    # calculate ohlcv candle start and end
    if df is not None and not df.empty:
        df['datetime'] = pd.to_datetime(df['date'], unit='ms')
        df['candle_start'] = df['datetime'].dt.floor(timeframe_frequency)
        df['candle_end'] = df['candle_start'] + pd.Timedelta(timeframe_minutes)
        df.drop(columns=['datetime'], inplace=True)


def populate_dataframe_with_trades(config: Config,
                                   dataframe: DataFrame,
                                   trades: DataFrame,
                                   *,
                                   pair: str) -> DataFrame:
    """
    Populates a dataframe with trades
    :param dataframe: Dataframe to populate
    :param trades: Trades to populate with
    :return: Dataframe with trades populated
    """
    config_orderflow = config['orderflow']
    timeframe = config['timeframe']

    # create columns for trades
    _init_dataframe_with_trades_columns(dataframe)
    df = dataframe.copy()

    try:
        start_time = time.time()
        # calculate ohlcv candle start and end
        _calculate_ohlcv_candle_start_and_end(df, timeframe)
        _calculate_ohlcv_candle_start_and_end(trades, timeframe)

        # slice of trades that are before current ohlcv candles to make groupby faster
        trades = trades.loc[trades.candle_start >= df.candle_start[0]]
        trades.reset_index(inplace=True, drop=True)

        # group trades by candle start
        trades_grouped_by_candle_start = trades.groupby(
            'candle_start', group_keys=False)
        # repair 'date' datetime type (otherwise crashes on each compare)
        if "date" in dataframe.columns:
            dataframe['date'] = pd.to_datetime(dataframe['date'])

        for candle_start in trades_grouped_by_candle_start.groups:
            trades_grouped_df = trades[candle_start == trades['candle_start']]
            is_between = (candle_start == df['candle_start'])
            if np.any(is_between == True):  # noqa: E712
                (timeframe_frequency, timeframe_minutes) = _convert_timeframe_to_pandas_frequency(
                    timeframe)
                candle_next = candle_start + \
                    pd.Timedelta(minutes=timeframe_minutes)
                # skip if there are no trades at next candle
                # because that this candle isn't finished yet
                if candle_next not in trades_grouped_by_candle_start.groups:
                    logger.warning(
                        f"candle at {candle_start} with {len(trades_grouped_df)} trades might be unfinished, because no finished trades at {candle_next}")  # noqa

                # add trades to each candle
                df.loc[is_between, 'trades'] = df.loc[is_between,
                                                      'trades'].apply(lambda _: trades_grouped_df)
                # calculate orderflow for each candle
                df.loc[is_between, 'orderflow'] = df.loc[is_between, 'orderflow'].apply(
                    lambda _: trades_to_volumeprofile_with_total_delta_bid_ask(
                        pd.DataFrame(trades_grouped_df),
                        scale=config_orderflow['scale']))
                # calculate imbalances for each candle's orderflow
                df.loc[is_between, 'imbalances'] = df.loc[is_between, 'orderflow'].apply(
                    lambda x: trades_orderflow_to_imbalances(x,
                                                             imbalance_ratio=config_orderflow['imbalance_ratio'],  # noqa: E501
                                                             imbalance_volume=config_orderflow['imbalance_volume']))  # noqa: E501

                df.loc[is_between, 'stacked_imbalances_bid'] = df.loc[is_between,
                                                                      'imbalances'].apply(
                                                                          lambda x: stacked_imbalance_bid(x,  # noqa: E501
                                                                                                          stacked_imbalance_range=config_orderflow['stacked_imbalance_range']))  # noqa: E501
                df.loc[is_between, 'stacked_imbalances_ask'] = df.loc[is_between,
                                                                      'imbalances'].apply(
                                                                          lambda x: stacked_imbalance_ask(x,  # noqa: E501
                                                                                                          stacked_imbalance_range=config_orderflow['stacked_imbalance_range']))  # noqa: E501

                buy = df.loc[is_between, 'bid'].apply(lambda _: np.where(
                    trades_grouped_df['side'].str.contains('buy'), 0, trades_grouped_df['amount']))
                sell = df.loc[is_between, 'ask'].apply(lambda _: np.where(
                    trades_grouped_df['side'].str.contains('sell'), 0, trades_grouped_df['amount']))
                deltas_per_trade = sell - buy
                min_delta = 0
                max_delta = 0
                delta = 0
                for deltas in deltas_per_trade:
                    for d in deltas:
                        delta += d
                        if delta > max_delta:
                            max_delta = delta
                        if delta < min_delta:
                            min_delta = delta
                df.loc[is_between, 'max_delta'] = max_delta
                df.loc[is_between, 'min_delta'] = min_delta

                df.loc[is_between, 'bid'] = np.where(trades_grouped_df['side'].str.contains(
                    'buy'), 0, trades_grouped_df['amount']).sum()
                df.loc[is_between, 'ask'] = np.where(trades_grouped_df['side'].str.contains(
                    'sell'), 0, trades_grouped_df['amount']).sum()
                df.loc[is_between, 'delta'] = df.loc[is_between,
                                                     'ask'] - df.loc[is_between, 'bid']
                min_delta = np.min(deltas_per_trade)
                max_delta = np.max(deltas_per_trade)

                df.loc[is_between, 'total_trades'] = len(trades_grouped_df)
                # copy to avoid memory leaks
                dataframe.loc[is_between] = df.loc[is_between].copy()
            else:
                logger.debug(
                    f"Found NO candles for trades starting with {candle_start}")
        logger.debug(
            f"trades.groups_keys in {time.time() - start_time} seconds")

        logger.debug(
            f"trades.singleton_iterate in {time.time() - start_time} seconds")

    except Exception as e:
        logger.exception("Error populating dataframe with trades:", e)

    return dataframe


def public_trades_to_dataframe(trades: List, pair: str) -> DataFrame:
    """
    Converts a list with candle (TRADES) data (in format returned by ccxt.fetch_trades)
    to a Dataframe
    :param trades: list with candle (TRADES) data, as returned by exchange.async_get_candle_history
    :param timeframe: timeframe (e.g. 5m). Used to fill up eventual missing data
    :param pair: Pair this data is for (used to warn if fillup was necessary)
    :param fill_missing: fill up missing candles with 0 candles
                         (see trades_fill_up_missing_data for details)
    :param drop_incomplete: Drop the last candle of the dataframe, assuming it's incomplete
    :return: DataFrame
    """
    logger.debug(
        f"Converting candle (TRADES) data to dataframe for pair {pair}.")
    cols = DEFAULT_TRADES_COLUMNS
    df = DataFrame(trades, columns=cols)
    df['date'] = pd.to_datetime(
        df['timestamp'], unit='ms', utc=True)

    # Some exchanges return int values for Volume and even for OHLC.
    # Convert them since TA-LIB indicators used in the strategy assume floats
    # and fail with exception...
    df = df.astype(dtype={'amount': 'float', 'cost': 'float',
                          'price': 'float'})
    return df


def trades_to_volumeprofile_with_total_delta_bid_ask(trades: DataFrame, scale: float):
    """
    :param trades: dataframe
    :param scale: scale aka bin size e.g. 0.5
    :return: trades binned to levels according to scale aka orderflow
    """
    df = pd.DataFrame([], columns=DEFAULT_ORDERFLOW_COLUMNS)
    # create bid, ask where side is sell or buy
    df['bid_amount'] = np.where(
        trades['side'].str.contains('buy'), 0, trades['amount'])
    df['ask_amount'] = np.where(
        trades['side'].str.contains('sell'), 0, trades['amount'])
    df['bid'] = np.where(
        trades['side'].str.contains('buy'), 0, 1)
    df['ask'] = np.where(
        trades['side'].str.contains('sell'), 0, 1)

    # round the prices to the nearest multiple of the scale
    df['price'] = ((trades['price'] / scale).round()
                   * scale).astype('float64').values
    if df.empty:
        df['total'] = np.nan
        df['delta'] = np.nan
        return df

    df['delta'] = df['ask_amount'] - df['bid_amount']
    df['total_volume'] = df['ask_amount'] + df['bid_amount']
    df['total_trades'] = df['ask'] + df['bid']

    # group to bins aka apply scale
    df = df.groupby('price').sum(numeric_only=True)
    return df


def trades_orderflow_to_imbalances(df: DataFrame, imbalance_ratio: int, imbalance_volume: int):
    """
    :param df: dataframes with bid and ask
    :param imbalance_ratio: imbalance_ratio e.g. 300
    :param imbalance_volume: imbalance volume e.g. 3)
    :return: dataframe with bid and ask imbalance
    """
    bid = df.bid
    ask = df.ask.shift(-1)
    bid_imbalance = (bid / ask) > (imbalance_ratio / 100)
    # overwrite bid_imbalance with False if volume is not big enough
    bid_imbalance_filtered = np.where(
        df.total_volume < imbalance_volume, False, bid_imbalance)
    ask_imbalance = (ask / bid) > (imbalance_ratio / 100)
    # overwrite ask_imbalance with False if volume is not big enough
    ask_imbalance_filtered = np.where(
        df.total_volume < imbalance_volume, False, ask_imbalance)
    dataframe = DataFrame(
        {"bid_imbalance": bid_imbalance_filtered,
         "ask_imbalance": ask_imbalance_filtered},
        index=df.index,
    )

    return dataframe


def stacked_imbalance(df: DataFrame,
                      label: str = "bid",
                      stacked_imbalance_range: int = 3,
                      should_reverse: bool = False):
    """
    y * (y.groupby((y != y.shift()).cumsum()).cumcount() + 1)
    https://stackoverflow.com/questions/27626542/counting-consecutive-positive-values-in-python-pandas-array
    """
    imbalance = df[f'{label}_imbalance']
    int_series = pd.Series(np.where(imbalance, 1, 0))
    stacked = int_series * \
        (int_series.groupby((int_series != int_series.shift()).cumsum()).cumcount() + 1)

    max_stacked_imbalance_idx = stacked.index[stacked >=
                                              stacked_imbalance_range]
    stacked_imbalance_price = np.nan
    if not max_stacked_imbalance_idx.empty:
        idx = max_stacked_imbalance_idx[0] if not should_reverse else np.flipud(
            max_stacked_imbalance_idx)[0]
        stacked_imbalance_price = imbalance.index[idx]
    return stacked_imbalance_price


def stacked_imbalance_bid(df: DataFrame, stacked_imbalance_range: int = 3):
    return stacked_imbalance(df, 'bid', stacked_imbalance_range)


def stacked_imbalance_ask(df: DataFrame, stacked_imbalance_range: int = 3):
    return stacked_imbalance(df, 'ask', stacked_imbalance_range, should_reverse=True)


def orderflow_to_volume_profile(df: DataFrame):
    """
    :param orderflow: dataframe
    :return: volume profile dataframe
    """
    bid = df.groupby('level').bid.sum()
    ask = df.groupby('level').ask.sum()
    df.groupby('level')['level'].sum()
    delta = df.groupby('level').ask.sum() - df.groupby('level').bid.sum()
    df = pd.DataFrame({'bid': bid, 'ask': ask, 'delta': delta})
    return df


def clean_ohlcv_dataframe(data: DataFrame, timeframe: str, pair: str, *,
                          fill_missing: bool, drop_incomplete: bool) -> DataFrame:
    """
    Cleanse a OHLCV dataframe by
      * Grouping it by date (removes duplicate tics)
      * dropping last candles if requested
      * Filling up missing data (if requested)
    :param data: DataFrame containing candle (OHLCV) data.
    :param timeframe: timeframe (e.g. 5m). Used to fill up eventual missing data
    :param pair: Pair this data is for (used to warn if fillup was necessary)
    :param fill_missing: fill up missing candles with 0 candles
                         (see ohlcv_fill_up_missing_data for details)
    :param drop_incomplete: Drop the last candle of the dataframe, assuming it's incomplete
    :return: DataFrame
    """
    # group by index and aggregate results to eliminate duplicate ticks
    data = data.groupby(by='date', as_index=False, sort=True).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'max',
    })
    # eliminate partial candle
    if drop_incomplete:
        data.drop(data.tail(1).index, inplace=True)
        logger.debug('Dropping last candle')

    if fill_missing:
        return ohlcv_fill_up_missing_data(data, timeframe, pair)
    else:
        return data


def warn_of_tick_duplicates(data: DataFrame, pair: str) -> None:
    no_dupes_colunms = ['id', 'timestamp', 'datetime']
    for col in no_dupes_colunms:
        if col in data.columns and data[col].duplicated().any():
            sum = data[col].duplicated().sum()
            message = f'{sum} duplicated ticks for {pair} in {col} detected.'
            if col == 'id':
                logger.warning(message)
            else:
                logger.debug(message)


def clean_duplicate_trades(trades: DataFrame, timeframe: str, pair: str, *,

                           fill_missing: bool, drop_incomplete: bool) -> DataFrame:
    """
    Cleanse a TRADES dataframe by
      * Grouping it by date (removes duplicate tics)
      * dropping last candles if requested
      * Filling up missing data (if requested)
    :param data: DataFrame containing candle (TRADES) data.
    :param timeframe: timeframe (e.g. 5m). Used to fill up eventual missing data
    :param pair: Pair this data is for (used to warn if fillup was necessary)
    :param fill_missing: fill up missing candles with 0 candles
                         (see trades_fill_up_missing_data for details)
    :param drop_incomplete: Drop the last candle of the dataframe, assuming it's incomplete
    :return: DataFrame
    """
    # group by index and aggregate results to eliminate duplicate ticks
    # check if data has duplicate ticks
    logger.debug(f"Clean duplicated ticks from Trades data {pair}")
    df = pd.DataFrame(trades_df_remove_duplicates(
        trades), columns=trades.columns)

    return df


def drop_incomplete_and_fill_missing_trades(data: DataFrame, timeframe: str, pair: str, *,
                                            fill_missing: bool, drop_incomplete: bool) -> DataFrame:

    # eliminate partial candle
    if drop_incomplete:
        # TODO: this is not correct, as it drops the last trade only
        # but we need to drop the last candle until closed
        pass
        data.drop(data.tail(1).index, inplace=True)
        logger.debug('Dropping last trade')

    return data


def ohlcv_fill_up_missing_data(dataframe: DataFrame, timeframe: str, pair: str) -> DataFrame:
    """
    Fills up missing data with 0 volume rows,
    using the previous close as price for "open", "high" "low" and "close", volume is set to 0

    """
    from freqtrade.exchange import timeframe_to_resample_freq

    ohlcv_dict = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }
    resample_interval = timeframe_to_resample_freq(timeframe)
    # Resample to create "NAN" values
    df = dataframe.resample(resample_interval, on='date').agg(ohlcv_dict)

    # Forwardfill close for missing columns
    df['close'] = df['close'].ffill()
    # Use close for "open, high, low"
    df.loc[:, ['open', 'high', 'low']] = df[['open', 'high', 'low']].fillna(
        value={'open': df['close'],
               'high': df['close'],
               'low': df['close'],
               })
    df.reset_index(inplace=True)
    len_before = len(dataframe)
    len_after = len(df)
    pct_missing = (len_after - len_before) / len_before if len_before > 0 else 0
    if len_before != len_after:
        message = (f"Missing data fillup for {pair}, {timeframe}: "
                   f"before: {len_before} - after: {len_after} - {pct_missing:.2%}")
        if pct_missing > 0.01:
            logger.info(message)
        else:
            # Don't be verbose if only a small amount is missing
            logger.debug(message)
    return df


def trim_dataframe(df: DataFrame, timerange, *, df_date_col: str = 'date',
                   startup_candles: int = 0) -> DataFrame:
    """
    Trim dataframe based on given timerange
    :param df: Dataframe to trim
    :param timerange: timerange (use start and end date if available)
    :param df_date_col: Column in the dataframe to use as Date column
    :param startup_candles: When not 0, is used instead the timerange start date
    :return: trimmed dataframe
    """
    if startup_candles:
        # Trim candles instead of timeframe in case of given startup_candle count
        df = df.iloc[startup_candles:, :]
    else:
        if timerange.starttype == 'date':
            df = df.loc[df[df_date_col] >= timerange.startdt, :]
    if timerange.stoptype == 'date':
        df = df.loc[df[df_date_col] <= timerange.stopdt, :]
    return df


def trim_dataframes(preprocessed: Dict[str, DataFrame], timerange,
                    startup_candles: int) -> Dict[str, DataFrame]:
    """
    Trim startup period from analyzed dataframes
    :param preprocessed: Dict of pair: dataframe
    :param timerange: timerange (use start and end date if available)
    :param startup_candles: Startup-candles that should be removed
    :return: Dict of trimmed dataframes
    """
    processed: Dict[str, DataFrame] = {}

    for pair, df in preprocessed.items():
        trimed_df = trim_dataframe(df, timerange, startup_candles=startup_candles)
        if not trimed_df.empty:
            processed[pair] = trimed_df
        else:
            logger.warning(f'{pair} has no data left after adjusting for startup candles, '
                           f'skipping.')
    return processed


def order_book_to_dataframe(bids: list, asks: list) -> DataFrame:
    """
    TODO: This should get a dedicated test
    Gets order book list, returns dataframe with below format per suggested by creslin
    -------------------------------------------------------------------
     b_sum       b_size       bids       asks       a_size       a_sum
    -------------------------------------------------------------------
    """
    cols = ['bids', 'b_size']

    bids_frame = DataFrame(bids, columns=cols)
    # add cumulative sum column
    bids_frame['b_sum'] = bids_frame['b_size'].cumsum()
    cols2 = ['asks', 'a_size']
    asks_frame = DataFrame(asks, columns=cols2)
    # add cumulative sum column
    asks_frame['a_sum'] = asks_frame['a_size'].cumsum()

    frame = pd.concat([bids_frame['b_sum'], bids_frame['b_size'], bids_frame['bids'],
                       asks_frame['asks'], asks_frame['a_size'], asks_frame['a_sum']], axis=1,
                      keys=['b_sum', 'b_size', 'bids', 'asks', 'a_size', 'a_sum'])
    # logger.info('order book %s', frame )
    return frame


def convert_ohlcv_format(
    config: Config,
    convert_from: str,
    convert_to: str,
    erase: bool,
):
    """
    Convert OHLCV from one format to another
    :param config: Config dictionary
    :param convert_from: Source format
    :param convert_to: Target format
    :param erase: Erase source data (does not apply if source and target format are identical)
    """
    from freqtrade.data.history.idatahandler import get_datahandler
    src = get_datahandler(config['datadir'], convert_from)
    trg = get_datahandler(config['datadir'], convert_to)
    timeframes = config.get('timeframes', [config.get('timeframe')])
    logger.info(f"Converting candle (OHLCV) for timeframe {timeframes}")

    candle_types = [CandleType.from_string(ct) for ct in config.get('candle_types', [
        c.value for c in CandleType])]
    logger.info(candle_types)
    paircombs = src.ohlcv_get_available_data(config['datadir'], TradingMode.SPOT)
    paircombs.extend(src.ohlcv_get_available_data(config['datadir'], TradingMode.FUTURES))

    if 'pairs' in config:
        # Filter pairs
        paircombs = [comb for comb in paircombs if comb[0] in config['pairs']]

    if 'timeframes' in config:
        paircombs = [comb for comb in paircombs if comb[1] in config['timeframes']]
    paircombs = [comb for comb in paircombs if comb[2] in candle_types]

    paircombs = sorted(paircombs, key=lambda x: (x[0], x[1], x[2].value))

    formatted_paircombs = '\n'.join([f"{pair}, {timeframe}, {candle_type}"
                                    for pair, timeframe, candle_type in paircombs])

    logger.info(f"Converting candle (OHLCV) data for the following pair combinations:\n"
                f"{formatted_paircombs}")
    for pair, timeframe, candle_type in paircombs:
        data = src.ohlcv_load(pair=pair, timeframe=timeframe,
                              timerange=None,
                              fill_missing=False,
                              drop_incomplete=False,
                              startup_candles=0,
                              candle_type=candle_type)
        logger.info(f"Converting {len(data)} {timeframe} {candle_type} candles for {pair}")
        if len(data) > 0:
            trg.ohlcv_store(
                pair=pair,
                timeframe=timeframe,
                data=data,
                candle_type=candle_type
            )
            if erase and convert_from != convert_to:
                logger.info(f"Deleting source data for {pair} / {timeframe}")
                src.ohlcv_purge(pair=pair, timeframe=timeframe, candle_type=candle_type)


def reduce_dataframe_footprint(df: DataFrame) -> DataFrame:
    """
    Ensure all values are float32 in the incoming dataframe.
    :param df: Dataframe to be converted to float/int 32s
    :return: Dataframe converted to float/int 32s
    """

    logger.debug(f"Memory usage of dataframe is "
                 f"{df.memory_usage().sum() / 1024**2:.2f} MB")

    df_dtypes = df.dtypes
    for column, dtype in df_dtypes.items():
        if column in ['open', 'high', 'low', 'close', 'volume']:
            continue
        if dtype == np.float64:
            df_dtypes[column] = np.float32
        elif dtype == np.int64:
            df_dtypes[column] = np.int32
    df = df.astype(df_dtypes)

    logger.debug(f"Memory usage after optimization is: "
                 f"{df.memory_usage().sum() / 1024**2:.2f} MB")

    return df
