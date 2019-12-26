"""
Functions to convert data from one format to another
"""
import logging
from datetime import datetime, timezone

import pandas as pd
from pandas import DataFrame, to_datetime

from freqtrade.configuration.timerange import TimeRange
from freqtrade.constants import DEFAULT_DATAFRAME_COLUMNS

logger = logging.getLogger(__name__)


def parse_ticker_dataframe(ticker: list, timeframe: str, pair: str, *,
                           fill_missing: bool = True,
                           drop_incomplete: bool = True) -> DataFrame:
    """
    Converts a ticker-list (format ccxt.fetch_ohlcv) to a Dataframe
    :param ticker: ticker list, as returned by exchange.async_get_candle_history
    :param timeframe: timeframe (e.g. 5m). Used to fill up eventual missing data
    :param pair: Pair this data is for (used to warn if fillup was necessary)
    :param fill_missing: fill up missing candles with 0 candles
                         (see ohlcv_fill_up_missing_data for details)
    :param drop_incomplete: Drop the last candle of the dataframe, assuming it's incomplete
    :return: DataFrame
    """
    logger.debug("Parsing tickerlist to dataframe")
    cols = DEFAULT_DATAFRAME_COLUMNS
    frame = DataFrame(ticker, columns=cols)

    frame['date'] = to_datetime(frame['date'],
                                unit='ms',
                                utc=True,
                                infer_datetime_format=True)

    # Some exchanges return int values for volume and even for ohlc.
    # Convert them since TA-LIB indicators used in the strategy assume floats
    # and fail with exception...
    frame = frame.astype(dtype={'open': 'float', 'high': 'float', 'low': 'float', 'close': 'float',
                                'volume': 'float'})
    return clean_ohlcv_dataframe(frame, timeframe, pair,
                                 fill_missing=fill_missing,
                                 drop_incomplete=drop_incomplete)


def clean_ohlcv_dataframe(data: DataFrame, timeframe: str, pair: str, *,
                          fill_missing: bool = True,
                          drop_incomplete: bool = True) -> DataFrame:
    """
    Clense a ohlcv dataframe by
      * Grouping it by date (removes duplicate tics)
      * dropping last candles if requested
      * Filling up missing data (if requested)
    :param data: DataFrame containing ohlcv data.
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


def ohlcv_fill_up_missing_data(dataframe: DataFrame, timeframe: str, pair: str) -> DataFrame:
    """
    Fills up missing data with 0 volume rows,
    using the previous close as price for "open", "high" "low" and "close", volume is set to 0

    """
    from freqtrade.exchange import timeframe_to_minutes

    ohlc_dict = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }
    ticker_minutes = timeframe_to_minutes(timeframe)
    # Resample to create "NAN" values
    df = dataframe.resample(f'{ticker_minutes}min', on='date').agg(ohlc_dict)

    # Forwardfill close for missing columns
    df['close'] = df['close'].fillna(method='ffill')
    # Use close for "open, high, low"
    df.loc[:, ['open', 'high', 'low']] = df[['open', 'high', 'low']].fillna(
        value={'open': df['close'],
               'high': df['close'],
               'low': df['close'],
               })
    df.reset_index(inplace=True)
    len_before = len(dataframe)
    len_after = len(df)
    if len_before != len_after:
        logger.info(f"Missing data fillup for {pair}: before: {len_before} - after: {len_after}")
    return df


def trim_dataframe(df: DataFrame, timerange: TimeRange, df_date_col: str = 'date') -> DataFrame:
    """
    Trim dataframe based on given timerange
    :param df: Dataframe to trim
    :param timerange: timerange (use start and end date if available)
    :param: df_date_col: Column in the dataframe to use as Date column
    :return: trimmed dataframe
    """
    if timerange.starttype == 'date':
        start = datetime.fromtimestamp(timerange.startts, tz=timezone.utc)
        df = df.loc[df[df_date_col] >= start, :]
    if timerange.stoptype == 'date':
        stop = datetime.fromtimestamp(timerange.stopts, tz=timezone.utc)
        df = df.loc[df[df_date_col] <= stop, :]
    return df


def order_book_to_dataframe(bids: list, asks: list) -> DataFrame:
    """
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


def trades_to_ohlcv(trades: list, timeframe: str) -> DataFrame:
    """
    Converts trades list to ohlcv list
    :param trades: List of trades, as returned by ccxt.fetch_trades.
    :param timeframe: Ticker timeframe to resample data to
    :return: ohlcv Dataframe.
    """
    from freqtrade.exchange import timeframe_to_minutes
    ticker_minutes = timeframe_to_minutes(timeframe)
    df = pd.DataFrame(trades)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.set_index('datetime')

    df_new = df['price'].resample(f'{ticker_minutes}min').ohlc()
    df_new['volume'] = df['amount'].resample(f'{ticker_minutes}min').sum()
    df_new['date'] = df_new.index
    # Drop 0 volume rows
    df_new = df_new.dropna()
    return df_new[DEFAULT_DATAFRAME_COLUMNS]
