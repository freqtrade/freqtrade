"""
Functions to convert data from one format to another
"""
import logging
import pandas as pd
from pandas import DataFrame, to_datetime

from freqtrade.constants import TICKER_INTERVAL_MINUTES

logger = logging.getLogger(__name__)


def parse_ticker_dataframe(ticker: list, ticker_interval: str,
                           fill_missing: bool = True) -> DataFrame:
    """
    Converts a ticker-list (format ccxt.fetch_ohlcv) to a Dataframe
    :param ticker: ticker list, as returned by exchange.async_get_candle_history
    :param ticker_interval: ticker_interval (e.g. 5m). Used to fill up eventual missing data
    :param fill_missing: fill up missing candles with 0 candles
                         (see ohlcv_fill_up_missing_data for details)
    :return: DataFrame
    """
    logger.debug("Parsing tickerlist to dataframe")
    cols = ['date', 'open', 'high', 'low', 'close', 'volume']
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

    # group by index and aggregate results to eliminate duplicate ticks
    frame = frame.groupby(by='date', as_index=False, sort=True).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'max',
    })
    frame.drop(frame.tail(1).index, inplace=True)     # eliminate partial candle
    logger.debug('Dropping last candle')

    if fill_missing:
        return ohlcv_fill_up_missing_data(frame, ticker_interval)
    else:
        return frame


def ohlcv_fill_up_missing_data(dataframe: DataFrame, ticker_interval: str) -> DataFrame:
    """
    Fills up missing data with 0 volume rows,
    using the previous close as price for "open", "high" "low" and "close", volume is set to 0

    """
    ohlc_dict = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }
    tick_mins = TICKER_INTERVAL_MINUTES[ticker_interval]
    # Resample to create "NAN" values
    df = dataframe.resample(f'{tick_mins}min', on='date').agg(ohlc_dict)

    # Forwardfill close for missing columns
    df['close'] = df['close'].fillna(method='ffill')
    # Use close for "open, high, low"
    df.loc[:, ['open', 'high', 'low']] = df[['open', 'high', 'low']].fillna(
        value={'open': df['close'],
               'high': df['close'],
               'low': df['close'],
               })
    df.reset_index(inplace=True)
    logger.debug(f"Missing data fillup: before: {len(dataframe)} - after: {len(df)}")
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
