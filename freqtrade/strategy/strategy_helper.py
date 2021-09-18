import pandas as pd

from freqtrade.exchange import timeframe_to_minutes


def merge_informative_pair(dataframe: pd.DataFrame, informative: pd.DataFrame,
                           timeframe: str, timeframe_inf: str, ffill: bool = True,
                           append_timeframe: bool = True,
                           date_column: str = 'date') -> pd.DataFrame:
    """
    Correctly merge informative samples to the original dataframe, avoiding lookahead bias.

    Since dates are candle open dates, merging a 15m candle that starts at 15:00, and a
    1h candle that starts at 15:00 will result in all candles to know the close at 16:00
    which they should not know.

    Moves the date of the informative pair by 1 time interval forward.
    This way, the 14:00 1h candle is merged to 15:00 15m candle, since the 14:00 1h candle is the
    last candle that's closed at 15:00, 15:15, 15:30 or 15:45.

    Assuming inf_tf = '1d' - then the resulting columns will be:
    date_1d, open_1d, high_1d, low_1d, close_1d, rsi_1d

    :param dataframe: Original dataframe
    :param informative: Informative pair, most likely loaded via dp.get_pair_dataframe
    :param timeframe: Timeframe of the original pair sample.
    :param timeframe_inf: Timeframe of the informative pair sample.
    :param ffill: Forwardfill missing values - optional but usually required
    :param append_timeframe: Rename columns by appending timeframe.
    :param date_column: A custom date column name.
    :return: Merged dataframe
    :raise: ValueError if the secondary timeframe is shorter than the dataframe timeframe
    """

    minutes_inf = timeframe_to_minutes(timeframe_inf)
    minutes = timeframe_to_minutes(timeframe)
    if minutes == minutes_inf:
        # No need to forwardshift if the timeframes are identical
        informative['date_merge'] = informative[date_column]
    elif minutes < minutes_inf:
        # Subtract "small" timeframe so merging is not delayed by 1 small candle
        # Detailed explanation in https://github.com/freqtrade/freqtrade/issues/4073
        informative['date_merge'] = (
            informative[date_column] + pd.to_timedelta(minutes_inf, 'm') -
            pd.to_timedelta(minutes, 'm')
        )
    else:
        raise ValueError("Tried to merge a faster timeframe to a slower timeframe."
                         "This would create new rows, and can throw off your regular indicators.")

    # Rename columns to be unique
    date_merge = 'date_merge'
    if append_timeframe:
        date_merge = f'date_merge_{timeframe_inf}'
        informative.columns = [f"{col}_{timeframe_inf}" for col in informative.columns]

    # Combine the 2 dataframes
    # all indicators on the informative sample MUST be calculated before this point
    dataframe = pd.merge(dataframe, informative, left_on='date',
                         right_on=date_merge, how='left')
    dataframe = dataframe.drop(date_merge, axis=1)

    if ffill:
        dataframe = dataframe.ffill()

    return dataframe


def stoploss_from_open(open_relative_stop: float, current_profit: float) -> float:
    """

    Given the current profit, and a desired stop loss value relative to the open price,
    return a stop loss value that is relative to the current price, and which can be
    returned from `custom_stoploss`.

    The requested stop can be positive for a stop above the open price, or negative for
    a stop below the open price. The return value is always >= 0.

    Returns 0 if the resulting stop price would be above the current price.

    :param open_relative_stop: Desired stop loss percentage relative to open price
    :param current_profit: The current profit percentage
    :return: Positive stop loss value relative to current price
    """

    # formula is undefined for current_profit -1, return maximum value
    if current_profit == -1:
        return 1

    stoploss = 1-((1+open_relative_stop)/(1+current_profit))

    # negative stoploss values indicate the requested stop price is higher than the current price
    return max(stoploss, 0.0)


def stoploss_from_absolute(stop_rate: float, current_rate: float) -> float:
    """
    Given current price and desired stop price, return a stop loss value that is relative to current
    price.

    The requested stop can be positive for a stop above the open price, or negative for
    a stop below the open price. The return value is always >= 0.

    Returns 0 if the resulting stop price would be above the current price.

    :param stop_rate: Stop loss price.
    :param current_rate: Current asset price.
    :return: Positive stop loss value relative to current price
    """

    # formula is undefined for current_rate 0, return maximum value
    if current_rate == 0:
        return 1

    stoploss = 1 - (stop_rate / current_rate)

    # negative stoploss values indicate the requested stop price is higher than the current price
    return max(stoploss, 0.0)
