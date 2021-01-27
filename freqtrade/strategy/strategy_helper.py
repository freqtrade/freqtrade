import pandas as pd

from freqtrade.exchange import timeframe_to_minutes


def merge_informative_pair(dataframe: pd.DataFrame, informative: pd.DataFrame,
                           timeframe: str, timeframe_inf: str, ffill: bool = True) -> pd.DataFrame:
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
    :return: Merged dataframe
    :raise: ValueError if the secondary timeframe is shorter than the dataframe timeframe
    """

    minutes_inf = timeframe_to_minutes(timeframe_inf)
    minutes = timeframe_to_minutes(timeframe)
    if minutes == minutes_inf:
        # No need to forwardshift if the timeframes are identical
        informative['date_merge'] = informative["date"]
    elif minutes < minutes_inf:
        # Subtract "small" timeframe so merging is not delayed by 1 small candle
        # Detailed explanation in https://github.com/freqtrade/freqtrade/issues/4073
        informative['date_merge'] = (
            informative["date"] + pd.to_timedelta(minutes_inf, 'm') - pd.to_timedelta(minutes, 'm')
            )
    else:
        raise ValueError("Tried to merge a faster timeframe to a slower timeframe."
                         "This would create new rows, and can throw off your regular indicators.")

    # Rename columns to be unique
    informative.columns = [f"{col}_{timeframe_inf}" for col in informative.columns]

    # Combine the 2 dataframes
    # all indicators on the informative sample MUST be calculated before this point
    dataframe = pd.merge(dataframe, informative, left_on='date',
                         right_on=f'date_merge_{timeframe_inf}', how='left')
    dataframe = dataframe.drop(f'date_merge_{timeframe_inf}', axis=1)

    if ffill:
        dataframe = dataframe.ffill()

    return dataframe
