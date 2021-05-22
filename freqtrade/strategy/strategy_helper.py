import pandas as pd

from freqtrade.exchange import timeframe_to_minutes, timeframe_to_prev_date
from freqtrade.strategy import IStrategy
from datetime import datetime
from freqtrade.persistence import Trade


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


class HelperMixin(IStrategy):
    custom_stoploss_config = {}

    def get_custom_dataframe(self, pair: str):
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        return dataframe

    def get_trade_candle(self, trade: 'Trade'):
        """
        search for nearest row of trade.open_date
        """
        trade_candle = self.find_candle_datetime(trade.open_date_utc, pair=trade.pair, now=None)
        return trade_candle

    def find_candle_datetime(self, query_date: datetime, pair: str, now: datetime):
        result = None
        dataframe = self.get_custom_dataframe(pair)
        candle = self.find_candle_datetime_safer(query_date, now, dataframe,)
        result = candle if candle.empty else candle.squeeze()
        return result

    def find_candle_datetime_faster(self, query_date: datetime, now: datetime, dataframe):
        if(now and now == query_date):
            candle = dataframe.iloc[-1]
        else:
            candle_date = timeframe_to_prev_date(self.timeframe, query_date)
            candle = dataframe.loc[dataframe.date == candle_date]
        return candle

    def find_candle_datetime_safer(self, query_date: datetime, now: datetime, dataframe):
        df = dataframe[['date']].set_index('date')

        try:
            date_mask = df.index.unique().get_loc(query_date, method='ffill')
            candle = dataframe.iloc[date_mask]  # use iloc because date_mask maybe :int
        except KeyError:  # trade.open_date may not exist yet
            candle = pd.DataFrame(index=dataframe.index)
        return candle

    def __init__(self, config: dict) -> None:
        super().__init__(config)
