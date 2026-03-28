from math import isnan

from pandas import DataFrame


def merge_informative_pair(
    dataframe: DataFrame,
    informative: DataFrame,
    timeframe: str,
    timeframe_inf: str,
    ffill: bool = True,
    append_timeframe: bool = True,
    date_column: str = "date",
    suffix: str | None = None,
) -> DataFrame:
    """
    Safely merge the informative sample(s) into the dataframe, avoiding lookahead bias.

    Because we're using a merge_asof, this is not a simple inner join.
    For every row in the dataframe, the most recent row from the informative dataframe
    that is older than the current row is used.

    :param dataframe: DataFrame with the base pair
    :param informative: DataFrame with the informative pair
    :param timeframe: Timeframe of the original pair sample.
    :param timeframe_inf: Timeframe of the informative pair sample.
    :param ffill: Forwardfill missing values - often desired so the most recent value is used.
    :param append_timeframe: Rename columns by appending timeframe.
    :param date_column: A column that will be used as the date column in the informative dataframe.
           Defaults to 'date'.
    :param suffix: A suffix to add to all columns of the informative dataframe. Useful when
                   combining multiple informative pairs with the same timeframe.
                   Incompatible with ``append_timeframe=True``.
    :return: Merged dataframe
    :raise: ValueError if suffix and append_timeframe are both specified.
    """
    minutes_inf = timeframe_to_minutes(timeframe_inf)
    minutes = timeframe_to_minutes(timeframe)
    if minutes == minutes_inf:
        # No need to adjust if the timeframes are the same
        informative_columns = list(informative.columns)
        if append_timeframe:
            for i, name in enumerate(informative_columns):
                if name != date_column:
                    informative_columns[i] = f"{name}_{timeframe_inf}"
        elif suffix:
            for i, name in enumerate(informative_columns):
                if name != date_column:
                    informative_columns[i] = f"{name}_{suffix}"
        informative.columns = informative_columns
        merged = dataframe.merge(informative, on=date_column, how="left")
        if ffill:
            merged = merged.ffill()
        return merged
    elif minutes < minutes_inf:
        raise ValueError(
            "Informative timeframe must be equal or higher than the dataframe timeframe!"
        )

    else:
        # Informative timeframe is higher than the base timeframe - use asof merge
        informative_columns = list(informative.columns)
        if suffix and append_timeframe:
            raise ValueError("You can not specify 'suffix' and 'append_timeframe' simultaneously.")
        if append_timeframe:
            for i, name in enumerate(informative_columns):
                if name != date_column:
                    informative_columns[i] = f"{name}_{timeframe_inf}"
        elif suffix:
            for i, name in enumerate(informative_columns):
                if name != date_column:
                    informative_columns[i] = f"{name}_{suffix}"
        informative.columns = informative_columns
        # Sort the dataframe and informative to ensure merge_asof works as expected
        dataframe = dataframe.sort_values(date_column)
        informative = informative.sort_values(date_column)
        # Shift the informative dataframe by 1 candle to avoid lookahead bias
        informative[date_column] = informative[date_column] + informative[date_column].diff().fillna(
            informative[date_column].diff().median()
        )
        import pandas as pd

        merged = pd.merge_asof(dataframe, informative, on=date_column)
        if ffill:
            merged = merged.ffill()
        return merged


def stoploss_from_open(
    open_relative_stop: float,
    current_profit: float,
    *,
    is_short: bool = False,
    leverage: float = 1.0,
) -> float:
    """
    Given the initial (open) candle stoploss, and current profit, return a stoploss value
    relative to current price that is appropriate.

    In normal situations, this function would be used to maintain a stoploss relative to the
    entry point, even when the price has moved significantly in your favor.

    Both open_relative_stop and current_profit should be given as relative values
    (expected profit/loss), not as the values returned from custom_stoploss.

    :param open_relative_stop: Desired stoploss relative to the open price
    :param current_profit: Current profit of the trade
    :param is_short: When true, perform the calculation using short formula
    :param leverage: Leverage used in the trade
    :return: Stop loss value relative to the current price
    """
    # Use this theoretical trade to scale out the open stoploss
    if is_short:
        if current_profit == -1:
            return 1
        current_price_norm = 1 / (1 + current_profit / leverage)
        desired_stop = 1 / (1 + open_relative_stop / leverage)
    else:
        if current_profit == -1:
            return 1
        current_price_norm = 1 + current_profit / leverage
        desired_stop = 1 + open_relative_stop / leverage

    # The current price is current_price_norm relative to the open.
    # The stop needs to be desired_stop relative to the open.
    # Stop relative to current price is therefore:
    if is_short:
        stoploss = 1 - desired_stop / current_price_norm
    else:
        stoploss = 1 - desired_stop / current_price_norm

    if isnan(stoploss):
        return 0
    return max(stoploss, 0.0)


def stoploss_from_absolute(
    stop_rate: float,
    current_rate: float,
    *,
    is_short: bool = False,
    leverage: float = 1.0,
) -> float:
    """
    Given current price and desired stop price, return a stoploss value (as used in custom_stoploss)
    that maintains the desired stop price.

    :param stop_rate: Stop loss target rate
    :param current_rate: Current asset price
    :param is_short: When true, perform the calculation using short formula
    :param leverage: Leverage used in the trade
    :return: Stop loss value relative to the current price
    """
    if current_rate == 0:
        return 1
    if is_short:
        stoploss = (stop_rate / current_rate) - 1
    else:
        stoploss = 1 - (stop_rate / current_rate)
    if isnan(stoploss):
        return 0
    return max(stoploss, 0.0) * leverage


def timeframe_to_minutes(timeframe: str) -> int:
    """
    Same as timeframe_to_seconds, but returns minutes.
    """
    from freqtrade.exchange import timeframe_to_seconds as _tts

    return _tts(timeframe) // 60
