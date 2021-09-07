from typing import Any, Callable, NamedTuple, Optional, Union

import pandas as pd
from mypy_extensions import KwArg
from pandas import DataFrame

from freqtrade.exceptions import OperationalException
from freqtrade.exchange import timeframe_to_minutes


PopulateIndicators = Callable[[Any, DataFrame, dict], DataFrame]


class InformativeData(NamedTuple):
    asset: Optional[str]
    timeframe: str
    fmt: Union[str, Callable[[KwArg(str)], str], None]
    ffill: bool


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
    :param stop_rate: Stop loss price.
    :param current_rate: Current asset price.
    :return: Positive stop loss value relative to current price
    """
    return 1 - (stop_rate / current_rate)


def informative(timeframe: str, asset: str = '',
                fmt: Optional[Union[str, Callable[[KwArg(str)], str]]] = None,
                ffill: bool = True) -> Callable[[PopulateIndicators], PopulateIndicators]:
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
    current pair.
    :param fmt: Column format (str) or column formatter (callable(name, asset, timeframe)). When not
    specified, defaults to:
    * {base}_{column}_{timeframe} if asset is specified and quote currency does match stake
    currency.
    * {base}_{quote}_{column}_{timeframe} if asset is specified and quote currency does not match
    stake currency.
    * {column}_{timeframe} if asset is not specified.
    Format string supports these format variables:
    * {asset} - full name of the asset, for example 'BTC/USDT'.
    * {base} - base currency in lower case, for example 'eth'.
    * {BASE} - same as {base}, except in upper case.
    * {quote} - quote currency in lower case, for example 'usdt'.
    * {QUOTE} - same as {quote}, except in upper case.
    * {column} - name of dataframe column.
    * {timeframe} - timeframe of informative dataframe.
    :param ffill: ffill dataframe after merging informative pair.
    """
    _asset = asset
    _timeframe = timeframe
    _fmt = fmt
    _ffill = ffill

    def decorator(fn: PopulateIndicators):
        informative_pairs = getattr(fn, '_ft_informative', [])
        informative_pairs.append(InformativeData(_asset, _timeframe, _fmt, _ffill))
        setattr(fn, '_ft_informative', informative_pairs)
        return fn
    return decorator


def _format_pair_name(config, pair: str) -> str:
    return pair.format(stake_currency=config['stake_currency'],
                       stake=config['stake_currency']).upper()


def _create_and_merge_informative_pair(strategy, dataframe: DataFrame,
                                       metadata: dict, informative_data: InformativeData,
                                       populate_indicators: Callable[[Any, DataFrame, dict],
                                                                     DataFrame]):
    asset = informative_data.asset or ''
    timeframe = informative_data.timeframe
    fmt = informative_data.fmt
    ffill = informative_data.ffill
    config = strategy.config
    dp = strategy.dp

    if asset:
        # Insert stake currency if needed.
        asset = _format_pair_name(config, asset)
    else:
        # Not specifying an asset will define informative dataframe for current pair.
        asset = metadata['pair']

    if '/' in asset:
        base, quote = asset.split('/')
    else:
        # When futures are supported this may need reevaluation.
        # base, quote = asset, None
        raise OperationalException('Not implemented.')

    # Default format. This optimizes for the common case: informative pairs using same stake
    # currency. When quote currency matches stake currency, column name will omit base currency.
    # This allows easily reconfiguring strategy to use different base currency. In a rare case
    # where it is desired to keep quote currency in column name at all times user should specify
    # fmt='{base}_{quote}_{column}_{timeframe}' format or similar.
    if not fmt:
        fmt = '{column}_{timeframe}'                # Informatives of current pair
        if quote != config['stake_currency']:
            fmt = '{quote}_' + fmt                  # Informatives of different quote currency
        if informative_data.asset:
            fmt = '{base}_' + fmt                   # Informatives of other pair

    inf_metadata = {'pair': asset, 'timeframe': timeframe}
    inf_dataframe = dp.get_pair_dataframe(asset, timeframe)
    inf_dataframe = populate_indicators(strategy, inf_dataframe, inf_metadata)

    formatter: Any = None
    if callable(fmt):
        formatter = fmt             # A custom user-specified formatter function.
    else:
        formatter = fmt.format      # A default string formatter.

    fmt_args = {
        'BASE': base.upper(),
        'QUOTE': quote.upper(),
        'base': base.lower(),
        'quote': quote.lower(),
        'asset': asset,
        'timeframe': timeframe,
    }
    inf_dataframe.rename(columns=lambda column: formatter(column=column, **fmt_args),
                         inplace=True)

    date_column = formatter(column='date', **fmt_args)
    if date_column in dataframe.columns:
        raise OperationalException(f'Duplicate column name {date_column} exists in '
                                   f'dataframe! Ensure column names are unique!')
    dataframe = merge_informative_pair(dataframe, inf_dataframe, strategy.timeframe, timeframe,
                                       ffill=ffill, append_timeframe=False,
                                       date_column=date_column)
    return dataframe
