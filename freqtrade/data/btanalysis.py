"""
Helpers when analyzing backtest data
"""
import logging
from pathlib import Path
from typing import Dict, Union, Tuple

import numpy as np
import pandas as pd
from datetime import timezone

from freqtrade import persistence
from freqtrade.misc import json_load
from freqtrade.persistence import Trade

logger = logging.getLogger(__name__)

# must align with columns in backtest.py
BT_DATA_COLUMNS = ["pair", "profitperc", "open_time", "close_time", "index", "duration",
                   "open_rate", "close_rate", "open_at_end", "sell_reason"]


def load_backtest_data(filename: Union[Path, str]) -> pd.DataFrame:
    """
    Load backtest data file.
    :param filename: pathlib.Path object, or string pointing to the file.
    :return: a dataframe with the analysis results
    """
    if isinstance(filename, str):
        filename = Path(filename)

    if not filename.is_file():
        raise ValueError(f"File {filename} does not exist.")

    with filename.open() as file:
        data = json_load(file)

    df = pd.DataFrame(data, columns=BT_DATA_COLUMNS)

    df['open_time'] = pd.to_datetime(df['open_time'],
                                     unit='s',
                                     utc=True,
                                     infer_datetime_format=True
                                     )
    df['close_time'] = pd.to_datetime(df['close_time'],
                                      unit='s',
                                      utc=True,
                                      infer_datetime_format=True
                                      )
    df['profit'] = df['close_rate'] - df['open_rate']
    df = df.sort_values("open_time").reset_index(drop=True)
    return df


def analyze_trade_parallelism(results: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """
    Find overlapping trades by expanding each trade once per period it was open
    and then counting overlaps.
    :param results: Results Dataframe - can be loaded
    :param timeframe: Timeframe used for backtest
    :return: dataframe with open-counts per time-period in timeframe
    """
    from freqtrade.exchange import timeframe_to_minutes
    timeframe_min = timeframe_to_minutes(timeframe)
    dates = [pd.Series(pd.date_range(row[1].open_time, row[1].close_time,
                                     freq=f"{timeframe_min}min"))
             for row in results[['open_time', 'close_time']].iterrows()]
    deltas = [len(x) for x in dates]
    dates = pd.Series(pd.concat(dates).values, name='date')
    df2 = pd.DataFrame(np.repeat(results.values, deltas, axis=0), columns=results.columns)

    df2 = pd.concat([dates, df2], axis=1)
    df2 = df2.set_index('date')
    df_final = df2.resample(f"{timeframe_min}min")[['pair']].count()
    df_final = df_final.rename({'pair': 'open_trades'}, axis=1)
    return df_final


def evaluate_result_multi(results: pd.DataFrame, timeframe: str,
                          max_open_trades: int) -> pd.DataFrame:
    """
    Find overlapping trades by expanding each trade once per period it was open
    and then counting overlaps
    :param results: Results Dataframe - can be loaded
    :param timeframe: Frequency used for the backtest
    :param max_open_trades: parameter max_open_trades used during backtest run
    :return: dataframe with open-counts per time-period in freq
    """
    df_final = analyze_trade_parallelism(results, timeframe)
    return df_final[df_final['open_trades'] > max_open_trades]


def load_trades_from_db(db_url: str) -> pd.DataFrame:
    """
    Load trades from a DB (using dburl)
    :param db_url: Sqlite url (default format sqlite:///tradesv3.dry-run.sqlite)
    :return: Dataframe containing Trades
    """
    trades: pd.DataFrame = pd.DataFrame([], columns=BT_DATA_COLUMNS)
    persistence.init(db_url, clean_open_orders=False)

    columns = ["pair", "open_time", "close_time", "profit", "profitperc",
               "open_rate", "close_rate", "amount", "duration", "sell_reason",
               "fee_open", "fee_close", "open_rate_requested", "close_rate_requested",
               "stake_amount", "max_rate", "min_rate", "id", "exchange",
               "stop_loss", "initial_stop_loss", "strategy", "ticker_interval"]

    trades = pd.DataFrame([(t.pair,
                            t.open_date.replace(tzinfo=timezone.utc),
                            t.close_date.replace(tzinfo=timezone.utc) if t.close_date else None,
                            t.calc_profit(), t.calc_profit_ratio(),
                            t.open_rate, t.close_rate, t.amount,
                            (round((t.close_date.timestamp() - t.open_date.timestamp()) / 60, 2)
                             if t.close_date else None),
                            t.sell_reason,
                            t.fee_open, t.fee_close,
                            t.open_rate_requested,
                            t.close_rate_requested,
                            t.stake_amount,
                            t.max_rate,
                            t.min_rate,
                            t.id, t.exchange,
                            t.stop_loss, t.initial_stop_loss,
                            t.strategy, t.ticker_interval
                            )
                           for t in Trade.get_trades().all()],
                          columns=columns)

    return trades


def load_trades(source: str, db_url: str, exportfilename: Path,
                no_trades: bool = False) -> pd.DataFrame:
    """
    Based on configuration option "trade_source":
    * loads data from DB (using `db_url`)
    * loads data from backtestfile (using `exportfilename`)
    :param source: "DB" or "file" - specify source to load from
    :param db_url: sqlalchemy formatted url to a database
    :param exportfilename: Json file generated by backtesting
    :param no_trades: Skip using trades, only return backtesting data columns
    :return: DataFrame containing trades
    """
    if no_trades:
        df = pd.DataFrame(columns=BT_DATA_COLUMNS)
        return df

    if source == "DB":
        return load_trades_from_db(db_url)
    elif source == "file":
        return load_backtest_data(exportfilename)


def extract_trades_of_period(dataframe: pd.DataFrame, trades: pd.DataFrame,
                             date_index=False) -> pd.DataFrame:
    """
    Compare trades and backtested pair DataFrames to get trades performed on backtested period
    :return: the DataFrame of a trades of period
    """
    if date_index:
        trades_start = dataframe.index[0]
        trades_stop = dataframe.index[-1]
    else:
        trades_start = dataframe.iloc[0]['date']
        trades_stop = dataframe.iloc[-1]['date']
    trades = trades.loc[(trades['open_time'] >= trades_start) &
                        (trades['close_time'] <= trades_stop)]
    return trades


def combine_dataframes_with_mean(data: Dict[str, pd.DataFrame],
                                 column: str = "close") -> pd.DataFrame:
    """
    Combine multiple dataframes "column"
    :param data: Dict of Dataframes, dict key should be pair.
    :param column: Column in the original dataframes to use
    :return: DataFrame with the column renamed to the dict key, and a column
        named mean, containing the mean of all pairs.
    """
    df_comb = pd.concat([data[pair].set_index('date').rename(
        {column: pair}, axis=1)[pair] for pair in data], axis=1)

    df_comb['mean'] = df_comb.mean(axis=1)

    return df_comb


def create_cum_profit(df: pd.DataFrame, trades: pd.DataFrame, col_name: str,
                      timeframe: str) -> pd.DataFrame:
    """
    Adds a column `col_name` with the cumulative profit for the given trades array.
    :param df: DataFrame with date index
    :param trades: DataFrame containing trades (requires columns close_time and profitperc)
    :param col_name: Column name that will be assigned the results
    :param timeframe: Timeframe used during the operations
    :return: Returns df with one additional column, col_name, containing the cumulative profit.
    :raise: ValueError if trade-dataframe was found empty.
    """
    if len(trades) == 0:
        raise ValueError("Trade dataframe empty.")
    from freqtrade.exchange import timeframe_to_minutes
    timeframe_minutes = timeframe_to_minutes(timeframe)
    # Resample to timeframe to make sure trades match candles
    _trades_sum = trades.resample(f'{timeframe_minutes}min', on='close_time')[['profitperc']].sum()
    df.loc[:, col_name] = _trades_sum.cumsum()
    # Set first value to 0
    df.loc[df.iloc[0].name, col_name] = 0
    # FFill to get continuous
    df[col_name] = df[col_name].ffill()
    return df


def calculate_max_drawdown(trades: pd.DataFrame, *, date_col: str = 'close_time',
                           value_col: str = 'profitperc'
                           ) -> Tuple[float, pd.Timestamp, pd.Timestamp]:
    """
    Calculate max drawdown and the corresponding close dates
    :param trades: DataFrame containing trades (requires columns close_time and profitperc)
    :param date_col: Column in DataFrame to use for dates (defaults to 'close_time')
    :param value_col: Column in DataFrame to use for values (defaults to 'profitperc')
    :return: Tuple (float, highdate, lowdate) with absolute max drawdown, high and low time
    :raise: ValueError if trade-dataframe was found empty.
    """
    if len(trades) == 0:
        raise ValueError("Trade dataframe empty.")
    profit_results = trades.sort_values(date_col).reset_index(drop=True)
    max_drawdown_df = pd.DataFrame()
    max_drawdown_df['cumulative'] = profit_results[value_col].cumsum()
    max_drawdown_df['high_value'] = max_drawdown_df['cumulative'].cummax()
    max_drawdown_df['drawdown'] = max_drawdown_df['cumulative'] - max_drawdown_df['high_value']

    idxmin = max_drawdown_df['drawdown'].idxmin()
    if idxmin == 0:
        raise ValueError("No losing trade, therefore no drawdown.")
    high_date = profit_results.loc[max_drawdown_df.iloc[:idxmin]['high_value'].idxmax(), date_col]
    low_date = profit_results.loc[idxmin, date_col]
    return abs(min(max_drawdown_df['drawdown'])), high_date, low_date
