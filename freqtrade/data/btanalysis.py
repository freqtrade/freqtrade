"""
Helpers when analyzing backtest data
"""
import logging
from pathlib import Path
from typing import Dict, Union, Tuple

import numpy as np
import pandas as pd
from datetime import timezone, datetime
from scipy.ndimage.interpolation import shift

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
    # compute how long each trade was left outstanding as date indexes
    dates = [pd.Series(pd.date_range(row[1].open_time, row[1].close_time,
                                     freq=f"{timeframe_min}min"))
             for row in results[['open_time', 'close_time']].iterrows()]
    # track the lifetime of each trade in number of candles
    deltas = [len(x) for x in dates]
    # concat expands and flattens the list of lists of dates
    dates = pd.Series(pd.concat(dates).values, name='date')
    # trades are repeated (column wise) according to their lifetime
    df2 = pd.DataFrame(np.repeat(results.values, deltas, axis=0), columns=results.columns)

    # the expanded dates list is added as a new column to the repeated trades (df2)
    df2 = pd.concat([dates, df2], axis=1)
    df2 = df2.set_index('date')
    # duplicate dates entries represent trades on the same candle
    # which resampling resolves through the applied function (count)
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


def load_trades(source: str, db_url: str, exportfilename: str) -> pd.DataFrame:
    """
    Based on configuration option "trade_source":
    * loads data from DB (using `db_url`)
    * loads data from backtestfile (using `exportfilename`)
    """
    if source == "DB":
        return load_trades_from_db(db_url)
    elif source == "file":
        return load_backtest_data(Path(exportfilename))


def extract_trades_of_period(dataframe: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    """
    Compare trades and backtested pair DataFrames to get trades performed on backtested period
    :return: the DataFrame of a trades of period
    """
    trades = trades.loc[(trades['open_time'] >= dataframe.iloc[0]['date']) &
                        (trades['close_time'] <= dataframe.iloc[-1]['date'])]
    return trades


def combine_tickers_with_mean(tickers: Dict[str, pd.DataFrame],
                              column: str = "close") -> pd.DataFrame:
    """
    Combine multiple dataframes "column"
    :param tickers: Dict of Dataframes, dict key should be pair.
    :param column: Column in the original dataframes to use
    :return: DataFrame with the column renamed to the dict key, and a column
        named mean, containing the mean of all pairs.
    """
    df_comb = pd.concat([tickers[pair].set_index('date').rename(
        {column: pair}, axis=1)[pair] for pair in tickers], axis=1)

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
    """
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
    profit_results = trades.sort_values(date_col)
    max_drawdown_df = pd.DataFrame()
    max_drawdown_df['cumulative'] = profit_results[value_col].cumsum()
    max_drawdown_df['high_value'] = max_drawdown_df['cumulative'].cummax()
    max_drawdown_df['drawdown'] = max_drawdown_df['cumulative'] - max_drawdown_df['high_value']

    high_date = profit_results.loc[max_drawdown_df['high_value'].idxmax(), date_col]
    low_date = profit_results.loc[max_drawdown_df['drawdown'].idxmin(), date_col]

    return abs(min(max_drawdown_df['drawdown'])), high_date, low_date


def calculate_outstanding_balance(
    results: pd.DataFrame,
    timeframe: str,
    min_date: datetime,
    max_date: datetime,
    hloc: Dict[str, pd.DataFrame],
    slippage=0,
) -> pd.DataFrame:
    """
    Sums the value of each trade (both open and closed) on each candle
    :param results: Results Dataframe
    :param timeframe: Frequency used for the backtest
    :param min_date: date of the first trade opened (results.open_time.min())
    :param max_date: date of the last trade closed (results.close_time.max())
    :param hloc: historical DataFrame of each pair tested
    :slippage: optional profit value to subtract per trade
    :return: DataFrame of outstanding balance at each timeframe
    """
    timedelta = pd.Timedelta(timeframe)

    date_index: pd.DatetimeIndex = pd.date_range(
        start=min_date, end=max_date, freq=timeframe, normalize=True
    )
    balance_total = []
    for pair in hloc:
        pair_candles = hloc[pair].set_index("date").reindex(date_index)
        # index becomes open_time
        pair_trades = (
            results.loc[results["pair"].values == pair]
            .set_index("open_time")
            .resample(timeframe)
            .asfreq()
            .reindex(date_index)
        )
        open_rate = pair_trades["open_rate"].fillna(0).values
        open_time = pair_trades.index.values
        close_time = pair_trades["close_time"].values
        close = pair_candles["close"].values
        profits = pair_trades["profit_percent"].values - slippage
        # at the open_time candle, the balance is matched to the close of the candle
        pair_balance = np.where(
            # only the rows with actual trades
            (open_rate > 0)
            # only if the trade is not also closed on the same candle
            & (open_time != close_time),
            1 - open_rate / close - slippage,
            0,
        )
        # at the close_time candle, the balance just uses the profits col
        pair_balance = pair_balance + np.where(
            # only rows with actual trades
            (open_rate > 0)
            # the rows where a close happens
            & (open_time == close_time),
            profits,
            pair_balance,
        )

        # how much time each trade was open, close - open time
        periods = close_time - open_time
        # how many candles each trade was open, set as a counter at each trade open_time index
        hops = np.nan_to_num(periods / timedelta).astype(int)

        # each loop update one timeframe forward, the balance on each timeframe
        # where there is at least one hop left to do (>0)
        for _ in range(1, hops.max() + 1):
            # move hops and open_rate by one
            hops = shift(hops, 1, cval=0)
            open_rate = shift(open_rate, 1, cval=0)
            pair_balance = np.where(
                hops > 0, pair_balance + (1 - open_rate / close) - slippage, pair_balance
            )
            hops -= 1

        # same as above but one loop per pair
        # trades_indexes = np.nonzero(hops)[0]
        # for i in trades_indexes:
        #     # start from 1 because counters are set at the open_time balance
        #     # which was already added previously
        #     for c in range(1, hops[i]):
        #         offset = i + c
        #         # the open rate is always for the current date, not the offset
        #         pair_balance[offset] += 1 - open_rate[i] / close[offset] - slippage

        # add the pair balance to the total
        balance_total.append(pair_balance)
    balance_total = np.array(balance_total).sum(axis=0)
    return pd.DataFrame({"balance": balance_total, "date": date_index})
