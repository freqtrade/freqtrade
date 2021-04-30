"""
Helpers when analyzing backtest data
"""
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from freqtrade.constants import LAST_BT_RESULT_FN
from freqtrade.misc import json_load
from freqtrade.persistence import LocalTrade, Trade, init_db


logger = logging.getLogger(__name__)

# Old format - maybe remove?
BT_DATA_COLUMNS_OLD = ["pair", "profit_percent", "open_date", "close_date", "index",
                       "trade_duration", "open_rate", "close_rate", "open_at_end", "sell_reason"]

# Mid-term format, crated by BacktestResult Named Tuple
BT_DATA_COLUMNS_MID = ['pair', 'profit_percent', 'open_date', 'close_date', 'trade_duration',
                       'open_rate', 'close_rate', 'open_at_end', 'sell_reason', 'fee_open',
                       'fee_close', 'amount', 'profit_abs', 'profit_ratio']

# Newest format
BT_DATA_COLUMNS = ['pair', 'stake_amount', 'amount', 'open_date', 'close_date',
                   'open_rate', 'close_rate',
                   'fee_open', 'fee_close', 'trade_duration',
                   'profit_ratio', 'profit_abs', 'sell_reason',
                   'initial_stop_loss_abs', 'initial_stop_loss_ratio', 'stop_loss_abs',
                   'stop_loss_ratio', 'min_rate', 'max_rate', 'is_open', ]


def get_latest_optimize_filename(directory: Union[Path, str], variant: str) -> str:
    """
    Get latest backtest export based on '.last_result.json'.
    :param directory: Directory to search for last result
    :param variant: 'backtest' or 'hyperopt' - the method to return
    :return: string containing the filename of the latest backtest result
    :raises: ValueError in the following cases:
        * Directory does not exist
        * `directory/.last_result.json` does not exist
        * `directory/.last_result.json` has the wrong content
    """
    if isinstance(directory, str):
        directory = Path(directory)
    if not directory.is_dir():
        raise ValueError(f"Directory '{directory}' does not exist.")
    filename = directory / LAST_BT_RESULT_FN

    if not filename.is_file():
        raise ValueError(
            f"Directory '{directory}' does not seem to contain backtest statistics yet.")

    with filename.open() as file:
        data = json_load(file)

    if f'latest_{variant}' not in data:
        raise ValueError(f"Invalid '{LAST_BT_RESULT_FN}' format.")

    return data[f'latest_{variant}']


def get_latest_backtest_filename(directory: Union[Path, str]) -> str:
    """
    Get latest backtest export based on '.last_result.json'.
    :param directory: Directory to search for last result
    :return: string containing the filename of the latest backtest result
    :raises: ValueError in the following cases:
        * Directory does not exist
        * `directory/.last_result.json` does not exist
        * `directory/.last_result.json` has the wrong content
    """
    return get_latest_optimize_filename(directory, 'backtest')


def get_latest_hyperopt_filename(directory: Union[Path, str]) -> str:
    """
    Get latest hyperopt export based on '.last_result.json'.
    :param directory: Directory to search for last result
    :return: string containing the filename of the latest hyperopt result
    :raises: ValueError in the following cases:
        * Directory does not exist
        * `directory/.last_result.json` does not exist
        * `directory/.last_result.json` has the wrong content
    """
    try:
        return get_latest_optimize_filename(directory, 'hyperopt')
    except ValueError:
        # Return default (legacy) pickle filename
        return 'hyperopt_results.pickle'


def get_latest_hyperopt_file(directory: Union[Path, str], predef_filename: str = None) -> Path:
    """
    Get latest hyperopt export based on '.last_result.json'.
    :param directory: Directory to search for last result
    :return: string containing the filename of the latest hyperopt result
    :raises: ValueError in the following cases:
        * Directory does not exist
        * `directory/.last_result.json` does not exist
        * `directory/.last_result.json` has the wrong content
    """
    if isinstance(directory, str):
        directory = Path(directory)
    if predef_filename:
        return directory / predef_filename
    return directory / get_latest_hyperopt_filename(directory)


def load_backtest_stats(filename: Union[Path, str]) -> Dict[str, Any]:
    """
    Load backtest statistics file.
    :param filename: pathlib.Path object, or string pointing to the file.
    :return: a dictionary containing the resulting file.
    """
    if isinstance(filename, str):
        filename = Path(filename)
    if filename.is_dir():
        filename = filename / get_latest_backtest_filename(filename)
    if not filename.is_file():
        raise ValueError(f"File {filename} does not exist.")
    logger.info(f"Loading backtest result from {filename}")
    with filename.open() as file:
        data = json_load(file)

    return data


def load_backtest_data(filename: Union[Path, str], strategy: Optional[str] = None) -> pd.DataFrame:
    """
    Load backtest data file.
    :param filename: pathlib.Path object, or string pointing to a file or directory
    :param strategy: Strategy to load - mainly relevant for multi-strategy backtests
                     Can also serve as protection to load the correct result.
    :return: a dataframe with the analysis results
    :raise: ValueError if loading goes wrong.
    """
    data = load_backtest_stats(filename)
    if not isinstance(data, list):
        # new, nested format
        if 'strategy' not in data:
            raise ValueError("Unknown dataformat.")

        if not strategy:
            if len(data['strategy']) == 1:
                strategy = list(data['strategy'].keys())[0]
            else:
                raise ValueError("Detected backtest result with more than one strategy. "
                                 "Please specify a strategy.")

        if strategy not in data['strategy']:
            raise ValueError(f"Strategy {strategy} not available in the backtest result.")

        data = data['strategy'][strategy]['trades']
        df = pd.DataFrame(data)
        df['open_date'] = pd.to_datetime(df['open_date'],
                                         utc=True,
                                         infer_datetime_format=True
                                         )
        df['close_date'] = pd.to_datetime(df['close_date'],
                                          utc=True,
                                          infer_datetime_format=True
                                          )
    else:
        # old format - only with lists.
        df = pd.DataFrame(data, columns=BT_DATA_COLUMNS_OLD)

        df['open_date'] = pd.to_datetime(df['open_date'],
                                         unit='s',
                                         utc=True,
                                         infer_datetime_format=True
                                         )
        df['close_date'] = pd.to_datetime(df['close_date'],
                                          unit='s',
                                          utc=True,
                                          infer_datetime_format=True
                                          )
        # Create compatibility with new format
        df['profit_abs'] = df['close_rate'] - df['open_rate']
    if 'profit_ratio' not in df.columns:
        df['profit_ratio'] = df['profit_percent']
    df = df.sort_values("open_date").reset_index(drop=True)
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
    dates = [pd.Series(pd.date_range(row[1]['open_date'], row[1]['close_date'],
                                     freq=f"{timeframe_min}min"))
             for row in results[['open_date', 'close_date']].iterrows()]
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


def trade_list_to_dataframe(trades: List[LocalTrade]) -> pd.DataFrame:
    """
    Convert list of Trade objects to pandas Dataframe
    :param trades: List of trade objects
    :return: Dataframe with BT_DATA_COLUMNS
    """
    df = pd.DataFrame.from_records([t.to_json() for t in trades], columns=BT_DATA_COLUMNS)
    if len(df) > 0:
        df.loc[:, 'close_date'] = pd.to_datetime(df['close_date'], utc=True)
        df.loc[:, 'open_date'] = pd.to_datetime(df['open_date'], utc=True)
        df.loc[:, 'close_rate'] = df['close_rate'].astype('float64')
    return df


def load_trades_from_db(db_url: str, strategy: Optional[str] = None) -> pd.DataFrame:
    """
    Load trades from a DB (using dburl)
    :param db_url: Sqlite url (default format sqlite:///tradesv3.dry-run.sqlite)
    :param strategy: Strategy to load - mainly relevant for multi-strategy backtests
                     Can also serve as protection to load the correct result.
    :return: Dataframe containing Trades
    """
    init_db(db_url, clean_open_orders=False)

    filters = []
    if strategy:
        filters.append(Trade.strategy == strategy)
    trades = trade_list_to_dataframe(Trade.get_trades(filters).all())

    return trades


def load_trades(source: str, db_url: str, exportfilename: Path,
                no_trades: bool = False, strategy: Optional[str] = None) -> pd.DataFrame:
    """
    Based on configuration option 'trade_source':
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
        return load_backtest_data(exportfilename, strategy)


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
    trades = trades.loc[(trades['open_date'] >= trades_start) &
                        (trades['close_date'] <= trades_stop)]
    return trades


def calculate_market_change(data: Dict[str, pd.DataFrame], column: str = "close") -> float:
    """
    Calculate market change based on "column".
    Calculation is done by taking the first non-null and the last non-null element of each column
    and calculating the pctchange as "(last - first) / first".
    Then the results per pair are combined as mean.

    :param data: Dict of Dataframes, dict key should be pair.
    :param column: Column in the original dataframes to use
    :return:
    """
    tmp_means = []
    for pair, df in data.items():
        start = df[column].dropna().iloc[0]
        end = df[column].dropna().iloc[-1]
        tmp_means.append((end - start) / start)

    return float(np.mean(tmp_means))


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
    :param trades: DataFrame containing trades (requires columns close_date and profit_abs)
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
    _trades_sum = trades.resample(f'{timeframe_minutes}min', on='close_date'
                                  )[['profit_abs']].sum()
    df.loc[:, col_name] = _trades_sum['profit_abs'].cumsum()
    # Set first value to 0
    df.loc[df.iloc[0].name, col_name] = 0
    # FFill to get continuous
    df[col_name] = df[col_name].ffill()
    return df


def calculate_max_drawdown(trades: pd.DataFrame, *, date_col: str = 'close_date',
                           value_col: str = 'profit_ratio'
                           ) -> Tuple[float, pd.Timestamp, pd.Timestamp, float, float]:
    """
    Calculate max drawdown and the corresponding close dates
    :param trades: DataFrame containing trades (requires columns close_date and profit_ratio)
    :param date_col: Column in DataFrame to use for dates (defaults to 'close_date')
    :param value_col: Column in DataFrame to use for values (defaults to 'profit_ratio')
    :return: Tuple (float, highdate, lowdate, highvalue, lowvalue) with absolute max drawdown,
             high and low time and high and low value.
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
    high_val = max_drawdown_df.loc[max_drawdown_df.iloc[:idxmin]
                                   ['high_value'].idxmax(), 'cumulative']
    low_val = max_drawdown_df.loc[idxmin, 'cumulative']
    return abs(min(max_drawdown_df['drawdown'])), high_date, low_date, high_val, low_val


def calculate_csum(trades: pd.DataFrame, starting_balance: float = 0) -> Tuple[float, float]:
    """
    Calculate min/max cumsum of trades, to show if the wallet/stake amount ratio is sane
    :param trades: DataFrame containing trades (requires columns close_date and profit_percent)
    :param starting_balance: Add starting balance to results, to show the wallets high / low points
    :return: Tuple (float, float) with cumsum of profit_abs
    :raise: ValueError if trade-dataframe was found empty.
    """
    if len(trades) == 0:
        raise ValueError("Trade dataframe empty.")

    csum_df = pd.DataFrame()
    csum_df['sum'] = trades['profit_abs'].cumsum()
    csum_min = csum_df['sum'].min() + starting_balance
    csum_max = csum_df['sum'].max() + starting_balance

    return csum_min, csum_max
