"""
Functions to convert orderflow data from public_trades
"""

import logging
import time
from datetime import datetime

import numpy as np
import pandas as pd

from freqtrade.constants import DEFAULT_ORDERFLOW_COLUMNS, ORDERFLOW_ADDED_COLUMNS, Config
from freqtrade.exceptions import DependencyException


logger = logging.getLogger(__name__)


def _init_dataframe_with_trades_columns(dataframe: pd.DataFrame):
    """
    Populates a dataframe with trades columns
    :param dataframe: Dataframe to populate
    """
    # Initialize columns with appropriate dtypes
    for column in ORDERFLOW_ADDED_COLUMNS:
        dataframe[column] = np.nan

    # Set columns to object type
    for column in (
        "trades",
        "orderflow",
        "imbalances",
        "stacked_imbalances_bid",
        "stacked_imbalances_ask",
    ):
        dataframe[column] = dataframe[column].astype(object)


def timeframe_to_DateOffset(timeframe: str) -> pd.DateOffset:
    """
    Translates the timeframe interval value written in the human readable
    form ('1m', '5m', '1h', '1d', '1w', etc.) to the number
    of seconds for one timeframe interval.
    """
    from freqtrade.exchange import timeframe_to_seconds

    timeframe_seconds = timeframe_to_seconds(timeframe)
    timeframe_minutes = timeframe_seconds // 60
    if timeframe_minutes < 1:
        return pd.DateOffset(seconds=timeframe_seconds)
    elif 59 < timeframe_minutes < 1440:
        return pd.DateOffset(hours=timeframe_minutes // 60)
    elif 1440 <= timeframe_minutes < 10080:
        return pd.DateOffset(days=timeframe_minutes // 1440)
    elif 10000 < timeframe_minutes < 43200:
        return pd.DateOffset(weeks=1)
    elif timeframe_minutes >= 43200 and timeframe_minutes < 525600:
        return pd.DateOffset(months=1)
    elif timeframe == "1y":
        return pd.DateOffset(years=1)
    else:
        return pd.DateOffset(minutes=timeframe_minutes)


def _calculate_ohlcv_candle_start_and_end(df: pd.DataFrame, timeframe: str):
    from freqtrade.exchange import timeframe_to_resample_freq

    if df is not None and not df.empty:
        timeframe_frequency = timeframe_to_resample_freq(timeframe)
        dofs = timeframe_to_DateOffset(timeframe)
        # calculate ohlcv candle start and end
        df["datetime"] = pd.to_datetime(df["date"], unit="ms")
        df["candle_start"] = df["datetime"].dt.floor(timeframe_frequency)
        # used in _now_is_time_to_refresh_trades
        df["candle_end"] = df["candle_start"] + dofs
        df.drop(columns=["datetime"], inplace=True)


def populate_dataframe_with_trades(
    cached_grouped_trades: pd.DataFrame | None,
    config: Config,
    dataframe: pd.DataFrame,
    trades: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Populates a dataframe with trades
    :param dataframe: Dataframe to populate
    :param trades: Trades to populate with
    :return: Dataframe with trades populated
    """

    timeframe = config["timeframe"]
    config_orderflow = config["orderflow"]

    # create columns for trades
    _init_dataframe_with_trades_columns(dataframe)
    if trades is None or trades.empty:
        return dataframe, cached_grouped_trades

    try:
        start_time = time.time()
        # calculate ohlcv candle start and end
        _calculate_ohlcv_candle_start_and_end(trades, timeframe)

        # get date of earliest max_candles candle
        max_candles = config_orderflow["max_candles"]
        start_date = dataframe.tail(max_candles).date.iat[0]
        # slice of trades that are before current ohlcv candles to make groupby faster
        trades = trades.loc[trades["candle_start"] >= start_date]
        trades.reset_index(inplace=True, drop=True)

        # group trades by candle start
        trades_grouped_by_candle_start = trades.groupby("candle_start", group_keys=False)

        candle_start: datetime
        for candle_start, trades_grouped_df in trades_grouped_by_candle_start:
            is_between = candle_start == dataframe["date"]
            if is_between.any():
                # there can only be one row with the same date
                index = dataframe.index[is_between][0]

                if (
                    cached_grouped_trades is not None
                    and (candle_start == cached_grouped_trades["date"]).any()
                ):
                    # Check if the trades are already in the cache
                    cache_idx = cached_grouped_trades.index[
                        cached_grouped_trades["date"] == candle_start
                    ][0]
                    for col in ORDERFLOW_ADDED_COLUMNS:
                        dataframe.at[index, col] = cached_grouped_trades.at[cache_idx, col]
                    continue

                dataframe.at[index, "trades"] = trades_grouped_df.drop(
                    columns=["candle_start", "candle_end"]
                ).to_dict(orient="records")

                # Calculate orderflow for each candle
                orderflow = trades_to_volumeprofile_with_total_delta_bid_ask(
                    trades_grouped_df, scale=config_orderflow["scale"]
                )
                dataframe.at[index, "orderflow"] = orderflow.to_dict(orient="index")
                # orderflow_series.loc[[index]] = [orderflow.to_dict(orient="index")]
                # Calculate imbalances for each candle's orderflow
                imbalances = trades_orderflow_to_imbalances(
                    orderflow,
                    imbalance_ratio=config_orderflow["imbalance_ratio"],
                    imbalance_volume=config_orderflow["imbalance_volume"],
                )
                dataframe.at[index, "imbalances"] = imbalances.to_dict(orient="index")

                stacked_imbalance_range = config_orderflow["stacked_imbalance_range"]
                dataframe.at[index, "stacked_imbalances_bid"] = stacked_imbalance(
                    imbalances, label="bid", stacked_imbalance_range=stacked_imbalance_range
                )

                dataframe.at[index, "stacked_imbalances_ask"] = stacked_imbalance(
                    imbalances, label="ask", stacked_imbalance_range=stacked_imbalance_range
                )

                bid = np.where(
                    trades_grouped_df["side"].str.contains("sell"), trades_grouped_df["amount"], 0
                )

                ask = np.where(
                    trades_grouped_df["side"].str.contains("buy"), trades_grouped_df["amount"], 0
                )
                deltas_per_trade = ask - bid
                dataframe.at[index, "max_delta"] = deltas_per_trade.cumsum().max()
                dataframe.at[index, "min_delta"] = deltas_per_trade.cumsum().min()

                dataframe.at[index, "bid"] = bid.sum()
                dataframe.at[index, "ask"] = ask.sum()
                dataframe.at[index, "delta"] = (
                    dataframe.at[index, "ask"] - dataframe.at[index, "bid"]
                )
                dataframe.at[index, "total_trades"] = len(trades_grouped_df)

        logger.debug(f"trades.groups_keys in {time.time() - start_time} seconds")

        # Cache the entire dataframe
        cached_grouped_trades = dataframe.tail(config_orderflow["cache_size"]).copy()

    except Exception as e:
        logger.exception("Error populating dataframe with trades")
        raise DependencyException(e)

    return dataframe, cached_grouped_trades


def trades_to_volumeprofile_with_total_delta_bid_ask(
    trades: pd.DataFrame, scale: float
) -> pd.DataFrame:
    """
    :param trades: dataframe
    :param scale: scale aka bin size e.g. 0.5
    :return: trades binned to levels according to scale aka orderflow
    """
    df = pd.DataFrame([], columns=DEFAULT_ORDERFLOW_COLUMNS)
    # create bid, ask where side is sell or buy
    df["bid_amount"] = np.where(trades["side"].str.contains("sell"), trades["amount"], 0)
    df["ask_amount"] = np.where(trades["side"].str.contains("buy"), trades["amount"], 0)
    df["bid"] = np.where(trades["side"].str.contains("sell"), 1, 0)
    df["ask"] = np.where(trades["side"].str.contains("buy"), 1, 0)
    # round the prices to the nearest multiple of the scale
    df["price"] = ((trades["price"] / scale).round() * scale).astype("float64").values
    if df.empty:
        df["total"] = np.nan
        df["delta"] = np.nan
        return df

    df["delta"] = df["ask_amount"] - df["bid_amount"]
    df["total_volume"] = df["ask_amount"] + df["bid_amount"]
    df["total_trades"] = df["ask"] + df["bid"]

    # group to bins aka apply scale
    df = df.groupby("price").sum(numeric_only=True)
    return df


def trades_orderflow_to_imbalances(df: pd.DataFrame, imbalance_ratio: int, imbalance_volume: int):
    """
    :param df: dataframes with bid and ask
    :param imbalance_ratio: imbalance_ratio e.g. 3
    :param imbalance_volume: imbalance volume e.g. 10
    :return: dataframe with bid and ask imbalance
    """
    bid = df.bid
    # compares bid and ask diagonally
    ask = df.ask.shift(-1)
    bid_imbalance = (bid / ask) > (imbalance_ratio)
    # overwrite bid_imbalance with False if volume is not big enough
    bid_imbalance_filtered = np.where(df.total_volume < imbalance_volume, False, bid_imbalance)
    ask_imbalance = (ask / bid) > (imbalance_ratio)
    # overwrite ask_imbalance with False if volume is not big enough
    ask_imbalance_filtered = np.where(df.total_volume < imbalance_volume, False, ask_imbalance)
    dataframe = pd.DataFrame(
        {"bid_imbalance": bid_imbalance_filtered, "ask_imbalance": ask_imbalance_filtered},
        index=df.index,
    )

    return dataframe


def stacked_imbalance(df: pd.DataFrame, label: str, stacked_imbalance_range: int):
    """
    y * (y.groupby((y != y.shift()).cumsum()).cumcount() + 1)
    https://stackoverflow.com/questions/27626542/counting-consecutive-positive-values-in-python-pandas-array
    """
    imbalance = df[f"{label}_imbalance"]
    int_series = pd.Series(np.where(imbalance, 1, 0))
    # Group consecutive True values and get their counts
    groups = (int_series != int_series.shift()).cumsum()
    counts = int_series.groupby(groups).cumsum()

    # Find indices where count meets or exceeds the range requirement
    valid_indices = counts[counts >= stacked_imbalance_range].index

    stacked_imbalance_prices = []
    if not valid_indices.empty:
        # Get all prices from valid indices from beginning of the range
        stacked_imbalance_prices = [
            imbalance.index.values[idx - (stacked_imbalance_range - 1)] for idx in valid_indices
        ]
    return stacked_imbalance_prices
