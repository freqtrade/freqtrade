"""
Functions to convert orderflow data from public_trades
"""

import logging
import time

import numpy as np
import pandas as pd

from freqtrade.constants import DEFAULT_ORDERFLOW_COLUMNS, Config
from freqtrade.exceptions import DependencyException


logger = logging.getLogger(__name__)


def _init_dataframe_with_trades_columns(dataframe: pd.DataFrame):
    """
    Populates a dataframe with trades columns
    :param dataframe: Dataframe to populate
    """
    dataframe["trades"] = dataframe.apply(lambda _: [], axis=1)
    dataframe["orderflow"] = dataframe.apply(lambda _: {}, axis=1)
    dataframe["bid"] = np.nan
    dataframe["ask"] = np.nan
    dataframe["delta"] = np.nan
    dataframe["min_delta"] = np.nan
    dataframe["max_delta"] = np.nan
    dataframe["total_trades"] = np.nan
    dataframe["stacked_imbalances_bid"] = np.nan
    dataframe["stacked_imbalances_ask"] = np.nan


def _calculate_ohlcv_candle_start_and_end(df: pd.DataFrame, timeframe: str):
    from freqtrade.exchange import timeframe_to_next_date, timeframe_to_resample_freq

    timeframe_frequency = timeframe_to_resample_freq(timeframe)
    # calculate ohlcv candle start and end
    if df is not None and not df.empty:
        df["datetime"] = pd.to_datetime(df["date"], unit="ms")
        df["candle_start"] = df["datetime"].dt.floor(timeframe_frequency)
        # used in _now_is_time_to_refresh_trades
        df["candle_end"] = df["candle_start"].apply(
            lambda candle_start: timeframe_to_next_date(timeframe, candle_start)
        )
        df.drop(columns=["datetime"], inplace=True)


def populate_dataframe_with_trades(
    config: Config, dataframe: pd.DataFrame, trades: pd.DataFrame
) -> pd.DataFrame:
    """
    Populates a dataframe with trades
    :param dataframe: Dataframe to populate
    :param trades: Trades to populate with
    :return: Dataframe with trades populated
    """
    config_orderflow = config["orderflow"]
    timeframe = config["timeframe"]

    # create columns for trades
    _init_dataframe_with_trades_columns(dataframe)
    df = dataframe.copy()

    try:
        start_time = time.time()
        # calculate ohlcv candle start and end
        _calculate_ohlcv_candle_start_and_end(trades, timeframe)

        # slice of trades that are before current ohlcv candles to make groupby faster
        trades = trades.loc[trades.candle_start >= df.date[0]]
        trades.reset_index(inplace=True, drop=True)

        # group trades by candle start
        trades_grouped_by_candle_start = trades.groupby("candle_start", group_keys=False)

        for candle_start in trades_grouped_by_candle_start.groups:
            trades_grouped_df = trades[candle_start == trades["candle_start"]]
            is_between = candle_start == df["date"]
            if np.any(is_between == True):  # noqa: E712
                from freqtrade.exchange import timeframe_to_next_date

                candle_next = timeframe_to_next_date(timeframe, candle_start)
                # skip if there are no trades at next candle
                # because that this candle isn't finished yet
                if candle_next not in trades_grouped_by_candle_start.groups:
                    logger.warning(
                        f"candle at {candle_start} with {len(trades_grouped_df)} trades "
                        f"might be unfinished, because no finished trades at {candle_next}"
                    )

                # add trades to each candle
                df.loc[is_between, "trades"] = df.loc[is_between, "trades"].apply(
                    lambda _: trades_grouped_df
                )
                # calculate orderflow for each candle
                df.loc[is_between, "orderflow"] = df.loc[is_between, "orderflow"].apply(
                    lambda _: trades_to_volumeprofile_with_total_delta_bid_ask(
                        trades_grouped_df, scale=config_orderflow["scale"]
                    )
                )
                # calculate imbalances for each candle's orderflow
                df.loc[is_between, "imbalances"] = df.loc[is_between, "orderflow"].apply(
                    lambda x: trades_orderflow_to_imbalances(
                        x,
                        imbalance_ratio=config_orderflow["imbalance_ratio"],
                        imbalance_volume=config_orderflow["imbalance_volume"],
                    )
                )

                _stacked_imb = config_orderflow["stacked_imbalance_range"]
                df.loc[is_between, "stacked_imbalances_bid"] = df.loc[
                    is_between, "imbalances"
                ].apply(lambda x: stacked_imbalance_bid(x, stacked_imbalance_range=_stacked_imb))
                df.loc[is_between, "stacked_imbalances_ask"] = df.loc[
                    is_between, "imbalances"
                ].apply(lambda x: stacked_imbalance_ask(x, stacked_imbalance_range=_stacked_imb))

                bid = np.where(
                    trades_grouped_df["side"].str.contains("sell"),
                    trades_grouped_df["amount"],
                    0,
                )
                ask = np.where(
                    trades_grouped_df["side"].str.contains("buy"),
                    trades_grouped_df["amount"],
                    0,
                )
                deltas_per_trade = ask - bid
                min_delta = 0
                max_delta = 0
                delta = 0
                for d in deltas_per_trade:
                    delta += d
                    if delta > max_delta:
                        max_delta = delta
                    if delta < min_delta:
                        min_delta = delta
                df.loc[is_between, "max_delta"] = max_delta
                df.loc[is_between, "min_delta"] = min_delta

                df.loc[is_between, "bid"] = np.where(
                    trades_grouped_df["side"].str.contains("sell"), trades_grouped_df["amount"], 0
                ).sum()
                df.loc[is_between, "ask"] = np.where(
                    trades_grouped_df["side"].str.contains("buy"), trades_grouped_df["amount"], 0
                ).sum()
                df.loc[is_between, "delta"] = df.loc[is_between, "ask"] - df.loc[is_between, "bid"]
                df.loc[is_between, "total_trades"] = len(trades_grouped_df)
                # copy to avoid memory leaks
                dataframe.loc[is_between] = df.loc[is_between].copy()
            else:
                logger.debug(f"Found NO candles for trades starting with {candle_start}")
        logger.debug(f"trades.groups_keys in {time.time() - start_time} seconds")

    except Exception as e:
        logger.exception("Error populating dataframe with trades:", e)
        raise DependencyException(e)

    return dataframe


def trades_to_volumeprofile_with_total_delta_bid_ask(trades: pd.DataFrame, scale: float):
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


def stacked_imbalance(
    df: pd.DataFrame, label: str, stacked_imbalance_range: int, should_reverse: bool
):
    """
    y * (y.groupby((y != y.shift()).cumsum()).cumcount() + 1)
    https://stackoverflow.com/questions/27626542/counting-consecutive-positive-values-in-python-pandas-array
    """
    imbalance = df[f"{label}_imbalance"]
    int_series = pd.Series(np.where(imbalance, 1, 0))
    stacked = int_series * (
        int_series.groupby((int_series != int_series.shift()).cumsum()).cumcount() + 1
    )

    max_stacked_imbalance_idx = stacked.index[stacked >= stacked_imbalance_range]
    stacked_imbalance_price = np.nan
    if not max_stacked_imbalance_idx.empty:
        idx = (
            max_stacked_imbalance_idx[0]
            if not should_reverse
            else np.flipud(max_stacked_imbalance_idx)[0]
        )
        stacked_imbalance_price = imbalance.index[idx]
    return stacked_imbalance_price


def stacked_imbalance_ask(df: pd.DataFrame, stacked_imbalance_range: int):
    return stacked_imbalance(df, "ask", stacked_imbalance_range, should_reverse=True)


def stacked_imbalance_bid(df: pd.DataFrame, stacked_imbalance_range: int):
    return stacked_imbalance(df, "bid", stacked_imbalance_range, should_reverse=False)
