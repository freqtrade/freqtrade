import logging
from datetime import datetime

import numpy as np
import pandas as pd

from freqtrade.constants import IntOrInf
from freqtrade.exchange import (
    timeframe_to_prev_date,
    timeframe_to_resample_freq,
)
from freqtrade.util import dt_from_ts


logger = logging.getLogger(__name__)


def analyze_trade_parallelism(trades: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """
    Find overlapping trades by expanding each trade once per period it was open
    and then counting overlaps.
    :param trades: Trades Dataframe - can be loaded from backtest, or created
        via trade_list_to_dataframe
    :param timeframe: Timeframe used for backtest
    :return: dataframe with open-counts per time-period in timeframe
    """
    from freqtrade.exchange import timeframe_to_resample_freq

    timeframe_freq = timeframe_to_resample_freq(timeframe)
    dates = [
        pd.Series(
            pd.date_range(
                row[1]["open_date"],
                row[1]["close_date"],
                freq=timeframe_freq,
                # Exclude right boundary - the date is the candle open date.
                inclusive="left",
            )
        )
        for row in trades[["open_date", "close_date"]].iterrows()
    ]
    deltas = [len(x) for x in dates]
    dates = pd.Series(pd.concat(dates).values, name="date")
    df2 = pd.DataFrame(np.repeat(trades.values, deltas, axis=0), columns=trades.columns)

    df2 = pd.concat([dates, df2], axis=1)
    df2 = df2.set_index("date")
    df_final = df2.resample(timeframe_freq)[["pair"]].count()
    df_final = df_final.rename({"pair": "open_trades"}, axis=1)
    return df_final


def evaluate_result_multi(
    trades: pd.DataFrame, timeframe: str, max_open_trades: IntOrInf
) -> pd.DataFrame:
    """
    Find overlapping trades by expanding each trade once per period it was open
    and then counting overlaps
    :param trades: Trades Dataframe - can be loaded from backtest, or created
        via trade_list_to_dataframe
    :param timeframe: Frequency used for the backtest
    :param max_open_trades: parameter max_open_trades used during backtest run
    :return: dataframe with open-counts per time-period in freq
    """
    df_final = analyze_trade_parallelism(trades, timeframe)
    return df_final[df_final["open_trades"] > max_open_trades]


def balance_distribution_over_time(
    trades: pd.DataFrame,
    min_date: datetime,
    max_date: datetime,
    timeframe: str,
    stake_currency: str,
    start_balance: float,
    pairlist: list[str],
) -> pd.DataFrame:
    """
    Return a dataframe with stake_currency and the pairlist as columns
    Each column will contain the amount of the currency at the given time
    Columns added are:
        - stake_currency: amount of stake currency
        - <pair>: amount of base currency in the pair
        - <pair>_leverage: leverage used for the pair at the time (NaN if no open trade)
        - <pair>_is_short: 1 if the open trade is short, 0 if long (NaN if no open trade)
        - <pair>_collateral: amount of stake currency used as collateral for open trades
    :param trades: Trades Dataframe - can be loaded from backtest, or created
        via trade_list_to_dataframe
    :param timeframe: Frequency to use for the resulting dataframe
    :param min_date: start date
    :param max_date: End date (will be rounded down to timeframe)
    :param stake_currency: The stake currency
    :param start_balance: Starting balance in stake currency
    :param pairlist: List of trading pairs to include in the dataframe
        Can be obtained via trade_df["pair"].unique()
        For pairs without trades, the column will be all zeros
    :return: Dataframe with balance distribution over time
    """
    min_date_res = timeframe_to_prev_date(timeframe, min_date)
    max_date_res = timeframe_to_prev_date(timeframe, max_date)
    index = pd.date_range(min_date_res, max_date_res, freq=timeframe_to_resample_freq(timeframe))
    pairs_lev = [f"{pair}_leverage" for pair in pairlist]
    pairs_is_short = [f"{pair}_is_short" for pair in pairlist]
    pairs_collateral = [f"{pair}_collateral" for pair in pairlist]
    pairs_lev += pairs_is_short

    df = pd.DataFrame(
        index=index, columns=[stake_currency] + pairlist + pairs_lev + pairs_collateral, dtype=float
    )
    # Initialize variables to starting values
    df[stake_currency] = float(start_balance)
    df[pairlist + pairs_collateral] = 0.0
    df[pairs_lev] = np.nan

    for trade in trades.sort_values(by=["open_date"]).itertuples():
        pair = trade.pair
        end_date = trade.close_date if trade.close_date is not pd.NaT else None
        # Exclude open orders - these won't have order_filled_timestamp set.
        df.loc[trade.open_date : end_date, f"{pair}_leverage"] = trade.leverage
        df.loc[trade.open_date : end_date, f"{pair}_is_short"] = 1 if trade.is_short else 0
        orders = [o for o in trade.orders if o["order_filled_timestamp"]]
        current_position = 0
        current_collateral = 0
        for order in sorted(orders, key=lambda x: x["order_filled_timestamp"]):
            filled_at = pd.Timestamp(dt_from_ts(order["order_filled_timestamp"]))
            real_amount = order.get("filled", order["amount"])
            stake = order["safe_price"] * real_amount
            stake_no_lev = stake / trade.leverage
            if order["ft_is_entry"]:
                # Entry order: lock collateral and pay fee
                # For both long and short: balance decreases by collateral + fee
                fee_open = stake * trade.fee_open
                current_position += real_amount
                current_collateral += stake_no_lev
                df.loc[filled_at:end_date, pair] += real_amount
                df.loc[filled_at:end_date, f"{pair}_collateral"] += stake_no_lev
                df.loc[filled_at:, stake_currency] -= stake_no_lev + fee_open
            else:
                # Exit order: release collateral and realize profit/loss
                fee_close = stake * trade.fee_close
                if trade.is_short:
                    # For SHORT
                    df.loc[filled_at:, stake_currency] += (
                        current_collateral * (1 + trade.leverage) - stake
                    ) - fee_close
                else:
                    # For LONG
                    df.loc[filled_at:, stake_currency] += (
                        stake - current_collateral * (trade.leverage - 1) - fee_close
                    )
                df.loc[filled_at:end_date, pair] -= real_amount
                df.loc[filled_at:end_date, f"{pair}_collateral"] -= stake_no_lev
                current_position -= real_amount
                current_collateral -= stake_no_lev

    # Round to avoid floating point issues
    df = df.round(14)
    return df
