import logging

import numpy as np
import pandas as pd

from freqtrade.constants import IntOrInf


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
