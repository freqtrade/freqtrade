"""EMA crossover trend-following strategy, filtered by a long-term trend regime EMA.

Research hypothesis, not an assumed edge -- BTC/USDT, 1h, meant to be run through the
research/ package's statistical validation gate (walk-forward + Deflated Sharpe Ratio +
Benjamini-Hochberg FDR + Probability of Backtest Overfitting) before ever touching paper
money. Designed and cross-checked with ChatGPT/Gemini via lmchatbot; see conversation for
the full design rationale.

Entry:  ema_fast crosses above ema_slow AND close is above ema_trend (trend regime filter)
Exit:   ema_fast crosses below ema_slow (no trend filter on exit)

The EMA period Parameters below use optimize=False and are read via .value directly inside
populate_indicators() -- this is deliberately NOT meant for freqtrade's own native hyperopt
(which expects a different, .range-based access pattern for that use case). Parameter
search for this strategy is owned by the external research/ package
(research/walkforward.py's WalkForwardRunner), which re-instantiates indicators fresh for
each grid point by setting `.value` directly and recomputing -- exactly the access pattern
used here.
"""

import talib.abstract as ta
from pandas import DataFrame
from technical import qtpylib

from freqtrade.strategy import IntParameter, IStrategy


class EmaTrendFollow(IStrategy):
    INTERFACE_VERSION = 3

    timeframe = "1h"

    # ROI disabled -- let the crossover exit and stoploss handle position exits, so the
    # backtest measures the strategy's raw signal edge rather than a hardcoded profit target.
    minimal_roi = {}

    # Fixed risk-control constant for this first experiment (not a tuned/gridded parameter
    # yet) -- there is no literature-canonical stoploss for BTC 1h; -8% is a loose guardrail.
    stoploss = -0.08

    # EMA(200) needs 200 candles of history before its value is meaningful.
    startup_candle_count = 200

    ema_fast = IntParameter(5, 20, default=12, space="buy", optimize=False)
    ema_slow = IntParameter(20, 50, default=26, space="buy", optimize=False)
    ema_trend = IntParameter(100, 300, default=200, space="buy", optimize=False)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["ema_fast"] = ta.EMA(dataframe, timeperiod=self.ema_fast.value)
        dataframe["ema_slow"] = ta.EMA(dataframe, timeperiod=self.ema_slow.value)
        dataframe["ema_trend"] = ta.EMA(dataframe, timeperiod=self.ema_trend.value)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            qtpylib.crossed_above(dataframe["ema_fast"], dataframe["ema_slow"])
            & (dataframe["close"] > dataframe["ema_trend"]),
            "enter_long",
        ] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            qtpylib.crossed_below(dataframe["ema_fast"], dataframe["ema_slow"]),
            "exit_long",
        ] = 1
        return dataframe
