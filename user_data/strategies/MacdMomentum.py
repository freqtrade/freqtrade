"""MACD crossover momentum strategy, inspired by the community "UniversalMACD"
strategy (freqtrade/freqtrade-strategies, C:\\dev\\freqtrade-strategies locally --
original author Masoud Azizi / @mablue).

Research hypothesis, not an assumed edge -- BTC/USDT, 1h (matching EmaTrendFollow's
and BandtasticMeanReversion's universe for a controlled comparison), meant to be run
through the research/ package's statistical validation gate (walk-forward + Deflated
Sharpe Ratio + Benjamini-Hochberg FDR + Probability of Backtest Overfitting) before
ever touching paper money.

**What changed from the original, and why:** the original UniversalMACD.py ships with
minimal_roi, stoploss, and all four buy/sell parameters already set to the output of
the original author's own hyperopt run (their own file's comment cites "16/100" epochs
and "Objective: -11.63412" -- a single cherry-picked result out of 100 trials). Worse,
the original's actual MECHANISM is itself overfit-prone independent of those specific
numbers: it enters/exits whenever a normalized MACD value (`(ema12/ema26) - 1`) falls
inside a hyperopt-FITTED numeric band only 0.0024 wide (-0.01416 to -0.01176) -- a
razor-thin window carved out of noise, not a real momentum signal, structurally the
same anti-pattern as Bandtastic's four parallel hyperopt-selected Bollinger widths.
This version drops that band-fit mechanism entirely and uses MACD's own standard,
textbook interpretation instead: the MACD line crossing its signal line. minimal_roi
is disabled (measure the raw signal edge, matching EmaTrendFollow/
BandtasticMeanReversion), stoploss is reset to the same plain -8% guardrail used by
both, and the three real MACD periods (fast/slow/signal EMA) become plain IntParameter
knobs at their canonical 12/26/9 defaults -- the actual historically-motivated MACD
parameterization, not a fitted one. The original's unused `pandas_ta` import (never
referenced in its own populate_indicators) is dropped rather than carried forward as a
new dependency this fork doesn't otherwise need.

Entry:  MACD line crosses above its signal line (bullish momentum)
Exit:   MACD line crosses below its signal line (bearish momentum)

The IntParameter knobs below use optimize=False and are read via .value directly
inside populate_indicators() -- this is deliberately NOT meant for freqtrade's own
native hyperopt. Parameter search is owned by the external research/ package
(research/walkforward.py's WalkForwardRunner), which re-instantiates indicators fresh
for each grid point by setting `.value` directly and recomputing -- same convention as
EmaTrendFollow and BandtasticMeanReversion.
"""

import talib.abstract as ta
from pandas import DataFrame
from technical import qtpylib

from freqtrade.strategy import IntParameter, IStrategy


class MacdMomentum(IStrategy):
    INTERFACE_VERSION = 3

    timeframe = "1h"

    # ROI disabled -- let the MACD crossover exit and stoploss handle position exits,
    # so the backtest measures the strategy's raw signal edge rather than a hardcoded
    # profit target. Same rationale as EmaTrendFollow/BandtasticMeanReversion.
    minimal_roi = {}

    # Fixed risk-control constant for this first experiment (not a tuned/gridded
    # parameter yet) -- there is no literature-canonical stoploss for BTC 1h; -8% is a
    # loose guardrail, same placeholder the other two strategies use.
    stoploss = -0.08

    # Covers the widest macd_slow + macd_signal range with headroom.
    startup_candle_count = 50

    macd_fast = IntParameter(8, 16, default=12, space="buy", optimize=False)
    macd_slow = IntParameter(20, 30, default=26, space="buy", optimize=False)
    macd_signal = IntParameter(5, 12, default=9, space="buy", optimize=False)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        macd = ta.MACD(
            dataframe,
            fastperiod=self.macd_fast.value,
            slowperiod=self.macd_slow.value,
            signalperiod=self.macd_signal.value,
        )
        dataframe["macd"] = macd["macd"]
        dataframe["macdsignal"] = macd["macdsignal"]
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            qtpylib.crossed_above(dataframe["macd"], dataframe["macdsignal"])
            & (dataframe["volume"] > 0),
            "enter_long",
        ] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            qtpylib.crossed_below(dataframe["macd"], dataframe["macdsignal"])
            & (dataframe["volume"] > 0),
            "exit_long",
        ] = 1
        return dataframe
