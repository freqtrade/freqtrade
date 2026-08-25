"""RSI + Bollinger Band mean-reversion strategy, adapted from the community "Bandtastic"
strategy (freqtrade/freqtrade-strategies, C:\\dev\\freqtrade-strategies locally --
original author Robert Roman, MIT-licensed).

Research hypothesis, not an assumed edge -- BTC/USDT, 1h (matching EmaTrendFollow's
universe for a controlled comparison), meant to be run through the research/ package's
statistical validation gate (walk-forward + Deflated Sharpe Ratio + Benjamini-Hochberg
FDR + Probability of Backtest Overfitting) before ever touching paper money.

**What changed from the original, and why:** the original Bandtastic.py ships with
minimal_roi, stoploss, trailing_stop*, and every buy/sell parameter default already set
to the output of the original author's own hyperopt run (their own file's comment cites
"199/40000" epochs -- a single cherry-picked result out of 40,000 trials, exactly the
kind of beautiful overfit this whole research/ package exists to catch). Reusing those
numbers unmodified would mean gating someone else's already-overfit result, not testing
an independent hypothesis. This version strips all of that: minimal_roi is disabled
(matching EmaTrendFollow's own rationale -- measure the raw signal edge, not a
hardcoded profit target), stoploss is reset to a plain, unoptimized guardrail, trailing
stop is dropped entirely, and the original's four parallel Bollinger Band widths plus
CategoricalParameter guard/trigger toggle machinery (a hyperopt-search-space pattern,
not something research/walkforward.py's grid-sweep mechanism is built to explore) are
collapsed to one canonical 20-period/2-std Bollinger Band and three plain numeric knobs.

Entry:  RSI below rsi_buy AND close below the Bollinger lower band (oversold + price
        extreme -- the core mean-reversion pair the original strategy was built around)
Exit:   RSI above rsi_sell AND close above the Bollinger upper band

The IntParameter knobs below use optimize=False and are read via .value directly
inside populate_indicators() -- this is deliberately NOT meant for freqtrade's own
native hyperopt. Parameter search is owned by the external research/ package
(research/walkforward.py's WalkForwardRunner), which re-instantiates indicators fresh
for each grid point by setting `.value` directly and recomputing -- same convention as
EmaTrendFollow.
"""

import talib.abstract as ta
from pandas import DataFrame
from technical import qtpylib

from freqtrade.strategy import IntParameter, IStrategy


class BandtasticMeanReversion(IStrategy):
    INTERFACE_VERSION = 3

    timeframe = "1h"

    # ROI disabled -- let the RSI/Bollinger exit and stoploss handle position exits, so
    # the backtest measures the strategy's raw signal edge rather than a hardcoded
    # profit target. Same rationale as EmaTrendFollow.
    minimal_roi = {}

    # Fixed risk-control constant for this first experiment (not a tuned/gridded
    # parameter yet) -- there is no literature-canonical stoploss for BTC 1h; -8% is a
    # loose guardrail, same placeholder EmaTrendFollow uses.
    stoploss = -0.08

    # Covers the widest Bollinger window in bb_window's range (40) with headroom.
    startup_candle_count = 50

    bb_window = IntParameter(10, 40, default=20, space="buy", optimize=False)
    rsi_buy = IntParameter(20, 40, default=30, space="buy", optimize=False)
    rsi_sell = IntParameter(60, 80, default=70, space="sell", optimize=False)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["rsi"] = ta.RSI(dataframe)
        bollinger = qtpylib.bollinger_bands(
            qtpylib.typical_price(dataframe), window=self.bb_window.value, stds=2
        )
        dataframe["bb_lowerband"] = bollinger["lower"]
        dataframe["bb_upperband"] = bollinger["upper"]
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe["rsi"] < self.rsi_buy.value)
            & (dataframe["close"] < dataframe["bb_lowerband"])
            & (dataframe["volume"] > 0),
            "enter_long",
        ] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe["rsi"] > self.rsi_sell.value)
            & (dataframe["close"] > dataframe["bb_upperband"])
            & (dataframe["volume"] > 0),
            "exit_long",
        ] = 1
        return dataframe
