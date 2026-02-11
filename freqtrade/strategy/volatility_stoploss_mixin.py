
from datetime import datetime
from pandas import DataFrame
from freqtrade.persistence import Trade
import pandas as pd

class VolatilityStoplossMixin:
    """
    Mixin to add volatility-based stoploss to a strategy.

    Usage:
    class MyStrategy(IStrategy, VolatilityStoplossMixin):
        stoploss_atr_multiplier = 3.0

        def populate_indicators(self, dataframe, metadata):
            dataframe['atr'] = ta.ATR(dataframe)
            return dataframe

    This mixin overrides `custom_stoploss`.
    """

    # Multiplier for ATR
    # Can be overridden in the strategy class
    stoploss_atr_multiplier = 2.0

    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:

        # Access the dataframe via dataprovider
        if not self.dp:
            return self.stoploss

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe is None or dataframe.empty:
            return self.stoploss

        # Get the relevant candle (avoid lookahead bias)
        # We need the candle corresponding to current_time (or the last closed one).
        # Since dataframe is sorted by date:
        # We select all rows where date <= current_time
        # And take the last one.

        # Note: In backtesting, 'dataframe' contains the full history.
        # In live, it contains history up to now.

        # Using searchsorted would be faster but requires converting to int64 or similar.
        # Boolean indexing is easier to implement correctly.

        # We assume dataframe['date'] exists and is datetime compatible.

        # Optimization: slicing the tail might be faster if we assume we are near the end in live?
        # But in backtesting we iterate through.

        # Check if 'date' column exists
        if 'date' not in dataframe.columns:
            return self.stoploss

        # Filter for past/present candles
        # Note: This might be slow in backtesting loop!
        # But it guarantees correctness.
        matches = dataframe.loc[dataframe['date'] <= current_time]

        if matches.empty:
            return self.stoploss

        candle = matches.iloc[-1]

        # Check if ATR is present
        if 'atr' not in candle:
            # Fallback to default stoploss if ATR is not calculated
            return self.stoploss

        atr = candle['atr']

        # Calculate dynamic stoploss based on current ATR
        # Stoploss distance = ATR * Multiplier
        # We return the ratio relative to current_rate (negative value)

        stop_dist = atr * self.stoploss_atr_multiplier

        if current_rate == 0:
            return self.stoploss

        stop_ratio = - (stop_dist / current_rate)

        # Sanity check: ensure it's negative
        if stop_ratio >= 0:
            return self.stoploss

        return stop_ratio
