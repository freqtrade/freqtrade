# pragma pylint: disable=missing-docstring, invalid-name, stateless-class
import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame

from datetime import datetime
from freqtrade.strategy import (
    IStrategy,
    DecimalParameter,
)
from freqtrade.persistence import Trade


class PortfolioStrategy(IStrategy):
    """
    This is a Portfolio Rebalancing Strategy template.
    It attempts to maintain a target allocation for each pair in the whitelist.

    How it works:
    - Defines a target percentage for each pair (1 / number of pairs).
    - Buys if the current holding is below the target.
    - Sells if the current holding is above the target (with a threshold).

    WARNING: This strategy behaves differently in Backtesting vs Live.
    In Backtesting, it behaves like a buy-and-hold or simple rebalance if logic allows.
    In Live, it checks actual wallet balances.
    """

    INTERFACE_VERSION = 3

    # Minimal ROI - set to very high because we exit based on rebalancing
    minimal_roi = {
        "0": 100
    }

    # Stoploss - effectively disabled
    stoploss = -0.99
    trailing_stop = False

    timeframe = '1h'

    # Rebalance threshold (percentage deviation allowed before rebalancing)
    rebalance_threshold = DecimalParameter(0.01, 0.1, default=0.05, space="buy")

    # Use custom stake amount to buy the exact needed amount
    use_custom_stoploss = False

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # No indicators needed for pure rebalancing
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # We signal a buy, but the custom_stake_amount will determine IF and HOW MUCH
        dataframe.loc[:, 'enter_long'] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # We signal a sell, but custom_exit will determine if we should actually sell
        dataframe.loc[:, 'exit_long'] = 1
        return dataframe

    def custom_stake_amount(self, pair: str, current_time: 'datetime', current_rate: float,
                            proposed_stake: float, min_stake: float, max_stake: float,
                            leverage: float, entry_tag: str | None, side: str,
                            **kwargs) -> float:

        # Calculate target allocation
        # For simplicity, equal weight for all whitelisted pairs
        # In a real strategy, you might want weights in config

        # Note: self.active_pair_whitelist is available in live/dry-run
        # In backtesting, we might need a fallback
        whitelist = self.dp.current_whitelist()
        if not whitelist:
            return proposed_stake

        target_weight = 1.0 / len(whitelist)

        # Get total portfolio value
        total_balance = self.wallets.get_total_stake_amount()

        # Target value for this pair
        target_value = total_balance * target_weight

        # Current value of this pair
        # We need to know how much we already hold.
        # This is tricky because 'wallets' tracks balances, but 'Trade' tracks open trades.
        # Freqtrade separates 'trade' balance from 'wallet' balance sometimes.
        # Assuming we use all available balance for trading.

        # If we have an open trade, we are holding.
        # But custom_stake_amount is called for NEW trades (or DCA).
        # If we have an open trade, we usually don't open another unless max_open_trades > 1.

        # If we are here, it means we are considering opening a trade.
        # So current holding in 'trades' logic is 0 (unless we consider existing trades).

        # If we interpret this as "Buy to reach target", then:
        amount_to_buy = target_value # - current_holding (which is 0 for new trade)

        if amount_to_buy < min_stake:
            return 0  # Don't buy if target is too small

        return min(amount_to_buy, max_stake)

    def custom_exit(self, pair: str, trade: Trade, current_time: 'datetime', current_rate: float,
                    current_profit: float, **kwargs):

        # Check if we are over-allocated
        whitelist = self.dp.current_whitelist()
        if not whitelist:
            return False

        target_weight = 1.0 / len(whitelist)
        total_balance = self.wallets.get_total_stake_amount()
        target_value = total_balance * target_weight

        current_value = trade.stake_amount * (1 + current_profit)

        # Threshold for selling
        # If current value > target * (1 + threshold), sell some or all
        # Freqtrade 'custom_exit' returns a reason (str) or True/False/None.
        # It triggers a full exit. Partial exit is not fully supported in simple mode.

        threshold = self.rebalance_threshold.value

        if current_value > target_value * (1 + threshold):
            return "rebalance_sell"

        return False
