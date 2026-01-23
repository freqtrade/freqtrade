"""
StacksEscrowStrategy - Demo strategy for ApexTrader-Stacks hackathon.
Integrates with Stacks blockchain escrow contract for secure trade execution.
"""

import logging
from datetime import datetime

import numpy as np
from pandas import DataFrame

from freqtrade.strategy import IStrategy, Order, Trade

import talib.abstract as ta


logger = logging.getLogger(__name__)


class StacksEscrowStrategy(IStrategy):
    """
    Demo strategy showcasing Stacks escrow integration.

    Trade flow:
    1. Signal detected via RSI/EMA indicators
    2. confirm_trade_entry() checks escrow balance and requests approval
    3. order_filled() updates escrow contract with trade result

    Config requirements:
    - exchange.name = "stacks"
    - stacks.enabled = true
    - stacks.escrow_contract = "<deployer>.usdcx-escrow"
    """

    INTERFACE_VERSION = 3

    # Minimal ROI - conservative for demo
    minimal_roi = {
        "60": 0.01,  # 1% after 60 mins
        "30": 0.02,  # 2% after 30 mins
        "0": 0.04,   # 4% immediately
    }

    stoploss = -0.10  # 10% max loss

    timeframe = "5m"
    process_only_new_candles = True
    use_exit_signal = True
    startup_candle_count = 50

    # Order configuration
    order_types = {
        "entry": "limit",
        "exit": "limit",
        "stoploss": "market",
        "stoploss_on_exchange": False,
    }

    # Track escrow state
    _escrow_manager = None
    _pending_escrow_trades: dict = {}

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self._stacks_config = config.get("stacks", {})
        self._escrow_enabled = self._stacks_config.get("enabled", False)
        self._max_escrow_pct = self._stacks_config.get("max_escrow_percentage", 0.1)

        if self._escrow_enabled:
            logger.info(
                f"StacksEscrowStrategy initialized with escrow integration. "
                f"Max escrow: {self._max_escrow_pct * 100}%"
            )
        else:
            logger.warning(
                "StacksEscrowStrategy running without escrow integration. "
                "Set stacks.enabled=true in config."
            )

    def _get_escrow_manager(self):
        """Lazy-load escrow manager from exchange."""
        if self._escrow_manager is None and self.dp is not None:
            try:
                exchange = self.dp._exchange
                if hasattr(exchange, "escrow"):
                    self._escrow_manager = exchange.escrow
                    logger.info("Escrow manager loaded from exchange")
            except Exception as e:
                logger.error(f"Failed to load escrow manager: {e}")
        return self._escrow_manager

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Add simple indicators for demo trading signals.
        Uses RSI and EMA for basic momentum detection.
        """
        # RSI - Relative Strength Index
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)

        # EMA - Exponential Moving Averages
        dataframe["ema_fast"] = ta.EMA(dataframe, timeperiod=8)
        dataframe["ema_slow"] = ta.EMA(dataframe, timeperiod=21)

        # EMA crossover signal
        dataframe["ema_diff"] = dataframe["ema_fast"] - dataframe["ema_slow"]

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Generate entry signals based on RSI and EMA crossover.
        - Long: RSI < 35 (oversold) and fast EMA crosses above slow EMA
        """
        dataframe.loc[
            (
                (dataframe["rsi"] < 35)  # Oversold condition
                & (dataframe["ema_diff"] > 0)  # Fast EMA above slow
                & (dataframe["ema_diff"].shift(1) <= 0)  # Just crossed
                & (dataframe["volume"] > 0)
            ),
            "enter_long",
        ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Generate exit signals.
        - Exit long: RSI > 70 (overbought) or EMA death cross
        """
        dataframe.loc[
            (
                (dataframe["rsi"] > 70)  # Overbought
                | (
                    (dataframe["ema_diff"] < 0)  # Fast EMA below slow
                    & (dataframe["ema_diff"].shift(1) >= 0)  # Just crossed down
                )
            )
            & (dataframe["volume"] > 0),
            "exit_long",
        ] = 1

        return dataframe

    def confirm_trade_entry(
        self,
        pair: str,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        current_time: datetime,
        entry_tag: str | None,
        side: str,
        **kwargs,
    ) -> bool:
        """
        Called before placing entry order.
        Checks escrow balance and requests trade approval from smart contract.
        """
        if not self._escrow_enabled:
            logger.debug("Escrow not enabled, allowing trade")
            return True

        escrow = self._get_escrow_manager()
        if escrow is None:
            logger.warning("Escrow manager not available, allowing trade")
            return True

        cost = amount * rate

        # Check escrow balance
        balance = escrow.get_escrow_balance()
        max_trade = balance * self._max_escrow_pct

        if cost > max_trade:
            logger.warning(
                f"Trade cost {cost} exceeds max escrow allowance {max_trade} "
                f"({self._max_escrow_pct * 100}% of {balance})"
            )
            return False

        if cost > balance:
            logger.warning(
                f"Insufficient escrow balance: {balance} < {cost}"
            )
            return False

        # Request trade approval from escrow contract
        trade_id = f"{pair}-{current_time.timestamp()}"
        try:
            tx_id = escrow.request_trade_approval(
                amount=cost,
                trade_id=trade_id,
                expiry_blocks=100,
            )
            if tx_id:
                self._pending_escrow_trades[trade_id] = {
                    "pair": pair,
                    "cost": cost,
                    "tx_id": tx_id,
                    "time": current_time,
                }
                logger.info(
                    f"Escrow trade approval requested: {trade_id}, tx: {tx_id}"
                )
                return True
            else:
                logger.error("Failed to get escrow approval")
                return False
        except Exception as e:
            logger.error(f"Escrow approval error: {e}")
            return False

    def confirm_trade_exit(
        self,
        pair: str,
        trade: Trade,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        exit_reason: str,
        current_time: datetime,
        **kwargs,
    ) -> bool:
        """
        Called before placing exit order.
        Verifies escrow contract is still active.
        """
        if not self._escrow_enabled:
            return True

        escrow = self._get_escrow_manager()
        if escrow is None:
            return True

        # Check if escrow is paused
        if escrow.is_paused():
            logger.warning("Escrow contract is paused, blocking exit")
            return False

        logger.info(
            f"Escrow exit confirmed for {pair}, reason: {exit_reason}"
        )
        return True

    def order_filled(
        self,
        pair: str,
        trade: Trade,
        order: Order,
        current_time: datetime,
        **kwargs,
    ) -> None:
        """
        Called after an order fills.
        Updates escrow contract with trade result (P/L).
        """
        if not self._escrow_enabled:
            return

        escrow = self._get_escrow_manager()
        if escrow is None:
            return

        # Only process exit orders for P/L settlement
        if order.ft_order_side == "sell":
            # Calculate profit/loss
            if trade.close_profit is not None:
                profit_loss = trade.stake_amount * trade.close_profit
            else:
                # Estimate based on order prices
                entry_cost = trade.stake_amount
                exit_value = order.safe_filled * order.safe_price
                profit_loss = exit_value - entry_cost

            try:
                # Execute trade settlement on escrow contract
                # Note: trade_id is simplified for demo
                tx_id = escrow.execute_trade(
                    trade_id=1,  # Would use actual contract trade ID
                    profit_loss=profit_loss,
                )
                logger.info(
                    f"Escrow trade executed: pair={pair}, P/L={profit_loss:.2f}, "
                    f"tx={tx_id}"
                )
            except Exception as e:
                logger.error(f"Escrow execution error: {e}")

        elif order.ft_order_side == "buy":
            logger.info(
                f"Entry order filled: pair={pair}, amount={order.safe_filled}, "
                f"price={order.safe_price}"
            )

    def bot_start(self, **kwargs) -> None:
        """Called when bot starts. Initialize escrow connection."""
        logger.info("StacksEscrowStrategy bot_start called")
        if self._escrow_enabled:
            escrow = self._get_escrow_manager()
            if escrow:
                status = escrow.get_status()
                logger.info(f"Escrow status: {status}")

    def informative_pairs(self):
        """No additional pairs needed for this simple strategy."""
        return []
