"""Deterministic smoke strategy for India equities."""

from __future__ import annotations

import logging
import os
from datetime import datetime

import talib.abstract as ta
from pandas import DataFrame

from freqtrade.strategy import IStrategy

logger = logging.getLogger(__name__)


class IndiaEquitySmokeStrategy(IStrategy):
    """Simple deterministic strategy used for plumbing validation."""

    INTERFACE_VERSION = 3

    timeframe = "5m"
    startup_candle_count = 50

    minimal_roi = {"0": 0.01}
    stoploss = -0.02

    process_only_new_candles = True
    can_short: bool = False

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Add EMA and RSI indicators needed for entry/exit rules."""
        dataframe["ema_9"] = ta.EMA(dataframe, timeperiod=9)
        dataframe["ema_21"] = ta.EMA(dataframe, timeperiod=21)
        dataframe["rsi_14"] = ta.RSI(dataframe, timeperiod=14)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Define deterministic entry conditions."""
        dataframe.loc[
            (dataframe["ema_9"] > dataframe["ema_21"]) & (dataframe["rsi_14"] > 52),
            "enter_long",
        ] = 1

        if os.environ.get("RISK_FORCE_SIGNAL"):
            dataframe.loc[:, "enter_long"] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Define deterministic exit conditions."""
        dataframe.loc[
            (dataframe["ema_9"] < dataframe["ema_21"]) | (dataframe["rsi_14"] < 48),
            "exit_long",
        ] = 1

        if os.environ.get("RISK_FORCE_SIGNAL"):
            dataframe.loc[:, "enter_long"] = 1

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
        Check risk guardrails before entering a trade.
        """
        try:
            from user_data.risk_guardrails.guardrails import RiskGuardrails

            guardrails = RiskGuardrails(self.config)

            # Context for P11
            context = {
                "open_trades_count": 0,
                "daily_trade_count": 0,
                "daily_profit_ratio": 0.0,
            }

            blocked, reason = guardrails.should_block_entry(context)
            if blocked:
                logger.info("RISK_BLOCK entry for %s: %s", pair, reason)
                return False

            logger.info("RISK_OK entry for %s: %s", pair, reason)
            return True
        except Exception as e:
            logger.error("Error in risk guardrail check: %s", e)
            return True
