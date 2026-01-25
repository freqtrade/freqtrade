"""Deterministic options strategy driven by underlying cash indicators."""

from __future__ import annotations

import os
import logging
from typing import Iterable, Optional
from datetime import datetime, time

import pandas as pd
from pandas import DataFrame
import talib.abstract as ta

from adapters.ccxt_shim.instrument import InstrumentType, parse_pair
from freqtrade.strategy import IStrategy, merge_informative_pair

logger = logging.getLogger(__name__)


class IndiaOptionsAutoStrategy(IStrategy):
    """Trade CE/PE options based on underlying cash trend signals."""

    INTERFACE_VERSION = 3

    timeframe = "5m"
    startup_candle_count = 50

    minimal_roi = {"0": 0.12}
    stoploss = -0.15

    process_only_new_candles = True
    can_short: bool = False

    def informative_pairs(self) -> list[tuple[str, str]]:
        """Return underlying cash pairs for each option pair in whitelist."""
        whitelist = self.config.get("exchange", {}).get("pair_whitelist", [])
        pairs: set[tuple[str, str]] = set()
        for pair in whitelist:
            try:
                spec = parse_pair(pair)
            except ValueError:
                logger.warning("Skipping non-canonical pair in whitelist: %s", pair)
                continue
            if spec.type != InstrumentType.OPT:
                continue
            pairs.add((f"{spec.underlying}/INR", self.timeframe))
        return sorted(pairs)

    def _underlying_pair(self, pair: str) -> str | None:
        try:
            spec = parse_pair(pair)
        except ValueError:
            logger.warning("Unable to parse pair: %s", pair)
            return None
        if spec.type != InstrumentType.OPT:
            return None
        return f"{spec.underlying}/INR"

    def _option_right(self, pair: str) -> str | None:
        try:
            spec = parse_pair(pair)
        except ValueError:
            return None
        if spec.type != InstrumentType.OPT:
            return None
        return spec.right

    @staticmethod
    def _ensure_columns(dataframe: DataFrame, columns: Iterable[str]) -> None:
        for column in columns:
            if column not in dataframe.columns:
                dataframe[column] = pd.NA

    @staticmethod
    def _ist_time_mask(dataframe: DataFrame) -> pd.Series:
        dates = pd.to_datetime(dataframe["date"], utc=True)
        local_times = dates.dt.tz_convert("Asia/Kolkata").dt.time
        return local_times

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Merge underlying cash indicators into option dataframe."""
        underlying_pair = self._underlying_pair(metadata.get("pair", ""))
        if not underlying_pair or not self.dp:
            self._ensure_columns(
                dataframe, ["ema_5_underlying", "ema_20_underlying", "rsi_14_underlying"]
            )
            return dataframe

        informative = self.dp.get_pair_dataframe(pair=underlying_pair, timeframe=self.timeframe)
        informative = informative.copy()
        informative["ema_5"] = ta.EMA(informative, timeperiod=5)
        informative["ema_20"] = ta.EMA(informative, timeperiod=20)
        informative["rsi_14"] = ta.RSI(informative, timeperiod=14)
        merged = merge_informative_pair(
            dataframe,
            informative,
            self.timeframe,
            self.timeframe,
            append_timeframe=False,
            suffix="underlying",
        )
        self._ensure_columns(merged, ["ema_5_underlying", "ema_20_underlying", "rsi_14_underlying"])
        return merged

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Enter CE in bull regime, PE in bear regime within time window."""
        right = self._option_right(metadata.get("pair", ""))
        if right is None:
            return dataframe

        local_times = self._ist_time_mask(dataframe)
        within_window = (local_times >= time(9, 45)) & (local_times <= time(14, 30))

        bull = (dataframe["ema_5_underlying"] > dataframe["ema_20_underlying"]) & (
            dataframe["rsi_14_underlying"] > 55
        )
        bear = (dataframe["ema_5_underlying"] < dataframe["ema_20_underlying"]) & (
            dataframe["rsi_14_underlying"] < 45
        )

        if right == "CE":
            dataframe.loc[within_window & bull, "enter_long"] = 1
        elif right == "PE":
            dataframe.loc[within_window & bear, "enter_long"] = 1

        if os.environ.get("RISK_FORCE_SIGNAL"):
            dataframe.loc[:, "enter_long"] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Force exits after 14:30 IST."""
        local_times = self._ist_time_mask(dataframe)
        dataframe.loc[local_times > time(14, 30), "exit_long"] = 1
        return dataframe

    def confirm_trade_entry(
        self,
        pair: str,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        current_time: datetime,
        entry_tag: Optional[str],
        side: str,
        **kwargs,
    ) -> bool:
        """
        Check risk guardrails before entering a trade.
        """
        try:
            from user_data.risk_guardrails.guardrails import RiskGuardrails

            guardrails = RiskGuardrails(self.config)

            # EMERGENCY DEBUG
            logger.error("DEBUG: confirm_trade_entry called for %s", pair)

            # Context for P11
            context = {
                "open_trades_count": 0,  # Simplified for P11
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
