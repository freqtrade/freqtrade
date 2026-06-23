"""Auto-generated source-translated strategy drafts.

Do not edit by hand. Re-generate with user_data/strategy_research/generate_source_strategies.py.
These classes are research-only and must not be promoted to live without manual approval.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

from pandas import DataFrame
import talib.abstract as ta

sys.path.append(str(Path(__file__).resolve().parents[1]))

from btc_eth_risk_controlled_strategies import BtcEthFuturesRegime10xOneMinuteStrategy


GENERATED_AT_UTC = '20260623T030135Z'


class SourceTranslatedFreqtradeOfficialSampleStrategy49665aee345bStrategy(BtcEthFuturesRegime10xOneMinuteStrategy):
    """Research-only source-translated mean-reversion strategy from freqtrade-official-sample-strategy-49665aee345b."""

    minimal_roi = {"0": 0.006, "90": 0.003, "240": 0}
    stoploss = -0.008
    startup_candle_count = 2400

    def leverage(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_leverage: float,
        max_leverage: float,
        entry_tag: str | None,
        side: str,
        **kwargs,
    ) -> float:
        return min(1.0, max_leverage)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = super().populate_indicators(dataframe, metadata)
        bb = ta.BBANDS(dataframe, timeperiod=120)
        dataframe["src_bb_upper"] = bb["upperband"]
        dataframe["src_bb_middle"] = bb["middleband"]
        dataframe["src_bb_lower"] = bb["lowerband"]
        dataframe["src_bb_width"] = (dataframe["src_bb_upper"] - dataframe["src_bb_lower"]) / dataframe["src_bb_middle"]
        dataframe["src_rsi"] = ta.RSI(dataframe, timeperiod=14)
        dataframe["src_volume_mean"] = dataframe["volume"].rolling(120).mean()
        dataframe["src_range_ok"] = (
            dataframe["range_regime"]
            & ~dataframe["high_vol_regime"]
            & (dataframe["src_bb_width"].between(0.0015, 0.025))
        )
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                dataframe["risk_allowed"]
                & dataframe["src_range_ok"]
                & (dataframe["close"] < dataframe["src_bb_lower"])
                & (dataframe["src_rsi"] < 32)
                & (dataframe["volume"] > dataframe["src_volume_mean"] * 0.6)
            ),
            ["enter_long", "enter_tag"],
        ] = (1, "src_freqtrade_official_sample_strategy_49665_mr_long")

        dataframe.loc[
            (
                dataframe["risk_allowed"]
                & dataframe["src_range_ok"]
                & (dataframe["close"] > dataframe["src_bb_upper"])
                & (dataframe["src_rsi"] > 68)
                & (dataframe["volume"] > dataframe["src_volume_mean"] * 0.6)
            ),
            ["enter_short", "enter_tag"],
        ] = (1, "src_freqtrade_official_sample_strategy_49665_mr_short")
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                ((dataframe["close"] >= dataframe["src_bb_middle"]) | ~dataframe["src_range_ok"])
                & (dataframe["volume"] > 0)
            ),
            ["exit_long", "exit_tag"],
        ] = (1, "src_freqtrade_official_sample_strategy_49665_mr_long_exit")

        dataframe.loc[
            (
                ((dataframe["close"] <= dataframe["src_bb_middle"]) | ~dataframe["src_range_ok"])
                & (dataframe["volume"] > 0)
            ),
            ["exit_short", "exit_tag"],
        ] = (1, "src_freqtrade_official_sample_strategy_49665_mr_short_exit")
        return dataframe


class SourceTranslatedSeedRsiEmaPullbackNoteStrategy(BtcEthFuturesRegime10xOneMinuteStrategy):
    """Research-only source-translated strategy from seed_rsi_ema_pullback_note."""

    minimal_roi = {"0": 0.012, "60": 0.006, "240": 0}
    stoploss = -0.012
    startup_candle_count = 2400

    def leverage(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_leverage: float,
        max_leverage: float,
        entry_tag: str | None,
        side: str,
        **kwargs,
    ) -> float:
        return min(1.0, max_leverage)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = super().populate_indicators(dataframe, metadata)
        dataframe["src_ema_fast"] = ta.EMA(dataframe, timeperiod=20)
        dataframe["src_ema_mid"] = ta.EMA(dataframe, timeperiod=60)
        dataframe["src_ema_slow"] = ta.EMA(dataframe, timeperiod=240)
        dataframe["src_rsi"] = ta.RSI(dataframe, timeperiod=14)
        dataframe["src_ret_5m"] = dataframe["close"] / dataframe["close"].shift(5) - 1.0
        dataframe["src_pullback_long"] = dataframe["low"].rolling(30).min() <= dataframe["src_ema_mid"] * 1.0015
        dataframe["src_pullback_short"] = dataframe["high"].rolling(30).max() >= dataframe["src_ema_mid"] * 0.9985
        dataframe["src_resume_long"] = (
            (dataframe["close"] > dataframe["src_ema_fast"])
            & (dataframe["close"] > dataframe["open"])
            & (dataframe["src_rsi"].between(45, 65))
            & (dataframe["src_ret_5m"] > 0.0005)
        )
        dataframe["src_resume_short"] = (
            (dataframe["close"] < dataframe["src_ema_fast"])
            & (dataframe["close"] < dataframe["open"])
            & (dataframe["src_rsi"].between(35, 55))
            & (dataframe["src_ret_5m"] < -0.0005)
        )
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                dataframe["risk_allowed"]
                & dataframe["trend_up_regime"]
                & (dataframe["close"] > dataframe["src_ema_slow"])
                & dataframe["src_pullback_long"]
                & dataframe["src_resume_long"]
                & (dataframe["volume"] > 0)
            ),
            ["enter_long", "enter_tag"],
        ] = (1, "src_seed_rsi_ema_pullback_note_long")

        dataframe.loc[
            (
                dataframe["risk_allowed"]
                & dataframe["trend_down_regime"]
                & (dataframe["close"] < dataframe["src_ema_slow"])
                & dataframe["src_pullback_short"]
                & dataframe["src_resume_short"]
                & (dataframe["volume"] > 0)
            ),
            ["enter_short", "enter_tag"],
        ] = (1, "src_seed_rsi_ema_pullback_note_short")
        return dataframe

