# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                       🌙 LUNAR PHASE STRATEGY 🌙                              ║
║                         Cyber-Mysticism Trading                               ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  "Crypto markets are driven by human emotion.                                ║
║   The moon controls human emotion.                                           ║
║   You do the math."                                                          ║
║                                                                              ║
║  This is a HODL Strategy based on ancient Lunar Cycles and Chinese           ║
║  Metaphysics (WuXing / Five Elements). It combines:                          ║
║                                                                              ║
║  1. LUNAR PHASES (月相)                                                       ║
║     - New Moon (朔月, Day 1): New beginnings, accumulation phase             ║
║     - Waxing Moon (上弦, Days 2-14): Growth energy, bullish momentum          ║
║     - Full Moon (望月, Day 15): Peak energy, profit-taking zone              ║
║     - Waning Moon (下弦, Days 16-29): Declining energy, bearish caution       ║
║                                                                              ║
║  2. WUXING / FIVE ELEMENTS (五行)                                             ║
║     The ancient Chinese system of elemental interactions:                    ║
║     - Wood (木) feeds Fire → Bullish for crypto (Fire element)               ║
║     - Fire (火) creates Earth → Neutral, consolidation                       ║
║     - Earth (土) bears Metal → Slight bearish, value extraction              ║
║     - Metal (金) collects Water → Bearish accumulation                       ║
║     - Water (水) nourishes Wood → Preparation for next cycle                 ║
║                                                                              ║
║  3. HEAVENLY STEMS & EARTHLY BRANCHES (天干地支)                              ║
║     Additional scoring based on the day's Gan-Zhi cycle.                     ║
║                                                                              ║
║  INSTALLATION:                                                               ║
║     pip install lunar_python                                                 ║
║                                                                              ║
║  DISCLAIMER: This strategy is for educational and entertainment purposes.    ║
║  Past lunar performance does not guarantee future astral returns.            ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
# --- Do not remove these imports ---
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from pandas import DataFrame
from typing import Optional, Union
import logging

from freqtrade.strategy import (
    IStrategy,
    Trade,
    Order,
    PairLocks,
    IntParameter,
    DecimalParameter,
)

# --------------------------------
# Lunar Python - Chinese Calendar Library
# Install with: pip install lunar_python
try:
    from lunar_python import Solar, Lunar
    LUNAR_AVAILABLE = True
except ImportError:
    LUNAR_AVAILABLE = False
    logging.warning(
        "lunar_python not installed. Please install with: pip install lunar_python\n"
        "LunarPhaseStrategy will use fallback calculations."
    )

logger = logging.getLogger(__name__)


class LunarPhaseStrategy(IStrategy):
    """
    Lunar Phase Strategy - Trading with the Cosmos

    This strategy uses Chinese lunar calendar data to generate trading signals:
    - Moon phases influence market sentiment (New Moon = accumulation, Full Moon = distribution)
    - WuXing (Five Elements) theory applied to crypto (Fire element)
    - Heavenly Stems provide additional timing signals

    The feng_shui_score combines all factors into a single indicator (-10 to +10).
    """

    # Strategy interface version
    INTERFACE_VERSION = 3

    # This strategy is long-only (HODL style)
    can_short: bool = False

    # Minimal ROI - Hold for longer periods aligned with lunar cycles
    # Lunar month is ~29.5 days, so we set generous ROI targets
    minimal_roi = {
        "0": 0.15,      # 15% profit target immediately
        "720": 0.10,    # 10% after 12 hours
        "1440": 0.05,   # 5% after 1 day
        "4320": 0.02,   # 2% after 3 days
        "10080": 0.01,  # 1% after 7 days (quarter moon cycle)
    }

    # Conservative stoploss - trust the cosmos, but verify
    stoploss = -0.15

    # Trailing stop to protect lunar gains
    trailing_stop = True
    trailing_stop_positive = 0.05
    trailing_stop_positive_offset = 0.10
    trailing_only_offset_is_reached = True

    # 4-hour timeframe - gives time for celestial energies to manifest
    timeframe = "4h"

    # Process only new candles for efficiency
    process_only_new_candles = True

    # Use exit signals from lunar phases
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # Hyperoptable parameters for fine-tuning cosmic alignment
    entry_threshold = DecimalParameter(
        3.0, 8.0, default=5.0, space="buy", optimize=True, load=True,
        decimals=1
    )
    exit_threshold = DecimalParameter(
        -8.0, -3.0, default=-5.0, space="sell", optimize=True, load=True,
        decimals=1
    )
    moon_weight = DecimalParameter(
        0.5, 2.0, default=1.0, space="buy", optimize=True, load=True,
        decimals=1
    )
    element_weight = DecimalParameter(
        0.5, 2.0, default=1.0, space="buy", optimize=True, load=True,
        decimals=1
    )

    # Startup candles needed (minimal, as lunar data is always available)
    startup_candle_count: int = 1

    # WuXing Element Scores for Crypto (Fire Element) Trading
    # Based on the generating (生) and overcoming (克) cycles
    WUXING_SCORES = {
        "木": 3,    # Wood FEEDS Fire → Very Bullish
        "火": 2,    # Fire is crypto's element → Bullish (self-strengthening)
        "土": 0,    # Fire creates Earth → Neutral (energy transformation)
        "金": -2,   # Fire melts Metal → Bearish (destructive cycle)
        "水": -3,   # Water DOUSES Fire → Very Bearish
    }

    # English translations for logging
    WUXING_NAMES = {
        "木": "Wood",
        "火": "Fire",
        "土": "Earth",
        "金": "Metal",
        "水": "Water",
    }

    # Heavenly Stems (天干) scores - Yang stems are more bullish
    TIANGAN_SCORES = {
        "甲": 2,   # Yang Wood - Strong growth
        "乙": 1,   # Yin Wood - Gentle growth
        "丙": 3,   # Yang Fire - Maximum bullish
        "丁": 2,   # Yin Fire - Bullish
        "戊": 0,   # Yang Earth - Stable
        "己": 0,   # Yin Earth - Stable
        "庚": -1,  # Yang Metal - Slight bearish
        "辛": -1,  # Yin Metal - Slight bearish
        "壬": -2,  # Yang Water - Bearish
        "癸": -2,  # Yin Water - Bearish
    }

    # Plot configuration for visualization
    plot_config = {
        "main_plot": {},
        "subplots": {
            "Moon Phase": {
                "moon_phase": {"color": "silver", "type": "line"},
                "full_moon_line": {"color": "yellow", "type": "line"},
            },
            "Feng Shui Score": {
                "feng_shui_score": {"color": "purple", "type": "bar"},
                "entry_line": {"color": "green", "type": "line"},
                "exit_line": {"color": "red", "type": "line"},
            },
            "WuXing": {
                "wuxing_score": {"color": "orange", "type": "bar"},
            },
        },
    }

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        if not LUNAR_AVAILABLE:
            logger.warning(
                "⚠️  lunar_python library not found! "
                "Install with: pip install lunar_python"
            )

    def _get_lunar_data(self, dt: datetime) -> dict:
        """
        Extract lunar calendar data for a given datetime.

        Returns a dictionary with:
        - moon_phase: Lunar day (1-30)
        - lunar_month: Lunar month (1-12, with leap months)
        - wuxing: Day's dominant element (五行)
        - tiangan: Heavenly Stem of the day (天干)
        - dizhi: Earthly Branch of the day (地支)
        - is_auspicious: Whether the day is traditionally auspicious
        """
        if not LUNAR_AVAILABLE:
            # Fallback: Simple moon phase calculation without library
            # Synodic month = 29.53059 days
            # Reference New Moon: Jan 6, 2000 18:14 UTC
            reference = datetime(2000, 1, 6, 18, 14, tzinfo=timezone.utc)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            days_since = (dt - reference).total_seconds() / 86400
            moon_phase = int(days_since % 29.53059) + 1

            # Fallback element cycle (simplified 5-day cycle)
            elements = ["木", "火", "土", "金", "水"]
            day_of_year = dt.timetuple().tm_yday
            wuxing = elements[day_of_year % 5]

            return {
                "moon_phase": moon_phase,
                "lunar_month": dt.month,  # Approximate
                "wuxing": wuxing,
                "tiangan": "甲",  # Default
                "dizhi": "子",    # Default
                "is_auspicious": moon_phase in [1, 2, 3, 8, 15, 16, 23],
            }

        # Use lunar_python for accurate calculations
        try:
            solar = Solar.fromYmdHms(
                dt.year, dt.month, dt.day,
                dt.hour, dt.minute, dt.second
            )
            lunar = solar.getLunar()

            # Get the day's GanZhi (干支)
            day_ganzhi = lunar.getDayInGanZhi()
            tiangan = day_ganzhi[0] if day_ganzhi else "甲"
            dizhi = day_ganzhi[1] if len(day_ganzhi) > 1 else "子"

            # Get WuXing from the day's Nayin (纳音五行) or stem
            # The day stem's element is more direct
            stem_elements = {
                "甲": "木", "乙": "木",
                "丙": "火", "丁": "火",
                "戊": "土", "己": "土",
                "庚": "金", "辛": "金",
                "壬": "水", "癸": "水",
            }
            wuxing = stem_elements.get(tiangan, "土")

            # Check for auspicious days (simplified)
            # Traditional: 1st, 15th are significant; 8 is lucky
            moon_phase = lunar.getDay()
            is_auspicious = moon_phase in [1, 2, 3, 8, 15, 16, 23, 30]

            return {
                "moon_phase": moon_phase,
                "lunar_month": lunar.getMonth(),
                "wuxing": wuxing,
                "tiangan": tiangan,
                "dizhi": dizhi,
                "is_auspicious": is_auspicious,
            }

        except Exception as e:
            logger.error(f"Error getting lunar data: {e}")
            return {
                "moon_phase": 15,
                "lunar_month": 1,
                "wuxing": "土",
                "tiangan": "甲",
                "dizhi": "子",
                "is_auspicious": False,
            }

    def _calculate_moon_score(self, moon_phase: int) -> float:
        """
        Calculate trading score based on moon phase.

        New Moon (Day 1-3): Maximum bullish - new cycle, accumulation
        Waxing Crescent (Day 4-7): Bullish - growing energy
        First Quarter (Day 8-10): Moderately bullish - momentum building
        Waxing Gibbous (Day 11-14): Slightly bullish - approaching peak
        Full Moon (Day 15-16): Neutral to slightly bearish - peak reached
        Waning Gibbous (Day 17-21): Bearish - energy declining
        Last Quarter (Day 22-25): Moderately bearish - contraction
        Waning Crescent (Day 26-29/30): Slightly bearish - cycle ending

        Returns score from -5 to +5
        """
        if moon_phase <= 3:
            # New Moon - Maximum bullish
            return 5.0
        elif moon_phase <= 7:
            # Waxing Crescent - Bullish
            return 4.0 - (moon_phase - 4) * 0.5
        elif moon_phase <= 10:
            # First Quarter - Moderately bullish
            return 2.5 - (moon_phase - 8) * 0.3
        elif moon_phase <= 14:
            # Waxing Gibbous - Slightly bullish
            return 1.5 - (moon_phase - 11) * 0.3
        elif moon_phase <= 16:
            # Full Moon - Peak, neutral to slight bearish
            return 0.0 - (moon_phase - 15) * 0.5
        elif moon_phase <= 21:
            # Waning Gibbous - Bearish
            return -1.0 - (moon_phase - 17) * 0.6
        elif moon_phase <= 25:
            # Last Quarter - Moderately bearish
            return -4.0 + (moon_phase - 22) * 0.2
        else:
            # Waning Crescent - Slightly bearish (approaching new cycle)
            return -3.0 + (moon_phase - 26) * 0.5

    def _calculate_element_score(self, wuxing: str, tiangan: str) -> float:
        """
        Calculate trading score based on WuXing (Five Elements) theory.

        Crypto is associated with Fire (火) element:
        - Volatile, transformative, associated with technology and speculation
        - Wood feeds Fire (bullish)
        - Water douses Fire (bearish)

        Returns score from -5 to +5
        """
        wuxing_score = self.WUXING_SCORES.get(wuxing, 0)
        tiangan_score = self.TIANGAN_SCORES.get(tiangan, 0)

        # Combine with 60% weight on element, 40% on stem
        combined = (wuxing_score * 0.6 + tiangan_score * 0.4) * 1.5

        # Clamp to -5 to +5
        return max(-5.0, min(5.0, combined))

    def _calculate_feng_shui_score(
        self, moon_phase: int, wuxing: str, tiangan: str, is_auspicious: bool
    ) -> float:
        """
        Calculate the composite Feng Shui trading score.

        Combines:
        - Moon phase score (weighted by moon_weight)
        - Element score (weighted by element_weight)
        - Auspicious day bonus

        Returns score from -10 to +10
        """
        moon_score = self._calculate_moon_score(moon_phase) * self.moon_weight.value
        element_score = self._calculate_element_score(wuxing, tiangan) * self.element_weight.value

        # Auspicious day bonus
        auspicious_bonus = 1.0 if is_auspicious else 0.0

        # Combine scores
        total = moon_score + element_score + auspicious_bonus

        # Clamp to -10 to +10
        return max(-10.0, min(10.0, total))

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Populate lunar and feng shui indicators for each candle.

        Adds:
        - moon_phase: Lunar day (1-30)
        - wuxing: Five Element of the day (Chinese character)
        - wuxing_english: Five Element (English)
        - tiangan: Heavenly Stem
        - moon_score: Score from moon phase alone
        - wuxing_score: Score from elements alone
        - feng_shui_score: Combined cosmic trading score
        - full_moon_line: Reference line at day 15
        - entry_line: Entry threshold line
        - exit_line: Exit threshold line
        """
        # Initialize columns
        dataframe["moon_phase"] = 0
        dataframe["wuxing"] = ""
        dataframe["wuxing_english"] = ""
        dataframe["tiangan"] = ""
        dataframe["moon_score"] = 0.0
        dataframe["wuxing_score"] = 0.0
        dataframe["feng_shui_score"] = 0.0
        dataframe["is_auspicious"] = False

        # Calculate lunar data for each candle
        for idx in dataframe.index:
            # Get the candle's datetime
            candle_time = dataframe.loc[idx, "date"]
            if isinstance(candle_time, pd.Timestamp):
                dt = candle_time.to_pydatetime()
            else:
                dt = candle_time

            # Get lunar data
            lunar_data = self._get_lunar_data(dt)

            # Store raw data
            dataframe.loc[idx, "moon_phase"] = lunar_data["moon_phase"]
            dataframe.loc[idx, "wuxing"] = lunar_data["wuxing"]
            dataframe.loc[idx, "wuxing_english"] = self.WUXING_NAMES.get(
                lunar_data["wuxing"], "Unknown"
            )
            dataframe.loc[idx, "tiangan"] = lunar_data["tiangan"]
            dataframe.loc[idx, "is_auspicious"] = lunar_data["is_auspicious"]

            # Calculate component scores
            moon_score = self._calculate_moon_score(lunar_data["moon_phase"])
            element_score = self._calculate_element_score(
                lunar_data["wuxing"], lunar_data["tiangan"]
            )
            feng_shui_score = self._calculate_feng_shui_score(
                lunar_data["moon_phase"],
                lunar_data["wuxing"],
                lunar_data["tiangan"],
                lunar_data["is_auspicious"],
            )

            dataframe.loc[idx, "moon_score"] = moon_score
            dataframe.loc[idx, "wuxing_score"] = element_score
            dataframe.loc[idx, "feng_shui_score"] = feng_shui_score

        # Add reference lines for plotting
        dataframe["full_moon_line"] = 15  # Full moon reference
        dataframe["entry_line"] = self.entry_threshold.value
        dataframe["exit_line"] = self.exit_threshold.value

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Generate entry signals based on feng shui score.

        Enter long when:
        - Feng Shui score exceeds entry threshold (default: 5)
        - This typically occurs during:
          - New Moon phase (days 1-3)
          - Wood or Fire element days
          - Auspicious days in the lunar calendar
        """
        dataframe.loc[
            (
                (dataframe["feng_shui_score"] >= self.entry_threshold.value)
                & (dataframe["volume"] > 0)  # Ensure there's market activity
            ),
            "enter_long",
        ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Generate exit signals based on feng shui score.

        Exit long when:
        - Feng Shui score drops below exit threshold (default: -5)
        - This typically occurs during:
          - Full Moon to Waning Moon (days 15-25)
          - Water or Metal element days
          - Generally inauspicious timing
        """
        dataframe.loc[
            (
                (dataframe["feng_shui_score"] <= self.exit_threshold.value)
                & (dataframe["volume"] > 0)
            ),
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
        entry_tag: Optional[str],
        side: str,
        **kwargs,
    ) -> bool:
        """
        Additional trade entry confirmation.

        Logs the cosmic conditions at entry for transparency.
        """
        lunar_data = self._get_lunar_data(current_time)
        logger.info(
            f"🌙 LUNAR ENTRY for {pair}: "
            f"Moon Day {lunar_data['moon_phase']}, "
            f"Element: {self.WUXING_NAMES.get(lunar_data['wuxing'], 'Unknown')} "
            f"({lunar_data['wuxing']}), "
            f"Stem: {lunar_data['tiangan']}, "
            f"Auspicious: {lunar_data['is_auspicious']}"
        )
        return True

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
        Additional trade exit confirmation.

        Logs the cosmic conditions at exit.
        """
        lunar_data = self._get_lunar_data(current_time)
        logger.info(
            f"🌑 LUNAR EXIT for {pair}: "
            f"Moon Day {lunar_data['moon_phase']}, "
            f"Element: {self.WUXING_NAMES.get(lunar_data['wuxing'], 'Unknown')}, "
            f"Reason: {exit_reason}"
        )
        return True
