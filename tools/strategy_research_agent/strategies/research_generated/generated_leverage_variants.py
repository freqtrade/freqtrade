"""Auto-generated isolated strategy variants.

Do not edit by hand. Re-generate with user_data/strategy_research/generate_variants.py.
These classes are research-only and must not be promoted to live without manual approval.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from btc_eth_risk_controlled_strategies import BtcEthFuturesEthSelfPullbackShortOnlyStrategy, BtcEthFuturesRegime10xPullbackShortOnlyStrategy


GENERATED_AT_UTC = '20260623T024630Z'


class GeneratedBtcEthFuturesEthSelfPullbackShortOnlyStrategyL3p0x(BtcEthFuturesEthSelfPullbackShortOnlyStrategy):
    """Research-only 3x leverage-cap variant of BtcEthFuturesEthSelfPullbackShortOnlyStrategy."""

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
        return min(3, max_leverage)


class GeneratedBtcEthFuturesEthSelfPullbackShortOnlyStrategyL5p0x(BtcEthFuturesEthSelfPullbackShortOnlyStrategy):
    """Research-only 5x leverage-cap variant of BtcEthFuturesEthSelfPullbackShortOnlyStrategy."""

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
        return min(5, max_leverage)


class GeneratedBtcEthFuturesEthSelfPullbackShortOnlyStrategyL10p0x(BtcEthFuturesEthSelfPullbackShortOnlyStrategy):
    """Research-only 10x leverage-cap variant of BtcEthFuturesEthSelfPullbackShortOnlyStrategy."""

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
        return min(10, max_leverage)


class GeneratedBtcEthFuturesRegime10xPullbackShortOnlyStrategyL3p0x(BtcEthFuturesRegime10xPullbackShortOnlyStrategy):
    """Research-only 3x leverage-cap variant of BtcEthFuturesRegime10xPullbackShortOnlyStrategy."""

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
        return min(3, max_leverage)


class GeneratedBtcEthFuturesRegime10xPullbackShortOnlyStrategyL5p0x(BtcEthFuturesRegime10xPullbackShortOnlyStrategy):
    """Research-only 5x leverage-cap variant of BtcEthFuturesRegime10xPullbackShortOnlyStrategy."""

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
        return min(5, max_leverage)


class GeneratedBtcEthFuturesRegime10xPullbackShortOnlyStrategyL10p0x(BtcEthFuturesRegime10xPullbackShortOnlyStrategy):
    """Research-only 10x leverage-cap variant of BtcEthFuturesRegime10xPullbackShortOnlyStrategy."""

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
        return min(10, max_leverage)

