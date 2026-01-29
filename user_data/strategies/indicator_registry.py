"""
Indicator Registry
Single source of truth for strategy indicators, defaults, and configuration.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any


@dataclass
class IndicatorSpec:
    name: str = ""
    kind: str = ""  # e.g., 'ema', 'rsi'
    params: Dict[str, Any] = field(default_factory=dict)
    source: str = "underlying"  # 'self' or 'underlying'


class IndicatorRegistry:
    # Defaults
    DEFAULTS = {
        "STOCK_OPT": {
            "timeframe": "5m",
            "startup_candle_count": 50,
            "entry_window_ist": ["09:45", "14:30"],
            "rsi_bull": 55,
            "rsi_bear": 45,
            "ema_fast": 5,
            "ema_slow": 20,
            "stale_tolerance_seconds": 600,
        },
        # INDEX_OPT and AUTO_OPT inherit from STOCK_OPT unless overridden
    }

    @staticmethod
    def get_defaults(strategy_id: str) -> Dict[str, Any]:
        """Returns default configuration for a strategy ID."""
        base = IndicatorRegistry.DEFAULTS.get("STOCK_OPT", {}).copy()
        specific = IndicatorRegistry.DEFAULTS.get(strategy_id, {})
        base.update(specific)
        return base

    @staticmethod
    def get_startup_candle_count(strategy_id: str) -> int:
        """Returns the canonical startup candle count."""
        defaults = IndicatorRegistry.get_defaults(strategy_id)
        return defaults.get("startup_candle_count", 50)

    @staticmethod
    def get_required_indicators(strategy_id: str) -> List[IndicatorSpec]:
        """Returns list of required indicators."""
        defaults = IndicatorRegistry.get_defaults(strategy_id)

        return [
            IndicatorSpec(
                name="ema_fast",
                kind="ema",
                params={"period": defaults.get("ema_fast", 5)},
                source="underlying",
            ),
            IndicatorSpec(
                name="ema_slow",
                kind="ema",
                params={"period": defaults.get("ema_slow", 20)},
                source="underlying",
            ),
            IndicatorSpec(name="rsi", kind="rsi", params={"period": 14}, source="underlying"),
        ]
