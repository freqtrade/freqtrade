"""Risk configuration namespace for strategy and account protection."""

from freqtrade_platform.risk.interface import SafetyGuard
from freqtrade_platform.risk.models import SafetyGuardPolicy, StrategyRiskConfig

__all__ = ["StrategyRiskConfig", "SafetyGuardPolicy", "SafetyGuard"]
