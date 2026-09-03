"""Trading profile domain namespace."""

from freqtrade_platform.profiles.manager import TradingProfileManager
from freqtrade_platform.profiles.models import TradingProfile, build_default_profile
from freqtrade_platform.profiles.repository import TradingProfileRepository

__all__ = ["TradingProfile", "TradingProfileManager", "TradingProfileRepository", "build_default_profile"]
