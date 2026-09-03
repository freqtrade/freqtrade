"""SQLite metadata persistence for the platform layer."""

from freqtrade_platform.storage.database import PlatformDatabase
from freqtrade_platform.storage.models import (
    AccountSnapshotRecord,
    CapitalAllocationRecord,
    PlatformProfileRecord,
    PlatformStrategyRecord,
    PlatformUniverseRecord,
    StrategyAssignmentRecord,
    StrategyPerformanceRecord,
)

__all__ = [
    "PlatformDatabase",
    "PlatformProfileRecord",
    "PlatformStrategyRecord",
    "PlatformUniverseRecord",
    "StrategyAssignmentRecord",
    "StrategyPerformanceRecord",
    "AccountSnapshotRecord",
    "CapitalAllocationRecord",
]
