"""Repository abstractions for platform metadata persistence."""

from __future__ import annotations

from freqtrade_platform.storage.models import (
    AccountSnapshotRecord,
    CapitalAllocationRecord,
    PlatformProfileRecord,
    PlatformStrategyRecord,
    StrategyAssignmentRecord,
    StrategyPerformanceRecord,
)


class PlatformProfileRepository:
    """Repository for profile records."""

    def __init__(self, session: object | None = None) -> None:
        self.session = session

    def add(self, record: PlatformProfileRecord) -> PlatformProfileRecord:
        return record


class PlatformStrategyRepository:
    """Repository for strategy records."""

    def __init__(self, session: object | None = None) -> None:
        self.session = session

    def add(self, record: PlatformStrategyRecord) -> PlatformStrategyRecord:
        return record


class StrategyAssignmentRepository:
    """Repository for strategy assignment records."""

    def __init__(self, session: object | None = None) -> None:
        self.session = session

    def add(self, record: StrategyAssignmentRecord) -> StrategyAssignmentRecord:
        return record


class StrategyPerformanceRepository:
    """Repository for strategy performance records."""

    def __init__(self, session: object | None = None) -> None:
        self.session = session

    def add(self, record: StrategyPerformanceRecord) -> StrategyPerformanceRecord:
        return record


class AccountSnapshotRepository:
    """Repository for snapshot records."""

    def __init__(self, session: object | None = None) -> None:
        self.session = session

    def add(self, record: AccountSnapshotRecord) -> AccountSnapshotRecord:
        return record


class CapitalAllocationRepository:
    """Repository for capital allocation records."""

    def __init__(self, session: object | None = None) -> None:
        self.session = session

    def add(self, record: CapitalAllocationRecord) -> CapitalAllocationRecord:
        return record
