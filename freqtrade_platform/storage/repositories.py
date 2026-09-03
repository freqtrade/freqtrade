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
        self._records: dict[str, PlatformProfileRecord] = {}

    def add(self, record: PlatformProfileRecord) -> PlatformProfileRecord:
        self._records[record.profile_id] = record
        if self.session is not None and hasattr(self.session, "add"):
            self.session.add(record)
        return record

    def get(self, profile_id: str) -> PlatformProfileRecord | None:
        if self.session is not None and hasattr(self.session, "query"):
            return self.session.query(PlatformProfileRecord).filter_by(profile_id=profile_id).one_or_none()
        return self._records.get(profile_id)

    def list(self) -> list[PlatformProfileRecord]:
        if self.session is not None and hasattr(self.session, "query"):
            return list(self.session.query(PlatformProfileRecord).all())
        return list(self._records.values())

    def remove(self, profile_id: str) -> None:
        self._records.pop(profile_id, None)
        if self.session is not None and hasattr(self.session, "query"):
            record = self.session.query(PlatformProfileRecord).filter_by(profile_id=profile_id).one_or_none()
            if record is not None:
                self.session.delete(record)


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
