"""Repository abstractions for platform metadata persistence."""

from __future__ import annotations

from freqtrade_platform.storage.models import (
    AccountSnapshotRecord,
    CapitalAllocationRecord,
    PlatformProfileRecord,
    PlatformRuntimeRecord,
    PlatformStrategyRecord,
    PlatformStrategySourceRecord,
    PlatformUniverseRecord,
    StrategyAssignmentRecord,
    StrategyPerformanceRecord,
)


class PlatformUniverseRepository:
    """Repository for universe records."""

    def __init__(self, session: object | None = None) -> None:
        self.session = session
        self._records: dict[str, PlatformUniverseRecord] = {}

    def add(self, record: PlatformUniverseRecord) -> PlatformUniverseRecord:
        self._records[record.universe_id] = record
        if self.session is not None and hasattr(self.session, "add"):
            self.session.add(record)
        return record

    def get(self, universe_id: str) -> PlatformUniverseRecord | None:
        if self.session is not None and hasattr(self.session, "query"):
            return self.session.query(PlatformUniverseRecord).filter_by(universe_id=universe_id).one_or_none()
        return self._records.get(universe_id)

    def list(self) -> list[PlatformUniverseRecord]:
        if self.session is not None and hasattr(self.session, "query"):
            return list(self.session.query(PlatformUniverseRecord).all())
        return list(self._records.values())


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
        self._records: dict[str, PlatformStrategyRecord] = {}

    def add(self, record: PlatformStrategyRecord) -> PlatformStrategyRecord:
        self._records[record.strategy_id] = record
        if self.session is not None and hasattr(self.session, "add"):
            self.session.add(record)
        return record

    def get(self, strategy_id: str) -> PlatformStrategyRecord | None:
        if self.session is not None and hasattr(self.session, "query"):
            return self.session.query(PlatformStrategyRecord).filter_by(strategy_id=strategy_id).one_or_none()
        return self._records.get(strategy_id)

    def list(self) -> list[PlatformStrategyRecord]:
        if self.session is not None and hasattr(self.session, "query"):
            return list(self.session.query(PlatformStrategyRecord).all())
        return list(self._records.values())


class PlatformStrategySourceRepository:
    """Repository for strategy source code and metadata records."""

    def __init__(self, session: object | None = None) -> None:
        self.session = session
        self._records: dict[str, PlatformStrategySourceRecord] = {}

    def add(self, record: PlatformStrategySourceRecord) -> PlatformStrategySourceRecord:
        self._records[record.strategy_id] = record
        if self.session is not None and hasattr(self.session, "add"):
            self.session.add(record)
        return record

    def get(self, strategy_id: str) -> PlatformStrategySourceRecord | None:
        if self.session is not None and hasattr(self.session, "query"):
            return self.session.query(PlatformStrategySourceRecord).filter_by(strategy_id=strategy_id).one_or_none()
        return self._records.get(strategy_id)

    def list(self) -> list[PlatformStrategySourceRecord]:
        if self.session is not None and hasattr(self.session, "query"):
            return list(self.session.query(PlatformStrategySourceRecord).all())
        return list(self._records.values())

    def remove(self, strategy_id: str) -> None:
        self._records.pop(strategy_id, None)
        if self.session is not None and hasattr(self.session, "query"):
            record = self.session.query(PlatformStrategySourceRecord).filter_by(strategy_id=strategy_id).one_or_none()
            if record is not None:
                self.session.delete(record)


class PlatformRuntimeRepository:
    """Repository for runtime instance state records."""

    def __init__(self, session: object | None = None) -> None:
        self.session = session
        self._records: dict[str, PlatformRuntimeRecord] = {}

    def add(self, record: PlatformRuntimeRecord) -> PlatformRuntimeRecord:
        self._records[record.runtime_id] = record
        if self.session is not None and hasattr(self.session, "add"):
            self.session.add(record)
        return record

    def get(self, runtime_id: str) -> PlatformRuntimeRecord | None:
        if self.session is not None and hasattr(self.session, "query"):
            return self.session.query(PlatformRuntimeRecord).filter_by(runtime_id=runtime_id).one_or_none()
        return self._records.get(runtime_id)

    def list(self) -> list[PlatformRuntimeRecord]:
        if self.session is not None and hasattr(self.session, "query"):
            return list(self.session.query(PlatformRuntimeRecord).all())
        return list(self._records.values())

    def get_active_for_profile(self, profile_id: str) -> PlatformRuntimeRecord | None:
        active_states = {"READY", "STARTING", "RUNNING"}
        if self.session is not None and hasattr(self.session, "query"):
            records = (
                self.session.query(PlatformRuntimeRecord)
                .filter_by(profile_id=profile_id)
                .all()
            )
            for r in records:
                if r.state in active_states:
                    return r
            return None
        for r in self._records.values():
            if r.profile_id == profile_id and r.state in active_states:
                return r
        return None

    def remove(self, runtime_id: str) -> None:
        self._records.pop(runtime_id, None)
        if self.session is not None and hasattr(self.session, "query"):
            record = self.session.query(PlatformRuntimeRecord).filter_by(runtime_id=runtime_id).one_or_none()
            if record is not None:
                self.session.delete(record)


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
