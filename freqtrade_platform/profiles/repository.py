"""Repository boundary for trading profile persistence."""

from __future__ import annotations

from sqlalchemy import select

from freqtrade_platform.profiles.models import TradingProfile
from freqtrade_platform.storage.database import PlatformDatabase
from freqtrade_platform.storage.models import PlatformProfileRecord


class TradingProfileRepository:
    """Persistence interface for profile metadata.

    The concrete storage implementation is purposely separated from the domain model.
    """

    def __init__(self, storage: PlatformDatabase | object | None = None) -> None:
        self._storage = storage
        self._profiles: dict[str, TradingProfile] | None = {} if storage is None else None

    @staticmethod
    def _profile_from_record(record: PlatformProfileRecord) -> TradingProfile:
        return TradingProfile(
            profile_id=record.profile_id,
            name=record.name,
            exchange=record.exchange,
            market_type=record.market_type,
            universe_id=record.universe_id,
            symbol_scope=[item for item in (record.symbol_scope or "").split(",") if item],
            capital_allocation=record.capital_allocation,
        )

    def _record_from_profile(self, profile: TradingProfile) -> PlatformProfileRecord:
        return PlatformProfileRecord(
            profile_id=profile.profile_id,
            name=profile.name,
            exchange=profile.exchange,
            market_type=profile.market_type,
            universe_id=profile.universe_id,
            symbol_scope=",".join(profile.symbol_scope) if profile.symbol_scope else None,
            capital_allocation=profile.capital_allocation,
        )

    def add(self, profile: TradingProfile) -> TradingProfile:
        if self._storage is None:
            self._profiles[profile.profile_id] = profile
            return profile

        with self._storage.session() as session:
            record = session.scalar(
                select(PlatformProfileRecord).where(PlatformProfileRecord.profile_id == profile.profile_id)
            )
            if record is None:
                record = self._record_from_profile(profile)
                session.add(record)
            else:
                record.name = profile.name
                record.exchange = profile.exchange
                record.market_type = profile.market_type
                record.universe_id = profile.universe_id
                record.symbol_scope = ",".join(profile.symbol_scope) if profile.symbol_scope else None
                record.capital_allocation = profile.capital_allocation
        return profile

    def get(self, profile_id: str) -> TradingProfile | None:
        if self._storage is None:
            return self._profiles.get(profile_id)

        with self._storage.session() as session:
            record = session.scalar(
                select(PlatformProfileRecord).where(PlatformProfileRecord.profile_id == profile_id)
            )
            if record is None:
                return None
            return self._profile_from_record(record)

    def list(self) -> list[TradingProfile]:
        if self._storage is None:
            return list(self._profiles.values())

        with self._storage.session() as session:
            records = session.scalars(select(PlatformProfileRecord)).all()
            return [self._profile_from_record(record) for record in records]

    def remove(self, profile_id: str) -> None:
        if self._storage is None:
            self._profiles.pop(profile_id, None)
            return

        with self._storage.session() as session:
            record = session.scalar(
                select(PlatformProfileRecord).where(PlatformProfileRecord.profile_id == profile_id)
            )
            if record is not None:
                session.delete(record)
