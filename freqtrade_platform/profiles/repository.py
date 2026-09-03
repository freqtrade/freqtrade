"""Repository boundary for trading profile persistence."""

from __future__ import annotations

from typing import Iterable

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
        self._profiles: dict[str, TradingProfile] = {}

    def _record_from_profile(self, profile: TradingProfile) -> PlatformProfileRecord:
        return PlatformProfileRecord(
            profile_id=profile.profile_id,
            name=profile.name,
            exchange=profile.exchange,
            market_type=profile.market_type,
            capital_allocation=profile.capital_allocation,
        )

    def add(self, profile: TradingProfile) -> TradingProfile:
        self._profiles[profile.profile_id] = profile
        if isinstance(self._storage, PlatformDatabase):
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
                    record.capital_allocation = profile.capital_allocation
        return profile

    def get(self, profile_id: str) -> TradingProfile | None:
        if isinstance(self._storage, PlatformDatabase):
            with self._storage.session() as session:
                record = session.scalar(
                    select(PlatformProfileRecord).where(PlatformProfileRecord.profile_id == profile_id)
                )
                if record is None:
                    return self._profiles.get(profile_id)
                return TradingProfile(
                    profile_id=record.profile_id,
                    name=record.name,
                    exchange=record.exchange,
                    market_type=record.market_type,
                    capital_allocation=record.capital_allocation,
                )
        return self._profiles.get(profile_id)

    def list(self) -> Iterable[TradingProfile]:
        if isinstance(self._storage, PlatformDatabase):
            with self._storage.session() as session:
                records = session.scalars(select(PlatformProfileRecord)).all()
                return [
                    TradingProfile(
                        profile_id=record.profile_id,
                        name=record.name,
                        exchange=record.exchange,
                        market_type=record.market_type,
                        capital_allocation=record.capital_allocation,
                    )
                    for record in records
                ]
        return list(self._profiles.values())

    def remove(self, profile_id: str) -> None:
        self._profiles.pop(profile_id, None)
        if isinstance(self._storage, PlatformDatabase):
            with self._storage.session() as session:
                record = session.scalar(
                    select(PlatformProfileRecord).where(PlatformProfileRecord.profile_id == profile_id)
                )
                if record is not None:
                    session.delete(record)
