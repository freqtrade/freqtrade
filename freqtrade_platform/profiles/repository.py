"""Repository boundary for trading profile persistence."""

from __future__ import annotations

from typing import Iterable

from freqtrade_platform.profiles.models import TradingProfile


class TradingProfileRepository:
    """Persistence interface for profile metadata.

    The concrete storage implementation is purposely separated from the domain model.
    """

    def __init__(self, storage: object | None = None) -> None:
        self._storage = storage
        self._profiles: dict[str, TradingProfile] = {}

    def add(self, profile: TradingProfile) -> TradingProfile:
        self._profiles[profile.profile_id] = profile
        return profile

    def get(self, profile_id: str) -> TradingProfile | None:
        return self._profiles.get(profile_id)

    def list(self) -> Iterable[TradingProfile]:
        return list(self._profiles.values())

    def remove(self, profile_id: str) -> None:
        self._profiles.pop(profile_id, None)
