"""Management boundary for trading profiles."""

from __future__ import annotations

from freqtrade_platform.profiles.models import TradingProfile


class TradingProfileManager:
    """Coordinates profile lifecycle operations without touching Freqtrade core."""

    def __init__(self, repository: object | None = None) -> None:
        self.repository = repository

    def validate(self, profile: TradingProfile) -> TradingProfile:
        profile._validate()
        return profile

    def add(self, profile: TradingProfile) -> TradingProfile:
        if self.repository is None:
            raise ValueError("repository is required for profile management")
        self.validate(profile)
        return self.repository.add(profile)

    def update(self, profile_id: str, **changes: object) -> TradingProfile:
        profile = self.get(profile_id)
        if profile is None:
            raise KeyError(profile_id)
        for key, value in changes.items():
            setattr(profile, key, value)
        self.validate(profile)
        self.repository.add(profile)
        return profile

    def delete(self, profile_id: str) -> None:
        if self.repository is not None:
            self.repository.remove(profile_id)

    def get(self, profile_id: str) -> TradingProfile | None:
        if self.repository is None:
            return None
        return self.repository.get(profile_id)

    def list(self) -> list[TradingProfile]:
        if self.repository is None:
            return []
        return list(self.repository.list())
