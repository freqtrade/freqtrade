"""Capital allocation domain models."""

from __future__ import annotations

from dataclasses import dataclass, field

from freqtrade_platform.core.exceptions import PlatformValidationError


@dataclass(slots=True)
class CapitalAllocation:
    """Allocates platform capital across profile scopes without changing wallet math."""

    profile_id: str
    allocation_percent: float
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if not self.profile_id or not self.profile_id.strip():
            raise PlatformValidationError("profile_id is required")
        if not 0.0 <= float(self.allocation_percent) <= 100.0:
            raise PlatformValidationError("allocation_percent must be between 0 and 100")
