"""Context container for platform-level runtime dependencies."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class PlatformContext:
    """Encapsulates platform dependencies without mutating Freqtrade runtime state."""

    app_name: str = "platform"
    config: dict[str, Any] = field(default_factory=dict)
    adapter: Any | None = None
    storage: Any | None = None

    def get(self, key: str, default: Any | None = None) -> Any:
        """Return a config value by key."""
        return self.config.get(key, default)
