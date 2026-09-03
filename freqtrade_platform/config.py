"""Platform configuration boundary.

The purpose of this module is to keep platform-level settings separate from Freqtrade's config
and prevent them from being bolted directly into the main Freqtrade configuration schema.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class PlatformConfig:
    """Top-level platform config container for future runtime settings."""

    app_name: str = "platform"
    profiles: list[dict[str, Any]] = field(default_factory=list)
    capital_allocation: list[dict[str, Any]] = field(default_factory=list)
    strategy_assignments: list[dict[str, Any]] = field(default_factory=list)
    safety_policy: dict[str, Any] = field(default_factory=dict)
    storage: dict[str, Any] = field(default_factory=dict)
    adapter: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "PlatformConfig":
        """Create a platform config object from a mapping without touching Freqtrade config."""
        return cls(
            app_name=data.get("app_name", "platform"),
            profiles=data.get("profiles", []),
            capital_allocation=data.get("capital_allocation", []),
            strategy_assignments=data.get("strategy_assignments", []),
            safety_policy=data.get("safety_policy", {}),
            storage=data.get("storage", {}),
            adapter=data.get("adapter", {}),
        )
