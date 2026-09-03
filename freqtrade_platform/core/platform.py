"""Top-level platform service boundary."""

from __future__ import annotations

from dataclasses import dataclass, field

from freqtrade_platform.core.context import PlatformContext
from freqtrade_platform.core.lifecycle import PlatformLifecycle


@dataclass(slots=True)
class Platform:
    """Root platform object that owns configuration, lifecycle, and adapters."""

    name: str = "platform"
    context: PlatformContext = field(default_factory=PlatformContext)
    lifecycle: PlatformLifecycle = field(default_factory=PlatformLifecycle)

    def initialize(self) -> None:
        """Prepare the platform without altering actual Freqtrade trading behavior."""
        self.lifecycle.state = self.lifecycle.state
