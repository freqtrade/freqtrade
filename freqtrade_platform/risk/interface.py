"""Interface boundary for future safety and risk orchestration."""

from __future__ import annotations

from abc import ABC, abstractmethod

from freqtrade_platform.risk.models import SafetyGuardPolicy


class SafetyGuard(ABC):
    """Account-level safety boundary for future enforcement of global rules."""

    @abstractmethod
    def evaluate(self, policy: SafetyGuardPolicy, context: object) -> bool:
        """Evaluate whether the platform is allowed to continue under the current state."""
