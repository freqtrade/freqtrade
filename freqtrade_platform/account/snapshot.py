"""Snapshot helpers for real versus simulated account states."""

from __future__ import annotations

from dataclasses import dataclass

from freqtrade_platform.account.models import RealAccountSnapshot, SimulationAccount


@dataclass(slots=True)
class AccountSnapshotSet:
    """Container for paired real and simulated account states."""

    real: RealAccountSnapshot
    simulated: SimulationAccount | None = None

    @property
    def has_simulated(self) -> bool:
        return self.simulated is not None
