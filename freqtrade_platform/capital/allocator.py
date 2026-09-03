"""Capital allocation boundary for future portfolio-level budget control."""

from __future__ import annotations

from freqtrade_platform.capital.models import CapitalAllocation


class CapitalAllocator:
    """Coordinates allocations without altering Freqtrade's wallet computations."""

    def __init__(self) -> None:
        self._allocations: dict[str, CapitalAllocation] = {}

    def add(self, allocation: CapitalAllocation) -> CapitalAllocation:
        self._allocations[allocation.profile_id] = allocation
        return allocation

    def total(self) -> float:
        return sum(item.allocation_percent for item in self._allocations.values())

    def list(self) -> list[CapitalAllocation]:
        return list(self._allocations.values())
