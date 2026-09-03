"""Capital allocation boundary for future portfolio-level budget control."""

from __future__ import annotations

from freqtrade_platform.capital.models import CapitalAllocation


class CapitalAllocator:
    """Coordinates allocations without altering Freqtrade's wallet computations."""

    def __init__(self) -> None:
        self._allocations: dict[str, CapitalAllocation] = {}

    def validate(self, allocation: CapitalAllocation) -> None:
        allocation.validate()
        current_total = sum(item.allocation_percent for item in self._allocations.values())
        if allocation.profile_id in self._allocations:
            current_total -= self._allocations[allocation.profile_id].allocation_percent
        if current_total + allocation.allocation_percent > 100.0:
            raise ValueError("total allocation must not exceed 100%")

    def add(self, allocation: CapitalAllocation) -> CapitalAllocation:
        self.validate(allocation)
        self._allocations[allocation.profile_id] = allocation
        return allocation

    def update(self, profile_id: str, allocation_percent: float) -> CapitalAllocation:
        allocation = self._allocations[profile_id]
        new_value = CapitalAllocation(profile_id=profile_id, allocation_percent=allocation_percent)
        self.validate(new_value)
        self._allocations[profile_id] = new_value
        return new_value

    def remove(self, profile_id: str) -> None:
        self._allocations.pop(profile_id, None)

    def total(self) -> float:
        return sum(item.allocation_percent for item in self._allocations.values())

    def remaining(self) -> float:
        return max(100.0 - self.total(), 0.0)

    def list(self) -> list[CapitalAllocation]:
        return list(self._allocations.values())

    def get(self, profile_id: str) -> CapitalAllocation | None:
        return self._allocations.get(profile_id)
