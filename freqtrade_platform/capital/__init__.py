"""Capital allocation namespace."""

from freqtrade_platform.capital.allocator import CapitalAllocator
from freqtrade_platform.capital.models import CapitalAllocation

__all__ = ["CapitalAllocation", "CapitalAllocator"]
