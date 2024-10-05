"""system specific and performance tuning"""

from freqtrade.system.asyncio_config import asyncio_setup
from freqtrade.system.gc_setup import gc_set_threshold


__all__ = ["asyncio_setup", "gc_set_threshold"]
