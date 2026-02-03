"""system specific and performance tuning"""

from freqtrade.system.asyncio_config import asyncio_setup
from freqtrade.system.gc_setup import gc_set_threshold
from freqtrade.system.set_mp_start_method import set_mp_start_method
from freqtrade.system.version_info import print_version_info


__all__ = ["asyncio_setup", "gc_set_threshold", "print_version_info", "set_mp_start_method"]
