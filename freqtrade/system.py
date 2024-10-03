"""System specific or performance tuning"""

import gc
import logging
import platform
import sys


logger = logging.getLogger(__name__)


def asyncio_setup() -> None:  # pragma: no cover
    # Set eventloop for win32 setups

    if sys.platform == "win32":
        import asyncio

        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


def gc_set_threshold():
    """
    Reduce number of GC runs to improve performance (explanation video)
    https://www.youtube.com/watch?v=p4Sn6UcFTOU

    """
    if platform.python_implementation() == "CPython":
        # allocs, g1, g2 = gc.get_threshold()
        gc.set_threshold(50_000, 500, 1000)
        logger.debug("Adjusting python allocations to reduce GC runs")
