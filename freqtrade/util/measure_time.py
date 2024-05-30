import logging
import time
from typing import Callable

from cachetools import TTLCache


logger = logging.getLogger(__name__)


class MeasureTime:
    """
    Measure the time of a block of code and call a callback if the time limit is exceeded.
    """

    def __init__(
        self, callback: Callable[[float, float], None], time_limit: float, ttl: int = 3600 * 4
    ):
        """
        :param callback: The callback to call if the time limit is exceeded.
            This callback will be called once every "ttl" seconds,
            with the parameters "duration" (in seconds) and
            "time limit" - representing the passed in time limit.
        :param time_limit: The time limit in seconds.
        :param ttl: The time to live of the cache in seconds.
            defaults to 4 hours.
        """
        self._callback = callback
        self._time_limit = time_limit
        self.__cache: TTLCache = TTLCache(maxsize=1, ttl=ttl)

    def __enter__(self):
        self._start = time.time()

    def __exit__(self, *args):
        end = time.time()
        if self.__cache.get("value"):
            return
        duration = end - self._start

        if duration < self._time_limit:
            return
        self._callback(duration, self._time_limit)

        self.__cache["value"] = True
