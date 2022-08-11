from datetime import datetime, timezone

from cachetools import TTLCache


class PeriodicCache(TTLCache):
    """
    Special cache that expires at "straight" times
    A timer with ttl of 3600 (1h) will expire at every full hour (:00).
    """

    def __init__(self, maxsize, ttl, getsizeof=None):
        def local_timer():
            ts = datetime.now(timezone.utc).timestamp()
            offset = (ts % ttl)
            return ts - offset

        # Init with smlight offset
        super().__init__(maxsize=maxsize, ttl=ttl - 1e-5, timer=local_timer, getsizeof=getsizeof)
