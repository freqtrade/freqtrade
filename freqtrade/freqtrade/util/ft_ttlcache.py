import time

from cachetools import TTLCache


class FtTTLCache(TTLCache):
    """
    A TTLCache with a different default timer to allow for easier mocking in tests.
    """

    def __init__(self, maxsize, ttl, timer=time.time, getsizeof=None):
        super().__init__(maxsize=maxsize, ttl=ttl, timer=timer, getsizeof=getsizeof)
