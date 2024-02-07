import time_machine

from freqtrade.util import PeriodicCache


def test_ttl_cache():

    with time_machine.travel("2021-09-01 05:00:00 +00:00") as t:

        cache = PeriodicCache(5, ttl=60)
        cache1h = PeriodicCache(5, ttl=3600)

        assert cache.timer() == 1630472400.0
        cache['a'] = 1235
        cache1h['a'] = 555123
        assert 'a' in cache
        assert 'a' in cache1h

        t.move_to("2021-09-01 05:00:59 +00:00")
        assert 'a' in cache
        assert 'a' in cache1h

        # Cache expired
        t.move_to("2021-09-01 05:01:00 +00:00")
        assert 'a' not in cache
        assert 'a' in cache1h

        t.move_to("2021-09-01 05:59:59 +00:00")
        assert 'a' not in cache
        assert 'a' in cache1h

        t.move_to("2021-09-01 06:00:00 +00:00")
        assert 'a' not in cache
        assert 'a' not in cache1h
