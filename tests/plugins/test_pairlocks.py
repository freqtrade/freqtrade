from datetime import datetime, timedelta, timezone

import arrow
import pytest

from freqtrade.persistence import PairLocks
from freqtrade.persistence.models import PairLock


@pytest.mark.parametrize('use_db', (False, True))
@pytest.mark.usefixtures("init_persistence")
def test_PairLocks(use_db):
    PairLocks.timeframe = '5m'
    PairLocks.use_db = use_db
    # No lock should be present
    if use_db:
        assert len(PairLock.query.all()) == 0

    assert PairLocks.use_db == use_db

    pair = 'ETH/BTC'
    assert not PairLocks.is_pair_locked(pair)
    PairLocks.lock_pair(pair, arrow.utcnow().shift(minutes=4).datetime)
    # ETH/BTC locked for 4 minutes (on both sides)
    assert PairLocks.is_pair_locked(pair)
    assert PairLocks.is_pair_locked(pair, side='long')
    assert PairLocks.is_pair_locked(pair, side='short')

    pair = 'BNB/BTC'
    PairLocks.lock_pair(pair, arrow.utcnow().shift(minutes=4).datetime, side='long')
    assert not PairLocks.is_pair_locked(pair)
    assert PairLocks.is_pair_locked(pair, side='long')
    assert not PairLocks.is_pair_locked(pair, side='short')

    pair = 'BNB/USDT'
    PairLocks.lock_pair(pair, arrow.utcnow().shift(minutes=4).datetime, side='short')
    assert not PairLocks.is_pair_locked(pair)
    assert not PairLocks.is_pair_locked(pair, side='long')
    assert PairLocks.is_pair_locked(pair, side='short')

    # XRP/BTC should not be locked now
    pair = 'XRP/BTC'
    assert not PairLocks.is_pair_locked(pair)
    # Unlocking a pair that's not locked should not raise an error
    PairLocks.unlock_pair(pair)

    PairLocks.lock_pair(pair, arrow.utcnow().shift(minutes=4).datetime)
    assert PairLocks.is_pair_locked(pair)

    # Get both locks from above
    locks = PairLocks.get_pair_locks(None)
    assert len(locks) == 2

    # Unlock original pair
    pair = 'ETH/BTC'
    PairLocks.unlock_pair(pair)
    assert not PairLocks.is_pair_locked(pair)
    assert not PairLocks.is_global_lock()

    pair = 'BTC/USDT'
    # Lock until 14:30
    lock_time = datetime(2020, 5, 1, 14, 30, 0, tzinfo=timezone.utc)
    PairLocks.lock_pair(pair, lock_time)

    assert not PairLocks.is_pair_locked(pair)
    assert PairLocks.is_pair_locked(pair, lock_time + timedelta(minutes=-10))
    assert not PairLocks.is_global_lock(lock_time + timedelta(minutes=-10))
    assert PairLocks.is_pair_locked(pair, lock_time + timedelta(minutes=-50))
    assert not PairLocks.is_global_lock(lock_time + timedelta(minutes=-50))

    # Should not be locked after time expired
    assert not PairLocks.is_pair_locked(pair, lock_time + timedelta(minutes=10))

    locks = PairLocks.get_pair_locks(pair, lock_time + timedelta(minutes=-2))
    assert len(locks) == 1
    assert 'PairLock' in str(locks[0])

    # Unlock all
    PairLocks.unlock_pair(pair, lock_time + timedelta(minutes=-2))
    assert not PairLocks.is_global_lock(lock_time + timedelta(minutes=-50))

    # Global lock
    PairLocks.lock_pair('*', lock_time)
    assert PairLocks.is_global_lock(lock_time + timedelta(minutes=-50))
    # Global lock also locks every pair separately
    assert PairLocks.is_pair_locked(pair, lock_time + timedelta(minutes=-50))
    assert PairLocks.is_pair_locked('XRP/USDT', lock_time + timedelta(minutes=-50))

    if use_db:
        locks = PairLocks.get_all_locks()
        locks_db = PairLock.query.all()
        assert len(locks) == len(locks_db)
        assert len(locks_db) > 0
    else:
        # Nothing was pushed to the database
        assert len(PairLocks.get_all_locks()) > 0
        assert len(PairLock.query.all()) == 0
    # Reset use-db variable
    PairLocks.reset_locks()
    PairLocks.use_db = True


@pytest.mark.parametrize('use_db', (False, True))
@pytest.mark.usefixtures("init_persistence")
def test_PairLocks_getlongestlock(use_db):
    PairLocks.timeframe = '5m'
    # No lock should be present
    PairLocks.use_db = use_db
    if use_db:
        assert len(PairLock.query.all()) == 0

    assert PairLocks.use_db == use_db

    pair = 'ETH/BTC'
    assert not PairLocks.is_pair_locked(pair)
    PairLocks.lock_pair(pair, arrow.utcnow().shift(minutes=4).datetime)
    # ETH/BTC locked for 4 minutes
    assert PairLocks.is_pair_locked(pair)
    lock = PairLocks.get_pair_longest_lock(pair)

    assert lock.lock_end_time.replace(tzinfo=timezone.utc) > arrow.utcnow().shift(minutes=3)
    assert lock.lock_end_time.replace(tzinfo=timezone.utc) < arrow.utcnow().shift(minutes=14)

    PairLocks.lock_pair(pair, arrow.utcnow().shift(minutes=15).datetime)
    assert PairLocks.is_pair_locked(pair)

    lock = PairLocks.get_pair_longest_lock(pair)
    # Must be longer than above
    assert lock.lock_end_time.replace(tzinfo=timezone.utc) > arrow.utcnow().shift(minutes=14)

    PairLocks.reset_locks()
    PairLocks.use_db = True


@pytest.mark.parametrize('use_db', (False, True))
@pytest.mark.usefixtures("init_persistence")
def test_PairLocks_reason(use_db):
    PairLocks.timeframe = '5m'
    PairLocks.use_db = use_db
    # No lock should be present
    if use_db:
        assert len(PairLock.query.all()) == 0

    assert PairLocks.use_db == use_db

    PairLocks.lock_pair('XRP/USDT', arrow.utcnow().shift(minutes=4).datetime, 'TestLock1')
    PairLocks.lock_pair('ETH/USDT', arrow.utcnow().shift(minutes=4).datetime, 'TestLock2')

    assert PairLocks.is_pair_locked('XRP/USDT')
    assert PairLocks.is_pair_locked('ETH/USDT')

    PairLocks.unlock_reason('TestLock1')
    assert not PairLocks.is_pair_locked('XRP/USDT')
    assert PairLocks.is_pair_locked('ETH/USDT')

    PairLocks.reset_locks()
    PairLocks.use_db = True
