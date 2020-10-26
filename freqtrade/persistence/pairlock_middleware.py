

import logging
from datetime import datetime, timezone
from typing import List, Optional

from freqtrade.persistence.models import PairLock
from freqtrade.exchange import timeframe_to_next_date


logger = logging.getLogger(__name__)


class PairLocks():
    """
    Pairlocks intermediate class

    """

    use_db = True
    locks: List[PairLock] = []

    timeframe: str = ''

    @staticmethod
    def lock_pair(pair: str, until: datetime, reason: str = None) -> None:
        lock = PairLock(
            pair=pair,
            lock_time=datetime.now(timezone.utc),
            lock_end_time=timeframe_to_next_date(PairLocks.timeframe, until),
            reason=reason,
            active=True
        )
        if PairLocks.use_db:
            PairLock.session.add(lock)
            PairLock.session.flush()
        else:
            PairLocks.locks.append(lock)

    @staticmethod
    def get_pair_locks(pair: Optional[str], now: Optional[datetime] = None) -> List[PairLock]:
        """
        Get all currently active locks for this pair
        :param pair: Pair to check for. Returns all current locks if pair is empty
        :param now: Datetime object (generated via datetime.now(timezone.utc)).
                    defaults to datetime.utcnow()
        """
        if not now:
            now = datetime.now(timezone.utc)

        if PairLocks.use_db:
            return PairLock.query_pair_locks(pair, now).all()
        else:
            locks = [lock for lock in PairLocks.locks if (
                lock.lock_end_time >= now
                and lock.active is True
                and (pair is None or lock.pair == pair)
            )]
            return locks

    @staticmethod
    def unlock_pair(pair: str, now: Optional[datetime] = None) -> None:
        """
        Release all locks for this pair.
        :param pair: Pair to unlock
        :param now: Datetime object (generated via datetime.now(timezone.utc)).
            defaults to datetime.now(timezone.utc)
        """
        if not now:
            now = datetime.now(timezone.utc)

        logger.info(f"Releasing all locks for {pair}.")
        locks = PairLocks.get_pair_locks(pair, now)
        for lock in locks:
            lock.active = False
        if PairLocks.use_db:
            PairLock.session.flush()

    @staticmethod
    def is_global_lock(now: Optional[datetime] = None) -> bool:
        """
        :param now: Datetime object (generated via datetime.now(timezone.utc)).
            defaults to datetime.now(timezone.utc)
        """
        if not now:
            now = datetime.now(timezone.utc)

        return len(PairLocks.get_pair_locks('*', now)) > 0

    @staticmethod
    def is_pair_locked(pair: str, now: Optional[datetime] = None) -> bool:
        """
        :param pair: Pair to check for
        :param now: Datetime object (generated via datetime.now(timezone.utc)).
            defaults to datetime.now(timezone.utc)
        """
        if not now:
            now = datetime.now(timezone.utc)

        return len(PairLocks.get_pair_locks(pair, now)) > 0 or PairLocks.is_global_lock(now)
