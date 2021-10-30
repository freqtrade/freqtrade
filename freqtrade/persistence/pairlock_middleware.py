import logging
from datetime import datetime, timezone
from typing import List, Optional

from freqtrade.exchange import timeframe_to_next_date
from freqtrade.persistence.models import PairLock


logger = logging.getLogger(__name__)


class PairLocks():
    """
    Pairlocks middleware class
    Abstracts the database layer away so it becomes optional - which will be necessary to support
    backtesting and hyperopt in the future.
    """

    use_db = True
    locks: List[PairLock] = []

    timeframe: str = ''

    @staticmethod
    def reset_locks() -> None:
        """
        Resets all locks. Only active for backtesting mode.
        """
        if not PairLocks.use_db:
            PairLocks.locks = []

    @staticmethod
    def lock_pair(pair: str, until: datetime, reason: str = None, *,
                  now: datetime = None) -> PairLock:
        """
        Create PairLock from now to "until".
        Uses database by default, unless PairLocks.use_db is set to False,
        in which case a list is maintained.
        :param pair: pair to lock. use '*' to lock all pairs
        :param until: End time of the lock. Will be rounded up to the next candle.
        :param reason: Reason string that will be shown as reason for the lock
        :param now: Current timestamp. Used to determine lock start time.
        """
        lock = PairLock(
            pair=pair,
            lock_time=now or datetime.now(timezone.utc),
            lock_end_time=timeframe_to_next_date(PairLocks.timeframe, until),
            reason=reason,
            active=True
        )
        if PairLocks.use_db:
            PairLock.query.session.add(lock)
            PairLock.query.session.commit()
        else:
            PairLocks.locks.append(lock)
        return lock

    @staticmethod
    def get_pair_locks(pair: Optional[str], now: Optional[datetime] = None) -> List[PairLock]:
        """
        Get all currently active locks for this pair
        :param pair: Pair to check for. Returns all current locks if pair is empty
        :param now: Datetime object (generated via datetime.now(timezone.utc)).
                    defaults to datetime.now(timezone.utc)
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
    def get_pair_longest_lock(pair: str, now: Optional[datetime] = None) -> Optional[PairLock]:
        """
        Get the lock that expires the latest for the pair given.
        """
        locks = PairLocks.get_pair_locks(pair, now)
        locks = sorted(locks, key=lambda l: l.lock_end_time, reverse=True)
        return locks[0] if locks else None

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
            PairLock.query.session.commit()

    @staticmethod
    def unlock_reason(reason: str, now: Optional[datetime] = None) -> None:
        """
        Release all locks for this reason.
        :param reason: Which reason to unlock
        :param now: Datetime object (generated via datetime.now(timezone.utc)).
            defaults to datetime.now(timezone.utc)
        """
        if not now:
            now = datetime.now(timezone.utc)

        if PairLocks.use_db:
            # used in live modes
            logger.info(f"Releasing all locks with reason '{reason}':")
            filters = [PairLock.lock_end_time > now,
                       PairLock.active.is_(True),
                       PairLock.reason == reason
                       ]
            locks = PairLock.query.filter(*filters)
            for lock in locks:
                logger.info(f"Releasing lock for {lock.pair} with reason '{reason}'.")
                lock.active = False
            PairLock.query.session.commit()
        else:
            # used in backtesting mode; don't show log messages for speed
            locks = PairLocks.get_pair_locks(None)
            for lock in locks:
                if lock.reason == reason:
                    lock.active = False

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

    @staticmethod
    def get_all_locks() -> List[PairLock]:
        """
        Return all locks, also locks with expired end date
        """
        if PairLocks.use_db:
            return PairLock.query.all()
        else:
            return PairLocks.locks
