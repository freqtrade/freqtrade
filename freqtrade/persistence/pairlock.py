from datetime import datetime, timezone
from typing import Any, Dict, Optional

from sqlalchemy import Boolean, Column, DateTime, Integer, String, or_
from sqlalchemy.orm import Query

from freqtrade.constants import DATETIME_PRINT_FORMAT
from freqtrade.persistence.base import _DECL_BASE


class PairLock(_DECL_BASE):
    """
    Pair Locks database model.
    """
    __tablename__ = 'pairlocks'

    id = Column(Integer, primary_key=True)

    pair = Column(String(25), nullable=False, index=True)
    # lock direction - long, short or * (for both)
    side = Column(String(25), nullable=False, default="*")
    reason = Column(String(255), nullable=True)
    # Time the pair was locked (start time)
    lock_time = Column(DateTime, nullable=False)
    # Time until the pair is locked (end time)
    lock_end_time = Column(DateTime, nullable=False, index=True)

    active = Column(Boolean, nullable=False, default=True, index=True)

    def __repr__(self):
        lock_time = self.lock_time.strftime(DATETIME_PRINT_FORMAT)
        lock_end_time = self.lock_end_time.strftime(DATETIME_PRINT_FORMAT)
        return (
            f'PairLock(id={self.id}, pair={self.pair}, side={self.side}, lock_time={lock_time}, '
            f'lock_end_time={lock_end_time}, reason={self.reason}, active={self.active})')

    @staticmethod
    def query_pair_locks(pair: Optional[str], now: datetime, side: str = '*') -> Query:
        """
        Get all currently active locks for this pair
        :param pair: Pair to check for. Returns all current locks if pair is empty
        :param now: Datetime object (generated via datetime.now(timezone.utc)).
        """
        filters = [PairLock.lock_end_time > now,
                   # Only active locks
                   PairLock.active.is_(True), ]
        if pair:
            filters.append(PairLock.pair == pair)
        if side != '*':
            filters.append(or_(PairLock.side == side, PairLock.side == '*'))
        else:
            filters.append(PairLock.side == '*')

        return PairLock.query.filter(
            *filters
        )

    def to_json(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'pair': self.pair,
            'lock_time': self.lock_time.strftime(DATETIME_PRINT_FORMAT),
            'lock_timestamp': int(self.lock_time.replace(tzinfo=timezone.utc).timestamp() * 1000),
            'lock_end_time': self.lock_end_time.strftime(DATETIME_PRINT_FORMAT),
            'lock_end_timestamp': int(self.lock_end_time.replace(tzinfo=timezone.utc
                                                                 ).timestamp() * 1000),
            'reason': self.reason,
            'side': self.side,
            'active': self.active,
        }
