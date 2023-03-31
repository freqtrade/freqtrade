from datetime import datetime, timezone
from typing import Any, ClassVar, Dict, Optional

from sqlalchemy import ScalarResult, String, or_, select
from sqlalchemy.orm import Mapped, mapped_column

from freqtrade.constants import DATETIME_PRINT_FORMAT
from freqtrade.persistence.base import ModelBase, SessionType


class PairLock(ModelBase):
    """
    Pair Locks database model.
    """
    __tablename__ = 'pairlocks'
    session: ClassVar[SessionType]

    id: Mapped[int] = mapped_column(primary_key=True)

    pair: Mapped[str] = mapped_column(String(25), nullable=False, index=True)
    # lock direction - long, short or * (for both)
    side: Mapped[str] = mapped_column(String(25), nullable=False, default="*")
    reason: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    # Time the pair was locked (start time)
    lock_time: Mapped[datetime] = mapped_column(nullable=False)
    # Time until the pair is locked (end time)
    lock_end_time: Mapped[datetime] = mapped_column(nullable=False, index=True)

    active: Mapped[bool] = mapped_column(nullable=False, default=True, index=True)

    def __repr__(self) -> str:
        lock_time = self.lock_time.strftime(DATETIME_PRINT_FORMAT)
        lock_end_time = self.lock_end_time.strftime(DATETIME_PRINT_FORMAT)
        return (
            f'PairLock(id={self.id}, pair={self.pair}, side={self.side}, lock_time={lock_time}, '
            f'lock_end_time={lock_end_time}, reason={self.reason}, active={self.active})')

    @staticmethod
    def query_pair_locks(
            pair: Optional[str], now: datetime, side: str = '*') -> ScalarResult['PairLock']:
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

        return PairLock.session.scalars(select(PairLock).filter(*filters))

    @staticmethod
    def get_all_locks() -> ScalarResult['PairLock']:
        return PairLock.session.scalars(select(PairLock))

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
