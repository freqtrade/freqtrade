"""
Pairs a user adds at runtime for the UserPairList handler.

The table starts empty. Rows stay in the database across restarts.
"""

import re
from typing import ClassVar

from sqlalchemy import String, select
from sqlalchemy.orm import Mapped, mapped_column

from freqtrade.persistence.base import ModelBase, SessionType


# Exact pair only. Futures pairs look like BTC/USDT:USDT.
_PAIR_RE = re.compile(r"^[A-Za-z0-9]{1,25}/[A-Za-z0-9]{1,25}(:[A-Za-z0-9]{1,25})?$")


def is_valid_user_pair(pair: str) -> bool:
    """True when pair is a concrete pair, not a wildcard or free text."""
    return bool(_PAIR_RE.fullmatch(pair))


class UserPair(ModelBase):
    """One pair stored for UserPairList."""

    __tablename__ = "user_pairs"
    session: ClassVar[SessionType]

    id: Mapped[int] = mapped_column(primary_key=True)
    pair: Mapped[str] = mapped_column(String(50), nullable=False, unique=True, index=True)

    def __repr__(self) -> str:
        return f"UserPair(id={self.id}, pair={self.pair})"

    @staticmethod
    def list_pairs() -> list[str]:
        """Pairs in the order they were added."""
        rows = UserPair.session.scalars(select(UserPair).order_by(UserPair.id)).all()
        return [row.pair for row in rows]

    @staticmethod
    def add_pair(pair: str) -> bool:
        """
        Store a pair.
        :return: False when the pair is already stored.
        """
        existing = UserPair.session.scalars(
            select(UserPair).where(UserPair.pair == pair).limit(1)
        ).first()
        if existing:
            return False
        UserPair.session.add(UserPair(pair=pair))
        UserPair.session.commit()
        return True

    @staticmethod
    def delete_pair(pair: str) -> bool:
        """
        Remove a pair.
        :return: False when the pair is not stored.
        """
        row = UserPair.session.scalars(
            select(UserPair).where(UserPair.pair == pair).limit(1)
        ).first()
        if row is None:
            return False
        UserPair.session.delete(row)
        UserPair.session.commit()
        return True
