from datetime import datetime
from typing import ClassVar, Optional, Self, Sequence

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, UniqueConstraint, select
from sqlalchemy.orm import Mapped, mapped_column, relationship

from freqtrade.constants import DATETIME_PRINT_FORMAT
from freqtrade.persistence.base import ModelBase, SessionType
from freqtrade.util import dt_now


class CustomData(ModelBase):
    """
    CustomData database model
    Keeps records of metadata as key/value store
    for trades or global persistant values
    One to many relationship with Trades:
      - One trade can have many metadata entries
      - One metadata entry can only be associated with one Trade
    """
    __tablename__ = 'trade_custom_data'
    session: ClassVar[SessionType]

    # Uniqueness should be ensured over pair, order_id
    # its likely that order_id is unique per Pair on some exchanges.
    __table_args__ = (UniqueConstraint('ft_trade_id', 'cd_key', name="_trade_id_cd_key"),)

    id = mapped_column(Integer, primary_key=True)
    ft_trade_id = mapped_column(Integer, ForeignKey('trades.id'), index=True, default=0)

    trade = relationship("Trade", back_populates="custom_data")

    cd_key: Mapped[str] = mapped_column(String(255), nullable=False)
    cd_type: Mapped[str] = mapped_column(String(25), nullable=False)
    cd_value: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=dt_now)
    updated_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    def __repr__(self):
        create_time = (self.created_at.strftime(DATETIME_PRINT_FORMAT)
                       if self.created_at is not None else None)
        update_time = (self.updated_at.strftime(DATETIME_PRINT_FORMAT)
                       if self.updated_at is not None else None)
        return (f'CustomData(id={self.id}, key={self.cd_key}, type={self.cd_type}, ' +
                f'value={self.cd_value}, trade_id={self.ft_trade_id}, created={create_time}, ' +
                f'updated={update_time})')

    @classmethod
    def query_cd(cls, key: Optional[str] = None,
                 trade_id: Optional[int] = None) -> Sequence['CustomData']:
        """
        Get all CustomData, if trade_id is not specified
        return will be for generic values not tied to a trade
        :param trade_id: id of the Trade
        """
        filters = []
        if trade_id is not None:
            filters.append(CustomData.ft_trade_id == trade_id)
        if key is not None:
            filters.append(CustomData.cd_key.ilike(key))

        return CustomData.session.scalars(select(CustomData)).all()
