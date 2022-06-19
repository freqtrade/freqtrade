from datetime import datetime
from typing import Optional

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Query, relationship

from freqtrade.constants import DATETIME_PRINT_FORMAT
from freqtrade.persistence.base import _DECL_BASE


class CustomData(_DECL_BASE):
    """
    CustomData database model
    Keeps records of metadata as key/value store
    for trades or global persistant values
    One to many relationship with Trades:
      - One trade can have many metadata entries
      - One metadata entry can only be associated with one Trade
    """
    __tablename__ = 'trade_custom_data'
    # Uniqueness should be ensured over pair, order_id
    # its likely that order_id is unique per Pair on some exchanges.
    __table_args__ = (UniqueConstraint('ft_trade_id', 'cd_key', name="_trade_id_cd_key"),)

    id = Column(Integer, primary_key=True)
    ft_trade_id = Column(Integer, ForeignKey('trades.id'), index=True, default=0)

    trade = relationship("Trade", back_populates="custom_data")

    cd_key = Column(String(255), nullable=False)
    cd_type = Column(String(25), nullable=False)
    cd_value = Column(Text, nullable=False)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=True)

    def __repr__(self):
        create_time = (self.created_at.strftime(DATETIME_PRINT_FORMAT)
                       if self.created_at is not None else None)
        update_time = (self.updated_at.strftime(DATETIME_PRINT_FORMAT)
                       if self.updated_at is not None else None)
        return (f'CustomData(id={self.id}, key={self.cd_key}, type={self.cd_type}, ' +
                f'value={self.cd_value}, trade_id={self.ft_trade_id}, created={create_time}, ' +
                f'updated={update_time})')

    @staticmethod
    def query_cd(key: Optional[str] = None, trade_id: Optional[int] = None) -> Query:
        """
        Get all CustomData, if trade_id is not specified
        return will be for generic values not tied to a trade
        :param trade_id: id of the Trade
        """
        filters = []
        filters.append(CustomData.ft_trade_id == trade_id if trade_id is not None else 0)
        if key is not None:
            filters.append(CustomData.cd_key.ilike(key))

        return CustomData.query.filter(*filters)
