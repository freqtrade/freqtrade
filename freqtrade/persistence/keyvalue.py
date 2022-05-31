from datetime import datetime
from typing import Optional

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Query, relationship

from freqtrade.constants import DATETIME_PRINT_FORMAT
from freqtrade.persistence.base import _DECL_BASE


class KeyValue(_DECL_BASE):
    """
    KeyValue database model
    Keeps records of metadata as key/value store
    for trades or global persistant values
    One to many relationship with Trades:
      - One trade can have many metadata entries
      - One metadata entry can only be associated with one Trade
    """
    __tablename__ = 'keyvalue'
    # Uniqueness should be ensured over pair, order_id
    # its likely that order_id is unique per Pair on some exchanges.
    __table_args__ = (UniqueConstraint('ft_trade_id', 'kv_key', name="_trade_id_kv_key"),)

    id = Column(Integer, primary_key=True)
    ft_trade_id = Column(Integer, ForeignKey('trades.id'), index=True, default=0)

    trade = relationship("Trade", back_populates="keyvalues")

    kv_key = Column(String(255), nullable=False)
    kv_type = Column(String(25), nullable=False)
    kv_value = Column(Text, nullable=False)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=True)

    def __repr__(self):
        create_time = (self.created_at.strftime(DATETIME_PRINT_FORMAT)
                       if self.created_at is not None else None)
        update_time = (self.updated_at.strftime(DATETIME_PRINT_FORMAT)
                       if self.updated_at is not None else None)
        return (f'KeyValue(id={self.id}, key={self.kv_key}, type={self.kv_type}, ',
                f'value={self.kv_value}, trade_id={self.ft_trade_id}, created={create_time}, ',
                f'updated={update_time})')

    @staticmethod
    def query_kv(key: Optional[str] = None, trade_id: Optional[int] = None) -> Query:
        """
        Get all keyvalues, if  trade_id is not specified
        return will be for generic values not tied to a trade
        :param trade_id: id of the Trade
        """
        key = key if key is not None else "%"

        filters = [KeyValue.ft_trade_id == trade_id if trade_id is not None else 0,
                   KeyValue.kv_key.ilike(key)]

        return KeyValue.query.filter(*filters)
