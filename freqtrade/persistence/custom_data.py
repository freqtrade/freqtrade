import json
import logging
from datetime import datetime
from typing import Any, ClassVar, List, Optional, Sequence

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, UniqueConstraint, select
from sqlalchemy.orm import Mapped, mapped_column, relationship

from freqtrade.constants import DATETIME_PRINT_FORMAT
from freqtrade.persistence.base import ModelBase, SessionType
from freqtrade.util import dt_now


logger = logging.getLogger(__name__)


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

        return CustomData.session.scalars(select(CustomData).filter(*filters)).all()


class CustomDataWrapper:
    """
    CustomData middleware class
    Abstracts the database layer away so it becomes optional - which will be necessary to support
    backtesting and hyperopt in the future.
    """

    use_db = True
    custom_data: List[CustomData] = []
    unserialized_types = ['bool', 'float', 'int', 'str']

    @staticmethod
    def reset_custom_data() -> None:
        """
        Resets all key-value pairs. Only active for backtesting mode.
        """
        if not CustomDataWrapper.use_db:
            CustomDataWrapper.custom_data = []

    @staticmethod
    def get_custom_data(key: Optional[str] = None,
                        trade_id: Optional[int] = None) -> CustomData:
        if trade_id is None:
            trade_id = 0

        if CustomDataWrapper.use_db:
            filtered_custom_data = []
            for data_entry in CustomData.query_cd(trade_id=trade_id, key=key):
                if data_entry.cd_type not in CustomDataWrapper.unserialized_types:
                    data_entry.cd_value = json.loads(data_entry.cd_value)
                filtered_custom_data.append(data_entry)
            return filtered_custom_data
        else:
            filtered_custom_data = [
                data_entry for data_entry in CustomDataWrapper.custom_data
                if (data_entry.ft_trade_id == trade_id)
            ]
            if key is not None:
                filtered_custom_data = [
                    data_entry for data_entry in filtered_custom_data
                    if (data_entry.cd_key.casefold() == key.casefold())
                ]
            return filtered_custom_data

    @staticmethod
    def set_custom_data(key: str, value: Any, trade_id: Optional[int] = None) -> None:

        value_type = type(value).__name__
        value_db = None

        if value_type not in CustomDataWrapper.unserialized_types:
            try:
                value_db = json.dumps(value)
            except TypeError as e:
                logger.warning(f"could not serialize {key} value due to {e}")
        else:
            value_db = str(value)

        if trade_id is None:
            trade_id = 0

        custom_data = CustomDataWrapper.get_custom_data(key=key, trade_id=trade_id)
        if custom_data:
            data_entry = custom_data[0]
            data_entry.cd_value = value_db
            data_entry.updated_at = dt_now()
        else:
            data_entry = CustomData(
                ft_trade_id=trade_id,
                cd_key=key,
                cd_type=value_type,
                cd_value=value_db,
                created_at=dt_now()
            )

        if CustomDataWrapper.use_db and value_db is not None:
            data_entry.cd_value = value_db
            CustomData.session.add(data_entry)
            CustomData.session.commit()
        elif not CustomDataWrapper.use_db:
            cd_index = -1
            for index, data_entry in enumerate(CustomDataWrapper.custom_data):
                if data_entry.ft_trade_id == trade_id and data_entry.cd_key == key:
                    cd_index = index
                    break

            if cd_index >= 0:
                data_entry.cd_type = value_type
                data_entry.cd_value = value_db
                data_entry.updated_at = dt_now()

                CustomDataWrapper.custom_data[cd_index] = data_entry
            else:
                CustomDataWrapper.custom_data.append(data_entry)

    @staticmethod
    def get_all_custom_data() -> List[CustomData]:

        if CustomDataWrapper.use_db:
            return list(CustomData.query_cd())
        else:
            return CustomDataWrapper.custom_data
