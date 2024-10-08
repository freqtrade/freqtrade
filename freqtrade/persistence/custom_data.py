import json
import logging
from collections.abc import Sequence
from datetime import datetime
from typing import Any, ClassVar, Optional

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, UniqueConstraint, select
from sqlalchemy.orm import Mapped, mapped_column, relationship

from freqtrade.constants import DATETIME_PRINT_FORMAT
from freqtrade.persistence.base import ModelBase, SessionType
from freqtrade.util import dt_now


logger = logging.getLogger(__name__)


class _CustomData(ModelBase):
    """
    CustomData database model
    Keeps records of metadata as key/value store
    for trades or global persistent values
    One to many relationship with Trades:
      - One trade can have many metadata entries
      - One metadata entry can only be associated with one Trade
    """

    __tablename__ = "trade_custom_data"
    __allow_unmapped__ = True
    session: ClassVar[SessionType]

    # Uniqueness should be ensured over pair, order_id
    # its likely that order_id is unique per Pair on some exchanges.
    __table_args__ = (UniqueConstraint("ft_trade_id", "cd_key", name="_trade_id_cd_key"),)

    id = mapped_column(Integer, primary_key=True)
    ft_trade_id = mapped_column(Integer, ForeignKey("trades.id"), index=True)

    trade = relationship("Trade", back_populates="custom_data")

    cd_key: Mapped[str] = mapped_column(String(255), nullable=False)
    cd_type: Mapped[str] = mapped_column(String(25), nullable=False)
    cd_value: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=dt_now)
    updated_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    # Empty container value - not persisted, but filled with cd_value on query
    value: Any = None

    def __repr__(self):
        create_time = (
            self.created_at.strftime(DATETIME_PRINT_FORMAT) if self.created_at is not None else None
        )
        update_time = (
            self.updated_at.strftime(DATETIME_PRINT_FORMAT) if self.updated_at is not None else None
        )
        return (
            f"CustomData(id={self.id}, key={self.cd_key}, type={self.cd_type}, "
            + f"value={self.cd_value}, trade_id={self.ft_trade_id}, created={create_time}, "
            + f"updated={update_time})"
        )

    @classmethod
    def query_cd(
        cls, key: Optional[str] = None, trade_id: Optional[int] = None
    ) -> Sequence["_CustomData"]:
        """
        Get all CustomData, if trade_id is not specified
        return will be for generic values not tied to a trade
        :param trade_id: id of the Trade
        """
        filters = []
        if trade_id is not None:
            filters.append(_CustomData.ft_trade_id == trade_id)
        if key is not None:
            filters.append(_CustomData.cd_key.ilike(key))

        return _CustomData.session.scalars(select(_CustomData).filter(*filters)).all()


class CustomDataWrapper:
    """
    CustomData middleware class
    Abstracts the database layer away so it becomes optional - which will be necessary to support
    backtesting and hyperopt in the future.
    """

    use_db = True
    custom_data: list[_CustomData] = []
    unserialized_types = ["bool", "float", "int", "str"]

    @staticmethod
    def _convert_custom_data(data: _CustomData) -> _CustomData:
        if data.cd_type in CustomDataWrapper.unserialized_types:
            data.value = data.cd_value
            if data.cd_type == "bool":
                data.value = data.cd_value.lower() == "true"
            elif data.cd_type == "int":
                data.value = int(data.cd_value)
            elif data.cd_type == "float":
                data.value = float(data.cd_value)
        else:
            data.value = json.loads(data.cd_value)
        return data

    @staticmethod
    def reset_custom_data() -> None:
        """
        Resets all key-value pairs. Only active for backtesting mode.
        """
        if not CustomDataWrapper.use_db:
            CustomDataWrapper.custom_data = []

    @staticmethod
    def delete_custom_data(trade_id: int) -> None:
        _CustomData.session.query(_CustomData).filter(_CustomData.ft_trade_id == trade_id).delete()
        _CustomData.session.commit()

    @staticmethod
    def get_custom_data(*, trade_id: int, key: Optional[str] = None) -> list[_CustomData]:
        if CustomDataWrapper.use_db:
            filters = [
                _CustomData.ft_trade_id == trade_id,
            ]
            if key is not None:
                filters.append(_CustomData.cd_key.ilike(key))
            filtered_custom_data = _CustomData.session.scalars(
                select(_CustomData).filter(*filters)
            ).all()

        else:
            filtered_custom_data = [
                data_entry
                for data_entry in CustomDataWrapper.custom_data
                if (data_entry.ft_trade_id == trade_id)
            ]
            if key is not None:
                filtered_custom_data = [
                    data_entry
                    for data_entry in filtered_custom_data
                    if (data_entry.cd_key.casefold() == key.casefold())
                ]
        return [CustomDataWrapper._convert_custom_data(d) for d in filtered_custom_data]

    @staticmethod
    def set_custom_data(trade_id: int, key: str, value: Any) -> None:
        value_type = type(value).__name__

        if value_type not in CustomDataWrapper.unserialized_types:
            try:
                value_db = json.dumps(value)
            except TypeError as e:
                logger.warning(f"could not serialize {key} value due to {e}")
                return
        else:
            value_db = str(value)

        if trade_id is None:
            trade_id = 0

        custom_data = CustomDataWrapper.get_custom_data(trade_id=trade_id, key=key)
        if custom_data:
            data_entry = custom_data[0]
            data_entry.cd_value = value_db
            data_entry.updated_at = dt_now()
        else:
            data_entry = _CustomData(
                ft_trade_id=trade_id,
                cd_key=key,
                cd_type=value_type,
                cd_value=value_db,
                created_at=dt_now(),
            )
        data_entry.value = value

        if CustomDataWrapper.use_db and value_db is not None:
            _CustomData.session.add(data_entry)
            _CustomData.session.commit()
        else:
            if not custom_data:
                CustomDataWrapper.custom_data.append(data_entry)
            # Existing data will have updated interactively.
